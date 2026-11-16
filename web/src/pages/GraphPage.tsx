import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { errText, fmt, get, send, type Catalog, type DatasetInfo, type Graph, type GraphKind, type GraphNode, type Job, type Target } from '../api'
import GraphEditor, { type NodeStatus } from '../components/GraphEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'
import { Empty, ErrorNote, PageHead, Pill } from '../components/ui'

const TEXT = {
  ingestion: { title: 'Ingestion pipelines', path: '/ingestion', noun: 'ingestion pipeline', job: 'ingest',
    blurb: 'Source → chunk → embed → target. Running a pipeline writes its chunks to the target, appending or overwriting as the target node says. Branch to fill several targets from one source.' },
  retrieval: { title: 'Retrieval pipelines', path: '/retrieval', noun: 'retrieval pipeline', job: 'benchmark',
    blurb: 'Target → retrieve → rerank → generate → benchmark. Running a pipeline answers the dataset\'s questions from the target and scores the result.' },
}
// Steps of a run in order, and which node each one belongs to.
const STEPS: Record<GraphKind, [string, string][]> = {
  ingestion: [['load', 'source'], ['transform', 'transform'], ['chunk', 'chunk'], ['embed', 'embed'], ['write', 'target']],
  retrieval: [['retrieve', 'retrieve'], ['rerank', 'rerank'], ['generate', 'generate'], ['evaluate', 'benchmark']],
}

type Summary = { job: Job; step?: string; index: number; total: number }

// One run's part on the canvas: a status per node, and a one-line summary for the banner.
// `job` is a job with its run's steps (from /api/jobs/<id> or the pipeline's history).
function runStatus(kind: GraphKind, graph: Graph, job: Job): { status: Record<string, NodeStatus>; summary: Summary; steps: Record<string, string> } {
  const logged = new Map((job.run?.stages ?? []).map(s => [s.stage, s]))
  const current = job.run?.current_stage ?? null
  const steps = STEPS[kind].filter(([, stage]) => graph.nodes.some(n => n.stage === stage))
  const firstMissing = steps.find(([step]) => !logged.has(step))?.[0]
  // The failed step: recorded by the run; older runs: the first step that never finished. None if it failed before any step.
  const failed = !job.run_id ? undefined : job.run?.failed_stage ?? firstMissing
  const status: Record<string, NodeStatus> = {}
  const nodeStep: Record<string, string> = {}     // node id -> the step of this run it shows
  let currentLabel: string | undefined
  steps.forEach(([step, stage]) => {
    const s = logged.get(step)
    const state: NodeStatus['state'] = s ? (s.cached ? 'cached' : 'done')
      : job.status === 'queued' ? 'queued'
      : job.status === 'running' && (current === step || (!current && step === firstMissing)) ? 'running'
      : job.status === 'failed' && step === failed ? 'failed'
      : job.status === 'cancelled' && step === firstMissing ? 'cancelled' : 'waiting'
    if (state === 'running' || state === 'failed') currentLabel = step
    const note = s ? `${s.cached ? 'cached' : `${fmt(s.duration_s, 1)}s`}` : undefined
    // Ingestion: the write step belongs to the target node this job writes to; others to every node of the stage.
    const nodes = stage === 'target' && kind === 'ingestion' ? graph.nodes.filter(n => n.id === job.spec?.meta?.node_id)
      : graph.nodes.filter(n => n.stage === stage)
    nodes.forEach(n => { status[n.id] = { state, note }; nodeStep[n.id] = step })
  })
  if (kind === 'retrieval') graph.nodes.filter(n => n.stage === 'target').forEach(n => { status[n.id] = { state: job.status === 'queued' ? 'queued' : 'done' } })
  const idx = steps.findIndex(([step]) => step === currentLabel)
  return { status, summary: { job, step: currentLabel, index: idx + 1, total: steps.length }, steps: nodeStep }
}

// Every run of this pipeline (newest first), refreshed whenever a job changes state,
// with the active run followed live.
function useHistory(kind: GraphKind, graph: Graph | undefined, jobs: Job[]) {
  const id = graph?.id
  const mine = id ? jobs.filter(j => j.spec?.meta?.pipeline_id === id) : []
  const active = mine.find(isActive)
  const [history, setHistory] = useState<Job[]>([])
  const [live, setLive] = useState<Job>()
  const jobState = mine.map(j => `${j.id}:${j.status}`).join()
  useEffect(() => {
    if (!id) { setHistory([]); return }
    get<Job[]>(`/api/graphs/${kind}/${id}/history`).then(setHistory).catch(() => setHistory([]))
  }, [kind, id, jobState])
  useEffect(() => {
    if (!active) { setLive(undefined); return }
    let stop = false
    const load = () => get<Job>(`/api/jobs/${active.id}`).then(d => { if (!stop) setLive(d) }).catch(() => undefined)
    load()
    const t = setInterval(load, 1500)
    return () => { stop = true; clearInterval(t) }
  }, [active?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  return history.map(h => live && h.id === live.id ? { ...live, spec: h.spec } : h)
}

const strip = (g: Graph) => JSON.stringify({ n: g.name, d: g.description, nodes: g.nodes, edges: g.edges })

function blank(kind: GraphKind, dataset: string, target: string): Graph {
  const node = (id: string, impl: string, params: Record<string, unknown>, i: number): GraphNode => ({ id, stage: id, impl, params, position: { x: 290 * i, y: 60 } })
  const nodes = kind === 'ingestion'
    ? [node('source', 'dataset', { dataset, subset: 'all', snapshot: 'latest' }, 0), node('chunk', 'fixed', {}, 1),
       node('embed', 'sentence_transformers', {}, 2), node('target', 'target', { target_id: target, mode: 'append' }, 3)]
    : [node('target', 'target', { target_id: target }, 0), node('retrieve', 'hybrid', {}, 1), node('benchmark', 'benchmark', { limit: 20, judge: 'auto' }, 2)]
  return { name: '', description: '', nodes, edges: nodes.slice(1).map((n, i) => ({ source: nodes[i].id, target: n.id })) }
}

export default function GraphPage({ kind }: { kind: GraphKind }) {
  const { id } = useParams()
  const nav = useNavigate()
  const t = TEXT[kind]
  const [catalog, setCatalog] = useState<Catalog>()
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [graphs, setGraphs] = useState<Graph[] | null>(null)
  const [graph, setGraph] = useState<Graph>()          // the one being edited
  const [saved, setSaved] = useState('')               // its last-saved content, to detect unsaved edits
  const [error, setError] = useState<string>()
  const [note, setNote] = useState<string>()
  const [runTargets, setRunTargets] = useState<string[] | null>(null)   // retrieval: targets ticked in the run menu
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length
  const history = useHistory(kind, id === 'new' ? undefined : graph, jobs)
  const [viewing, setViewing] = useState<string | null>(null)       // a past run shown on the canvas; null = the latest
  const [logStep, setLogStep] = useState<string | null>(null)       // the step whose log is open
  const shown = (viewing && history.find(h => h.id === viewing)) || history[0]
  const live = graph && shown ? runStatus(kind, graph, shown) : undefined
  const failedStep = (j: Job) => {
    if (!graph || j.status !== 'failed') return null
    return runStatus(kind, graph, j).summary.step ?? 'setup'
  }
  useEffect(() => { setViewing(null); setLogStep(null) }, [graph?.id])
  // When the latest run fails, open its failed step's log.
  const latestKey = `${history[0]?.id}:${history[0]?.status}`
  useEffect(() => { if (history[0]?.status === 'failed') { setViewing(null); setLogStep(failedStep(history[0])) } }, [latestKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const show = (jobId: string, step?: string | null) => {
    const j = history.find(h => h.id === jobId)
    setViewing(jobId === history[0]?.id ? null : jobId)
    setLogStep(step !== undefined ? step : j ? failedStep(j) : null)
  }
  const pick = (nodeId: string | null) => {
    setSelected(nodeId)
    const step = nodeId ? live?.steps[nodeId] : undefined
    if (step && live?.status[nodeId!]?.state !== 'waiting') setLogStep(step)
  }

  const reload = useCallback(() => get<Graph[]>(`/api/graphs/${kind}`).then(setGraphs).catch(e => setError(errText(e))), [kind])
  useEffect(() => {
    get<Catalog>('/api/catalog').then(setCatalog)
    get<DatasetInfo[]>('/api/datasets').then(setDatasets).catch(() => undefined)
    reload()
  }, [reload])
  useEffect(() => { get<Target[]>('/api/targets').then(setTargets).catch(() => undefined) }, [busy])

  // Open the pipeline named in the URL; without one the page lists every pipeline.
  useEffect(() => {
    if (id === 'new') return
    if (!id) { setGraph(undefined); return }
    if (!graphs) return
    const g = graphs.find(x => x.id === id)
    if (!g) { nav(t.path, { replace: true }); return }
    setGraph(g); setSaved(strip(g)); setRunTargets(null); setNote(undefined)
  }, [id, graphs, nav, t.path])

  const dirty = graph !== undefined && strip(graph) !== saved
  const latest = useRef<Graph | undefined>(undefined)       // the graph as the editor last reported it
  latest.current = graph
  const [selected, setSelected] = useState<string | null>(null)   // kept here so a save (which remounts the editor) keeps it
  const save = async (override?: Graph): Promise<Graph | undefined> => {
    const body = override ?? graph
    if (!body) return
    try {
      const g = await send<Graph>('PUT', `/api/graphs/${kind}`, body)
      setError(undefined); setGraph(g); setSaved(strip(g)); await reload()
      nav(`${t.path}/${g.id}`, { replace: true })
      return g
    } catch (e) { setError(errText(e)) }
  }
  const create = () => {
    const name = window.prompt(`Name of the new ${t.noun}:`)
    if (!name) return
    setGraph({ ...blank(kind, datasets.find(d => d.n_questions)?.name ?? '', targets[0]?.id ?? ''), name }); setSaved(''); nav(`${t.path}/new`)
  }
  const duplicate = () => graph && (setGraph({ ...graph, id: undefined, name: `${graph.name} copy` }), setSaved(''), nav(`${t.path}/new`))
  const del = async () => {
    if (!graph?.id || !window.confirm(`Delete “${graph.name}”? Its past runs are kept.`)) return
    await send('DELETE', `/api/graphs/${kind}/${graph.id}`)
    setGraph(undefined); await reload(); nav(t.path)
  }

  const ownTarget = String(graph?.nodes.find(n => n.stage === 'target')?.params.target_id ?? '')
  const chosen = runTargets ?? (ownTarget ? [ownTarget] : [])
  const run = async () => {
    setNote(undefined)
    const g = dirty ? await save() : graph          // a run always uses the saved pipeline
    if (!g?.id) return
    try {
      const body = kind === 'ingestion' ? { pipeline_id: g.id } : { pipeline_id: g.id, target_ids: chosen }
      const out = await send<Job[]>('POST', `/api/jobs/${t.job}`, body)
      setError(undefined); setNote(`${out.length} job${out.length === 1 ? '' : 's'} queued`)
    } catch (e) { setError(errText(e)) }
  }

  if (!catalog || graphs === null) return <p className="muted">Loading…</p>
  if (!id) return (
    <>
      <PageHead title={t.title} sub={t.blurb}><button className="primary" onClick={create}>New pipeline</button></PageHead>
      {datasets.filter(d => d.n_questions).length === 0 && <div className="notice warn" style={{ marginBottom: 12 }}>No dataset is ready yet. Create one on the <Link to="/datasets">Datasets</Link> page first.</div>}
      <PipelineList kind={kind} graphs={graphs} jobs={jobs} targets={targets} onOpen={g => nav(`${t.path}/${g.id}`)} onCreate={create} />
      <div className="section-title">Recent runs <span className="muted">· <Link to="/jobs">all jobs</Link></span></div>
      <JobList jobs={jobs} kinds={[t.job]} limit={5} />
    </>)
  const nTargets = graph?.nodes.filter(n => n.stage === 'target').length ?? 0
  return (
    <>
      <PageHead title={graph?.name || t.title} sub={graph?.description || t.blurb}>
        <button className="ghost" onClick={() => nav(t.path)}>All pipelines</button>
        <button onClick={create}>New pipeline</button>
      </PageHead>
      {datasets.filter(d => d.n_questions).length === 0 && <div className="notice warn" style={{ marginBottom: 12 }}>No dataset is ready yet. Create one on the <Link to="/datasets">Datasets</Link> page first.</div>}

      {!graph ? <Empty>No {t.noun}s yet. <a href="#" onClick={e => { e.preventDefault(); create() }}>Create one</a>.</Empty> : <>
        <div className="card" style={{ marginBottom: 12, padding: '10px 12px' }}>
          <div className="row">
            <select value={id === 'new' ? '' : graph.id ?? ''} onChange={e => nav(`${t.path}/${e.target.value}`)} style={{ maxWidth: 220 }}>
              {id === 'new' && <option value="">(unsaved)</option>}
              {graphs.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
            </select>
            <input value={graph.name} style={{ width: 190 }} aria-label="Pipeline name" onChange={e => setGraph({ ...graph, name: e.target.value })} />
            <input className="grow" value={graph.description} placeholder="Description" style={{ minWidth: 140 }} onChange={e => setGraph({ ...graph, description: e.target.value })} />
            <button className="ghost sm" onClick={duplicate}>Duplicate</button>
            <button className="ghost sm danger" onClick={del} disabled={!graph.id}>Delete</button>
            <button onClick={() => save()} disabled={!dirty}>{dirty ? 'Save' : 'Saved'}</button>
            <button className="primary" onClick={run} disabled={kind === 'ingestion' ? nTargets === 0 : chosen.length === 0}>
              {kind === 'ingestion' ? `▶ Run ingestion${nTargets > 1 ? ` (${nTargets} targets)` : ''}` : `▶ Run benchmark${chosen.length > 1 ? ` (${chosen.length} targets)` : ''}`}</button>
          </div>
          {kind === 'retrieval' && targets.length > 1 && (
            <div className="row small" style={{ marginTop: 8 }}><span className="muted">Benchmark against:</span>
              {targets.map(tg => (
                <label key={tg.id} className={tg.version ? '' : 'faint'}><input type="checkbox" disabled={!tg.version} checked={chosen.includes(tg.id!)}
                  onChange={() => setRunTargets(chosen.includes(tg.id!) ? chosen.filter(x => x !== tg.id) : [...chosen, tg.id!])} /> {tg.name}{tg.version ? '' : ' (empty)'}</label>))}
            </div>)}
          {note && <div className="small muted" style={{ marginTop: 6 }}>{note}{dirty ? '' : ' · unsaved edits are saved before a run'}</div>}
        </div>
        <ErrorNote error={error} />
        {live && <RunBanner s={live.summary} past={viewing !== null && viewing !== history[0]?.id} onLatest={() => show(history[0].id)} />}

        <GraphEditor status={live?.status ?? {}} key={`${kind}:${graph.id ?? 'new'}:${graph.updated_at ?? ''}`} kind={kind} graph={graph} catalog={catalog} datasets={datasets} targets={targets}
          selected={selected} onSelect={pick}
          onSaveScript={(nodeId, code, patch) => {      // save one node's script now: the whole pipeline is saved, recording a version
            const g = latest.current!
            return save({ ...g, nodes: g.nodes.map(n => n.id === nodeId ? { ...n, ...patch, params: { ...(patch?.params ?? n.params), code } } : n) })
          }}
          onChange={setGraph} extra={(n, g) => kind !== 'ingestion' ? null : n.stage === 'chunk' ? <ChunkPreview node={n} graph={g} />
            : n.stage === 'transform' ? <TransformPreview node={n} graph={g} /> : null} />

        {shown && logStep && <StepLog job={shown} step={logStep} onClose={() => setLogStep(null)} />}
        {kind === 'retrieval' && <TryQuery graph={graph} targets={targets.filter(tg => tg.version)} defaultTarget={ownTarget} />}

        <div className="section-title">Run history <span className="muted">· click a run to show it on the pipeline, a step to read its log · <Link to="/jobs">all jobs</Link></span></div>
        <RunHistory kind={kind} graph={graph} history={history} shown={shown?.id}
          onShow={show} />
      </>}
    </>
  )
}

const STAGE_ORDER = ['source', 'transform', 'chunk', 'embed', 'target', 'retrieve', 'rerank', 'generate', 'benchmark']

// Every pipeline of this kind, with what it does at a glance and how its last run went.
function PipelineList({ kind, graphs, jobs, targets, onOpen, onCreate }: {
  kind: GraphKind; graphs: Graph[]; jobs: Job[]; targets: Target[]; onOpen: (g: Graph) => void; onCreate: () => void
}) {
  if (graphs.length === 0) return <Empty>No {TEXT[kind].noun}s yet. <a href="#" onClick={e => { e.preventDefault(); onCreate() }}>Create one</a>.</Empty>
  const targetName = (tid: unknown) => targets.find(tg => tg.id === tid)?.name ?? String(tid || '?')
  const label = (n: GraphNode) => n.stage === 'source' ? String(n.params.dataset || 'no dataset') + (n.params.subset && n.params.subset !== 'all' ? ` · ${n.params.subset}` : '')
    : n.stage === 'target' ? targetName(n.params.target_id) : n.stage === 'benchmark' ? `${n.params.limit ? `${n.params.limit} questions` : 'all questions'}` : n.impl
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>pipeline</th><th>{kind === 'ingestion' ? 'source → steps → targets' : 'target → steps'}</th><th>last run</th><th>updated</th></tr></thead>
      <tbody>{graphs.map(g => {
        const nodes = [...g.nodes].sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage) || a.position.y - b.position.y)
        const last = jobs.find(j => (j as Job & { spec?: { meta?: { pipeline_id?: string } } }).spec?.meta?.pipeline_id === g.id)
        return (
          <tr key={g.id} className="clickable" onClick={() => onOpen(g)}>
            <td><Link to={`${TEXT[kind].path}/${g.id}`} onClick={e => e.stopPropagation()}><b>{g.name}</b></Link>
              {g.description && <div className="small muted">{g.description}</div>}</td>
            <td><div className="chain">{nodes.map((n, i) => <span key={n.id} className="chain-step">{i > 0 && <span className="faint">→</span>}
              <span className="chip" title={`${n.stage}: ${n.impl}`}><span className="faint">{n.stage}</span> {label(n)}</span></span>)}</div></td>
            <td className="small">{last ? <><Pill status={last.status} />
              <div className="faint">{(last.finished_at ?? last.created_at).replace('T', ' ').slice(0, 16)}</div></> : <span className="faint">never run</span>}</td>
            <td className="small faint">{g.updated_at ? g.updated_at.replace('T', ' ').slice(0, 16) : '–'}</td>
          </tr>)})}</tbody>
    </table></div>)
}

function RunBanner({ s, past, onLatest }: { s: Summary; past: boolean; onLatest: () => void }) {
  const j = s.job
  const back = past ? <button className="sm ghost" onClick={onLatest}>Show latest run</button> : null
  const since = (t?: string) => t ? Math.max(0, Math.round((Date.now() - new Date(t).getTime()) / 1000)) : 0
  const dur = (sec: number) => sec >= 60 ? `${Math.floor(sec / 60)}m ${sec % 60}s` : `${sec}s`
  // The run's own progress lines are indented or start with "[stage]"; skip library warnings.
  const last = (j.log ?? '').split('\n').filter(l => /^\s{2,}\S|^\s*\[/.test(l)).slice(-1)[0]?.trim()
  const runLink = j.run_id ? <Link to={`/runs/${j.run_id}`}>{j.run_id}</Link> : null
  if (j.status === 'queued') return <div className="run-banner queued"><span className="pill queued">queued</span> {j.title} · waiting for the job before it to finish</div>
  if (j.status === 'running') return (
    <div className="run-banner running"><span className="pill running">running</span>
      <span><b>{j.title}</b> · step <b>{s.step ?? (s.index === 0 && (j.run?.stages?.length ?? 0) > 0 ? 'finishing' : 'starting')}</b>{s.index > 0 ? ` (${s.index} of ${s.total})` : ''} · {dur(since(j.started_at))}</span>
      {last && <span className="small muted mono banner-log">{last}</span>}</div>)
  if (j.status === 'failed') return (
    <div className="run-banner failed"><span className="pill failed">failed</span>
      <span><b>{j.title}</b>{s.step ? <> failed at step <b>{s.step}</b></> : ' failed before its first step'}: {j.error}</span>
      <span className="small">{runLink}{past ? ` · ${(j.finished_at ?? j.created_at).replace('T', ' ').slice(0, 16)}` : ''}</span>{back}</div>)
  return (
    <div className={`run-banner ${j.status}`}><span className={`pill ${j.status}`}>{j.status === 'ok' ? (past ? 'ok' : 'last run ok') : j.status}</span>
      <span className="small">{past ? 'Showing an earlier run: ' : ''}{j.title} · finished {(j.finished_at ?? '').replace('T', ' ').slice(0, 16)}</span>
      {runLink && <span className="small">{runLink}{kind(j) === 'benchmark' && j.run_id ? <> · <Link to={`/runs/${j.run_id}`}>results →</Link></> : null}</span>}{back}</div>)
}
const kind = (j: Job) => j.kind

const when = (j: Job) => (j.started_at ?? j.created_at).replace('T', ' ').slice(0, 16)
const took = (j: Job) => {
  if (!j.started_at) return '–'
  const sec = Math.round(((j.finished_at ? new Date(j.finished_at) : new Date()).getTime() - new Date(j.started_at).getTime()) / 1000)
  return sec >= 60 ? `${Math.floor(sec / 60)}m ${sec % 60}s` : `${sec}s`
}

// Every run of the pipeline: when, how it went, and its steps as chips coloured by state.
function RunHistory({ kind, graph, history, shown, onShow }: {
  kind: GraphKind; graph: Graph; history: Job[]; shown?: string; onShow: (jobId: string, step?: string | null) => void
}) {
  if (history.length === 0) return <Empty>This pipeline has not run yet.</Empty>
  return (
    <div className="table-wrap"><table className="run-history">
      <thead><tr><th>started</th><th>status</th><th>steps</th><th className="num">took</th><th>run</th></tr></thead>
      <tbody>{history.map(j => {
        const r = runStatus(kind, graph, j)
        const order = STEPS[kind].map(([step]) => step).filter(step => Object.values(r.steps).includes(step))
        const stateOf = (step: string) => r.status[Object.keys(r.steps).find(n => r.steps[n] === step)!]
        return (
          <tr key={j.id} className={`clickable ${j.id === shown ? 'selected' : ''}`} onClick={() => onShow(j.id)}>
            <td className="small">{when(j)}</td>
            <td><Pill status={j.status} /></td>
            <td><div className="chain">
              {!j.run_id && j.status === 'failed' && <button className="step-chip failed" onClick={e => { e.stopPropagation(); onShow(j.id, 'setup') }}>✕ before the first step · log</button>}
              {order.map(step => { const st = stateOf(step); return (
                <button key={step} className={`step-chip ${st?.state}`} title={`${step}: ${st?.state}${st?.note ? ` (${st.note})` : ''}. Click for its log.`}
                  disabled={st?.state === 'waiting' || st?.state === 'queued'} onClick={e => { e.stopPropagation(); onShow(j.id, step) }}>
                  {st?.state === 'failed' ? '✕ ' : st?.state === 'running' ? '● ' : ''}{step}{st?.note ? <span className="faint"> {st.note}</span> : null}</button>) })}
            </div>
              {j.status === 'failed' && j.error && <div className="small bad-text run-error">{j.error}</div>}</td>
            <td className="num small">{took(j)}</td>
            <td className="small">{j.run_id ? <Link to={j.kind === 'benchmark' && j.status === 'ok' ? `/runs/${j.run_id}` : '#'} onClick={e => { e.stopPropagation(); if (!(j.kind === 'benchmark' && j.status === 'ok')) e.preventDefault() }} className="mono">{j.run_id}</Link> : <span className="faint">no run</span>}</td>
          </tr>)
      })}</tbody>
    </table></div>)
}

// The log of one step of a run (runs/<id>/logs/<step>.log). Runs from before step logs
// existed, and jobs that failed before their run started, show the job's whole log instead.
function StepLog({ job, step, onClose }: { job: Job; step: string; onClose: () => void }) {
  const [text, setText] = useState<{ log: string; whole: boolean }>()
  const box = useRef<HTMLPreElement>(null)
  const live = isActive(job)
  useEffect(() => {
    let stop = false
    const whole = () => get<{ log: string }>(`/api/jobs/${job.id}/log`).then(r => ({ log: r.log, whole: true }))
    const load = () => (job.run_id
      ? get<{ log: string }>(`/api/runs/${job.run_id}/log?step=${encodeURIComponent(step)}`).then(r => ({ log: r.log, whole: false })).catch(whole)
      : whole()).then(r => { if (!stop) setText(r) }).catch(() => undefined)
    load()
    const t = live ? setInterval(load, 2000) : undefined
    return () => { stop = true; if (t) clearInterval(t) }
  }, [job.id, job.run_id, step, live])
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight }, [text])
  const failed = job.status === 'failed' && (job.run?.failed_stage ?? null) === step
  return (
    <div className={`card step-log ${failed ? 'failed' : ''}`}>
      <div className="row">
        <b>Log · {step === 'setup' ? 'before the first step' : <>step <code>{step}</code></>}</b>
        <span className="small muted">{job.run_id ?? job.title} · {when(job)}</span>
        {text?.whole && <span className="small muted">(this run has no per-step log, so this is the whole job’s log)</span>}
        <span className="grow" /><button className="sm ghost" onClick={onClose}>Close</button>
      </div>
      <pre ref={box}>{text === undefined ? 'Loading…' : text.log || '(no output)'}</pre>
    </div>)
}

// Chunk a random real document with the node's current settings (saved or not).
function ChunkPreview({ node, graph }: { node: GraphNode; graph: Graph }) {
  const [out, setOut] = useState<{ doc: { title: string; source: string; chars: number }; chunks: string[] }>()
  const [err, setErr] = useState<string>()
  const dataset = useMemo(() => String(graph.nodes.find(n => n.stage === 'source')?.params.dataset ?? ''), [graph])
  return (
    <div style={{ marginTop: 12 }}>
      <button className="sm" disabled={!dataset} onClick={() => {
        setErr(undefined)
        send<typeof out>('POST', '/api/preview/chunk', { impl: node.impl, params: node.params, dataset }).then(setOut).catch(e => { setOut(undefined); setErr(errText(e)) })
      }}>Preview on a random document</button>
      <ErrorNote error={err} />
      {out && <>
        <p className="small muted">{out.doc.source} · “{out.doc.title.slice(0, 60)}” · {out.doc.chars.toLocaleString()} chars → <b>{out.chunks.length} chunks</b></p>
        {out.chunks.slice(0, 12).map((c, i) => <details key={i}><summary className="small">chunk {i} · {c.length} chars</summary><pre>{c}</pre></details>)}
      </>}
    </div>
  )
}

interface PreviewDoc { doc_id: string; source: string; title: string; text: string; metadata: Record<string, unknown> }

// Run the transform node on one document and show it before and after.
function TransformPreview({ node, graph }: { node: GraphNode; graph: Graph }) {
  const [out, setOut] = useState<{ before: PreviewDoc; after: PreviewDoc | null }>()
  const [err, setErr] = useState<string>()
  const dataset = useMemo(() => String(graph.nodes.find(n => n.stage === 'source')?.params.dataset ?? ''), [graph])
  return (
    <div style={{ marginTop: 12 }}>
      <button className="sm" disabled={!dataset} onClick={() => {
        setErr(undefined)
        send<typeof out>('POST', '/api/preview/transform', { impl: node.impl, params: node.params, dataset }).then(setOut).catch(e => { setOut(undefined); setErr(errText(e)) })
      }}>Preview on a random document</button>
      <ErrorNote error={err} />
      {out && <>
        <p className="small muted">{out.before.doc_id} · {out.before.text.length.toLocaleString()} chars →{' '}
          {out.after === null ? <b className="bad-text">dropped</b> : <b>{out.after.text.length.toLocaleString()} chars{out.after.text === out.before.text && out.after.title === out.before.title ? ' (unchanged)' : ''}</b>}</p>
        <details><summary className="small">before</summary><pre>{out.before.text.slice(0, 4000)}</pre></details>
        {out.after && <details open><summary className="small">after{out.after.title !== out.before.title ? ` · title “${out.after.title}”` : ''}</summary><pre>{out.after.text.slice(0, 4000)}</pre></details>}
      </>}
    </div>
  )
}

interface Hit { chunk_id: string; doc_id: string; score: number; text: string }

// Run one query through retrieve + rerank against a target, without saving anything.
function TryQuery({ graph, targets, defaultTarget }: { graph: Graph; targets: Target[]; defaultTarget: string }) {
  const [tid, setTid] = useState('')
  const [query, setQuery] = useState('')
  const [res, setRes] = useState<{ seconds: number; hits: Hit[] }>()
  const [err, setErr] = useState<string>()
  const [loading, setLoading] = useState(false)
  const target = tid || (targets.some(t => t.id === defaultTarget) ? defaultTarget : targets[0]?.id) || ''
  const run = () => {
    if (!target || !query.trim()) return
    setLoading(true); setErr(undefined)
    send<typeof res>('POST', '/api/preview/query', { pipeline: graph, target_id: target, query })
      .then(setRes).catch(e => { setRes(undefined); setErr(errText(e)) }).finally(() => setLoading(false))
  }
  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="card-head"><h2>Try a query</h2><span className="small muted">Runs retrieve + rerank as drawn above, saved or not. The first query loads models (~10s).</span></div>
      {targets.length === 0 ? <p className="muted small">No target has data yet. Run an <Link to="/ingestion">ingestion pipeline</Link> first.</p> : (
        <div className="row">
          <select value={target} onChange={e => setTid(e.target.value)}>{targets.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</select>
          <input className="grow" style={{ minWidth: 240 }} value={query} placeholder="Ask a question…" onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && run()} />
          <button className="primary" onClick={run} disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
        </div>)}
      <ErrorNote error={err} />
      {res && <>
        <p className="small muted">{res.hits.length} chunks in {res.seconds}s</p>
        {res.hits.map((h, i) => (
          <div key={h.chunk_id} className="doc">
            <div className="small"><b>#{i + 1}</b> · score {fmt(h.score, 3)} · <span className="mono faint">{h.chunk_id}</span></div>
            <details open={i < 3}><summary className="small">{h.text.split('\n')[0].slice(0, 110)}</summary><pre>{h.text}</pre></details>
          </div>))}
      </>}
    </div>
  )
}
