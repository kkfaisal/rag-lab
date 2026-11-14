import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { errText, fmt, get, send, type Catalog, type DatasetInfo, type Graph, type GraphKind, type GraphNode, type Job, type Target } from '../api'
import GraphEditor from '../components/GraphEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'
import { Empty, ErrorNote, PageHead } from '../components/ui'

const TEXT = {
  ingestion: { title: 'Ingestion pipelines', path: '/ingestion', noun: 'ingestion pipeline', job: 'ingest',
    blurb: 'Source → chunk → embed → target. Running a pipeline writes its chunks to the target, appending or overwriting as the target node says. Branch to fill several targets from one source.' },
  retrieval: { title: 'Retrieval pipelines', path: '/retrieval', noun: 'retrieval pipeline', job: 'benchmark',
    blurb: 'Target → retrieve → rerank → generate → benchmark. Running a pipeline answers the dataset\'s questions from the target and scores the result.' },
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

  const reload = useCallback(() => get<Graph[]>(`/api/graphs/${kind}`).then(setGraphs).catch(e => setError(errText(e))), [kind])
  useEffect(() => {
    get<Catalog>('/api/catalog').then(setCatalog)
    get<DatasetInfo[]>('/api/datasets').then(setDatasets).catch(() => undefined)
    reload()
  }, [reload])
  useEffect(() => { get<Target[]>('/api/targets').then(setTargets).catch(() => undefined) }, [busy])

  // Open the pipeline named in the URL (or the first one).
  useEffect(() => {
    if (id === 'new' || !graphs?.length) return
    const g = graphs.find(x => x.id === id) ?? graphs[0]
    if (g.id !== id) { nav(`${t.path}/${g.id}`, { replace: true }); return }
    setGraph(g); setSaved(strip(g)); setRunTargets(null); setNote(undefined)
  }, [id, graphs, nav, t.path])

  const dirty = graph !== undefined && strip(graph) !== saved
  const save = async (): Promise<Graph | undefined> => {
    if (!graph) return
    try {
      const g = await send<Graph>('PUT', `/api/graphs/${kind}`, graph)
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
  const nTargets = graph?.nodes.filter(n => n.stage === 'target').length ?? 0
  return (
    <>
      <PageHead title={t.title} sub={t.blurb}>
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

        <GraphEditor key={`${kind}:${graph.id ?? 'new'}:${graph.updated_at ?? ''}`} kind={kind} graph={graph} catalog={catalog} datasets={datasets} targets={targets}
          onChange={setGraph} extra={(n, g) => kind !== 'ingestion' ? null : n.stage === 'chunk' ? <ChunkPreview node={n} graph={g} />
            : n.stage === 'transform' ? <TransformPreview node={n} graph={g} /> : null} />

        {kind === 'retrieval' && <TryQuery graph={graph} targets={targets.filter(tg => tg.version)} defaultTarget={ownTarget} />}
        <div className="section-title">Runs of this page <span className="muted">· <Link to="/jobs">all jobs</Link></span></div>
        <JobList jobs={jobs} kinds={[t.job]} limit={5} />
      </>}
    </>
  )
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
