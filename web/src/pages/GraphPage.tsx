import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { fmt, get, send, type Catalog, type DatasetInfo, type Graph, type GraphKind, type GraphNode, type IndexStatus } from '../api'
import GraphEditor from '../components/GraphEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'

const TEXT = {
  pipeline: { title: 'Pipelines', path: '/pipelines', blurb: 'Turn documents into a searchable index: source → chunk → embed → index. Branch to try several strategies side by side; every index node is saved and reusable.' },
  retriever: { title: 'Retrieval', path: '/retrieval', blurb: 'How questions are answered from an index: retrieve → rerank → generate. Pair any setup with any index on the Experiments page.' },
}
const BLANK: Record<GraphKind, (ds: string) => Graph> = {
  pipeline: ds => ({ name: '', description: '', edges: [], nodes: [{ id: 'source', stage: 'source', impl: 'dataset', params: { dataset: ds, subset: 'all', snapshot: 'latest' }, position: { x: 20, y: 80 } }] }),
  retriever: () => ({ name: '', description: '', edges: [], nodes: [{ id: 'retrieve', stage: 'retrieve', impl: 'hybrid', params: {}, position: { x: 20, y: 80 } }] }),
}
const strip = (g: Graph) => JSON.stringify({ n: g.name, d: g.description, nodes: g.nodes, edges: g.edges })

export default function GraphPage({ kind }: { kind: GraphKind }) {
  const { id } = useParams()
  const nav = useNavigate()
  const t = TEXT[kind]
  const [catalog, setCatalog] = useState<Catalog>()
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [graphs, setGraphs] = useState<Graph[]>([])
  const [graph, setGraph] = useState<Graph>()          // the one being edited
  const [saved, setSaved] = useState('')               // its last-saved content, to detect unsaved edits
  const [indexes, setIndexes] = useState<IndexStatus[]>([])
  const [error, setError] = useState<string>()
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length

  const reload = useCallback(() => get<Graph[]>(`/api/graphs/${kind}`).then(setGraphs).catch(e => setError(String(e))), [kind])
  useEffect(() => {
    get<Catalog>('/api/catalog').then(setCatalog)
    get<DatasetInfo[]>('/api/datasets').then(setDatasets)
    reload()
  }, [reload])
  useEffect(() => { get<IndexStatus[]>('/api/indexes').then(setIndexes).catch(() => undefined) }, [busy, saved])

  // Open the graph named in the URL (or the first one).
  useEffect(() => {
    if (id === 'new' || graphs.length === 0) return
    const g = graphs.find(x => x.id === id) ?? graphs[0]
    if (g.id !== id) { nav(`${t.path}/${g.id}`, { replace: true }); return }
    setGraph(g); setSaved(strip(g))
  }, [id, graphs, nav, t.path])

  const dirty = graph !== undefined && strip(graph) !== saved
  const save = async () => {
    if (!graph) return
    try {
      const g = await send<Graph>('PUT', `/api/graphs/${kind}`, graph)
      setError(undefined); setGraph(g); setSaved(strip(g)); await reload()
      nav(`${t.path}/${g.id}`, { replace: true })
    } catch (e) { setError(String(e)) }
  }
  const create = () => {
    const name = window.prompt(`Name of the new ${kind === 'pipeline' ? 'pipeline' : 'retrieval setup'}:`)
    if (!name) return
    setGraph({ ...BLANK[kind](datasets[0]?.name ?? ''), name }); setSaved(''); nav(`${t.path}/new`)
  }
  const duplicate = () => graph && (setGraph({ ...graph, id: undefined, name: `${graph.name} copy` }), setSaved(''), nav(`${t.path}/new`))
  const del = async () => {
    if (!graph?.id || !window.confirm(`Delete "${graph.name}"? Its past runs are kept.`)) return
    await send('DELETE', `/api/graphs/${kind}/${graph.id}`)
    setGraph(undefined); await reload(); nav(t.path)
  }

  const statusOf = useCallback((nodeId: string) => indexes.find(i => i.pipeline_id === graph?.id && i.node_id === nodeId), [indexes, graph?.id])
  const badge = useCallback((n: GraphNode) => {
    if (kind !== 'pipeline' || n.stage !== 'index') return null
    const s = statusOf(n.id)?.state ?? 'not built'
    return <span className={`badge ${s === 'built' ? 'ok' : s === 'failed' || s === 'invalid' ? 'failed' : ''}`}>{s}</span>
  }, [kind, statusOf])

  if (!catalog) return <p className="muted">Loading…</p>
  return (
    <>
      <h1>{t.title}</h1>
      <p className="muted small">{t.blurb}</p>
      <div className="row" style={{ margin: '10px 0' }}>
        <select value={id === 'new' ? '' : graph?.id ?? ''} onChange={e => nav(`${t.path}/${e.target.value}`)}>
          {id === 'new' && <option value="">(unsaved)</option>}
          {graphs.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
        </select>
        <button onClick={create}>New</button>
        {graph && <>
          <button onClick={duplicate}>Duplicate</button>
          <button onClick={del} disabled={!graph.id}>Delete</button>
          <input value={graph.name} style={{ width: 200 }} onChange={e => setGraph({ ...graph, name: e.target.value })} />
          <input value={graph.description} placeholder="description" style={{ flex: 1, minWidth: 160 }} onChange={e => setGraph({ ...graph, description: e.target.value })} />
          <button className="primary" onClick={save} disabled={!dirty}>{dirty ? 'Save' : 'Saved'}</button>
        </>}
      </div>
      {error && <p className="badge failed">{error}</p>}
      {datasets.length === 0 && kind === 'pipeline' && <p className="muted">No datasets yet — add one on the <Link to="/datasets">Datasets</Link> page first.</p>}

      {graph && (
        <GraphEditor key={`${kind}:${graph.id ?? 'new'}:${graph.updated_at ?? ''}`} kind={kind} graph={graph} catalog={catalog} datasets={datasets}
          onChange={setGraph} badge={badge}
          extra={(n, g) => kind === 'pipeline'
            ? <PipelineExtra node={n} graph={g} dirty={dirty} status={statusOf(n.id)} />
            : null} />)}

      {graph && kind === 'pipeline' && (
        <div className="grid2" style={{ marginTop: 16 }}>
          <div>
            <h2>Indexes produced by this pipeline</h2>
            <table><thead><tr><th>index</th><th>state</th><th>chunks</th><th>built</th><th></th></tr></thead>
              <tbody>{indexes.filter(i => i.pipeline_id === graph.id).map(i => (
                <tr key={i.node_id}>
                  <td><b>{i.label}</b>{i.error && <div className="small" style={{ color: 'var(--miss)' }}>{i.error}</div>}</td>
                  <td><span className={`badge ${i.state === 'built' ? 'ok' : i.state === 'failed' || i.state === 'invalid' ? 'failed' : ''}`}>{i.state}</span></td>
                  <td className="num">{fmt(i.stages.find(s => s.stage === 'chunk')?.stats?.n_chunks)}</td>
                  <td className="small muted">{i.built_at?.slice(0, 16) ?? '–'}{i.snapshot ? ` · snap ${i.snapshot}` : ''}</td>
                  <td><BuildButton pipelineId={graph.id!} nodeId={i.node_id} disabled={dirty || i.state === 'invalid'} label={i.state === 'built' ? 'Rebuild' : 'Build'} /></td>
                </tr>))}</tbody></table>
            {dirty && <p className="small muted">Save the pipeline to build its indexes.</p>}
          </div>
          <div><h2>Jobs</h2><JobList jobs={jobs} kinds={['build']} limit={5} /></div>
        </div>)}

      {graph && kind === 'retriever' && <TryQuery graph={graph} indexes={indexes.filter(i => i.state === 'built')} />}
    </>
  )
}

function BuildButton({ pipelineId, nodeId, disabled, label }: { pipelineId: string; nodeId: string; disabled: boolean; label: string }) {
  const [msg, setMsg] = useState<string>()
  return <>
    <button disabled={disabled} onClick={() => send('POST', '/api/jobs/build', { pipeline_id: pipelineId, node_id: nodeId }).then(() => setMsg('queued')).catch(e => setMsg(String(e)))}>{label}</button>
    {msg && <span className="small muted"> {msg}</span>}
  </>
}

// Extra controls in the settings panel: chunk preview on a real document, and build state for index nodes.
function PipelineExtra({ node, graph, dirty, status }: { node: GraphNode; graph: Graph; dirty: boolean; status?: IndexStatus }) {
  const [out, setOut] = useState<{ doc: { title: string; source: string; chars: number }; chunks: string[] }>()
  const [err, setErr] = useState<string>()
  const dataset = useMemo(() => String(graph.nodes.find(n => n.stage === 'source')?.params.dataset ?? ''), [graph])
  if (node.stage === 'chunk') return (
    <div style={{ marginTop: 12 }}>
      <button disabled={!dataset} onClick={() => {
        setErr(undefined)
        send<typeof out>('POST', '/api/preview/chunk', { impl: node.impl, params: node.params, dataset }).then(setOut).catch(e => { setOut(undefined); setErr(String(e)) })
      }}>Preview on a random document</button>
      {err && <pre style={{ color: 'var(--miss)' }}>{err}</pre>}
      {out && <>
        <p className="small muted">{out.doc.source} · “{out.doc.title.slice(0, 60)}” · {out.doc.chars.toLocaleString()} chars → <b>{out.chunks.length} chunks</b></p>
        {out.chunks.slice(0, 12).map((c, i) => <details key={i}><summary className="small">chunk {i} · {c.length} chars</summary><pre>{c}</pre></details>)}
      </>}
    </div>)
  if (node.stage === 'index' && graph.id) return (
    <div style={{ marginTop: 12 }}>
      <p className="small">State: <b>{status?.state ?? 'not built'}</b>{status?.state === 'stale' && ' — the pipeline changed since the last build'}</p>
      {status?.stages.map(s => (
        <div key={s.stage} className="small muted">{s.stage}: {Object.entries(s.stats ?? {}).map(([k, v]) => `${k}=${v}`).join(', ')}</div>))}
      <div style={{ marginTop: 8 }}><BuildButton pipelineId={graph.id} nodeId={node.id} disabled={dirty} label="Build this index" />
        {dirty && <span className="small muted"> save first</span>}</div>
    </div>)
  return null
}

interface Hit { chunk_id: string; doc_id: string; score: number; text: string }

// Run one query through retrieve + rerank against a built index, without saving anything.
function TryQuery({ graph, indexes }: { graph: Graph; indexes: IndexStatus[] }) {
  const [ix, setIx] = useState('')
  const [query, setQuery] = useState('')
  const [res, setRes] = useState<{ seconds: number; hits: Hit[] }>()
  const [err, setErr] = useState<string>()
  const [loading, setLoading] = useState(false)
  const chosen = indexes.find(i => `${i.pipeline_id}/${i.node_id}` === ix) ?? indexes[0]
  const run = () => {
    if (!chosen || !query.trim()) return
    setLoading(true); setErr(undefined)
    send<typeof res>('POST', '/api/preview/query', { pipeline_id: chosen.pipeline_id, node_id: chosen.node_id, retriever: graph, query })
      .then(setRes).catch(e => { setRes(undefined); setErr(String(e)) }).finally(() => setLoading(false))
  }
  return (
    <div style={{ marginTop: 16 }}>
      <h2>Try a query</h2>
      <p className="muted small">Runs the retrieve and rerank nodes as currently drawn (saved or not) against a built index. The first query loads models and can take ~10s.</p>
      {indexes.length === 0 ? <p className="muted">No built index yet — build one on the <Link to="/pipelines">Pipelines</Link> page.</p> : (
        <div className="row">
          <select value={chosen ? `${chosen.pipeline_id}/${chosen.node_id}` : ''} onChange={e => setIx(e.target.value)}>
            {indexes.map(i => <option key={i.pipeline_id + i.node_id} value={`${i.pipeline_id}/${i.node_id}`}>{i.pipeline_name} / {i.label} ({i.dataset})</option>)}
          </select>
          <input style={{ flex: 1, minWidth: 240 }} value={query} placeholder="Ask something…" onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && run()} />
          <button className="primary" onClick={run} disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
        </div>)}
      {err && <pre style={{ color: 'var(--miss)' }}>{err}</pre>}
      {res && <>
        <p className="small muted">{res.hits.length} chunks in {res.seconds}s</p>
        {res.hits.map((h, i) => (
          <div key={h.chunk_id} className="doc">
            <div className="small"><b>#{i + 1}</b> score {fmt(h.score, 3)} <span className="mono muted">{h.chunk_id}</span></div>
            <details open={i < 3}><summary className="small">{h.text.split('\n')[0].slice(0, 110)}</summary><pre>{h.text}</pre></details>
          </div>))}
      </>}
    </div>
  )
}
