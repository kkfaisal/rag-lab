import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fmt, get, heat, isPct, send, type Catalog, type Graph, type IndexStatus, type MatrixCell } from '../api'
import { isActive, JobList, useJobs } from '../components/Jobs'

const METRICS = ['recall@10', 'recall@1', 'recall@5', 'recall@20', 'context_recall', 'answer_recall', 'extra_docs', 'correctness', 'completeness', 'abstained']

export default function Experiments() {
  const nav = useNavigate()
  const [indexes, setIndexes] = useState<IndexStatus[]>([])
  const [loaded, setLoaded] = useState(false)
  const [retrievers, setRetrievers] = useState<Graph[]>([])
  const [catalog, setCatalog] = useState<Catalog>()
  const [cells, setCells] = useState<MatrixCell[]>([])
  const [selIx, setSelIx] = useState<string[]>([])
  const [selRet, setSelRet] = useState<string[]>([])
  const [limit, setLimit] = useState('20')
  const [judge, setJudge] = useState('auto')
  const [metric, setMetric] = useState('recall@10')
  const [dataset, setDataset] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [msg, setMsg] = useState<string>()
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length

  useEffect(() => {
    get<Catalog>('/api/catalog').then(setCatalog)
    get<Graph[]>('/api/graphs/retriever').then(setRetrievers)
  }, [])
  useEffect(() => {
    get<IndexStatus[]>('/api/indexes').then(i => { setIndexes(i); setLoaded(true) })
    get<MatrixCell[]>('/api/matrix').then(setCells)
  }, [busy])

  const key = (i: IndexStatus) => `${i.pipeline_id}/${i.node_id}`
  const toggle = (list: string[], set: (v: string[]) => void, v: string) => set(list.includes(v) ? list.filter(x => x !== v) : [...list, v])
  const launch = async () => {
    setMsg(undefined)
    try {
      const out = await send<unknown[]>('POST', '/api/jobs/experiments', {
        indexes: indexes.filter(i => selIx.includes(key(i))).map(i => ({ pipeline_id: i.pipeline_id, node_id: i.node_id })),
        retriever_ids: selRet, limit: limit ? Number(limit) : null, judge: judge === 'off' ? false : judge })
      setMsg(`${out.length} experiment${out.length === 1 ? '' : 's'} queued`)
    } catch (e) { setMsg(String(e)) }
  }

  const datasets = [...new Set(cells.map(c => c.dataset))].sort()
  const ds = dataset || datasets[0] || ''
  const grid = useMemo(() => {
    const mine = cells.filter(c => c.dataset === ds)
    const rows = [...new Map(mine.map(c => [c.index_key, c.index])).entries()]
    const cols = [...new Map(mine.map(c => [c.retriever_id, c.retriever])).entries()]
    const at = (r: string, c: string) => mine.find(x => x.index_key === r && x.retriever_id === c)
    return { rows, cols, at }
  }, [cells, ds])
  const pick = (runId: string) => setPicked(p => p.includes(runId) ? p.filter(x => x !== runId) : [...p.slice(-1), runId])

  return (
    <>
      <h1>Experiments</h1>
      <p className="muted small">An experiment runs one index with one retrieval setup against the dataset's questions. Pick several of each to compare every combination; anything already computed is reused from the cache.</p>

      <div className="grid2" style={{ marginTop: 12 }}>
        <div className="panel">
          <b>1. Indexes</b> <span className="muted small">(from <Link to="/pipelines">Pipelines</Link>)</span>
          {indexes.map(i => (
            <label key={key(i)} className="check">
              <input type="checkbox" disabled={i.state === 'invalid'} checked={selIx.includes(key(i))} onChange={() => toggle(selIx, setSelIx, key(i))} />
              <span>{i.pipeline_name} / <b>{i.label}</b> <span className="muted small">{i.dataset} · {i.subset}</span></span>
              <span className={`badge ${i.state === 'built' ? 'ok' : i.state === 'invalid' ? 'failed' : ''}`}>{i.state === 'built' ? 'built' : i.state === 'invalid' ? 'invalid' : 'will be built'}</span>
            </label>))}
          {indexes.length === 0 && <p className="muted small">{loaded ? 'No pipelines yet.' : 'Loading…'}</p>}
        </div>
        <div className="panel">
          <b>2. Retrieval setups</b> <span className="muted small">(from <Link to="/retrieval">Retrieval</Link>)</span>
          {retrievers.map(r => (
            <label key={r.id} className="check">
              <input type="checkbox" checked={selRet.includes(r.id!)} onChange={() => toggle(selRet, setSelRet, r.id!)} />
              <span><b>{r.name}</b> <span className="muted small">{r.nodes.map(n => n.impl).join(' → ')}</span></span>
              {r.nodes.some(n => n.stage === 'generate' && n.impl === 'llm') && !catalog?.has_api_key && <span className="badge failed">needs API key</span>}
            </label>))}
        </div>
      </div>
      <div className="row" style={{ margin: '12px 0' }}>
        <b>3.</b>
        <label className="small">questions <input style={{ width: 70 }} type="number" value={limit} placeholder="all" onChange={e => setLimit(e.target.value)} /></label>
        <label className="small">answer judge <select value={judge} onChange={e => setJudge(e.target.value)}>
          <option value="auto">auto (when answers + API key exist)</option><option value="off">off (retrieval metrics only)</option></select></label>
        <button className="primary" disabled={selIx.length === 0 || selRet.length === 0} onClick={launch}>
          Run {selIx.length * selRet.length || ''} experiment{selIx.length * selRet.length === 1 ? '' : 's'}</button>
        {msg && <span className="small muted">{msg}</span>}
        {catalog && !catalog.has_api_key && <span className="small muted">No ANTHROPIC_API_KEY in .env: LLM answers and the judge are unavailable; retrieval metrics still work.</span>}
      </div>

      <h2>Jobs</h2>
      <JobList jobs={jobs} kinds={['experiment', 'build']} limit={6} />

      <h2>Comparison: index × retrieval setup</h2>
      <div className="row" style={{ marginBottom: 8 }}>
        <select value={ds} onChange={e => setDataset(e.target.value)}>{datasets.map(d => <option key={d}>{d}</option>)}</select>
        <select value={metric} onChange={e => setMetric(e.target.value)}>{METRICS.map(m => <option key={m}>{m}</option>)}</select>
        <span className="muted small">Latest successful run per cell. Click a value for details; tick two cells to diff them question by question.</span>
        <span style={{ flex: 1 }} />
        <button disabled={picked.length !== 2} onClick={() => nav(`/diff?a=${picked[0]}&b=${picked[1]}`)}>Diff {picked.length}/2</button>
      </div>
      <div className="scroll"><table className="heat">
        <thead><tr><th>index ↓ / retrieval →</th>{grid.cols.map(([id, name]) => <th key={id} className="num">{name}</th>)}</tr></thead>
        <tbody>{grid.rows.map(([rk, rname]) => (
          <tr key={rk}><td><b>{rname}</b></td>
            {grid.cols.map(([ck]) => {
              const c = grid.at(rk, ck)
              return <td key={ck} className="cell" style={{ background: c && isPct(metric) && metric !== 'abstained' ? heat(c.overall[metric]) : undefined }}>
                {c ? <>
                  <input type="checkbox" checked={picked.includes(c.run_id)} onChange={() => pick(c.run_id)} />{' '}
                  <Link to={`/runs/${c.run_id}`}>{fmt(c.overall[metric])}</Link>
                  <div className="small muted">{c.n_questions} q · snap {c.snapshot}</div></> : <span className="muted">–</span>}
              </td>
            })}</tr>))}</tbody>
      </table></div>
      {cells.length === 0 && <p className="muted">No finished experiments yet.</p>}
      <p className="small" style={{ marginTop: 12 }}><Link to="/runs">All runs (sortable, filter by any setting) →</Link> · <Link to="/heatmap">Per-category heatmap →</Link></p>
    </>
  )
}
