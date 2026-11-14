import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fmt, get, heat, isPct, type MatrixCell } from '../api'
import { MetricHelp } from '../components/Glossary'
import { isActive, JobList, useJobs } from '../components/Jobs'
import { Empty, PageHead } from '../components/ui'

const METRICS = ['recall@10', 'recall@1', 'recall@5', 'recall@20', 'context_recall', 'answer_recall', 'extra_docs', 'correctness', 'completeness', 'abstained']

// Benchmark results at a glance: every target against every retrieval pipeline.
export default function Results() {
  const nav = useNavigate()
  const [cells, setCells] = useState<MatrixCell[] | null>(null)
  const [metric, setMetric] = useState('recall@10')
  const [dataset, setDataset] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length
  useEffect(() => { get<MatrixCell[]>('/api/matrix').then(setCells).catch(() => setCells([])) }, [busy])

  const datasets = [...new Set((cells ?? []).map(c => c.dataset))].sort()
  const ds = dataset || datasets[0] || ''
  const grid = useMemo(() => {
    const mine = (cells ?? []).filter(c => c.dataset === ds)
    const rows = [...new Map(mine.map(c => [c.target_id, c.target])).entries()]
    const cols = [...new Map(mine.map(c => [c.pipeline_id, c.pipeline])).entries()]
    const at = (r: string, c: string) => mine.find(x => x.target_id === r && x.pipeline_id === c)
    const best = Math.max(...mine.map(c => c.overall[metric] ?? -Infinity))
    return { rows, cols, at, best }
  }, [cells, ds, metric])
  const pick = (runId: string) => setPicked(p => p.includes(runId) ? p.filter(x => x !== runId) : [...p.slice(-1), runId])

  return (
    <>
      <PageHead title="Results" sub="The latest successful benchmark for each target × retrieval pipeline. Run benchmarks from a retrieval pipeline; tick two cells to compare them question by question.">
        <select value={ds} onChange={e => setDataset(e.target.value)} aria-label="Dataset">{datasets.map(d => <option key={d}>{d}</option>)}</select>
        <select value={metric} onChange={e => setMetric(e.target.value)} aria-label="Metric">{METRICS.map(m => <option key={m}>{m}</option>)}</select>
        <MetricHelp metric={metric} />
        <button disabled={picked.length !== 2} onClick={() => nav(`/diff?a=${picked[0]}&b=${picked[1]}`)}>Compare {picked.length}/2</button>
      </PageHead>

      {cells === null ? <p className="muted">Loading…</p> : cells.length === 0 ? (
        <Empty>No benchmark results yet. Open a <Link to="/retrieval">retrieval pipeline</Link> and press “Run benchmark”.</Empty>) : (
        <div className="table-wrap"><table className="heat">
          <thead><tr><th>target ↓ · retrieval pipeline →</th>{grid.cols.map(([id, name]) => <th key={id} className="num">{name}</th>)}</tr></thead>
          <tbody>{grid.rows.map(([rk, rname]) => (
            <tr key={rk}><td><b>{rname}</b></td>
              {grid.cols.map(([ck]) => {
                const c = grid.at(rk, ck)
                const v = c?.overall[metric]
                return <td key={ck} className="cell" style={{ background: c && isPct(metric) && metric !== 'abstained' ? heat(v) : undefined }}>
                  {c ? <>
                    <input type="checkbox" checked={picked.includes(c.run_id)} onChange={() => pick(c.run_id)} aria-label="Select for comparison" />{' '}
                    <Link to={`/runs/${c.run_id}`} style={{ fontWeight: v === grid.best ? 700 : 500 }}>{fmt(v)}</Link>
                    <div className="small faint">{c.n_questions} questions · target v{c.target_version}{c.judged ? ' · judged' : ''}</div></> : <span className="faint">not run</span>}
                </td>
              })}</tr>))}</tbody>
        </table></div>)}

      <div className="section-title">Recent benchmark jobs</div>
      <JobList jobs={jobs} kinds={['benchmark']} limit={5} />
    </>
  )
}
