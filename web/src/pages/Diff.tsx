import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { fmt, get, type RunRow } from '../api'

interface DiffRow { question_id: string; category: string; question: string; a: number | null; b: number | null; status: string }
interface DiffData { metric: string; snapshots: number[]; metrics: string[]; counts: Record<string, number>; rows: DiffRow[] }

export default function Diff() {
  const [params, setParams] = useSearchParams()
  const a = params.get('a') ?? '', b = params.get('b') ?? '', metric = params.get('metric') ?? ''
  const [runs, setRuns] = useState<RunRow[]>([])
  const [data, setData] = useState<DiffData>()
  const [tab, setTab] = useState('fixed')
  const [error, setError] = useState<string>()

  useEffect(() => { get<RunRow[]>('/api/runs').then(setRuns) }, [])
  useEffect(() => {
    setError(undefined)
    if (a && b) get<DiffData>(`/api/diff?a=${a}&b=${b}${metric ? `&metric=${encodeURIComponent(metric)}` : ''}`).then(setData).catch(e => { setData(undefined); setError(String(e)) })
  }, [a, b, metric])

  const set = (k: string, v: string) => { const p = new URLSearchParams(params); p.set(k, v); setParams(p) }
  const pick = (k: 'a' | 'b', v: string) => (
    <select value={v} onChange={e => set(k, e.target.value)}>
      <option value="">run {k.toUpperCase()}…</option>
      {runs.filter(r => k === 'a' || !runA || r.dataset === runA.dataset).map(r => <option key={r.run_id} value={r.run_id}>{r.run_id} ({r.dataset})</option>)}
    </select>
  )
  const runA = runs.find(r => r.run_id === a)
  const rows = data?.rows.filter(r => r.status === tab) ?? []
  return (
    <>
      <h1>Compare two runs</h1>
      <p className="muted small">"Fixed" = the metric went up from A to B for that question; "broken" = it went down.</p>
      <div className="row" style={{ margin: '8px 0' }}>
        {pick('a', a)} → {pick('b', b)}
        {data && <select value={data.metric} onChange={e => set('metric', e.target.value)}>
          {data.metrics.map(m => <option key={m}>{m}</option>)}</select>}
      </div>
      {error && <p className="badge miss">{error}</p>}
      {data && data.snapshots[0] !== data.snapshots[1] &&
        <p className="small muted">Note: A uses snapshot {data.snapshots[0]}, B uses snapshot {data.snapshots[1]} — the documents differ, not just the pipeline.</p>}
      {data && <>
        <div className="tabs">{['fixed', 'broken', 'same', 'n/a'].map(s => (
          <button key={s} className={tab === s ? 'active' : ''} onClick={() => setTab(s)}>{s} ({data.counts[s]})</button>))}</div>
        <div className="scroll"><table>
          <thead><tr><th>question</th><th>category</th><th></th><th className="num">A</th><th className="num">B</th></tr></thead>
          <tbody>{rows.map(r => (
            <tr key={r.question_id}>
              <td className="mono">{r.question_id}<div className="small">
                <Link to={`/runs/${a}/q/${r.question_id}`}>in A</Link> · <Link to={`/runs/${b}/q/${r.question_id}`}>in B</Link></div></td>
              <td className="small">{r.category}</td>
              <td className="small">{r.question}</td>
              <td className="num">{fmt(r.a)}</td><td className="num">{fmt(r.b)}</td>
            </tr>))}</tbody>
        </table></div>
      </>}
    </>
  )
}
