import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { fmt, get, type RunRow } from '../api'
import { PageHead } from '../components/ui'

interface DiffRow { question_id: string; category: string; question: string; a: number | null; b: number | null; status: string }
interface PairedTest { n: number; mean_a: number | null; mean_b: number | null; delta: number | null; ci95: [number, number] | null; p_value: number | null
  b_better: number; a_better: number; ties: number; significant: boolean; verdict: string }
interface DiffData { metric: string; test: PairedTest; snapshots: (number | null)[]; metrics: string[]; counts: Record<string, number>; rows: DiffRow[] }

// Is the difference between the runs real? Mean of each, the gap with its 95% interval, and a paired test.
function Verdict({ t, metric }: { t: PairedTest; metric: string }) {
  if (!t.n || t.delta === null) return null
  const sign = (v: number) => (v > 0 ? '+' : '') + fmt(v)
  return (
    <div className={`card verdict ${t.significant ? (t.delta > 0 ? 'better' : 'worse') : 'noise'}`} style={{ margin: '10px 0' }}>
      <div className="stats">
        <div className="stat"><div className="v">{fmt(t.mean_a)}</div><div className="k">A · mean {metric}</div></div>
        <div className="stat"><div className="v">{fmt(t.mean_b)}</div><div className="k">B · mean {metric}</div></div>
        <div className="stat"><div className="v">{sign(t.delta)}</div><div className="k">difference B − A</div></div>
        <div className="stat"><div className="v">{t.ci95 ? `${sign(t.ci95[0])} … ${sign(t.ci95[1])}` : '–'}</div><div className="k">95% confidence interval</div></div>
        <div className="stat"><div className="v">{t.b_better} / {t.a_better} / {t.ties}</div><div className="k">questions B better / A better / tied</div></div>
      </div>
      <p style={{ margin: '8px 0 0' }}><b>{t.verdict}.</b> <span className="small muted">The verdict comes from a paired permutation test over {t.n} questions. The interval (a bootstrap over questions) shows the likely size of the gap; with few questions it is too narrow, so trust the verdict.</span></p>
    </div>)
}

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
      {runs.filter(r => k === 'a' || !runA || r.dataset === runA.dataset).map(r => <option key={r.run_id} value={r.run_id}>{r.description || r.name} · {String(r.created_at).replace('T', ' ').slice(5, 16)}</option>)}
    </select>
  )
  const runA = runs.find(r => r.run_id === a)
  const rows = data?.rows.filter(r => r.status === tab) ?? []
  return (
    <>
      <PageHead title="Compare two runs" sub="Whether B really beats A on the same questions, or the gap is noise; then question by question: “fixed” means the metric went up from A to B, “broken” means it went down." />
      <div className="row" style={{ margin: '8px 0' }}>
        {pick('a', a)} → {pick('b', b)}
        {data && <select value={data.metric} onChange={e => set('metric', e.target.value)}>
          {data.metrics.map(m => <option key={m}>{m}</option>)}</select>}
      </div>
      {error && <p className="pill miss">{error}</p>}
      {data && data.snapshots[0] !== null && data.snapshots[0] !== data.snapshots[1] &&
        <p className="small muted">Note: A uses snapshot {data.snapshots[0]}, B uses snapshot {data.snapshots[1]} — the documents differ, not just the pipeline.</p>}
      {data && <>
        <Verdict t={data.test} metric={data.metric} />
        <div className="tabs">{['fixed', 'broken', 'same', 'n/a'].map(s => (
          <button key={s} className={tab === s ? 'active' : ''} onClick={() => setTab(s)}>{s} ({data.counts[s]})</button>))}</div>
        <div className="table-wrap"><table>
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
