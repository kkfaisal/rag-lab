import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { fmt, get, heat, isPct } from '../api'
import { MetricHelp } from '../components/Glossary'
import { PageHead } from '../components/ui'

interface Cell { run_id: string; name: string; created_at: string; snapshot: number; category: string; value: number }
interface HeatData { metrics: string[]; dataset: string | null; datasets: string[]; cells: Cell[] }

export default function Heatmap() {
  const [metric, setMetric] = useState('recall@10')
  const [dataset, setDataset] = useState('')
  const [data, setData] = useState<HeatData>({ metrics: [], dataset: null, datasets: [], cells: [] })
  useEffect(() => { get<HeatData>(`/api/heatmap?metric=${encodeURIComponent(metric)}${dataset ? `&dataset=${dataset}` : ''}`).then(setData) }, [metric, dataset])

  const { runs, cats, value } = useMemo(() => {
    const runs = [...new Map(data.cells.map(c => [c.run_id, c])).values()]
    const cats = [...new Set(data.cells.map(c => c.category))].filter(c => c !== '_overall').sort()
    const map = new Map(data.cells.map(c => [`${c.run_id}|${c.category}`, c.value]))
    return { runs, cats: ['_overall', ...cats], value: (r: string, c: string) => map.get(`${r}|${c}`) ?? null }
  }, [data])

  return (
    <>
      <PageHead title="By category" sub="One metric broken down by question category, for every benchmark run on a dataset." />
      <div className="row" style={{ margin: '8px 0 12px' }}>
        <select value={data.dataset ?? ''} onChange={e => setDataset(e.target.value)}>
          {data.datasets.map(d => <option key={d}>{d}</option>)}
        </select>
        <select value={metric} onChange={e => setMetric(e.target.value)}>
          {data.metrics.filter(m => m !== 'n').map(m => <option key={m}>{m}</option>)}
        </select>
        <MetricHelp metric={metric} />
        <span className="muted small">Green = high, red = low. Empty = metric not computed for that run (e.g. no judge).</span>
      </div>
      <div className="table-wrap"><table className="heat">
        <thead><tr><th>run</th>{cats.map(c => <th key={c} className="num">{c === '_overall' ? 'overall' : c}</th>)}</tr></thead>
        <tbody>{runs.map(r => (
          <tr key={r.run_id}>
            <td><Link to={`/runs/${r.run_id}`}>{r.name || r.run_id}</Link> <span className="muted small">snap {r.snapshot}</span></td>
            {cats.map(c => {
              const v = value(r.run_id, c)
              return <td key={c} className="cell" style={{ background: isPct(metric) ? heat(v) : undefined }}>{fmt(v)}</td>
            })}
          </tr>))}</tbody>
      </table></div>
      {runs.length === 0 && <p className="muted">No runs have this metric yet.</p>}
    </>
  )
}
