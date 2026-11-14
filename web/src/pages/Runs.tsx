import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fmt, get, heat, type RunRow } from '../api'
import { Empty, PageHead } from '../components/ui'
import { Term } from '../components/Glossary'

// Columns shown by default; any other param is still filterable.
const PARAM_COLS = ['target.name', 'retrieve.impl', 'retrieve.top_k', 'rerank.impl', 'generate.impl']
const METRIC_COLS = ['recall@10', 'context_recall', 'answer_recall', 'correctness', 'completeness']

type Filter = { key: string; value: string }

export default function Runs() {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [error, setError] = useState<string>()
  const [sort, setSort] = useState<{ col: string; desc: boolean }>({ col: 'created_at', desc: true })
  const [filters, setFilters] = useState<Filter[]>([])
  const [newKey, setNewKey] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const nav = useNavigate()

  useEffect(() => { get<RunRow[]>('/api/runs').then(setRuns).catch(e => setError(String(e))) }, [])

  const paramKeys = useMemo(() => [...new Set(runs.flatMap(r => Object.keys(r.params)))].sort(), [runs])
  const valuesFor = (k: string) => [...new Set(runs.map(r => JSON.stringify(r.params[k] ?? null)))].sort()

  const value = (r: RunRow, col: string): unknown =>
    col in r.overall ? r.overall[col] : col in r.params ? r.params[col] : (r as unknown as Record<string, unknown>)[col]

  const rows = useMemo(() => {
    const filtered = runs.filter(r => (!dataset || r.dataset === dataset) && filters.every(f => JSON.stringify(r.params[f.key] ?? null) === f.value))
    return [...filtered].sort((a, b) => {
      const va = value(a, sort.col), vb = value(b, sort.col)
      if (va === vb) return 0
      if (va === null || va === undefined) return 1
      if (vb === null || vb === undefined) return -1
      return (va < vb ? -1 : 1) * (sort.desc ? -1 : 1)
    })
  }, [runs, filters, sort, dataset])
  const datasets = [...new Set(runs.map(r => r.dataset))].sort()
  const sameDataset = selected.length === 2 && runs.filter(r => selected.includes(r.run_id)).every((r, _, a) => r.dataset === a[0].dataset)

  const Th = ({ col, label, num }: { col: string; label?: string; num?: boolean }) => (
    <th className={`sortable ${num ? 'num' : ''}`} onClick={() => setSort({ col, desc: sort.col === col ? !sort.desc : true })}>
      <Term k={col === 'run_id' ? 'run' : col}>{label ?? col}</Term>{sort.col === col ? (sort.desc ? ' ▾' : ' ▴') : ''}
    </th>
  )
  const toggle = (id: string) => setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s.slice(-1), id])

  if (error) return <p className="muted">Could not load runs: {error}. Is <code>rag-lab serve</code> running?</p>
  return (
    <>
      <PageHead title="Runs" sub={<>Every benchmark run, newest first. Each is a folder in <code>runs/</code> with its full config. Click a header to sort, filter by any setting, tick two to compare.</>} />
      <div className="row" style={{ margin: '12px 0' }}>
        <select value={dataset} onChange={e => setDataset(e.target.value)}>
          <option value="">All datasets</option>
          {datasets.map(d => <option key={d}>{d}</option>)}
        </select>
        <select value={newKey} onChange={e => setNewKey(e.target.value)}>
          <option value="">Filter by config param…</option>
          {paramKeys.map(k => <option key={k}>{k}</option>)}
        </select>
        {newKey && (
          <select value="" onChange={e => { setFilters([...filters, { key: newKey, value: e.target.value }]); setNewKey('') }}>
            <option value="">value…</option>
            {valuesFor(newKey).map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        )}
        {filters.map((f, i) => (
          <span className="chip" key={i}>{f.key} = {f.value}
            <a href="#" onClick={e => { e.preventDefault(); setFilters(filters.filter((_, j) => j !== i)) }}>×</a></span>
        ))}
        <span style={{ flex: 1 }} />
        {selected.length === 2 && !sameDataset && <span className="small muted">pick two runs on the same dataset</span>}
        <button className="primary" disabled={!sameDataset}
          onClick={() => nav(`/diff?a=${selected[0]}&b=${selected[1]}`)}>Compare {selected.length}/2</button>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr>
            <th></th><Th col="run_id" label="run" /><Th col="status" /><Th col="dataset" /><Th col="snapshot" label="snap" num /><Th col="n_questions" label="questions" num />
            {PARAM_COLS.map(c => <Th key={c} col={c} />)}
            {METRIC_COLS.map(c => <Th key={c} col={c} num />)}
            <Th col="cost_usd" label="cost $" num /><Th col="duration_s" label="time s" num />
          </tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.run_id}>
                <td><input type="checkbox" checked={selected.includes(r.run_id)} onChange={() => toggle(r.run_id)} /></td>
                <td><Link to={`/runs/${r.run_id}`}>{r.description || r.run_id}</Link>
                  <div className="faint small">{String(r.created_at).replace('T', ' ').slice(0, 16)} · {r.git_sha?.slice(0, 7) ?? 'no commit'}{r.git_dirty ? ' (dirty)' : ''}{r.judged ? ' · judged' : ''}</div></td>
                <td><span className={`pill ${r.status}`}>{r.status}</span></td>
                <td className="small">{r.dataset}<div className="muted">{r.subset}</div></td>
                <td className="num">{r.snapshot}</td>
                <td className="num">{r.n_questions}</td>
                {PARAM_COLS.map(c => <td key={c} className="small">{fmt(r.params[c])}</td>)}
                {METRIC_COLS.map(c => <td key={c} className="num" style={{ background: heat(r.overall[c]) }}>{fmt(r.overall[c])}</td>)}
                <td className="num">{fmt(r.cost_usd, 4)}</td>
                <td className="num">{fmt(r.duration_s, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {runs.length === 0 && <Empty>No benchmark runs yet. Open a <Link to="/retrieval">retrieval pipeline</Link> and press “Run benchmark”.</Empty>}
    </>
  )
}
