import { useEffect, useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { get, type DatasetInfo } from '../api'

export default function Datasets() {
  const [ds, setDs] = useState<DatasetInfo[] | null>(null)
  useEffect(() => { get<DatasetInfo[]>('/api/datasets').then(setDs) }, [])
  return (
    <>
      <h1>Datasets</h1>
      <p className="muted small">Each <code>rag-lab data add</code> that changes something creates a snapshot. Runs record the snapshot they used.</p>
      {ds === null && <p className="muted">Loading…</p>}
      {ds?.map(d => (
        <div key={d.name} className="panel" style={{ marginTop: 16 }}>
          <h2 style={{ marginTop: 0 }}>{d.name} <span className="badge">{d.adapter}</span></h2>
          <p className="small muted">latest snapshot {d.latest_snapshot} · {Object.values(d.by_source).reduce((a, b) => a + b, 0).toLocaleString()} docs ·
            {' '}{d.n_questions} questions · subsets: {Object.entries(d.subsets).map(([k, v]) => `${k} (${v} q)`).join(', ') || 'all only'}</p>
          <div className="grid2">
            <ResponsiveContainer width="100%" height={Math.max(120, Object.keys(d.by_source).length * 26)}>
              <BarChart layout="vertical" data={Object.entries(d.by_source).map(([source, n]) => ({ source, n }))} margin={{ left: 40 }}>
                <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                <YAxis type="category" dataKey="source" tick={{ fontSize: 11, fill: 'var(--muted)' }} width={90} />
                <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)' }} />
                <Bar dataKey="n" fill="var(--accent)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="scroll"><table>
              <thead><tr><th className="num">#</th><th>created</th><th className="num">added</th><th className="num">changed</th>
                <th className="num">deleted</th><th className="num">docs</th><th>note</th></tr></thead>
              <tbody>{[...d.snapshots].reverse().map(s => (
                <tr key={s.snapshot}><td className="num">{s.snapshot}</td><td className="small">{s.created_at.slice(0, 19)}</td>
                  <td className="num">+{s.added.toLocaleString()}</td><td className="num">~{s.changed.toLocaleString()}</td>
                  <td className="num">-{s.deleted.toLocaleString()}</td><td className="num">{s.n_docs.toLocaleString()}</td>
                  <td className="small muted">{s.note}</td></tr>))}</tbody>
            </table></div>
          </div>
        </div>))}
      {ds?.length === 0 && <p className="muted">No datasets yet.</p>}
    </>
  )
}
