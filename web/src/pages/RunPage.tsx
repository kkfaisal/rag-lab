import { Fragment, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmt, get, heat, isPct, type QuestionRow, type RunDetail } from '../api'
import { MetricHelp, Term } from '../components/Glossary'

const display = (v: unknown) => (typeof v === 'object' ? JSON.stringify(v) : String(v))
const STAGES = ['target', 'retrieve', 'rerank', 'generate', 'evaluate']
// Per-question columns in the Questions table.
const Q_METRICS = ['recall@1', 'recall@5', 'recall@10', 'answer_recall', 'correctness', 'completeness']

export default function RunPage() {
  const { runId } = useParams()
  const [d, setD] = useState<RunDetail>()
  const [qs, setQs] = useState<QuestionRow[]>([])
  const [metric, setMetric] = useState('recall@10')
  const [error, setError] = useState<string>()

  useEffect(() => {
    get<RunDetail>(`/api/runs/${runId}`).then(setD).catch(e => setError(String(e)))
    get<QuestionRow[]>(`/api/runs/${runId}/questions`).then(setQs).catch(() => setQs([]))
  }, [runId])

  const metricNames = useMemo(() => Object.keys(d?.metrics.overall ?? {}).filter(m => m !== 'n'), [d])
  const chart = useMemo(() => Object.entries(d?.metrics.by_category ?? {}).map(([cat, m]) => ({ cat, value: m[metric] })), [d, metric])

  if (error) return <p>{error}</p>
  if (!d) return <p className="muted">Loading…</p>
  const stageLog = Object.fromEntries(d.stages.map(s => [s.stage, s]))
  const u = d.usage.generate
  return (
    <>
      <p className="small"><Link to="/runs">← Runs</Link></p>
      <h1 style={{ fontSize: 20 }}>{d.config.description || d.run.run_id} <span className={`pill ${d.run.status === 'ok' ? 'ok' : 'failed'}`}>{d.run.status}</span></h1>
      <p className="faint small mono">{d.run.run_id}</p>
      <p className="small muted">
        {d.run.n_questions} questions · subset <b>{String(d.config.data.subset)}</b> · {fmt(d.run.duration_s, 0)}s ·
        git {d.run.git.sha?.slice(0, 10) ?? 'none'}{d.run.git.dirty ? ' (dirty — diff saved)' : ''} ·
        judge {d.metrics.judged ? 'ran' : 'off'}{d.run.bench_commit ? ` (bench ${d.run.bench_commit.slice(0, 7)})` : ''}
      </p>
      {d.error && <details open><summary>Error</summary><pre>{d.error}</pre></details>}

      <div className="section-title">Pipeline</div>
      <div className="pipeline">
        {STAGES.map(s => {
          const cfg = d.config[s] ?? {}
          const log = stageLog[s]
          return (
            <div className="stage" key={s}>
              <h3>{s}</h3>
              <div className="impl">{String(cfg.impl ?? cfg.name ?? '')}</div>
              <dl>
                {Object.entries(cfg).filter(([k]) => !['impl', 'system_prompt', 'code', 'name', 'id'].includes(k)).map(([k, v]) => (
                  <Fragment key={k}><dt>{k}</dt><dd title={display(v)}>{display(v).length > 40 ? display(v).slice(0, 38) + '…' : display(v)}</dd></Fragment>
                ))}
              </dl>
              {log && <div className="small" style={{ marginTop: 8 }}>
                {log.cached ? <span className="pill plain">cached</span> : <span className="pill plain">ran {fmt(log.duration_s)}s</span>}{' '}
                <span className="mono muted" title={`code ${d.code_hashes[s]?.hash}`}>{log.key}</span>
              </div>}
            </div>
          )
        })}
      </div>
      {typeof d.config.generate?.system_prompt === 'string' &&
        <details><summary>generate.system_prompt</summary><pre>{d.config.generate.system_prompt}</pre></details>}

      <div className="section-title">Metrics by category</div>
      <div className="grid2">
        <div className="card">
          <div className="row"><select value={metric} onChange={e => setMetric(e.target.value)}>
            {metricNames.map(m => <option key={m}>{m}</option>)}</select>
            <span className="muted small">overall: <b>{fmt(d.metrics.overall?.[metric])}</b></span></div>
          <MetricHelp metric={metric} />
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chart} margin={{ top: 12, right: 8, bottom: 48, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis dataKey="cat" angle={-35} textAnchor="end" interval={0} tick={{ fontSize: 11, fill: 'var(--muted)' }} />
              <YAxis domain={isPct(metric) ? [0, 100] : [0, 'auto']} tick={{ fontSize: 11, fill: 'var(--muted)' }} />
              <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)' }} />
              <Bar dataKey="value" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3 style={{ margin: '0 0 8px', fontSize: 13 }}>Usage & cost</h3>
          {u ? <table><tbody>
            <tr><td>LLM requests</td><td className="num">{u.requests}</td></tr>
            <tr><td>input tokens</td><td className="num">{u.input_tokens.toLocaleString()}</td></tr>
            <tr><td>output tokens</td><td className="num">{u.output_tokens.toLocaleString()}</td></tr>
            <tr><td>cache read / write tokens</td><td className="num">{u.cache_read_input_tokens.toLocaleString()} / {u.cache_creation_input_tokens.toLocaleString()}</td></tr>
            <tr><td>generation cost</td><td className="num">${fmt(u.cost_usd, 4)}</td></tr>
            <tr><td>incurred this run</td><td className="num">${fmt(d.usage.cost_usd_incurred, 4)}{d.usage.generate_cached ? ' (cached)' : ''}</td></tr>
            <tr><td>judge</td><td className="small muted">{d.usage.judge}</td></tr>
          </tbody></table> : <p className="muted">no LLM usage</p>}
          {d.git_diff && <details><summary>git diff at run time</summary><pre>{d.git_diff}</pre></details>}
          <details><summary>full config.json</summary><pre>{JSON.stringify(d.config, null, 2)}</pre></details>
        </div>
      </div>

      <div className="table-wrap"><table style={{ marginTop: 12 }}>
        <thead><tr><th>category</th>{metricNames.concat('n').map(m => <th key={m} className="num"><Term k={m} /></th>)}</tr></thead>
        <tbody>{Object.entries({ overall: d.metrics.overall ?? {}, ...d.metrics.by_category }).map(([cat, m]) => (
          <tr key={cat}><td><b>{cat}</b></td>{metricNames.concat('n').map(k => (
            <td key={k} className="num" style={{ background: isPct(k) && k !== 'abstained' ? heat(m[k]) : undefined }}>{fmt(m[k])}</td>))}</tr>
        ))}</tbody>
      </table></div>

      <div className="section-title">Questions</div>
      <div className="table-wrap"><table>
        <thead><tr><th>id</th><th>category</th><th>question</th>
          {Q_METRICS.map(m => <th key={m} className="num"><Term k={m} /></th>)}</tr></thead>
        <tbody>{qs.map(q => (
          <tr key={q.question_id}>
            <td><Link to={`/runs/${runId}/q/${q.question_id}`}>{q.question_id}</Link></td>
            <td className="small">{q.category}</td>
            <td className="small">{q.question}</td>
            {Q_METRICS.map(m => (
              <td key={m} className="num" style={{ background: heat(q[m] as number) }}>{fmt(q[m])}</td>))}
          </tr>))}</tbody>
      </table></div>
    </>
  )
}
