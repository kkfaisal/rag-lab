import { Fragment, useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { get } from '../api'
import { ErrorNote, PageHead } from '../components/ui'

// Guided lessons: small experiments, each with a goal score that is checked
// against your benchmark runs (or checked off by hand when there is no score).

interface Goal { dataset: string; where: Record<string, string>; metric: string; at_least: number; min_questions: number }
interface Lesson {
  id: string; level: string; title: string; learn: string; needs?: string; steps: string[]; hints?: string[]; takeaway: string
  goal?: Goal; status: { tried?: number; best?: number | null; best_run?: string | null; passed?: boolean }
}

// [label](/path) → link, **text** → bold, `code` → code
function rich(text: string): ReactNode[] {
  return text.split(/(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`)/).map((part, i) => {
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
    if (link) return <Link key={i} to={link[2]}>{link[1]}</Link>
    if (part.startsWith('**')) return <b key={i}>{part.slice(2, -2)}</b>
    if (part.startsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>
    return <Fragment key={i}>{part}</Fragment>
  })
}

const DONE_KEY = 'rag-lab.lessons.done'
function readDone(): string[] {
  try { return JSON.parse(localStorage.getItem(DONE_KEY) ?? '[]') } catch { return [] }
}

export default function Lessons() {
  const [lessons, setLessons] = useState<Lesson[]>()
  const [done, setDone] = useState<string[]>(readDone)
  const [error, setError] = useState('')
  useEffect(() => { get<Lesson[]>('/api/lessons').then(setLessons).catch(e => setError(String(e))) }, [])
  const toggle = (id: string) => {
    const next = done.includes(id) ? done.filter(d => d !== id) : [...done, id]
    setDone(next)
    try { localStorage.setItem(DONE_KEY, JSON.stringify(next)) } catch { /* private window: keep it in memory */ }
  }
  const passed = (l: Lesson) => (l.goal ? !!l.status.passed : done.includes(l.id))
  const count = lessons?.filter(passed).length ?? 0
  return (
    <div className="lessons">
      <PageHead title="Lessons" sub={<>Small experiments that teach how to build a really good RAG pipeline. Each has a goal score, checked
        against your benchmark runs. Start with <code>uv run rag-lab quickstart</code>, then work down the list.</>}>
        {lessons && <span className="lesson-progress"><b>{count}</b> of {lessons.length} done</span>}
      </PageHead>
      <ErrorNote error={error} />
      {lessons?.map((l, i) => {
        const ok = passed(l)
        const s = l.status
        return (
          <section key={l.id} className={`card lesson ${ok ? 'passed' : ''}`}>
            <div className="lesson-head">
              <span className="lesson-n">{ok ? '✓' : i + 1}</span>
              <div>
                <div className="lesson-level">{l.level}</div>
                <h2>{l.title}</h2>
              </div>
              <div className="lesson-goal">
                {l.goal ? <>
                  <span className={`pill ${ok ? 'ok' : s.tried ? 'warn' : 'plain'}`}>{ok ? 'Goal reached' : s.tried ? 'Not yet' : 'Not started'}</span>
                  <div className="goal-text">Goal: <b>{l.goal.metric} ≥ {l.goal.at_least}</b> on {l.goal.dataset}{l.goal.min_questions > 1 && <>, {l.goal.min_questions}+ questions</>}</div>
                  {s.best != null && <div className="goal-text">Best so far: <Link to={`/runs/${s.best_run}`}><b>{s.best}</b></Link> ({s.tried} matching {s.tried === 1 ? 'run' : 'runs'})</div>}
                </> : <label className="goal-text"><input type="checkbox" checked={ok} onChange={() => toggle(l.id)} /> Done (no score to check: tick it yourself)</label>}
              </div>
            </div>
            <p className="lesson-learn"><b>You'll learn:</b> {l.learn}</p>
            {l.needs && <p className="lesson-needs"><b>Needs:</b> {l.needs}</p>}
            <ol className="lesson-steps">{l.steps.map((st, k) => <li key={k}>{rich(st)}</li>)}</ol>
            {l.hints && <details className="lesson-hints"><summary>Hints</summary><ul>{l.hints.map((h, k) => <li key={k}>{rich(h)}</li>)}</ul></details>}
            <p className="lesson-takeaway"><b>Takeaway:</b> {l.takeaway}</p>
          </section>)
      })}
    </div>)
}
