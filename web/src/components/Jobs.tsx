import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fmt, get, send, type Job } from '../api'

const ACTIVE = ['queued', 'running']

// Poll the job list; fast while something is queued or running, slowly otherwise.
export function useJobs(): Job[] {
  const [jobs, setJobs] = useState<Job[]>([])
  useEffect(() => {
    let timer: number, stopped = false
    const tick = async () => {
      let next = 6000
      try {
        const js = await get<Job[]>('/api/jobs')
        if (stopped) return
        setJobs(js)
        if (js.some(j => ACTIVE.includes(j.status))) next = 1500
      } catch { /* server restarting */ }
      timer = window.setTimeout(tick, next)
    }
    tick()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])
  return jobs
}

export const isActive = (j: Job) => ACTIVE.includes(j.status)

function JobRow({ job }: { job: Job }) {
  const [detail, setDetail] = useState<Job>()
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open && job.status !== 'running') return
    get<Job>(`/api/jobs/${job.id}`).then(setDetail).catch(() => undefined)
  }, [job, open])
  const run = detail?.run
  return (
    <div className="job">
      <div className="row" style={{ gap: 8 }}>
        <span className={`badge ${job.status}`}>{job.status}</span>
        <span className="badge">{job.kind}</span>
        <span style={{ flex: 1 }}>{job.title}</span>
        {job.run_id && job.kind === 'experiment' && job.status === 'ok' && <Link to={`/runs/${job.run_id}`}>results →</Link>}
        {isActive(job) && <button onClick={() => send('POST', `/api/jobs/${job.id}/cancel`)}>Cancel</button>}
        <button onClick={() => setOpen(!open)}>{open ? 'Hide log' : 'Log'}</button>
      </div>
      {job.status === 'running' && run && (
        <div className="small muted" style={{ marginTop: 4 }}>
          {run.stages.map(s => <span key={s.stage} className="chip" style={{ marginRight: 4 }}>{s.stage} {s.cached ? 'cached' : `${fmt(s.duration_s)}s`}</span>)}
          {run.current_stage && <span className="chip running">{run.current_stage}…</span>}
        </div>)}
      {job.error && !open && <div className="small" style={{ color: 'var(--miss)', marginTop: 4 }}>{job.error}</div>}
      {open && <pre>{detail?.log || '(no output yet)'}</pre>}
    </div>
  )
}

export function JobList({ jobs, kinds, limit = 8 }: { jobs: Job[]; kinds?: string[]; limit?: number }) {
  const shown = jobs.filter(j => !kinds || kinds.includes(j.kind)).slice(0, limit)
  if (shown.length === 0) return <p className="muted small">No jobs yet.</p>
  return <div>{shown.map(j => <JobRow key={j.id} job={j} />)}</div>
}
