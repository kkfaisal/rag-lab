import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fmt, get, send, type Job } from '../api'
import { Pill } from './ui'

const ACTIVE = ['queued', 'running']
export const isActive = (j: Job) => ACTIVE.includes(j.status)

// Poll the job list: fast while something is queued or running, slowly otherwise.
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
        if (js.some(isActive)) next = 1500
      } catch { /* server restarting */ }
      timer = window.setTimeout(tick, next)
    }
    tick()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])
  return jobs
}

const KIND: Record<string, string> = { dataset: 'dataset', ingest: 'ingestion', benchmark: 'benchmark' }

function JobRow({ job }: { job: Job }) {
  const [detail, setDetail] = useState<Job>()
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open && job.status !== 'running') return
    get<Job>(`/api/jobs/${job.id}`).then(setDetail).catch(() => undefined)
  }, [job, open])
  const run = detail?.run
  const write = run?.stages.find(s => s.stage === 'write')?.stats
  return (
    <div className="job">
      <div className="row" style={{ gap: 8 }}>
        <Pill status={job.status} />
        <span className="pill plain">{KIND[job.kind] ?? job.kind}</span>
        <span className="grow">{job.title}</span>
        <span className="faint small">{(job.finished_at ?? job.started_at ?? job.created_at).slice(11, 16)}</span>
        {job.run_id && job.kind === 'benchmark' && job.status === 'ok' && <Link to={`/runs/${job.run_id}`}>Results →</Link>}
        {isActive(job) && <button className="sm" onClick={() => send('POST', `/api/jobs/${job.id}/cancel`)}>Cancel</button>}
        <button className="sm ghost" onClick={() => setOpen(!open)}>{open ? 'Hide log' : 'Log'}</button>
      </div>
      {run && (job.status === 'running' || open) && (
        <div className="row" style={{ marginTop: 6, gap: 5 }}>
          {run.stages.map(s => <span key={s.stage} className="chip">{s.stage} · {s.cached ? 'cached' : `${fmt(s.duration_s)}s`}</span>)}
          {run.current_stage && <span className="chip running">{run.current_stage}…</span>}
        </div>)}
      {write && open && <div className="small muted" style={{ marginTop: 6 }}>
        Wrote {String(write.docs_written)} documents ({String(write.chunks_written)} chunks), skipped {String(write.docs_skipped)} unchanged · target version {String(write.target_version)}</div>}
      {job.error && !open && <div className="small" style={{ color: 'var(--bad)', marginTop: 5 }}>{job.error}</div>}
      {open && <pre>{detail?.log || '(no output yet)'}</pre>}
    </div>
  )
}

export function JobList({ jobs, kinds, limit = 8 }: { jobs: Job[]; kinds?: string[]; limit?: number }) {
  const shown = jobs.filter(j => !kinds || kinds.includes(j.kind)).slice(0, limit)
  if (shown.length === 0) return <p className="muted small">No jobs yet.</p>
  return <div>{shown.map(j => <JobRow key={j.id} job={j} />)}</div>
}
