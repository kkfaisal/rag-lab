import type { ReactNode } from 'react'

// Small presentational building blocks shared by every page.

export function PageHead({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  return (
    <div className="page-head">
      <div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>
      {children && <div className="actions">{children}</div>}
    </div>
  )
}

const TONE: Record<string, string> = { ok: 'ok', built: 'ok', ready: 'ok', failed: 'failed', invalid: 'failed', running: 'running', queued: '', cancelled: 'cancelled', empty: 'warn' }

export function Pill({ status, children }: { status: string; children?: ReactNode }) {
  return <span className={`pill ${TONE[status] ?? status}`}>{children ?? status}</span>
}

export function Stat({ value, label }: { value: ReactNode; label: string }) {
  return <div className="stat"><div className="v">{value}</div><div className="k">{label}</div></div>
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

export function ErrorNote({ error }: { error?: string }) {
  return error ? <div className="notice bad" style={{ margin: '8px 0', whiteSpace: 'pre-wrap' }}>{error}</div> : null
}

const P = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
// Minimal 16px line icons for the sidebar.
export const Icon = {
  data: <svg viewBox="0 0 16 16" {...P}><ellipse cx="8" cy="4" rx="5" ry="2" /><path d="M3 4v4c0 1.1 2.2 2 5 2s5-.9 5-2V4M3 8v4c0 1.1 2.2 2 5 2s5-.9 5-2V8" /></svg>,
  target: <svg viewBox="0 0 16 16" {...P}><circle cx="8" cy="8" r="5.5" /><circle cx="8" cy="8" r="2" /></svg>,
  ingest: <svg viewBox="0 0 16 16" {...P}><rect x="1.5" y="6" width="4" height="4" rx="1" /><rect x="10.5" y="6" width="4" height="4" rx="1" /><path d="M5.5 8h5M8.5 6l2 2-2 2" /></svg>,
  retrieve: <svg viewBox="0 0 16 16" {...P}><circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5 14 14" /></svg>,
  results: <svg viewBox="0 0 16 16" {...P}><path d="M2 13.5h12M4 11V7M8 11V3M12 11V8.5" /></svg>,
  runs: <svg viewBox="0 0 16 16" {...P}><path d="M3 4h10M3 8h10M3 12h10" /></svg>,
  diff: <svg viewBox="0 0 16 16" {...P}><path d="M5 2v12M11 2v12M5 5H2.5M5 11H2.5M11 5h2.5M11 11h2.5" /></svg>,
  heat: <svg viewBox="0 0 16 16" {...P}><rect x="2" y="2" width="5" height="5" rx="1" /><rect x="9" y="2" width="5" height="5" rx="1" /><rect x="2" y="9" width="5" height="5" rx="1" /><rect x="9" y="9" width="5" height="5" rx="1" /></svg>,
  help: <svg viewBox="0 0 16 16" {...P}><circle cx="8" cy="8" r="6" /><path d="M6.3 6.2a1.8 1.8 0 0 1 3.5.6c0 1.2-1.8 1.5-1.8 2.6M8 11.6v.1" /></svg>,
  jobs: <svg viewBox="0 0 16 16" {...P}><circle cx="8" cy="8" r="5.5" /><path d="M8 5v3l2 1.5" /></svg>,
}
