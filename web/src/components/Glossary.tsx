import { useEffect, useRef, useState, type ReactNode } from 'react'
import { GLOSSARY, lookup, type Entry } from '../glossary'

// Open the glossary panel, optionally scrolled to one term.
export const openGlossary = (key?: string) => window.dispatchEvent(new CustomEvent('glossary', { detail: key }))

// A term with an explanation on hover; click opens the glossary panel at it.
// Unknown keys render as plain text, so it is safe to wrap any metric name.
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const e = lookup(k)
  if (!e) return <>{children ?? k}</>
  return (
    <span className="term" tabIndex={0} onClick={ev => { ev.stopPropagation(); openGlossary(k) }}
      onKeyDown={ev => { if (ev.key === 'Enter') openGlossary(k) }}>
      {children ?? k}
      <span className="term-tip" role="tooltip"><b>{e.title}</b>{e.short}<span className="faint"> Click for more.</span></span>
    </span>)
}

// The one-line explanation of a metric, for under a metric picker.
export function MetricHelp({ metric }: { metric: string }) {
  const e = lookup(metric)
  return e ? <span className="small muted metric-help">{e.short} <a onClick={() => openGlossary(metric)}>More</a></span> : null
}

// A side panel with every term, grouped, searchable. Mounted once in App.
export function GlossaryPanel() {
  const [open, setOpen] = useState(false)
  const [focus, setFocus] = useState<string>()
  const [q, setQ] = useState('')
  const body = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const on = (e: Event) => { const key = (e as CustomEvent).detail; setOpen(true); setQ(''); setFocus(key ? (lookup(key) ? keyOf(key) : undefined) : undefined) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('glossary', on)
    window.addEventListener('keydown', esc)
    return () => { window.removeEventListener('glossary', on); window.removeEventListener('keydown', esc) }
  }, [])
  useEffect(() => { if (open && focus) body.current?.querySelector(`[data-key="${CSS.escape(focus)}"]`)?.scrollIntoView({ block: 'start' }) }, [open, focus])
  if (!open) return null
  const match = (e: Entry) => !q || `${e.title} ${e.short} ${e.long ?? ''}`.toLowerCase().includes(q.toLowerCase())
  const groups = [...new Set(Object.values(GLOSSARY).map(e => e.group))]
  return (
    <aside className="glossary" aria-label="Glossary">
      <div className="glossary-head"><b>Glossary</b><span className="grow" /><button className="sm ghost" onClick={() => setOpen(false)}>Close</button></div>
      <input placeholder="Search terms…" value={q} onChange={e => setQ(e.target.value)} autoFocus />
      <div className="glossary-body" ref={body}>
        {groups.map(g => {
          const items = Object.entries(GLOSSARY).filter(([, e]) => e.group === g && match(e))
          return items.length ? <section key={g}><div className="section-title">{g}</div>
            {items.map(([k, e]) => (
              <div key={k} data-key={k} className={`glossary-item ${focus === k ? 'focus' : ''}`}>
                <b>{e.title}</b><p>{e.short}</p>{e.long && <p className="muted">{e.long}</p>}</div>))}</section> : null
        })}
      </div>
    </aside>)
}

const keyOf = (k: string) => Object.keys(GLOSSARY).find(x => GLOSSARY[x] === lookup(k))
