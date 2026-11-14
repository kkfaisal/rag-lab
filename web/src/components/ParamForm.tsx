import type { ImplInfo, SchemaProp } from '../api'
import CodeEditor from './CodeEditor'

const typesOf = (p: SchemaProp): string[] => (p.type ? [p.type] : (p.anyOf ?? []).map(a => a.type ?? '').filter(Boolean))

// A settings form generated from an implementation's JSON schema (which the
// backend derives from its Pydantic config class), so new implementations and
// new parameters show up in the UI without any frontend change.
export default function ParamForm({ info, params, onChange, scriptKind, versionKey }: { info: ImplInfo; params: Record<string, unknown>; onChange: (p: Record<string, unknown>) => void; scriptKind?: string; versionKey?: string }) {
  const props = info.schema.properties ?? {}
  const set = (k: string, v: unknown) => onChange({ ...params, [k]: v })
  return (
    <div className="form">
      {Object.entries(props).map(([key, p]) => {
        const types = typesOf(p)
        const nullable = types.includes('null')
        const v = params[key] ?? info.defaults[key] ?? null
        if (key === 'code') return (
          <div key={key} className="field wide"><label>Python</label><CodeEditor value={String(v ?? '')} onChange={c => set(key, c)} kind={scriptKind} versionKey={versionKey} /></div>)
        if (key === 'system_prompt') return (
          <div key={key} className="field wide"><label>{key}</label>
            <textarea rows={8} value={String(v ?? '')} onChange={e => set(key, e.target.value)} /></div>)
        if (types.includes('boolean')) return (
          <div key={key} className="field"><label><input type="checkbox" checked={Boolean(v)} onChange={e => set(key, e.target.checked)} /> {key}</label></div>)
        if (types.includes('integer') || types.includes('number')) return (
          <div key={key} className="field"><label>{key}</label>
            <input type="number" value={v === null ? '' : Number(v)} step={types.includes('integer') ? 1 : 'any'}
              onChange={e => set(key, e.target.value === '' ? (nullable ? null : 0) : Number(e.target.value))} /></div>)
        if (types.includes('array')) return (
          <div key={key} className="field"><label>{key} <span className="muted">(comma-separated)</span></label>
            <input defaultValue={Array.isArray(v) ? v.join(', ') : ''} placeholder={nullable ? 'all' : ''}
              onBlur={e => {
                const items = e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                const typed = p.items?.type === 'integer' || p.items?.type === 'number' ? items.map(Number).filter(n => !Number.isNaN(n)) : items
                set(key, typed.length === 0 && nullable ? null : typed)
              }} /></div>)
        return (
          <div key={key} className="field"><label>{key}</label>
            <input value={v === null ? '' : String(v)} placeholder={nullable ? 'default' : ''}
              onChange={e => set(key, e.target.value === '' && nullable ? null : e.target.value)} /></div>)
      })}
      {Object.keys(props).length === 0 && <p className="muted small">No settings.</p>}
    </div>
  )
}
