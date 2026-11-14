import { useCallback, useEffect, useState } from 'react'
import { errText, get } from '../api'

interface Version { version: number; saved_at: string; code: string }

// Above a script node's code: which saved version it is, a picker to load any version,
// and Save, which saves the pipeline and so records the script as a new version.
export default function ScriptVersions({ code, versionKey, onPick, onSave }: {
  code: string; versionKey?: string; onPick: (code: string) => void; onSave?: (code: string) => Promise<unknown>
}) {
  const [versions, setVersions] = useState<Version[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string>()
  const load = useCallback(() => {
    if (versionKey) get<Version[]>(`/api/scripts/versions?code=true&key=${encodeURIComponent(versionKey)}`).then(setVersions).catch(() => setVersions([]))
  }, [versionKey])
  useEffect(() => { load() }, [load])

  const match = versions.find(v => v.code === code)          // the saved version the editor currently holds, if any
  const latest = versions[0]
  const pick = (n: number) => {
    const v = versions.find(x => x.version === n)
    if (!v || v.code === code) return
    if (!match && !window.confirm('Your unsaved changes to this script will be replaced. Continue?')) return
    onPick(v.code)
  }
  const save = () => { if (!onSave) return; setBusy(true); setErr(undefined); onSave(code).then(load).catch(e => setErr(errText(e))).finally(() => setBusy(false)) }

  return (
    <div className="script-versions">
      <label>Python</label>
      {versionKey ? <>
        <select value={match?.version ?? ''} onChange={e => pick(Number(e.target.value))} aria-label="Script version" title="Load a saved version of this script">
          {!match && <option value="">unsaved changes</option>}
          {versions.map(v => <option key={v.version} value={v.version}>v{v.version}{v === latest ? ' (latest)' : ''} · {v.saved_at.replace('T', ' ').slice(5, 16)}</option>)}
        </select>
        {match && match !== latest && <span className="small warn-text">older version loaded</span>}
        {!match && <span className="small warn-text">● unsaved</span>}
      </> : <span className="small muted">save the pipeline to start keeping versions</span>}
      <span className="grow" />
      {onSave && <button className="sm primary" disabled={busy || (!!latest && code === latest.code)} onClick={save}
        title="Save this script now (saves the pipeline) and record it as a new version">
        {busy ? 'Saving…' : latest && code === latest.code ? `Saved · v${latest.version}` : match ? `Use v${match.version} (save as new version)` : 'Save version'}</button>}
      {err && <span className="small bad-text">{err}</span>}
    </div>)
}
