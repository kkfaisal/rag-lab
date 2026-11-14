import { useEffect, useMemo, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { unifiedMergeView } from '@codemirror/merge'
import { get, send, errText } from '../api'

const dark = () => window.matchMedia('(prefers-color-scheme: dark)').matches

interface Contract { template: string; required: string[]; optional: string[] }
interface ScriptsInfo { contracts: Record<string, Contract>; assist: boolean; assist_model: string }
interface Check { ok: boolean; message: string; line: number | null }
interface Proposal { code: string; explanation: string; model: string; cost_usd: number; version?: number }
interface Version { version: number; saved_at: string; note: string; lines: number }

let infoPromise: Promise<ScriptsInfo> | null = null
const scriptsInfo = () => (infoPromise ??= get<ScriptsInfo>('/api/scripts/info'))

const KIND_LABEL: Record<string, string> = {
  chunk: 'Chunk script', embed: 'Embed script', retrieve: 'Retrieve script', rerank: 'Rerank script',
  extract: 'Data source script', target: 'Target script',
}

// A Python editor for script nodes, data-source scripts and script targets.
// Inline it is compact; "Open editor" opens a full-window editor with a static check,
// the stage's contract, saved versions, and optional AI help that proposes changes as a diff.
//   versionKey   where this script's saved versions live (see backend versions.py)
//   onStudioSave lets the full-window editor edit a read-only script (e.g. a built-in node's
//                code: saving turns the node into a Python node); `notice` explains that
export default function CodeEditor({ value, onChange, height = '320px', readOnly = false, kind, title, versionKey, onStudioSave, notice }: {
  value: string; onChange?: (v: string) => void; height?: string; readOnly?: boolean; kind?: string; title?: string
  versionKey?: string; onStudioSave?: (v: string) => void; notice?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="code-editor">
      <div className="code-bar">
        <span className="small faint">{readOnly ? 'Python · read-only' : 'Python'}</span>
        <button className="sm ghost" onClick={() => setOpen(true)} title="Open a full-window editor">⤢ {readOnly && !onStudioSave ? 'Expand' : 'Open editor'}</button>
      </div>
      <CodeMirror value={value} height={height} theme={dark() ? 'dark' : 'light'} extensions={[python()]}
        basicSetup={{ foldGutter: false, highlightActiveLine: false }} onChange={onChange} readOnly={readOnly} editable={!readOnly} />
      {open && <Studio initial={value} readOnly={readOnly && !onStudioSave} kind={kind} title={title} versionKey={versionKey} notice={notice}
        onClose={() => setOpen(false)} onSave={v => { (onStudioSave ?? onChange)?.(v); setOpen(false) }} />}
    </div>
  )
}

function Studio({ initial, readOnly, kind, title, versionKey, notice, onClose, onSave }: {
  initial: string; readOnly: boolean; kind?: string; title?: string; versionKey?: string; notice?: string
  onClose: () => void; onSave: (v: string) => void
}) {
  const [draft, setDraft] = useState(initial)
  const [info, setInfo] = useState<ScriptsInfo | null>(null)
  const [check, setCheck] = useState<Check | null>(null)
  const [tab, setTab] = useState<'ai' | 'contract' | 'versions'>(readOnly ? 'contract' : 'ai')
  const [versions, setVersions] = useState<Version[] | null>(null)
  useEffect(() => { if (versionKey) get<Version[]>(`/api/scripts/versions?key=${encodeURIComponent(versionKey)}`).then(setVersions).catch(() => setVersions([])) }, [versionKey])
  const pickVersion = (v: Version) => get<{ code: string }>(`/api/scripts/versions/${v.version}?key=${encodeURIComponent(versionKey!)}`)
    .then(r => setProposal({ code: r.code, explanation: `Version ${v.version} · saved ${v.saved_at.replace('T', ' ')}`, model: '', cost_usd: 0, version: v.version }))
    .catch(e => setErr(errText(e)))
  const [ask, setAsk] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [proposal, setProposal] = useState<Proposal | null>(null)
  const [log, setLog] = useState<{ ask: string; explanation: string; accepted?: boolean }[]>([])
  const contract = kind ? info?.contracts[kind] : undefined
  const changed = draft !== initial

  useEffect(() => { scriptsInfo().then(setInfo).catch(() => setInfo({ contracts: {}, assist: false, assist_model: '' })) }, [])
  useEffect(() => {      // re-check a moment after typing stops
    const t = setTimeout(() => send<Check>('POST', '/api/scripts/check', { kind, code: draft }).then(setCheck).catch(() => {}), 400)
    return () => clearTimeout(t)
  }, [draft, kind])

  const close = () => { if (!changed || readOnly || confirm('Discard your changes to this script?')) onClose() }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { if (proposal?.version !== undefined) setProposal(null); else if (!proposal) close() }   // Esc = Back from a version
      if ((e.metaKey || e.ctrlKey) && e.key === 's' && !readOnly) { e.preventDefault(); onSave(draft) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const run = (instruction: string) => {
    setBusy(true); setErr(null)
    send<Proposal>('POST', '/api/assist', { kind, code: draft, instruction, error: check && !check.ok ? check.message : undefined })
      .then(p => { setProposal(p); setLog(l => [...l, { ask: instruction, explanation: p.explanation }]); setAsk('') })
      .catch(e => setErr(errText(e))).finally(() => setBusy(false))
  }
  const decide = (accept: boolean) => {
    if (accept && proposal) setDraft(proposal.code)
    if (proposal?.version === undefined) setLog(l => l.map((x, i) => i === l.length - 1 ? { ...x, accepted: accept } : x))
    setProposal(null)
  }
  const mergeExt = useMemo(() => proposal ? [python(), unifiedMergeView({ original: draft, mergeControls: false })] : [], [proposal]) // eslint-disable-line react-hooks/exhaustive-deps

  const suggestions = [
    ...(check && !check.ok ? ['Fix the problem the check reports'] : []),
    'Explain what this code does, step by step, in comments',
    'Make it faster for large inputs',
    'Add error handling for malformed input',
  ]

  return (
    <div className="studio-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) close() }}>
      <div className="studio" role="dialog" aria-label="Code editor">
        <div className="studio-head">
          <b>{title ?? (kind ? KIND_LABEL[kind] ?? kind : 'Python')}</b>
          {readOnly && <span className="pill plain">read-only</span>}
          {notice && !readOnly && <span className="small muted">{notice}</span>}
          {changed && !readOnly && <span className="pill warn">unsaved</span>}
          {check && <span className={`small ${check.ok ? 'ok-text' : 'bad-text'}`} title={check.message}>{check.ok ? '✓ ' : '✕ '}{check.message}</span>}
          <span className="grow" />
          {contract && !readOnly && <button className="sm" onClick={() => { if (confirm('Replace the code with the starting template?')) setDraft(contract.template) }}>Reset to template</button>}
          {readOnly ? <button className="sm primary" onClick={onClose}>Close</button> : <>
            <button className="sm" onClick={close}>Cancel</button>
            <button className="sm primary" disabled={!!proposal} onClick={() => onSave(draft)} title="⌘S">Save</button></>}
        </div>
        <div className="studio-body">
          <div className="studio-code">
            {proposal ? <>
              <div className="proposal-bar">
                {proposal.version !== undefined
                  ? <span className="small"><b>{proposal.explanation}</b> · shown as changes from the code in the editor</span>
                  : <span className="small"><b>Suggested change</b> · {proposal.model} · ${proposal.cost_usd.toFixed(4)}</span>}
                <span className="grow" />
                <button className="sm" onClick={() => decide(false)}>{proposal.version !== undefined ? 'Back' : 'Discard'}</button>
                {!readOnly && <button className="sm primary" onClick={() => decide(true)}>{proposal.version !== undefined ? 'Use this version' : 'Accept change'}</button>}
              </div>
              <CodeMirror key="proposal" value={proposal.code} height="100%" className="cm-fill" theme={dark() ? 'dark' : 'light'}
                extensions={mergeExt} readOnly editable={false} />
            </> : (
              <CodeMirror key="draft" value={draft} height="100%" className="cm-fill" theme={dark() ? 'dark' : 'light'} extensions={[python()]}
                onChange={setDraft} readOnly={readOnly} editable={!readOnly} autoFocus />)}
          </div>
          <aside className="studio-side">
            <div className="tabs">
              {!readOnly && <button className={tab === 'ai' ? 'active' : ''} onClick={() => setTab('ai')}>AI assist</button>}
              <button className={tab === 'contract' ? 'active' : ''} onClick={() => setTab('contract')}>Contract</button>
              {versionKey && <button className={tab === 'versions' ? 'active' : ''} onClick={() => setTab('versions')}>Versions{versions?.length ? ` (${versions.length})` : ''}</button>}
            </div>
            {tab === 'versions' && (
              <div className="side-pane">
                <p className="small muted" style={{ marginTop: 0 }}>Every save that changes this script records a version. Pick one to see how it differs from the code in the editor, then use it or go back.</p>
                {err && <div className="notice bad small">{err}</div>}
                {versions === null ? <p className="small muted">Loading…</p> : versions.length === 0 ? <p className="small muted">No saved versions yet. The first one is recorded when you save.</p> : (
                  <div className="version-list">{versions.map((v, i) => (
                    <button key={v.version} className={`version ${proposal?.version === v.version ? 'active' : ''}`} onClick={() => pickVersion(v)}>
                      <b>v{v.version}</b>{i === 0 && <span className="pill plain">latest</span>}
                      <span className="small muted">{v.saved_at.replace('T', ' ').slice(0, 16)} · {v.lines} lines</span>
                    </button>))}</div>)}
              </div>)}
            {tab === 'ai' && !readOnly && (
              <div className="side-pane">
                {info && !info.assist ? (
                  <div className="notice">AI help is off. Put <code>ANTHROPIC_API_KEY</code> in <code>.env</code> and restart rag-lab to turn it on. Everything else works without it.</div>
                ) : <>
                  <p className="small muted" style={{ marginTop: 0 }}>Describe a change. Claude ({info?.assist_model ?? '…'}) rewrites the script within this stage's contract and you review the diff before anything changes.</p>
                  <textarea rows={4} value={ask} placeholder="e.g. split on markdown headings and keep each heading with its section"
                    onChange={e => setAsk(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && ask.trim() && !busy) run(ask) }} />
                  <div className="row" style={{ marginTop: 6 }}><span className="small faint">⌘↵ to send</span><span className="grow" />
                    <button className="primary sm" disabled={busy || !ask.trim() || !!proposal} onClick={() => run(ask)}>{busy ? 'Thinking…' : 'Suggest change'}</button></div>
                  <div className="chips">{suggestions.map(s => <button key={s} className="chip" disabled={busy || !!proposal} onClick={() => run(s)}>{s}</button>)}</div>
                  {err && <div className="notice bad small" style={{ marginTop: 8 }}>{err}</div>}
                  {log.length > 0 && <div className="section-title">This session</div>}
                  {log.slice().reverse().map((x, i) => (
                    <div key={i} className="assist-log">
                      <div className="small"><b>{x.ask}</b> {x.accepted === true ? <span className="pill ok">accepted</span> : x.accepted === false ? <span className="pill plain">discarded</span> : <span className="pill warn">review</span>}</div>
                      <div className="small muted">{x.explanation}</div>
                    </div>))}
                </>}
              </div>)}
            {tab === 'contract' && (
              <div className="side-pane">
                {contract ? <>
                  <p className="small" style={{ marginTop: 0 }}>rag-lab calls {contract.required.map(f => <code key={f}>{f}</code>).reduce((a, b) => <>{a}, {b}</>)}
                    {contract.optional.length > 0 && <> and, if defined, {contract.optional.map(f => <code key={f}>{f}</code>).reduce((a, b) => <>{a}, {b}</>)}</>}.</p>
                  <p className="small muted">The starting template, with each function's arguments and return shape in its docstring:</p>
                  <CodeMirror value={contract.template} theme={dark() ? 'dark' : 'light'} extensions={[python()]} readOnly editable={false}
                    basicSetup={{ lineNumbers: false, foldGutter: false, highlightActiveLine: false }} />
                </> : <p className="small muted">This code has no fixed contract.</p>}
              </div>)}
          </aside>
        </div>
      </div>
    </div>
  )
}
