import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { get, send, type Catalog, type DatasetInfo } from '../api'
import CodeEditor from '../components/CodeEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'

interface Files { path: string; count: number; files: { path: string; size: number }[] }

// Upload files (or a zip), optionally write an extractor script, and ingest.
// Each ingest that changes something becomes a new snapshot of the dataset.
function AddData({ catalog, existing, initial, onQueued }: { catalog: Catalog; existing: DatasetInfo[]; initial: string; onQueued: () => void }) {
  const [name, setName] = useState(initial)
  const [files, setFiles] = useState<Files>()
  const [adapter, setAdapter] = useState<'folder' | 'script'>('folder')
  const [code, setCode] = useState(catalog.templates.extract)
  const [sync, setSync] = useState(false)
  const [msg, setMsg] = useState<string>()
  const input = useRef<HTMLInputElement>(null)
  const valid = /^[A-Za-z0-9_-]{1,64}$/.test(name)
  const ds = existing.find(d => d.name === name)

  useEffect(() => { setName(initial) }, [initial])
  useEffect(() => {
    if (!valid) { setFiles(undefined); return }
    get<Files>(`/api/datasets/${name}/files`).then(setFiles).catch(() => setFiles(undefined))
    if (ds?.adapter === 'script' && ds.code) { setAdapter('script'); setCode(ds.code) }
  }, [name, valid, ds?.adapter, ds?.code])

  const upload = async (list: FileList | null) => {
    if (!list?.length) return
    const form = new FormData()
    for (const f of list) form.append('files', f, (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name)
    setMsg('Uploading…')
    try { setFiles(await send<Files>('POST', `/api/datasets/${name}/upload`, form)); setMsg(`${list.length} file(s) uploaded`) }
    catch (e) { setMsg(String(e)) }
    if (input.current) input.current.value = ''
  }
  const ingest = () => send('POST', `/api/datasets/${name}/ingest`, { adapter, code, sync })
    .then(() => { setMsg('Ingest queued — see Jobs below'); onQueued() }).catch(e => setMsg(String(e)))

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>Add or update a dataset</h2>
      <div className="row">
        <label className="small">name <input value={name} placeholder="my_docs" onChange={e => setName(e.target.value.trim())} /></label>
        {name && !valid && <span className="small" style={{ color: 'var(--miss)' }}>letters, digits, _ and - only</span>}
        {ds && <span className="small muted">existing dataset — new and changed documents become snapshot #{ds.latest_snapshot + 1}</span>}
        {ds?.adapter === 'enterprise_rag_bench' && <span className="small" style={{ color: 'var(--miss)' }}>the benchmark dataset is managed from the CLI</span>}
      </div>
      {valid && ds?.adapter !== 'enterprise_rag_bench' && <>
        <div className="row" style={{ marginTop: 10 }}>
          <input ref={input} type="file" multiple onChange={e => upload(e.target.files)} />
          <span className="muted small">.txt / .md / .jsonl files, a .zip of them, and optionally questions.jsonl</span>
        </div>
        {files && files.count > 0 && (
          <details style={{ marginTop: 6 }}><summary className="small">{files.count} file(s) in the upload folder</summary>
            <pre>{files.files.map(f => `${f.path}  (${f.size.toLocaleString()} B)`).join('\n')}</pre></details>)}
        <div className="row" style={{ marginTop: 10 }}>
          <span className="small">Extract documents with:</span>
          <label className="small"><input type="radio" checked={adapter === 'folder'} onChange={() => setAdapter('folder')} /> built-in reader (one doc per text file / jsonl line)</label>
          <label className="small"><input type="radio" checked={adapter === 'script'} onChange={() => setAdapter('script')} /> my Python script</label>
        </div>
        {adapter === 'script' && <div style={{ marginTop: 8 }}><CodeEditor value={code} onChange={setCode} height="300px" /></div>}
        <div className="row" style={{ marginTop: 10 }}>
          <label className="small"><input type="checkbox" checked={sync} onChange={e => setSync(e.target.checked)} /> remove documents that are no longer in the upload folder</label>
          <button className="primary" disabled={!files || files.count === 0} onClick={ingest}>Ingest</button>
          {msg && <span className="small muted">{msg}</span>}
        </div>
        <p className="small muted">Questions file format, one JSON object per line: <code>{'{"question_id", "question", "expected_doc_ids": [...], "gold_answer", "category"}'}</code>. With the built-in reader a document's id is its file path.</p>
      </>}
    </div>
  )
}

export default function Datasets() {
  const [ds, setDs] = useState<DatasetInfo[] | null>(null)
  const [catalog, setCatalog] = useState<Catalog>()
  const [target, setTarget] = useState('')
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length
  useEffect(() => { get<Catalog>('/api/catalog').then(setCatalog) }, [])
  useEffect(() => { get<DatasetInfo[]>('/api/datasets').then(setDs).catch(() => setDs([])) }, [busy])
  return (
    <>
      <h1>Datasets</h1>
      <p className="muted small">Documents and questions to build pipelines on. Every ingest that changes something creates a snapshot; runs record the snapshot they used.</p>
      {catalog && <AddData catalog={catalog} existing={ds ?? []} initial={target} onQueued={() => undefined} />}
      <h2>Jobs</h2>
      <JobList jobs={jobs} kinds={['ingest']} limit={4} />
      {ds === null && <p className="muted">Loading…</p>}
      {ds?.map(d => (
        <div key={d.name} className="panel" style={{ marginTop: 16 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h2 style={{ margin: 0 }}>{d.name} <span className="badge">{d.adapter}</span></h2>
            {d.adapter !== 'enterprise_rag_bench' && <button onClick={() => { setTarget(d.name); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Add data…</button>}
          </div>
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
