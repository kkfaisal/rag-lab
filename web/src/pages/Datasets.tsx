import { useEffect, useRef, useState } from 'react'
import { errText, get, send, type Catalog, type DatasetInfo } from '../api'
import CodeEditor from '../components/CodeEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'
import { Empty, ErrorNote, PageHead, Pill, Stat } from '../components/ui'

interface Files { path: string; count: number; has_questions: boolean; files: { path: string; size: number }[] }
const SOURCE_LABEL: Record<string, string> = { upload: 'File upload', folder: 'Local folder', enterprise_rag_bench: 'EnterpriseRAG-Bench', script: 'Python script' }
const QUESTION_FORMAT = '{"question_id": "q1", "question": "…", "expected_doc_ids": ["…"], "gold_answer": "…", "category": "…"}'

// Create or update a dataset: pick a data source, supply benchmark questions, ingest.
function DatasetForm({ catalog, existing, initial, onClose }: { catalog: Catalog; existing: DatasetInfo[]; initial: string; onClose: () => void }) {
  const [name, setName] = useState(initial)
  const [source, setSource] = useState('upload')
  const [path, setPath] = useState('')
  const [code, setCode] = useState(catalog.templates.extract)
  const [sync, setSync] = useState(false)
  const [files, setFiles] = useState<Files>()
  const [msg, setMsg] = useState<string>()
  const [error, setError] = useState<string>()
  const docsInput = useRef<HTMLInputElement>(null)
  const ds = existing.find(d => d.name === name)
  const valid = /^[A-Za-z0-9_-]{1,64}$/.test(name)
  const usesUploads = source === 'upload' || (source === 'script' && !path)

  useEffect(() => {
    if (source === 'enterprise_rag_bench' && !name) setName('enterprise_rag_bench')
  }, [source, name])
  useEffect(() => {
    if (ds?.adapter === 'script' && ds.code) { setSource('script'); setCode(ds.code) }
    if (ds?.adapter === 'enterprise_rag_bench') setSource('enterprise_rag_bench')
  }, [ds?.adapter, ds?.code])
  useEffect(() => {
    if (valid) get<Files>(`/api/datasets/${name}/files`).then(setFiles).catch(() => setFiles(undefined))
    else setFiles(undefined)
  }, [name, valid])

  const upload = async (list: FileList | null, questions: boolean) => {
    if (!list?.length) return
    const form = new FormData()
    for (const f of list) form.append('files', f, (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name)
    setError(undefined); setMsg('Uploading…')
    try {
      setFiles(await send<Files>('POST', `/api/datasets/${name}/upload${questions ? '?questions=true' : ''}`, form))
      setMsg(questions ? 'Questions file uploaded' : `${list.length} file(s) uploaded`)
    } catch (e) { setMsg(undefined); setError(errText(e)) }
    if (docsInput.current) docsInput.current.value = ''
  }
  const hasQuestions = Boolean(ds?.n_questions) || Boolean(files?.has_questions) || source === 'enterprise_rag_bench' || source === 'script'
  const hasDocs = source === 'enterprise_rag_bench' || source === 'folder' ? (source === 'folder' ? path.trim() !== '' : true)
    : source === 'script' ? true : (files?.count ?? 0) > (files?.has_questions ? 1 : 0)
  const ingest = () => send('POST', `/api/datasets/${name}/ingest`, { source, sync, params: { path: path.trim() || undefined, code: source === 'script' ? code : undefined } })
    .then(() => { setError(undefined); setMsg('Ingest queued — follow it under Jobs below') }).catch(e => setError(errText(e)))

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-head"><h2>{ds ? `Update “${ds.name}”` : 'New dataset'}</h2><div className="actions"><button className="ghost sm" onClick={onClose}>Close</button></div></div>

      <div className="section-title" style={{ marginTop: 0 }}>1 · Data source</div>
      <div className="option-cards">
        {Object.entries(catalog.sources).map(([key, desc]) => (
          <div key={key} className={`option ${source === key ? 'selected' : ''}`} onClick={() => setSource(key)}>
            <b>{SOURCE_LABEL[key] ?? key}</b><span>{desc}</span></div>))}
      </div>

      <div className="form" style={{ marginTop: 14 }}>
        <div className="field"><label>Dataset name</label>
          <input value={name} placeholder="my_docs" onChange={e => setName(e.target.value.trim())} />
          {name && !valid ? <span className="hint" style={{ color: 'var(--bad)' }}>Letters, digits, _ and - only</span>
            : ds ? <span className="hint">Existing dataset: new and changed documents become snapshot #{ds.latest_snapshot + 1}</span> : null}</div>
        {(source === 'folder' || source === 'script') && (
          <div className="field"><label>Folder on this machine{source === 'script' ? ' (optional)' : ''}</label>
            <input value={path} placeholder="/path/to/documents" onChange={e => setPath(e.target.value)} />
            <span className="hint">{source === 'script' ? 'Passed to your script as `path`. Empty = the upload folder below.' : 'Must contain the documents and a questions.jsonl'}</span></div>)}
      </div>

      {source === 'enterprise_rag_bench' && <div className="notice">Downloads the benchmark release (1.26 GB), loads ~512k documents with its 500 questions, and builds the “dev” subset (100 questions, their gold documents and 20k distractors). Takes a few minutes and ~6 GB of disk.</div>}

      {valid && usesUploads && <>
        <div className="row" style={{ marginTop: 4 }}>
          <input ref={docsInput} type="file" multiple onChange={e => upload(e.target.files, false)} />
          <span className="small muted">.txt, .md, .jsonl files or a .zip of them{source === 'script' ? ' — or anything your script can read' : ''}</span>
        </div>
        {files && files.count > 0 && <details style={{ marginTop: 6 }}><summary className="small">{files.count} file(s) in the upload folder</summary>
          <pre>{files.files.map(f => `${f.path}  (${f.size.toLocaleString()} B)`).join('\n')}</pre></details>}
      </>}
      {source === 'script' && <div style={{ marginTop: 10 }}><CodeEditor value={code} onChange={setCode} height="280px" kind="extract" /></div>}

      <div className="section-title">2 · Benchmark questions <span className="muted">required</span>
        {hasQuestions ? <Pill status="ok">{ds?.n_questions ? `${ds.n_questions} loaded` : source === 'script' ? 'from script' : source === 'enterprise_rag_bench' ? 'included' : 'file ready'}</Pill> : <Pill status="warn">missing</Pill>}</div>
      {source === 'script' ? <p className="small muted">Your script's <code>questions(path)</code> function must yield them.</p>
        : source === 'enterprise_rag_bench' ? <p className="small muted">The benchmark ships its own 500 questions.</p>
        : source === 'folder' ? <p className="small muted">Put a <code>questions.jsonl</code> in the folder.</p>
        : valid && <div className="row"><input type="file" accept=".jsonl,.json" onChange={e => upload(e.target.files, true)} /><span className="small muted">questions.jsonl</span></div>}
      {source !== 'enterprise_rag_bench' && <p className="small faint">One JSON object per line: <code>{QUESTION_FORMAT}</code>. <code>expected_doc_ids</code> are document ids (file paths for uploaded files) and drive the recall metrics; <code>gold_answer</code> enables answer judging.</p>}

      <div className="row" style={{ marginTop: 12 }}>
        {source !== 'enterprise_rag_bench' && <label className="small"><input type="checkbox" checked={sync} onChange={e => setSync(e.target.checked)} /> Remove documents that are no longer in the source</label>}
        <span className="grow" />
        {msg && <span className="small muted">{msg}</span>}
        <button className="primary" disabled={!valid || !hasDocs || !hasQuestions} onClick={ingest}>{ds ? 'Ingest changes' : 'Create dataset'}</button>
      </div>
      <ErrorNote error={error} />
    </div>
  )
}

export default function Datasets() {
  const [ds, setDs] = useState<DatasetInfo[] | null>(null)
  const [catalog, setCatalog] = useState<Catalog>()
  const [form, setForm] = useState<string | null>(null)       // null = closed, '' = new, name = update
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length
  useEffect(() => { get<Catalog>('/api/catalog').then(setCatalog) }, [])
  useEffect(() => { get<DatasetInfo[]>('/api/datasets').then(setDs).catch(() => setDs([])) }, [busy])
  return (
    <>
      <PageHead title="Datasets" sub="A dataset is source documents plus benchmark questions. Every ingest that changes something creates a snapshot, so results stay tied to the exact documents they used.">
        <button className="primary" onClick={() => setForm('')}>New dataset</button>
      </PageHead>
      {catalog && form !== null && <DatasetForm key={form} catalog={catalog} existing={ds ?? []} initial={form} onClose={() => setForm(null)} />}
      {jobs.some(j => j.kind === 'dataset') && <><div className="section-title">Recent dataset jobs</div><JobList jobs={jobs} kinds={['dataset']} limit={3} /></>}

      {ds === null && <p className="muted">Loading…</p>}
      {ds?.length === 0 && form === null && <Empty>No datasets yet. Create one from uploaded files, a local folder, EnterpriseRAG-Bench or your own Python script.</Empty>}
      <div style={{ marginTop: 14 }}>
        {ds?.map(d => {
          const docs = Object.values(d.by_source).reduce((a, b) => a + b, 0)
          return (
            <div key={d.name} className="card">
              <div className="card-head">
                <h2>{d.name}</h2><span className="pill plain">{SOURCE_LABEL[d.adapter === 'folder' ? 'upload' : d.adapter] ?? d.adapter}</span>
                {d.n_questions > 0 ? <Pill status="ok">ready</Pill> : <Pill status="warn">no questions</Pill>}
                <div className="actions"><button className="sm" onClick={() => { setForm(d.name); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Add data</button></div>
              </div>
              <div className="stats">
                <Stat value={docs.toLocaleString()} label="documents" />
                <Stat value={d.n_questions.toLocaleString()} label="questions" />
                <Stat value={`#${d.latest_snapshot}`} label="latest snapshot" />
                <Stat value={Object.keys(d.by_source).length} label="source types" />
                {Object.entries(d.subsets).map(([k, v]) => <Stat key={k} value={v} label={`questions in “${k}” subset`} />)}
              </div>
              <details style={{ marginTop: 10 }}><summary className="small">Source types and snapshot history</summary>
                <div className="grid2" style={{ marginTop: 8 }}>
                  <div className="table-wrap"><table><thead><tr><th>source type</th><th className="num">documents</th></tr></thead>
                    <tbody>{Object.entries(d.by_source).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.toLocaleString()}</td></tr>)}</tbody></table></div>
                  <div className="table-wrap"><table><thead><tr><th className="num">#</th><th>created</th><th className="num">added</th><th className="num">changed</th><th className="num">deleted</th><th className="num">total</th></tr></thead>
                    <tbody>{[...d.snapshots].reverse().map(s => (
                      <tr key={s.snapshot}><td className="num">{s.snapshot}</td><td className="small">{s.created_at.slice(0, 16)}</td>
                        <td className="num">+{s.added.toLocaleString()}</td><td className="num">~{s.changed.toLocaleString()}</td>
                        <td className="num">−{s.deleted.toLocaleString()}</td><td className="num">{s.n_docs.toLocaleString()}</td></tr>))}</tbody></table></div>
                </div></details>
            </div>)
        })}
      </div>
    </>
  )
}
