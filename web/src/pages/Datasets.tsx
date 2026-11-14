import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { bytes, errText, get, send, type DatasetInfo, type DatasetMetrics, type FolderSummary } from '../api'
import CodeEditor from '../components/CodeEditor'
import { isActive, JobList, useJobs } from '../components/Jobs'
import { Empty, ErrorNote, PageHead, Pill, Stat } from '../components/ui'

const BENCH = 'enterprise_rag_bench'
const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/
const QUESTION_FORMAT = '{"question_id": "q1", "question": "…", "expected_doc_ids": ["…"], "gold_answer": "…", "category": "…"}'

interface Row { name: string; folder?: FolderSummary; loaded?: DatasetInfo }
const docCount = (d?: DatasetInfo) => d ? Object.values(d.by_source).reduce((a, b) => a + b, 0) : 0

// A dataset has two explicit steps, each with its own button:
//   Download  raw files into data/datasets/<name>/ (a Python extractor, or files you add)
//   Load      read that folder into rag-lab's internal document store, which pipelines read from
// This page lists datasets, creates new ones, and shows each one's files, metrics, questions and snapshots.
export default function Datasets() {
  const { name } = useParams()
  const [loaded, setLoaded] = useState<DatasetInfo[] | null>(null)
  const [folders, setFolders] = useState<(FolderSummary & { name: string })[] | null>(null)
  const [creating, setCreating] = useState<string | null>(null)     // null = closed, '' = new, name = a public dataset
  const jobs = useJobs()
  const [error, setError] = useState<string>()
  const loadIntoStore = (n: string) => send('POST', `/api/datasets/${n}/ingest`, { source: 'folder', sync: true })
    .then(() => { setError(undefined); reload() }).catch(e => setError(errText(e)))
  const working = (n: string) => jobs.some(j => j.kind === 'dataset' && isActive(j) && j.title.endsWith(` ${n}`))
  const reload = useCallback(() => {
    get<DatasetInfo[]>('/api/datasets').then(setLoaded).catch(() => setLoaded([]))
    get<(FolderSummary & { name: string })[]>('/api/folders').then(setFolders).catch(() => setFolders([]))
  }, [])
  // Refresh whenever a dataset job changes state (a quick job can start and finish between two polls).
  const jobState = jobs.filter(j => j.kind === 'dataset').map(j => `${j.id}:${j.status}`).join()
  useEffect(() => { reload() }, [reload, jobState])

  const rows: Row[] = []
  for (const f of folders ?? []) rows.push({ name: f.name, folder: f, loaded: loaded?.find(d => d.name === f.name) })
  for (const d of loaded ?? []) if (!rows.some(r => r.name === d.name)) rows.push({ name: d.name, folder: d.folder, loaded: d })
  rows.sort((a, b) => a.name.localeCompare(b.name))

  if (name) {
    const row = rows.find(r => r.name === name)
    return loaded === null || folders === null ? <p className="muted">Loading…</p>
      : <DatasetPage key={name} row={row ?? { name }} onChange={reload} jobs={jobs} />
  }
  return (
    <>
      <PageHead title="Datasets" sub="Two steps per dataset. Download puts the raw files (documents + questions.jsonl) in data/datasets/<name>/. Load reads that folder into rag-lab's internal document store: normalized documents and questions, kept as versioned snapshots. Ingestion pipelines read from the store, then chunk, embed or index into a target.">
        <button className="primary" onClick={() => setCreating(c => c === '' ? null : '')}>New dataset</button>
      </PageHead>
      {creating !== null && <NewDataset key={creating} existing={rows.map(r => r.name)} initial={creating || undefined}
        onClose={() => setCreating(null)} onDone={reload} />}
      <ErrorNote error={error} />

      {loaded === null || folders === null ? <p className="muted">Loading…</p> : rows.length === 0 ? (
        <Empty>No datasets on this machine yet. Get one of the public datasets below (each is a Python extractor script you can read first), create your own with <b>New dataset</b>, or put a folder in <code>data/datasets/</code>.</Empty>
      ) : (
        <div className="table-wrap" style={{ marginTop: 14 }}><table>
          <thead><tr><th>dataset</th><th>state</th><th className="num">documents</th><th className="num">questions</th><th className="num">files</th><th className="num">size on disk</th><th>origin</th><th /></tr></thead>
          <tbody>{rows.map(r => (
            <tr key={r.name}>
              <td><Link to={`/datasets/${r.name}`}><b>{r.name}</b></Link></td>
              <td><StateBadge row={r} /></td>
              <td className="num">{r.loaded ? docCount(r.loaded).toLocaleString() : '–'}</td>
              <td className="num">{r.loaded ? r.loaded.n_questions.toLocaleString() : r.folder?.has_questions ? 'file ready' : '–'}</td>
              <td className="num">{r.folder?.exists ? r.folder.n_files.toLocaleString() : '–'}</td>
              <td className="num">{r.folder?.exists ? bytes(r.folder.size) : '–'}</td>
              <td className="small muted">{r.folder?.has_extractor ? 'Python extractor' : r.name === BENCH ? 'EnterpriseRAG-Bench' : 'local files'}</td>
              <td className="num"><LoadButton loaded={!!r.loaded} downloaded={!!r.folder?.exists} working={working(r.name)} onLoad={() => loadIntoStore(r.name)} /></td>
            </tr>))}</tbody>
        </table></div>)}

      <div className="section-title" style={{ marginTop: 22 }}>Public datasets <span className="muted">· Get downloads the raw files with a Python extractor (not part of the repository); Load reads them into the internal store</span></div>
      <PublicDatasets loaded={(loaded ?? []).map(d => d.name)} downloaded={(folders ?? []).map(f => f.name)} working={working} busy={jobState}
        onGet={name => { setCreating(name); window.scrollTo({ top: 0, behavior: 'smooth' }) }} onLoad={loadIntoStore} />
      <p className="small faint">Each dataset keeps its own license (EnterpriseRAG-Bench: MIT; BEIR sets: see their Hugging Face cards). From a terminal: <code>uv run python scripts/get_datasets.py --list</code>.</p>
      {jobs.some(j => j.kind === 'dataset') && <><div className="section-title" style={{ marginTop: 22 }}>Recent dataset jobs <span className="muted">· <Link to="/jobs">all jobs</Link></span></div><JobList jobs={jobs} kinds={['dataset']} limit={5} /></>}
    </>
  )
}

function StateBadge({ row }: { row: Row }) {
  if (!row.folder?.exists && !(row.name === BENCH && row.loaded)) return <Pill status="bad">folder missing</Pill>
  if (!row.loaded) return <Pill status="warn">downloaded · not loaded</Pill>
  if (!row.loaded.n_questions) return <Pill status="warn">no questions</Pill>
  return <Pill status="ok">loaded · snapshot #{row.loaded.latest_snapshot}</Pill>
}

const LOAD_HINT = 'Read the folder into rag-lab’s internal document store (only new or changed documents are added, as a new snapshot). Pipelines use the store.'

function LoadButton({ loaded, downloaded, working, onLoad }: { loaded: boolean; downloaded: boolean; working: boolean; onLoad: () => void }) {
  return <button className={`sm ${loaded ? '' : 'primary'}`} disabled={!downloaded || working} title={downloaded ? LOAD_HINT : 'Download it first'} onClick={onLoad}>
    {working ? 'Working…' : loaded ? 'Reload' : 'Load'}</button>
}

interface PublicDataset { name: string; description: string; size: string; downloaded: boolean }

// Public datasets, each fetched by a Python extractor script you can read and edit first.
// On a fresh install this is how data gets onto the machine; nothing is in the repository.
function PublicDatasets({ loaded, downloaded, working, busy, onGet, onLoad }: {
  loaded: string[]; downloaded: string[]; working: (name: string) => boolean; busy: string
  onGet: (name: string) => void; onLoad: (name: string) => void
}) {
  const [list, setList] = useState<PublicDataset[]>()
  useEffect(() => { get<PublicDataset[]>('/api/benchmarks').then(setList).catch(() => setList([])) }, [busy])
  if (!list) return null
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>public dataset</th><th className="num">download</th><th>what it is</th><th>on this machine</th><th className="num">1 · download</th><th className="num">2 · load</th></tr></thead>
      <tbody>{list.map(d => {
        const isDown = d.downloaded || downloaded.includes(d.name), isLoaded = loaded.includes(d.name)
        return (
        <tr key={d.name}>
          <td><b>{d.name}</b></td><td className="num small">{d.size}</td><td className="small muted">{d.description}</td>
          <td>{isLoaded ? <Pill status="ok">loaded</Pill> : isDown ? <Pill status="warn">downloaded</Pill> : <span className="small faint">–</span>}</td>
          <td className="num"><button className="sm" title="Download the raw files into data/datasets/ with this dataset’s Python extractor (you can read it first)" onClick={() => onGet(d.name)}>{isDown ? 'Re-download' : 'Get'}</button></td>
          <td className="num"><LoadButton loaded={isLoaded} downloaded={isDown} working={working(d.name)} onLoad={() => onLoad(d.name)} /></td>
        </tr>)})}</tbody>
    </table></div>)
}

function NewDataset({ existing, initial, onClose, onDone }: { existing: string[]; initial?: string; onClose: () => void; onDone: () => void }) {
  const nav = useNavigate()
  const [kind, setKind] = useState<'extract' | 'folder'>('extract')
  const [name, setName] = useState(initial ?? 'my_dataset')
  const [code, setCode] = useState<string>()
  const [error, setError] = useState<string>()
  const valid = NAME_RE.test(name)
  const taken = existing.includes(name)
  useEffect(() => {      // a public dataset's own extractor, or the template for a new one
    const url = initial ? `/api/benchmarks/${initial}/extractor` : null
    ;(url ? get<{ code: string }>(url).then(r => r.code)
      : get<{ contracts: Record<string, { template: string }> }>('/api/scripts/info').then(r => r.contracts.extract.template))
      .then(setCode).catch(e => setError(errText(e)))
  }, [initial])

  const go = async () => {
    setError(undefined)
    try {
      if (kind === 'extract') await send('POST', `/api/datasets/${name}/ingest`, { source: 'extract', params: { code } })
      else await send('POST', `/api/datasets/${name}/folder`)
      onDone(); onClose(); nav(`/datasets/${name}`)
    } catch (e) { setError(errText(e)) }
  }

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-head"><h2>{initial ? `Get ${initial}` : 'New dataset'}</h2><div className="actions"><button className="ghost sm" onClick={onClose}>Close</button></div></div>
      {!initial && <div className="option-cards">
        <div className={`option ${kind === 'extract' ? 'selected' : ''}`} onClick={() => setKind('extract')}>
          <b>Python extractor</b><span>A script downloads or converts data into the dataset’s folder. Load it into the store afterwards. The template fetches a BEIR benchmark.</span></div>
        <div className={`option ${kind === 'folder' ? 'selected' : ''}`} onClick={() => setKind('folder')}>
          <b>Empty folder</b><span>Creates data/datasets/&lt;name&gt;/. Add files with the file explorer or copy them in, then press Load.</span></div>
      </div>}
      {!initial && (
        <div className="form" style={{ marginTop: 14 }}>
          <div className="field"><label>Dataset name (= folder name)</label>
            <input value={name} placeholder="my_docs" onChange={e => setName(e.target.value.trim())} />
            {name && !valid ? <span className="hint bad-text">Letters, digits, _ and - only</span>
              : taken ? <span className="hint">A dataset with this name exists. {kind === 'extract' ? 'The extractor will download into its folder again.' : ''}</span>
              : <span className="hint">data/datasets/{name || '…'}/</span>}</div>
        </div>)}
      {kind === 'extract' && <>
        <p className="small muted">{initial ? <>This script downloads the dataset into <code>data/datasets/{initial}/</code> on this machine. Read or edit it, then run it. </> : null}
          It is saved in the folder as <code>extract.py</code>, so you can re-run it later; <code>download(folder)</code> must leave documents and a <code>questions.jsonl</code> there.
          This only downloads: press <b>Load</b> afterwards to read the files into the internal store.
          From a terminal: <code>uv run python scripts/get_datasets.py {initial ?? '<name>'}</code> (add <code>--load</code> to do both).</p>
        {code !== undefined && <CodeEditor value={code} onChange={setCode} height="300px" kind="extract" title={`Extractor · ${name}`} />}</>}
      {initial === BENCH && <div className="notice" style={{ marginTop: 12 }}>A 1.3 GB download that takes a few minutes. Loading it afterwards takes a couple of minutes and ~5 GB of disk. Files already downloaded are skipped.</div>}
      <div className="row" style={{ marginTop: 12 }}><span className="grow" />
        <button className="primary" disabled={!valid || (kind === 'folder' && taken) || (kind === 'extract' && code === undefined)} onClick={go}>
          {kind === 'extract' ? 'Run extractor (download)' : 'Create folder'}</button></div>
      <ErrorNote error={error} />
    </div>
  )
}

type Tab = 'overview' | 'files' | 'questions' | 'snapshots' | 'extractor'

function DatasetPage({ row, onChange, jobs }: { row: Row; onChange: () => void; jobs: ReturnType<typeof useJobs> }) {
  const nav = useNavigate()
  const { name, loaded, folder } = row
  const [tab, setTab] = useState<Tab>(loaded ? 'overview' : 'files')
  const [sync, setSync] = useState(true)
  const [msg, setMsg] = useState<string>()
  const [error, setError] = useState<string>()
  const running = jobs.some(j => j.kind === 'dataset' && isActive(j) && j.title.endsWith(` ${name}`))
  const act = (p: Promise<unknown>, text: string) => p.then(() => { setError(undefined); setMsg(text); onChange() }).catch(e => setError(errText(e)))
  const load = () => act(send('POST', `/api/datasets/${name}/ingest`, { source: 'folder', sync }), 'Load queued')
  const remove = () => {
    if (window.confirm(`Remove “${name}” from the internal document store? Its folder and downloaded files are kept, so you can load it again.`))
      act(send('DELETE', `/api/datasets/${name}`), 'Removal queued')
  }
  const tabs: [Tab, string][] = [['overview', 'Overview'], ['files', 'Files'], ['questions', 'Questions'], ['snapshots', 'Snapshots'],
    ...(folder?.has_extractor ? [['extractor', 'Extractor'] as [Tab, string]] : [])]

  return (
    <>
      <PageHead title={name} sub={name === BENCH ? 'EnterpriseRAG-Bench release: synthetic company documents with benchmark questions.' : `data/datasets/${name}/`}>
        <button className="ghost" onClick={() => nav('/datasets')}>All datasets</button>
        {loaded && <button className="danger" onClick={remove} title="Remove from the internal store; the folder is kept">Remove from store</button>}
        <button className="primary" title={LOAD_HINT} disabled={running || (!folder?.exists && name !== BENCH)} onClick={load}>{running ? 'Working…' : loaded ? 'Reload into store' : 'Load into store'}</button>
      </PageHead>
      <div className="row" style={{ marginBottom: 10, gap: 14 }}>
        <StateBadge row={row} />
        {loaded && <span className="small muted">{docCount(loaded).toLocaleString()} documents · {loaded.n_questions.toLocaleString()} questions · snapshot #{loaded.latest_snapshot}</span>}
        {folder?.exists && <span className="small muted">{folder.n_files.toLocaleString()} files · {bytes(folder.size)} on disk</span>}
        <span className="grow" />
        {name !== BENCH && <label className="small" title="Documents no longer in the folder are removed from the new snapshot"><input type="checkbox" checked={sync} onChange={e => setSync(e.target.checked)} /> reload removes deleted files</label>}
        {msg && <span className="small muted">{msg}</span>}
      </div>
      <ErrorNote error={error} />
      {jobs.some(j => j.kind === 'dataset' && j.title.endsWith(` ${name}`)) && <div style={{ marginBottom: 12 }}><JobList jobs={jobs.filter(j => j.title.endsWith(` ${name}`))} kinds={['dataset']} limit={2} /></div>}

      <div className="tabs" style={{ marginBottom: 14 }}>
        {tabs.map(([k, label]) => <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>)}
      </div>
      {tab === 'overview' && (loaded ? <Overview name={name} info={loaded} folder={folder} /> : <Empty>Downloaded, not loaded. Check the files, then press <b>Load into store</b> to read them into rag-lab’s internal document store, which pipelines use. The folder needs documents and a <code>questions.jsonl</code> ({QUESTION_FORMAT}).</Empty>)}
      {tab === 'files' && <Explorer name={name} onChange={onChange} />}
      {tab === 'questions' && (loaded ? <Questions name={name} /> : <Empty>Load the dataset into the store to browse its questions.</Empty>)}
      {tab === 'snapshots' && (loaded ? <Snapshots info={loaded} /> : <Empty>No snapshots yet.</Empty>)}
      {tab === 'extractor' && <Extractor name={name} onRun={() => { setMsg('Extractor queued'); onChange() }} />}
    </>
  )
}

function Overview({ name, info, folder }: { name: string; info: DatasetInfo; folder?: FolderSummary }) {
  const [m, setM] = useState<DatasetMetrics>()
  const [error, setError] = useState<string>()
  useEffect(() => { get<DatasetMetrics>(`/api/datasets/${name}/metrics`).then(setM).catch(e => setError(errText(e))) }, [name, info.latest_snapshot])
  if (error) return <ErrorNote error={error} />
  if (!m) return <p className="muted">Computing metrics…</p>
  const q = m.questions
  const pct = (a: number, b: number) => b ? `${Math.round((100 * a) / b)}%` : '–'
  return (
    <>
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="stats">
          <Stat value={m.docs.n.toLocaleString()} label="documents" />
          <Stat value={`~${(m.docs.approx_tokens / 1e6).toFixed(m.docs.approx_tokens > 1e7 ? 0 : 2)}M`} label="tokens (chars ÷ 4)" />
          <Stat value={m.docs.p50_chars.toLocaleString()} label="median doc length (chars)" />
          <Stat value={m.docs.p90_chars.toLocaleString()} label="90th percentile length" />
          <Stat value={q.n.toLocaleString()} label="questions" />
          <Stat value={pct(q.with_gold_docs, q.n)} label="questions with gold documents" />
          <Stat value={pct(q.with_gold_answer, q.n)} label="questions with a gold answer" />
          <Stat value={pct(q.gold_docs_found, q.gold_doc_ids)} label={`gold documents present (${q.gold_docs_found.toLocaleString()} of ${q.gold_doc_ids.toLocaleString()})`} />
        </div>
        {q.gold_doc_ids > q.gold_docs_found && <div className="notice warn" style={{ marginTop: 10 }}>{(q.gold_doc_ids - q.gold_docs_found).toLocaleString()} gold document ids in the questions are not in the documents. No retriever can find those, so they cap recall.</div>}
      </div>
      <div className="grid2">
        <div className="card">
          <div className="section-title" style={{ marginTop: 0 }}>Document length (characters)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={m.length_hist} margin={{ top: 6, right: 6, bottom: 30, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" />
              <XAxis dataKey="bucket" angle={-25} textAnchor="end" interval={0} tick={{ fontSize: 11, fill: 'var(--text-2)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-2)' }} />
              <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)' }} />
              <Bar dataKey="docs" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="small muted" style={{ margin: 0 }}>Average {m.docs.avg_chars.toLocaleString()} · longest {m.docs.max_chars.toLocaleString()}. Compare with your chunk size: documents longer than it are split.</p>
        </div>
        <div className="card">
          <div className="section-title" style={{ marginTop: 0 }}>Source types</div>
          <div className="table-wrap"><table><thead><tr><th>source</th><th className="num">documents</th><th className="num">avg length</th></tr></thead>
            <tbody>{m.by_source.map(s => <tr key={s.source}><td>{s.source}</td><td className="num">{s.docs.toLocaleString()}</td><td className="num">{s.avg_chars.toLocaleString()}</td></tr>)}</tbody></table></div>
          {m.categories.length > 1 && <>
            <div className="section-title">Question categories</div>
            <div className="table-wrap"><table><thead><tr><th>category</th><th className="num">questions</th></tr></thead>
              <tbody>{m.categories.map(c => <tr key={c.category}><td>{c.category}</td><td className="num">{c.questions}</td></tr>)}</tbody></table></div></>}
          {folder && Object.keys(folder.by_ext).length > 0 && <>
            <div className="section-title">Files on disk</div>
            <div className="table-wrap"><table><thead><tr><th>type</th><th className="num">files</th><th className="num">size</th></tr></thead>
              <tbody>{Object.entries(folder.by_ext).map(([k, v]) => <tr key={k}><td><code>{k}</code></td><td className="num">{v.files.toLocaleString()}</td><td className="num">{bytes(v.size)}</td></tr>)}</tbody></table></div></>}
        </div>
      </div>
      {Object.keys(info.subsets).length > 0 && <p className="small muted">Subsets: {Object.entries(info.subsets).map(([k, v]) => `${k} (${v} questions)`).join(', ')}</p>}
    </>
  )
}

interface Entry { name: string; path: string; dir: boolean; size: number; files?: number; modified: string }
type Preview = { path: string; kind: 'text'; size: number; text: string; truncated: boolean } | { path: string; kind: 'zip'; size: number; members: { name: string; size: number }[]; n_members: number } | { path: string; kind: 'binary'; size: number }

function Explorer({ name, onChange }: { name: string; onChange: () => void }) {
  const [path, setPath] = useState('')
  const [tree, setTree] = useState<{ exists: boolean; root?: string; entries: Entry[] }>()
  const [preview, setPreview] = useState<Preview>()
  const [error, setError] = useState<string>()
  const [msg, setMsg] = useState<string>()
  const input = useRef<HTMLInputElement>(null)
  const list = useCallback(() => get<typeof tree>(`/api/datasets/${name}/tree?path=${encodeURIComponent(path)}`).then(setTree).catch(e => setError(errText(e))), [name, path])
  useEffect(() => { list() }, [list])
  const open = (e: Entry) => {
    if (e.dir) { setPath(e.path); setPreview(undefined); return }
    get<Preview>(`/api/datasets/${name}/file?path=${encodeURIComponent(e.path)}`).then(p => setPreview({ ...p, path: e.path })).catch(err => setError(errText(err)))
  }
  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    const form = new FormData()
    for (const f of files) form.append('files', f, (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name)
    setMsg('Uploading…')
    try { const r = await send<{ saved: number }>('POST', `/api/datasets/${name}/upload?path=${encodeURIComponent(path)}`, form); setMsg(`${r.saved} file(s) added. Press Reload into store to load them.`); list(); onChange() }
    catch (e) { setMsg(undefined); setError(errText(e)) }
    if (input.current) input.current.value = ''
  }
  const del = (e: Entry) => {
    if (!window.confirm(`Delete ${e.dir ? 'folder' : 'file'} “${e.path}” from disk? This cannot be undone.`)) return
    send('DELETE', `/api/datasets/${name}/file?path=${encodeURIComponent(e.path)}`).then(() => { if (preview?.path === e.path) setPreview(undefined); list(); onChange() }).catch(err => setError(errText(err)))
  }
  const crumbs = path ? path.split('/') : []

  return (
    <div className="explorer">
      <div className="card explorer-list">
        <div className="row" style={{ marginBottom: 8 }}>
          <span className="small crumbs">
            <a onClick={() => setPath('')}>{name}</a>
            {crumbs.map((c, i) => <span key={i}> / <a onClick={() => setPath(crumbs.slice(0, i + 1).join('/'))}>{c}</a></span>)}
          </span>
          <span className="grow" />
          <input ref={input} type="file" multiple hidden onChange={e => upload(e.target.files)} />
          <button className="sm" onClick={() => input.current?.click()} title=".txt, .md, .jsonl, questions.jsonl, or a .zip (unpacked here)">Upload files</button>
        </div>
        {msg && <p className="small muted" style={{ marginTop: 0 }}>{msg}</p>}
        <ErrorNote error={error} />
        {!tree ? <p className="muted">Loading…</p> : !tree.exists ? <Empty>This folder doesn’t exist yet.</Empty> : tree.entries.length === 0 ? <Empty>Empty folder. Upload files, or copy them into <code>{tree.root}</code>.</Empty> : (
          <table className="files"><tbody>
            {path && <tr className="file-row" onClick={() => setPath(crumbs.slice(0, -1).join('/'))}><td colSpan={4}>↑ ..</td></tr>}
            {tree.entries.map(e => (
              <tr key={e.path} className={`file-row ${preview?.path === e.path ? 'selected' : ''}`} onClick={() => open(e)}>
                <td>{e.dir ? '📁' : '📄'} {e.name}{e.name === 'questions.jsonl' && <span className="pill plain" style={{ marginLeft: 6 }}>questions</span>}{e.name === 'extract.py' && <span className="pill plain" style={{ marginLeft: 6 }}>extractor</span>}</td>
                <td className="num small muted">{e.dir ? `${e.files?.toLocaleString()} files` : ''}</td>
                <td className="num small">{bytes(e.size)}</td>
                <td className="num"><span className="small faint">{e.modified}</span> <button className="sm ghost" title="Delete from disk" onClick={ev => { ev.stopPropagation(); del(e) }}>✕</button></td>
              </tr>))}
          </tbody></table>)}
      </div>
      <div className="card explorer-preview">
        {!preview ? <p className="small muted">Select a file to preview it.</p> : <>
          <div className="small" style={{ marginBottom: 6 }}><b>{preview.path}</b> <span className="muted">· {bytes(preview.size)}</span></div>
          {preview.kind === 'text' && <><pre className="preview-text">{preview.text}</pre>{preview.truncated && <p className="small faint">Showing the first 64 KB.</p>}</>}
          {preview.kind === 'zip' && <><p className="small muted">Zip archive, {preview.n_members.toLocaleString()} files{preview.n_members > preview.members.length ? ` (first ${preview.members.length} shown)` : ''}:</p>
            <pre className="preview-text">{preview.members.map(m => `${m.name}  ${bytes(m.size)}`).join('\n')}</pre></>}
          {preview.kind === 'binary' && <p className="small muted">Binary file, no preview.</p>}
        </>}
      </div>
    </div>
  )
}

function Questions({ name }: { name: string }) {
  const [qs, setQs] = useState<{ question_id: string; category: string; question: string; gold_answer: string | null; n_gold_docs: number }[]>()
  useEffect(() => { get<typeof qs>(`/api/datasets/${name}/questions?limit=200`).then(setQs) }, [name])
  if (!qs) return <p className="muted">Loading…</p>
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>id</th><th>category</th><th>question</th><th className="num">gold docs</th><th>gold answer</th></tr></thead>
      <tbody>{qs.map(q => <tr key={q.question_id}><td className="small mono">{q.question_id}</td><td className="small">{q.category}</td><td>{q.question}</td>
        <td className="num">{q.n_gold_docs}</td><td className="small muted">{q.gold_answer ? q.gold_answer.slice(0, 160) : '–'}</td></tr>)}</tbody>
    </table>{qs.length === 200 && <p className="small faint" style={{ padding: '0 12px' }}>First 200 shown.</p>}</div>)
}

function Snapshots({ info }: { info: DatasetInfo }) {
  return (
    <div className="table-wrap"><table>
      <thead><tr><th className="num">#</th><th>created</th><th className="num">added</th><th className="num">changed</th><th className="num">deleted</th><th className="num">documents</th><th>note</th></tr></thead>
      <tbody>{[...info.snapshots].reverse().map(s => (
        <tr key={s.snapshot}><td className="num">{s.snapshot}</td><td className="small">{s.created_at.slice(0, 16)}</td>
          <td className="num">+{s.added.toLocaleString()}</td><td className="num">~{s.changed.toLocaleString()}</td>
          <td className="num">−{s.deleted.toLocaleString()}</td><td className="num">{s.n_docs.toLocaleString()}</td><td className="small muted">{s.note}</td></tr>))}</tbody>
    </table></div>)
}

function Extractor({ name, onRun }: { name: string; onRun: () => void }) {
  const [code, setCode] = useState<string>()
  const [error, setError] = useState<string>()
  useEffect(() => { get<{ code: string }>(`/api/datasets/${name}/extractor`).then(r => setCode(r.code)) }, [name])
  const run = () => send('POST', `/api/datasets/${name}/ingest`, { source: 'extract', params: { code } }).then(() => { setError(undefined); onRun() }).catch(e => setError(errText(e)))
  if (code === undefined) return <p className="muted">Loading…</p>
  return (
    <div className="card">
      <p className="small muted" style={{ marginTop: 0 }}>This folder’s <code>extract.py</code>. Running it downloads into the folder again (files the script no longer writes stay on disk). It doesn’t touch the store: press <b>Reload into store</b> afterwards.</p>
      <CodeEditor value={code} onChange={setCode} height="360px" kind="extract" title={`Extractor · ${name}`} versionKey={`datasets/${name}`} />
      <div className="row" style={{ marginTop: 10 }}><span className="grow" /><button className="primary" onClick={run}>Save and run (download)</button></div>
      <ErrorNote error={error} />
    </div>)
}
