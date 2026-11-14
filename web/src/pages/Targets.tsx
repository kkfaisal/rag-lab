import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { errText, get, send, type Catalog, type Target } from '../api'
import CodeEditor from '../components/CodeEditor'
import { isActive, useJobs } from '../components/Jobs'
import ParamForm from '../components/ParamForm'
import { Empty, ErrorNote, PageHead, Pill, Stat } from '../components/ui'

const TYPE_LABEL: Record<string, string> = {
  local: 'Local store', pgvector: 'PostgreSQL + pgvector', sql: 'SQL database', qdrant: 'Qdrant', opensearch: 'OpenSearch',
  elasticsearch: 'Elasticsearch', python: 'Python script',
}
const TYPE_HINT: Record<string, string> = {
  local: 'Vectors + BM25 in a file on disk. No setup.', qdrant: 'Vector database: embedded on disk, or a server URL.',
  pgvector: 'Vectors (HNSW) and full-text search in Postgres.',
  sql: 'Any SQLAlchemy database (Postgres, MySQL, SQL Server, Oracle, SQLite…): the Python counterpart of JDBC.',
  opensearch: 'BM25 and k-NN on an OpenSearch cluster.', elasticsearch: 'BM25 and k-NN on Elasticsearch 9.',
  python: 'Your own store: write add / delete / search in Python.',
}
const BLANK = { name: '', description: '', type: 'local', params: {}, version: 0, n_docs: 0, n_chunks: 0, dataset: null, subset: null, snapshot: null, embed: null, history: [] } as Target

export default function Targets() {
  const { id } = useParams()
  const nav = useNavigate()
  const [catalog, setCatalog] = useState<Catalog>()
  const [targets, setTargets] = useState<Target[] | null>(null)
  const [edit, setEdit] = useState<Target>()
  const [error, setError] = useState<string>()
  const [conn, setConn] = useState<null | 'testing' | { ok: boolean; message: string }>(null)
  const [services, setServices] = useState<{ type: string; label: string; url: string; reachable: boolean }[]>()
  const jobs = useJobs()
  const busy = jobs.filter(isActive).length
  const reload = useCallback(() => get<Target[]>('/api/targets').then(setTargets).catch(e => setError(errText(e))), [])
  useEffect(() => { get<Catalog>('/api/catalog').then(setCatalog) }, [])
  useEffect(() => { get<typeof services>('/api/services').then(setServices).catch(() => {}) }, [])
  useEffect(() => { setConn(null) }, [edit?.id, edit?.type])
  useEffect(() => { reload() }, [reload, busy])
  useEffect(() => {
    if (id === 'new') setEdit(e => e && !e.id ? e : { ...BLANK })
    else if (id && targets) setEdit(targets.find(t => t.id === id))
    else setEdit(undefined)
  }, [id, targets])

  const info = catalog?.target_types.find(t => t.impl === edit?.type)
  const save = async () => {
    if (!edit) return
    try {
      const t = await send<Target>('PUT', '/api/targets', { id: edit.id, name: edit.name, description: edit.description, type: edit.type, params: edit.params })
      setError(undefined); await reload(); nav(`/targets/${t.id}`)
    } catch (e) { setError(errText(e)) }
  }
  const remove = async (purge: boolean) => {
    if (!edit?.id || !window.confirm(purge ? `Delete “${edit.name}” and everything stored in it? This cannot be undone.` : `Remove the definition of “${edit.name}”? Stored data is left in place.`)) return
    try { await send('DELETE', `/api/targets/${edit.id}?purge=${purge}`); await reload(); nav('/targets') } catch (e) { setError(errText(e)) }
  }

  return (
    <>
      <PageHead title="Targets" sub="A target is where an ingestion pipeline writes chunks and where a retrieval pipeline searches. Use a built-in store or write your own in Python.">
        <button className="primary" onClick={() => nav('/targets/new')}>New target</button>
      </PageHead>
      <ErrorNote error={error} />

      {services && !edit && (
        <div className="services" title="Database services docker-compose can start (docker compose --profile search up -d)">
          <span className="small muted" style={{ alignSelf: 'center' }}>Database services</span>
          {services.map(s => (
            <div key={s.type} className="service" title={s.url}>
              <span className={`dot ${s.reachable ? 'on' : ''}`} />
              <span className="small"><b>{s.label}</b> <span className="faint">{s.reachable ? 'running' : 'not running'}</span></span>
              {s.reachable && <button className="sm ghost" onClick={() => { setEdit({ ...BLANK, type: s.type, name: s.label, params: s.type === 'qdrant' ? { url: s.url } : {} }); nav('/targets/new') }}>Add target</button>}
            </div>))}
        </div>)}

      {edit && catalog && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-head"><h2>{edit.id ? edit.name : 'New target'}</h2>
            {edit.id && (edit.version ? <Pill status="ok">version {edit.version}</Pill> : <Pill status="warn">empty</Pill>)}
            <div className="actions">
              {edit.id && <><button className="sm danger" onClick={() => remove(true)}>Delete with data</button><button className="sm" onClick={() => remove(false)}>Remove definition</button></>}
              <button className="ghost sm" onClick={() => nav('/targets')}>Close</button></div></div>

          {!edit.id && <div className="option-cards" style={{ marginBottom: 12 }}>
            {catalog.target_types.map(t => (
              <div key={t.impl} className={`option ${edit.type === t.impl ? 'selected' : ''}`} onClick={() => setEdit({ ...edit, type: t.impl, params: {} })}>
                <b>{TYPE_LABEL[t.impl] ?? t.impl}</b><span>{TYPE_HINT[t.impl]}</span></div>))}
          </div>}
          <div className="form">
            <div className="field"><label>Name</label><input value={edit.name} placeholder="e.g. Local store" onChange={e => setEdit({ ...edit, name: e.target.value })} /></div>
            <div className="field"><label>Description</label><input value={edit.description} onChange={e => setEdit({ ...edit, description: e.target.value })} /></div>
          </div>
          <div className="section-title">{TYPE_LABEL[edit.type] ?? edit.type} settings</div>
          {info && <ParamForm key={(edit.id ?? 'new') + edit.type} info={info} params={edit.params} onChange={p => { setEdit({ ...edit, params: p }); setConn(null) }} scriptKind="target" versionKey={edit.id ? `targets/${edit.id}` : undefined} />}
          {['opensearch', 'elasticsearch', 'qdrant'].includes(edit.type) && <div className="notice">Credentials are read from environment variables you name here (set them in <code>.env</code>), never stored in the target.</div>}
          {['pgvector', 'sql'].includes(edit.type) && <div className="notice">Write secrets in the URL as <code>{'${NAME}'}</code>, e.g. <code>{'postgresql://raglab:${POSTGRES_PASSWORD}@localhost:5433/raglab'}</code>. They are read from the environment (<code>.env</code>) when connecting, never stored in the target.</div>}
          <div className="row conn-test">
            <button className="sm" disabled={conn === 'testing'} onClick={() => { setConn('testing'); send<{ ok: boolean; message: string }>('POST', '/api/targets/test', { type: edit.type, params: edit.params }).then(setConn).catch(e => setConn({ ok: false, message: errText(e) })) }}>
              {conn === 'testing' ? 'Testing…' : 'Test connection'}</button>
            {conn && conn !== 'testing' && <span className={`small ${conn.ok ? 'ok-text' : 'bad-text'}`}>{conn.ok ? '✓ ' : '✕ '}{conn.message}</span>}
          </div>
          {info?.source && edit.type !== 'python' && <details style={{ marginTop: 10 }}><summary className="small">Python source of this target type · {info.source_file}</summary>
            <CodeEditor value={info.source} readOnly height="260px" /></details>}
          <div className="row" style={{ marginTop: 12 }}><span className="grow" />
            <button className="primary" disabled={!edit.name.trim()} onClick={save}>{edit.id ? 'Save changes' : 'Create target'}</button></div>

          {edit.id && <>
            <div className="section-title">Contents</div>
            <div className="stats">
              <Stat value={edit.n_chunks.toLocaleString()} label="chunks" /><Stat value={edit.n_docs.toLocaleString()} label="documents" />
              <Stat value={edit.dataset ?? '–'} label={edit.subset && edit.subset !== 'all' ? `dataset · ${edit.subset} subset` : 'dataset'} />
              <Stat value={edit.embed ? String(edit.embed.model ?? 'script') : 'none'} label="embedder" />
            </div>
            <div className="section-title">Write history</div>
            {edit.history.length === 0 ? <p className="muted small">Nothing written yet. Add this target to an <Link to="/ingestion">ingestion pipeline</Link> and run it.</p> : (
              <div className="table-wrap"><table>
                <thead><tr><th className="num">version</th><th>when</th><th>mode</th><th>pipeline</th><th className="num">docs written</th><th className="num">skipped (unchanged)</th><th className="num">chunks written</th></tr></thead>
                <tbody>{edit.history.map(h => (
                  <tr key={h.version}><td className="num">{h.version}</td><td className="small">{h.at.replace('T', ' ').slice(0, 16)}</td>
                    <td><span className="pill plain">{h.mode}</span></td><td>{h.pipeline ?? '–'}</td>
                    <td className="num">{h.docs_written.toLocaleString()}</td><td className="num">{h.docs_skipped.toLocaleString()}</td><td className="num">{h.chunks_written.toLocaleString()}</td></tr>))}</tbody>
              </table></div>)}
          </>}
        </div>)}

      {targets === null ? <p className="muted">Loading…</p> : targets.length === 0 && !edit ? <Empty>No targets yet.</Empty> : (
        <div className="table-wrap"><table>
          <thead><tr><th>target</th><th>type</th><th>state</th><th className="num">chunks</th><th className="num">documents</th><th>dataset</th><th>embedder</th><th>last write</th></tr></thead>
          <tbody>{targets.map(t => (
            <tr key={t.id}>
              <td><Link to={`/targets/${t.id}`}><b>{t.name}</b></Link><div className="small muted">{t.description}</div></td>
              <td><span className="pill plain">{TYPE_LABEL[t.type] ?? t.type}</span></td>
              <td>{t.version ? <Pill status="ok">v{t.version}</Pill> : <Pill status="warn">empty</Pill>}</td>
              <td className="num">{t.n_chunks.toLocaleString()}</td><td className="num">{t.n_docs.toLocaleString()}</td>
              <td>{t.dataset ?? '–'}{t.subset && t.subset !== 'all' ? <span className="muted"> · {t.subset}</span> : null}</td>
              <td className="small">{t.embed ? String(t.embed.model ?? 'script') : '–'}</td>
              <td className="small muted">{t.history[0] ? `${t.history[0].at.replace('T', ' ').slice(0, 16)} · ${t.history[0].mode}` : '–'}</td>
            </tr>))}</tbody>
        </table></div>)}
    </>
  )
}
