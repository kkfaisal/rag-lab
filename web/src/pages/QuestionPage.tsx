import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fmt, get, type Chunk, type QuestionDetail } from '../api'

function FullDoc({ docId, d }: { docId: string; d: QuestionDetail }) {
  const [text, setText] = useState<string>()
  return (
    <details onToggle={e => { if ((e.target as HTMLDetailsElement).open && !text) get<{ text: string }>(`/api/doc?dataset=${encodeURIComponent(d.dataset)}&snapshot=${d.snapshot}&doc_id=${encodeURIComponent(docId)}`).then(d => setText(d.text)) }}>
      <summary className="small">full document</summary>
      <pre>{text ?? 'Loading…'}</pre>
    </details>
  )
}

export default function QuestionPage() {
  const { runId, qid } = useParams()
  const [d, setD] = useState<QuestionDetail>()
  const [error, setError] = useState<string>()
  useEffect(() => { get<QuestionDetail>(`/api/runs/${runId}/questions/${qid}`).then(setD).catch(e => setError(String(e))) }, [runId, qid])
  if (error) return <p>{error}</p>
  if (!d) return <p className="muted">Loading…</p>

  const gold = new Set(d.question.expected_doc_ids)
  const cited = new Set(d.answer?.document_ids ?? [])
  const context = new Set(d.generation?.context_doc_ids ?? [])
  const final: Chunk[] = d.reranked ?? d.retrieval
  const retrievedRank = new Map<string, number>()
  d.retrieval.forEach((c, i) => { if (!retrievedRank.has(c.doc_id)) retrievedRank.set(c.doc_id, i + 1) })
  const title = (id: string) => d.docs[id]?.title ?? id
  const m = d.metrics ?? {}

  return (
    <>
      <p className="small"><Link to={`/runs/${runId}`}>← {runId}</Link></p>
      <h1>{d.question.question_id} <span className="badge">{d.question.category}</span></h1>
      <p className="small muted">{d.dataset} · snapshot {d.snapshot}</p>
      <p style={{ fontSize: 16 }}>{d.question.question}</p>
      <div className="row small muted">
        {['recall@1', 'recall@5', 'recall@10', 'recall@20', 'context_recall', 'answer_recall', 'extra_docs', 'correctness', 'completeness', 'invalid_extra_docs']
          .filter(k => k in m).map(k => <span key={k} className="chip">{k} <b>{fmt(m[k])}</b></span>)}
      </div>

      <div className="grid2" style={{ marginTop: 16 }}>
        <div>
          <h2>Generated answer {d.generation?.abstained && <span className="badge extra">abstained</span>}
            {d.generation?.stop_reason === 'refusal' && <span className="badge miss">refusal</span>}</h2>
          <div className="panel answer">{d.answer?.answer ?? <span className="muted">(no answer — retrieval-only run)</span>}</div>
          {typeof m.judge_reasoning === 'string' && <>
            <h2>Judge: {m.correctness === 100 ? <span className="badge hit">correct</span> : <span className="badge miss">incorrect</span>}</h2>
            <div className="panel small answer">{m.judge_reasoning}</div></>}
          <h2>Gold answer</h2>
          <div className="panel answer">{d.question.gold_answer}</div>
          <h2>Facts the judge checks</h2>
          <ul className="small">{d.question.answer_facts.map((f, i) => <li key={i}>{f}</li>)}</ul>
        </div>

        <div>
          <h2>Gold documents ({gold.size})</h2>
          {gold.size === 0 && <p className="muted small">None — this category has no ground-truth docs (high level / info not found).</p>}
          {[...gold].map(id => {
            const rank = retrievedRank.get(id)
            return (
              <div key={id} className={`doc ${cited.has(id) ? 'hit' : 'miss'}`}>
                <div><b>{title(id)}</b> <span className="muted small">{d.docs[id]?.source}</span></div>
                <div className="small">
                  {rank ? <span className="badge">retrieved #{rank}</span> : <span className="badge miss">not retrieved</span>}{' '}
                  {context.has(id) ? <span className="badge">in prompt</span> : <span className="badge miss">not in prompt</span>}{' '}
                  {cited.has(id) ? <span className="badge hit">cited</span> : <span className="badge miss">not cited</span>}
                  <span className="mono muted"> {id}</span>
                </div>
                <FullDoc docId={id} d={d} />
              </div>)
          })}

          <h2>{d.reranked ? 'Reranked' : 'Retrieved'} chunks ({final.length})</h2>
          <p className="small muted">Green = chunk from a gold doc (hit); amber = not gold (extra). ★ = cited in the answer.</p>
          {final.map((c, i) => (
            <div key={c.chunk_id} className={`doc ${gold.has(c.doc_id) ? 'hit' : 'extra'}`}>
              <div className="small"><b>#{i + 1}</b> {cited.has(c.doc_id) ? '★ ' : ''}<b>{title(c.doc_id)}</b>{' '}
                <span className="muted">{d.docs[c.doc_id]?.source} · score {fmt(c.score, 3)}
                  {c.retrieval_score !== undefined ? ` (retrieval ${fmt(c.retrieval_score, 3)})` : ''}</span></div>
              {d.chunk_texts[c.chunk_id] && <details><summary className="small">chunk text <span className="mono">{c.chunk_id}</span></summary>
                <pre>{d.chunk_texts[c.chunk_id]}</pre></details>}
            </div>))}

          {[...cited].filter(id => !gold.has(id)).length > 0 && <>
            <h2>Cited but not gold</h2>
            {[...cited].filter(id => !gold.has(id)).map(id => (
              <div key={id} className="doc extra"><b>{title(id)}</b> <span className="mono muted small">{id}</span><FullDoc docId={id} d={d} /></div>))}
          </>}
        </div>
      </div>
    </>
  )
}
