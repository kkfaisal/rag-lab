// Typed helpers for the read-only rag-lab API.

export type Metrics = Record<string, number | null>

export interface RunRow {
  run_id: string
  name: string
  description: string
  created_at: string
  status: string
  dataset: string
  snapshot: number
  subset: string
  n_questions: number
  limit: number | null
  git_sha: string | null
  git_dirty: boolean
  duration_s: number
  cost_usd: number
  judged: boolean
  error: string | null
  params: Record<string, unknown>
  overall: Metrics
}

export interface StageLog {
  stage: string
  impl: string
  key: string
  cached: boolean
  duration_s: number
  original_duration_s: number | null
}

export interface RunDetail {
  run: { run_id: string; name: string; created_at: string; status: string; git: { sha: string | null; dirty: boolean; branch: string | null }; limit: number | null; n_questions: number; bench_commit: string | null; duration_s: number; error?: string }
  config: Record<string, Record<string, unknown>> & { name: string; description: string }
  stages: StageLog[]
  usage: { generate?: { input_tokens: number; output_tokens: number; cache_read_input_tokens: number; cache_creation_input_tokens: number; requests: number; cost_usd: number; by_model: Record<string, Metrics> }; generate_cached?: boolean; cost_usd_incurred?: number; judge?: string }
  code_hashes: Record<string, { impl: string; hash: string }>
  metrics: { overall?: Metrics; by_category?: Record<string, Metrics>; judged?: boolean }
  git_diff: string | null
  error: string | null
}

export interface QuestionRow extends Record<string, unknown> {
  question_id: string
  category: string
  question: string
  answer: string | null
}

export interface Chunk { chunk_id: string; doc_id: string; score: number; retrieval_score?: number }

export interface QuestionDetail {
  dataset: string
  snapshot: number
  question: { question_id: string; category: string; question: string; gold_answer: string; expected_doc_ids: string[]; answer_facts: string[]; source_types: string[] }
  retrieval: Chunk[]
  reranked: Chunk[] | null
  generation: { context_doc_ids: string[]; cited_doc_ids: string[]; abstained: boolean; stop_reason: string | null; model: string | null; raw: string | null } | null
  answer: { answer: string | null; document_ids: string[] } | null
  metrics: Record<string, unknown> | null
  docs: Record<string, { doc_id: string; source: string; title: string; path: string }>
  chunk_texts: Record<string, string>
}

export interface DatasetInfo {
  name: string
  adapter: string
  code: string | null
  created_at: string
  latest_snapshot: number
  n_questions: number
  by_source: Record<string, number>
  subsets: Record<string, number>
  snapshots: { snapshot: number; created_at: string; added: number; changed: number; deleted: number; n_docs: number; note: string }[]
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const fmt = (v: unknown, digits = 1): string =>
  v === null || v === undefined ? '–' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(digits)) : String(v)

// 0..100 -> red..green background, for heatmaps and metric cells.
export const heat = (v: number | null | undefined): string =>
  v === null || v === undefined ? 'transparent' : `hsl(${Math.round((v / 100) * 120)} 65% 50% / 0.35)`

// Metrics expressed as percentages (others, like extra_docs, are counts).
export const isPct = (m: string) => !['extra_docs', 'invalid_extra_docs', 'n'].includes(m)

// ---------------------------------------------------------------- playground types

export interface GraphNode { id: string; stage: string; impl: string; params: Record<string, unknown>; label?: string; position: { x: number; y: number } }
export interface Graph { id?: string; name: string; description: string; nodes: GraphNode[]; edges: { source: string; target: string }[]; updated_at?: string }
export type GraphKind = 'pipeline' | 'retriever'

export interface SchemaProp { type?: string; title?: string; default?: unknown; anyOf?: { type?: string }[]; items?: { type?: string }; description?: string }
export interface ImplInfo { impl: string; doc: string; schema: { properties?: Record<string, SchemaProp> }; defaults: Record<string, unknown> }
export interface Catalog { stages: Record<string, ImplInfo[]>; kinds: Record<GraphKind, string[]>; templates: { extract: string }; has_api_key: boolean; bench_ready: boolean }

export interface IndexStatus {
  pipeline_id: string; pipeline_name: string; node_id: string; label: string
  state: 'built' | 'stale' | 'not built' | 'failed' | 'invalid'
  error: string | null; run_id: string | null; dataset?: string; subset?: string; snapshot?: number; built_at?: string
  stages: (StageLog & { stats?: Record<string, unknown> })[]
}

export interface Job {
  id: string; kind: 'build' | 'experiment' | 'ingest'; title: string; status: 'queued' | 'running' | 'ok' | 'failed' | 'cancelled'
  created_at: string; started_at?: string; finished_at?: string; run_id?: string; error?: string; log?: string
  spec: { meta?: Record<string, string>; dataset?: string }
  run?: { current_stage: string | null; stages: StageLog[] }
}

export interface MatrixCell { run_id: string; index: string; index_key: string; retriever_id: string; retriever: string; dataset: string; snapshot: number; n_questions: number; judged: unknown; overall: Metrics; cost_usd: number }

export async function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, { method, headers: body instanceof FormData ? undefined : { 'content-type': 'application/json' },
    body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body) })
  if (!r.ok) {
    const text = await r.text()
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json()
}
