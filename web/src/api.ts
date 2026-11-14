// Typed helpers for the read-only rag-lab API.

export type Metrics = Record<string, number | null>

export interface RunRow {
  run_id: string
  name: string
  description: string
  created_at: string
  status: string
  kind: string
  meta: Record<string, string>
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
  key: string | null
  stats?: Record<string, unknown>
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
  generation: { context_doc_ids: string[]; cited_doc_ids: string[]; abstained: boolean; stop_reason: string | null; model: string | null; raw: string | null; searches?: { query: string; hits: number; new_docs: number }[] } | null
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
  folder?: FolderSummary
}

// A folder under data/datasets/: what is on disk, loaded or not.
export interface FolderSummary {
  name?: string; exists: boolean; n_files: number; size: number; has_questions: boolean; has_extractor: boolean
  by_ext: Record<string, { files: number; size: number }>
}

export interface DatasetMetrics {
  snapshot: number
  docs: { n: number; chars: number; avg_chars: number; p50_chars: number; p90_chars: number; max_chars: number; approx_tokens: number }
  length_hist: { bucket: string; docs: number }[]
  by_source: { source: string; docs: number; avg_chars: number }[]
  questions: { n: number; with_gold_docs: number; with_gold_answer: number; avg_gold_docs: number; avg_question_chars: number; gold_doc_ids: number; gold_docs_found: number }
  categories: { category: string; questions: number }[]
}

export const bytes = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : n < 1024 ** 3 ? `${(n / 1024 ** 2).toFixed(1)} MB` : `${(n / 1024 ** 3).toFixed(2)} GB`

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
export type GraphKind = 'ingestion' | 'retrieval'

export interface SchemaProp { type?: string; title?: string; default?: unknown; anyOf?: { type?: string }[]; items?: { type?: string }; description?: string }
export interface ImplInfo { impl: string; doc: string; source?: string; fn_source?: string; source_file?: string; scriptable?: boolean; schema: { properties?: Record<string, SchemaProp> }; defaults: Record<string, unknown> }
export interface Catalog {
  stages: Record<string, ImplInfo[]>; target_types: ImplInfo[]; kinds: Record<GraphKind, string[]>
  sources: Record<string, string>; templates: Record<string, string>; has_api_key: boolean; bench_ready: boolean
}

export interface TargetWrite { version: number; at: string; mode: string; run_id: string; pipeline: string | null; docs_written: number; docs_skipped: number; chunks_written: number }
export interface Target {
  id?: string; name: string; description: string; type: string; params: Record<string, unknown>
  version: number; n_docs: number; n_chunks: number; dataset: string | null; subset: string | null; snapshot: number | null
  embed: Record<string, unknown> | null; history: TargetWrite[]
}

export interface Job {
  id: string; kind: 'dataset' | 'ingest' | 'benchmark'; title: string; status: 'queued' | 'running' | 'ok' | 'failed' | 'cancelled'
  created_at: string; started_at?: string; finished_at?: string; run_id?: string; error?: string; log?: string
  spec: { meta?: Record<string, string>; dataset?: string }
  run?: { current_stage: string | null; stages: (StageLog & { stats?: Record<string, unknown> })[] }
}

export interface MatrixCell { run_id: string; target_id: string; target: string; pipeline_id: string; pipeline: string; dataset: string; target_version: number; n_questions: number; judged: unknown; overall: Metrics; cost_usd: number; created_at: string }

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

export const errText = (e: unknown) => String(e instanceof Error ? e.message : e)
