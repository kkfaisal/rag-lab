// Plain-language explanations of every metric and concept the UI shows.
// Used by <Term> (hover) and the Glossary panel (the "?" in the sidebar).

export interface Entry { title: string; short: string; long?: string; group: 'Metrics' | 'Answer judge' | 'Comparing runs' | 'Data' | 'Pipelines' }

export const GLOSSARY: Record<string, Entry> = {
  // ---- retrieval metrics (free, no LLM)
  'recall@k': { group: 'Metrics', title: 'recall@k (recall@1, @5, @10, @20)',
    short: 'Share of a question’s gold documents found among the top k retrieved documents.',
    long: 'Chunks are folded into their documents first (a document ranks where its best chunk ranks). 100 = every gold document is in the top k. Averaged over questions that have gold documents. Higher k is more forgiving: recall@20 ≥ recall@10 ≥ recall@1.' },
  'ndcg@10': { group: 'Metrics', title: 'nDCG@10',
    short: 'Like recall@10, but a gold document ranked higher counts more.',
    long: 'Normalised Discounted Cumulative Gain: each gold document in the top 10 earns 1 / log2(rank + 1), divided by the best possible score. 100 = all gold documents at the very top. This is the metric BEIR papers and leaderboards report.' },
  'mrr@10': { group: 'Metrics', title: 'MRR@10',
    short: 'How soon the first gold document appears: 100 at rank 1, 50 at rank 2, 33 at rank 3, … 0 if not in the top 10.',
    long: 'Mean Reciprocal Rank. Good for questions with one right document, where what matters is whether it comes first.' },
  context_recall: { group: 'Metrics', title: 'context_recall',
    short: 'Share of gold documents that made it into the prompt sent to Claude.',
    long: 'For single-pass generation: the top context_chunks chunks after reranking. For the agent: every document it saw across all its searches. If a gold document never reaches the prompt, the answer can’t use it.' },
  answer_recall: { group: 'Metrics', title: 'answer_recall',
    short: 'Share of gold documents the answer cites as its sources.',
    long: 'Without a generate node, the answer’s documents are simply the top retrieved documents, so this mirrors retrieval.' },
  extra_docs: { group: 'Metrics', title: 'extra_docs',
    short: 'Number of documents the answer cites that are not gold documents. Lower is better.' },
  abstained: { group: 'Metrics', title: 'abstained',
    short: 'Share of questions where the model said the answer is not in the documents.',
    long: 'Some abstaining is right (EnterpriseRAG-Bench has questions whose answer is not in the documents). Abstaining on answerable questions usually means retrieval missed the gold documents.' },
  n: { group: 'Metrics', title: 'n', short: 'Number of questions behind the numbers in this row.' },
  // ---- judge
  correctness: { group: 'Answer judge', title: 'correctness',
    short: 'Share of answers the judge rates correct: they state the gold answer’s key facts with nothing contradicting or invented.',
    long: 'Each answer is right (100) or wrong (0); the number shown is the average. Needs a generate node, gold answers and an API key. EnterpriseRAG-Bench uses the benchmark’s own judge; other datasets use rag-lab’s Claude judge.' },
  completeness: { group: 'Answer judge', title: 'completeness',
    short: 'Share of the gold answer’s facts that the answer states (0–100 per question).' },
  judge_doc_recall: { group: 'Answer judge', title: 'judge_doc_recall',
    short: 'EnterpriseRAG-Bench judge: share of gold documents among the answer’s cited documents.' },
  invalid_extra_docs: { group: 'Answer judge', title: 'invalid_extra_docs',
    short: 'EnterpriseRAG-Bench judge: cited documents that are wrong for the question. Lower is better.' },
  judge: { group: 'Answer judge', title: 'Judge',
    short: 'An LLM that grades each generated answer against the gold answer.',
    long: '“auto” turns it on when it can run (generate node, gold answers, API key) and says why when it can’t.' },
  'gold documents': { group: 'Data', title: 'Gold documents',
    short: 'The documents that contain the answer to a question, listed with the question in questions.jsonl.' },
  'gold answer': { group: 'Data', title: 'Gold answer', short: 'The reference answer a judge compares generated answers with. BEIR datasets have none.' },
  // ---- comparing runs
  'p-value': { group: 'Comparing runs', title: 'p-value',
    short: 'If A and B were equally good, how often a gap this large would appear by chance. Below 0.05 = the difference is real.',
    long: 'From a paired permutation test: on each question, A’s and B’s scores are randomly swapped many times, and we count how often the shuffled gap is as large as the real one.' },
  'confidence interval': { group: 'Comparing runs', title: '95% confidence interval',
    short: 'The range the true difference between B and A probably lies in, from resampling the questions.',
    long: 'With few questions it comes out too narrow, so the verdict uses the p-value.' },
  'questions won': { group: 'Comparing runs', title: 'B better / A better / tied',
    short: 'How many questions each run scored higher on. A big gap from only a few questions is not reliable.' },
  // ---- data
  dataset: { group: 'Data', title: 'Dataset', short: 'A folder of documents plus a questions.jsonl, downloaded then loaded into the internal store.' },
  snapshot: { group: 'Data', title: 'Snapshot',
    short: 'A version of a dataset in the internal store. Each load records only what changed; runs say which snapshot they used.' },
  subset: { group: 'Data', title: 'Subset',
    short: '“all”, or a named part of a dataset. EnterpriseRAG-Bench’s “dev”: 100 questions, their gold documents and 20,000 others, for quick runs.' },
  category: { group: 'Data', title: 'Category', short: 'A question type (EnterpriseRAG-Bench has 10). Metrics are also shown per category.' },
  // ---- pipelines
  chunk: { group: 'Pipelines', title: 'Chunk', short: 'A piece of a document. Search works on chunks; scores are folded back to documents.' },
  embedding: { group: 'Pipelines', title: 'Embedding / vector search',
    short: 'Text turned into a vector so that similar meanings are close; vector search finds chunks near the question’s vector.' },
  bm25: { group: 'Pipelines', title: 'BM25', short: 'Keyword search: ranks chunks by the question’s words, weighting rare words more.' },
  hybrid: { group: 'Pipelines', title: 'Hybrid (RRF)',
    short: 'BM25 and vector search together, merged with Reciprocal Rank Fusion: each result scores 1 / (60 + its rank) in each list.' },
  rerank: { group: 'Pipelines', title: 'Rerank', short: 'A cross-encoder re-reads the question with each candidate chunk and re-orders them; slower but more accurate.' },
  generate: { group: 'Pipelines', title: 'Generate',
    short: 'Claude answers from the retrieved chunks. “llm” = one pass; “agent” = Claude may search again with its own queries first.' },
  target: { group: 'Pipelines', title: 'Target', short: 'A store chunks are written to and searched in (local, pgvector, Qdrant, OpenSearch, …).' },
  'target version': { group: 'Pipelines', title: 'Target version', short: 'Goes up with every write to a target; a benchmark records which version it searched.' },
  cached: { group: 'Pipelines', title: 'Cached step', short: 'Reused from an earlier run with exactly the same settings, code and input, so it took no time.' },
  'append / overwrite': { group: 'Pipelines', title: 'Append / overwrite',
    short: 'Append writes only new or changed documents to the target; overwrite clears it and writes everything.' },
  run: { group: 'Pipelines', title: 'Run identifier',
    short: 'pipeline_target_yyyy_mm_dd_weekday_hh_mm, e.g. hybrid-rerank-agent_local-store_2026_11_14_sat_18_47.' },
}

export function lookup(key: string): Entry | undefined {
  return GLOSSARY[key] ?? GLOSSARY[key.replace(/^recall@\d+$/, 'recall@k')] ?? GLOSSARY[key.toLowerCase()]
}
