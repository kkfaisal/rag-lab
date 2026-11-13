# rag-lab

An open-source, end-to-end **playground for RAG pipelines**. Bring data, build ingestion and
retrieval pipelines on a node canvas, run them from the browser, and benchmark every combination
against a question set.

## The model

| Thing | What it is |
|---|---|
| **Dataset** | Source documents **plus benchmark questions** (both required). Comes from a built-in data source or a Python script. Versioned: every ingest that changes something is a snapshot. |
| **Target** | A named, persistent store that chunks are written to and searched in. Built-in: local (vectors + BM25), Qdrant, OpenSearch. Or a Python script for your own store. |
| **Ingestion pipeline** | `source → chunk → embed → target`. Running it writes to the target, **appending** or **overwriting** as the target node says. Can branch to fill several targets. |
| **Retrieval pipeline** | `target → retrieve → rerank → generate → benchmark`. Running it answers the dataset's questions from the target and scores the result. |

Every node is either a built-in with a settings form or a **Python script you edit in the browser**.
Built-in nodes show their Python too, with an *Edit as Python script* button that swaps the node for
an equivalent editable script.

Every run is recorded with its full resolved config (scripts included), code version, dataset
snapshot, target version, per-stage timings, token usage and cost.

> **Local tool.** Scripts run as ordinary Python on your machine with your privileges.
> `rag-lab serve` binds to 127.0.0.1; don't expose it to a network you don't trust.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, git.

```bash
git clone <this repo> rag-lab && cd rag-lab
uv sync                                   # Python deps (Python 3.12 is pinned)
cp .env.example .env                      # optional: put your ANTHROPIC_API_KEY in .env
cd web && npm install && npm run build && cd ..
uv run rag-lab serve                      # http://127.0.0.1:8000
```

Then, in the browser:

1. **Datasets → New dataset.** Pick a source (file upload, local folder, EnterpriseRAG-Bench, or a
   Python script), supply the questions, ingest.
2. **Targets.** A local store is created for you; add Qdrant, OpenSearch or a script target if you like.
3. **Ingestion.** Open the example pipeline, check the source and target nodes, press **Run ingestion**.
4. **Retrieval.** Open an example pipeline, try a query, press **Run benchmark**.
5. **Results.** Target × retrieval pipeline matrix; drill into a run, a two-run comparison
   (questions fixed/broken), or a single question (gold vs retrieved documents and the answer).

**Without an API key** everything retrieval-related works and is free (recall@k, context recall,
answer-document recall). **With `ANTHROPIC_API_KEY`**, generate nodes answer with Claude
(`claude-sonnet-5-5` by default) and answers are judged.

For EnterpriseRAG-Bench's own judge, also run `uv run rag-lab bench setup` once. Saved pipelines can
be run without a browser: `uv run rag-lab ingest <pipeline-id>` and
`uv run rag-lab benchmark <pipeline-id> [--target <id>] [--limit N]`.

For UI development, run `uv run rag-lab serve` and `cd web && npm run dev` (Vite proxies `/api`).

## Datasets

Built-in sources:

- **File upload**: `.txt`, `.md`, `.jsonl` files or a `.zip` of them, plus `questions.jsonl`.
- **Local folder**: the same, read from a path on this machine.
- **EnterpriseRAG-Bench** ([repo](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), MIT): one
  click downloads the release (1.26 GB), loads ~512k synthetic company documents with 500 questions
  in 10 categories, and builds a `dev` subset (100 questions, their gold documents, 20k distractors).
- **Python script**: `extract(path)` yields documents, `questions(path)` yields questions.

Questions file, one JSON object per line:

```json
{"question_id": "q1", "question": "…", "expected_doc_ids": ["…"], "gold_answer": "…", "category": "…"}
```

`expected_doc_ids` drive the recall metrics; `gold_answer` enables answer judging. A dataset without
questions is refused at ingest.

Each ingest compares incoming documents with the dataset's current state by content hash and stores
only the difference as a new snapshot, so old snapshots stay reconstructable.

## Targets, append and overwrite

- **Overwrite** clears the target, then writes everything.
- **Append** writes only documents that are new, or whose content, chunking or embedding settings
  changed since they were last written (their old chunks are replaced). Unchanged documents are
  skipped. Documents that disappeared from the source are left in the target.
- Appending with a different embedder than the one the target was written with is refused: mixed
  vector spaces can't be searched. Overwrite instead.
- A target has a **version** that increases on every write that changes it. Benchmark runs record
  the version they read, and results are cached per version.

| Target type | Keyword search | Vector search | Needs |
|---|---|---|---|
| `local` | BM25 (DuckDB FTS) | exact, in numpy | nothing |
| `qdrant` | – | yes | nothing (embedded on disk), or a Qdrant URL |
| `opensearch` | BM25 | k-NN | a reachable OpenSearch cluster |
| `python` | your `search()` | your `search()` | your code |

Credentials are never stored: Qdrant and OpenSearch settings take the *names* of environment
variables (set them in `.env`).

## Script nodes

```python
def extract(path):                       # data source: path is the upload folder (or one you name)
    yield {"doc_id": "a", "text": "...", "title": "...", "source": "..."}

def chunk(doc):                          # chunk node: doc has doc_id, source, title, text, body
    return [p for p in doc["body"].split("\n\n") if p.strip()]

def embed(texts):                        # embed node: one vector per text; optional embed_query(text)
    return my_model.encode(texts)

def retrieve(question, search, top_k):   # retrieve node: search.bm25(q, k) / search.vector(q, k)
    return search.bm25(question, top_k)

def rerank(question, chunks, top_n):     # rerank node: chunks have chunk_id, doc_id, score, text
    return sorted(chunks, key=lambda c: -len(c["text"]))[:top_n]

# target script: clear(), delete_docs(doc_ids), add(chunks), search(query, query_vector, k)
```

A script is part of its node's settings, so it is hashed for caching (edit it and only that node and
what follows re-run) and saved in every run's `config.json`.

## Caching

Stage outputs live under `cache/<stage>/<key>/`, where
`key = hash(stage settings + upstream key + source hash of the stage's module)`. A change
invalidates that stage and everything after it, nothing before it.

Underneath, chunks are cached per document content hash and embeddings per chunk-text hash
(`cache/store/`), so a dataset that grows only chunks and embeds the new material, and embedding
progress survives an interrupted run.

## Metrics

Free on every benchmark (no LLM):

- `recall@k`: share of gold documents among the top-k retrieved documents
- `context_recall`: share of gold documents that reached the LLM prompt
- `answer_recall` / `extra_docs`: gold documents among, and non-gold documents in, the answer's document ids
- `abstained`: share of questions the model declined to answer

With answers and an API key: `correctness` and `completeness`. EnterpriseRAG-Bench uses the
benchmark's own judge (run with `--no-correction`, which also yields `judge_doc_recall` and
`invalid_extra_docs`; its token costs are not tracked). Other datasets use the built-in generic
judge: one Claude call per question comparing the answer with the gold answer; its cost is tracked.

## What a run folder holds

```
runs/<run_id>/
  run.json            kind (ingest | benchmark), status, dataset + snapshot, target version, git, timings
  config.json         full resolved config, scripts included
  code_hashes.json    source hash of every stage implementation used
  git.diff            only if the working tree was dirty
  stages.json         per stage: key, cached?, duration, stats
  usage.json          tokens (incl. cache read/write) and cost            (benchmark)
  retrieval.jsonl  reranked.jsonl  answers.jsonl  generations.jsonl       (benchmark)
  per_question.jsonl  metrics.json  judge_results.json  judge.log         (benchmark)
```

`runs/index.duckdb` indexes all folders for the UI; rebuild it with `uv run rag-lab runs reindex`.
Pipelines and targets you define are JSON files under `workspace/`.

## Architecture

```
backend/rag_lab/
  cli.py          typer CLI
  datasets/       dataset adapters (enterprise_rag_bench, folder, script)
  data.py         versioned document store, snapshots, subsets, built-in sources
  targets/        target types (local, qdrant, opensearch, script) + per-target manifest
  stages/         load, chunk, embed, retrieve, rerank, generate, evaluate
  registry.py     stage implementation registry
  scripting.py    loading user scripts, their templates, built-in → script equivalents
  workspace.py    targets and pipeline graphs (workspace/*.json) → runnable configs
  pipeline.py     the engine: run_ingest, run_benchmark, run recording
  cache.py        content-addressed stage cache
  jobs.py         background job queue; worker.py runs one job in a subprocess
  llm/            provider interface, Anthropic provider, pricing
  bench.py        EnterpriseRAG-Bench judge setup + subprocess call
  tracking.py     run folders → runs/index.duckdb
  api.py          FastAPI backend for the UI
web/              Vite + React + TypeScript UI (React Flow canvas, CodeMirror, Recharts)
```

Anything slow or that writes (dataset ingests, ingestion runs, benchmarks) is a background job: one
at a time, each in its own subprocess, with a log you can open in the UI.

## Extending

- **Stage implementation**: `@register("retrieve", "my_retriever", MyConfig)` on a function in
  `stages/`; its Pydantic config becomes a settings form automatically.
- **Target type**: subclass `Target` in `targets/` and decorate with `@register_target("name", Config)`.
- **Dataset adapter**: subclass `Dataset` in `datasets/`.

## Notes and limits

- The EnterpriseRAG-Bench `dev` subset is much easier than the full corpus (fewer distractors).
- The local target's vector search is exact brute force; fine for ~100k chunks, memory-hungry for millions.
- Benchmark runs store retrieved chunk text in the run folder (self-contained, but a few MB per run).
- Claude requests opt into server-side refusal fallback (`refusal_fallback` on the generate node).

## License

MIT. EnterpriseRAG-Bench data and code are MIT-licensed by their authors and are downloaded, not
redistributed, by this repo.
