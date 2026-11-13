# rag-lab

An open-source, end-to-end **playground for RAG pipelines**: bring data, build indexing pipelines on a
node canvas, design retrieval setups, run every combination against a question set, and see exactly
what each change did.

The workflow, all in the browser:

1. **Datasets** — upload files (or a zip), optionally write a Python extractor script, add a
   questions file. Every ingest is an incremental, versioned snapshot.
2. **Pipelines** — a node canvas, like an ETL tool: `source → chunk → embed → index`. Branch it to try
   several strategies side by side. Nodes are built-ins with settings forms, or a **Python script
   node** you edit in the browser (with a live preview on a real document). Every index node's
   result is saved and reusable.
3. **Retrieval** — a second canvas: `retrieve → rerank → generate`, again with built-ins or your own
   script, and a "try a query" box against any built index.
4. **Experiments** — tick indexes × retrieval setups, press Run, watch the jobs, and compare the
   results in a matrix. Drill down to a two-run diff (questions fixed/broken) and a per-question
   page showing gold vs retrieved documents and the answer.

Every run is recorded with its full resolved config (scripts included), code version, dataset
snapshot, per-stage timings, token usage and cost. Everything the UI does is also available from
the CLI with YAML configs.

It ships with an adapter for [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
(MIT): ~500k synthetic company documents (Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies,
GitHub, Jira, Confluence) with deliberate noise, and 500 questions across 10 categories, scored by
the benchmark's own LLM judge. Other benchmarks and your own data use the same tool: retrieval
metrics come from gold document ids, and a built-in generic Claude judge scores answers against
gold answers.

> **Local tool.** Script nodes and extractor scripts run as ordinary Python on your machine with
> your privileges. `rag-lab serve` binds to 127.0.0.1; don't expose it to a network you don't trust.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, git.

```bash
git clone <this repo> rag-lab && cd rag-lab
uv sync                                   # Python deps (Python 3.12 is pinned)
cp .env.example .env                      # optional: put your ANTHROPIC_API_KEY in .env
cd web && npm install && npm run build && cd ..
uv run rag-lab serve                      # http://127.0.0.1:8000
```

That is enough to start with your own data: open **Datasets**, upload files, ingest, then build a
pipeline. To work with EnterpriseRAG-Bench (~6 GB of disk), load it first:

```bash
uv run rag-lab data download              # questions.jsonl + all_documents.zip (1.26 GB) -> data/raw/
uv run rag-lab data load                  # ingest into data/corpus.duckdb (~1 min)
uv run rag-lab data subset                # dev subset: 100 questions (10/category) + gold docs + 20k distractors
uv run rag-lab bench setup                # clone the benchmark's judge into .bench/ with its own venv
```

On first start the UI seeds an example pipeline (one source, two chunkers, two indexes) and three
retrieval setups. On **Experiments**, tick both indexes and a couple of setups and press Run.
The first build embeds ~105k chunks with a local model (about 9 minutes on an Apple-silicon
laptop); after that, chunks and embeddings are reused.

**Without an API key** everything retrieval-related works and is free (recall@k, context recall,
answer-doc recall). **With `ANTHROPIC_API_KEY`**, generate nodes answer with Claude
(`claude-sonnet-5-5` by default) and answers are judged: by the benchmark's own judge for
EnterpriseRAG-Bench, by the generic judge elsewhere.

The same from the CLI:

```bash
uv run rag-lab run configs/baseline.yaml --limit 20
uv run rag-lab run configs/baseline.yaml --limit 20 --set generate.impl=none    # no LLM
```

For UI development, run `uv run rag-lab serve` and `cd web && npm run dev` (Vite proxies `/api`).

### Script nodes

Every node shows its Python. Built-in nodes show their implementation read-only with an
**Edit as Python script** button, which swaps the node for an editable script that does exactly the
same thing with the node's current settings — the starting point for your own variant. Chunk,
embed, retrieve and rerank can be scripts (index and generate are built-in only for now).

```python
def chunk(doc):                      # chunk node: doc has doc_id, source, title, text, body
    return [p for p in doc["body"].split("\n\n") if p.strip()]

def retrieve(question, search, top_k):   # retrieve node: search.bm25(q, k) / search.vector(q, k)
    return search.bm25(question, top_k)

def embed(texts):                    # embed node: one vector per text; optional embed_query(text)
    return my_model.encode(texts)

def rerank(question, chunks, top_n):     # rerank node: chunks have chunk_id, doc_id, score, text
    return sorted(chunks, key=lambda c: -len(c["text"]))[:top_n]

def extract(path):                   # dataset extractor: path is the dataset's upload folder
    yield {"doc_id": "a", "text": "...", "title": "...", "source": "..."}
```

The script text is part of the node's settings, so it is hashed for caching (edit the script and
only that node and what follows it re-run) and saved in every run's `config.json`.

## Concepts

### The pipeline

```
load → chunk → embed → index → retrieve → rerank → generate → evaluate
```

| stage    | implementations (v1)                               | output                         |
|----------|----------------------------------------------------|--------------------------------|
| load     | `corpus` (dataset + snapshot + subset)             | docs.parquet                   |
| chunk    | `fixed`, `slack_aware` (whole messages, thread root repeated), `python` (your script) | chunks.parquet |
| embed    | `sentence_transformers` (default `BAAI/bge-small-en-v1.5`), `python`, `none` | embeddings.npy |
| index    | `duckdb` (BM25 via DuckDB FTS; vectors searched with numpy) | index.duckdb          |
| retrieve | `bm25`, `vector`, `hybrid` (Reciprocal Rank Fusion), `python` (your script) | retrieval.jsonl |
| rerank   | `none`, `cross_encoder`, `python`                  | reranked.jsonl                 |
| generate | `llm` (provider interface; `anthropic` first), `none` | answers.jsonl, generations.jsonl |
| evaluate | `standard` (free recall metrics + optional LLM judge) | metrics.json, per_question.jsonl |

`uv run rag-lab impls` lists every implementation and its parameters.

### Configs

A canvas path (one index node + one retrieval setup) compiles to exactly this format, so the
engine is the same for the UI and the CLI:

```yaml
name: baseline
data:     {dataset: enterprise_rag_bench, subset: dev, snapshot: latest}
chunk:    {impl: fixed, chunk_size: 1200, overlap: 150}
retrieve: {impl: vector, top_k: 20}
...
```

Each stage names an `impl`; the other keys are validated by that implementation's Pydantic model and
missing ones are filled with defaults. Override anything from the CLI: `--set retrieve.top_k=50`,
`--set rerank.impl=cross_encoder`. Two configs ship: `configs/baseline.yaml` and
`configs/hybrid_rerank.yaml`.

### Caching

Each stage's output is stored under `cache/<stage>/<key>/` where

```
key = hash(stage config + upstream stage's key + source hash of the stage's module)
```

So a change invalidates that stage and everything after it, and nothing before it: change only the
reranker and load/chunk/embed/index/retrieve are reused. Editing a stage's code changes its hash too.

Underneath, chunks and embeddings are also cached **per item** in `cache/store/`: chunks per
document content hash, embeddings per chunk-text hash. When a dataset grows, only new or changed
documents are chunked and only new chunk texts are embedded. Embedding progress is saved every 10k
chunks, so an interrupted run resumes where it stopped.

The chunk store is tied to the chunker's source code (edit the chunker, get fresh chunks). The embed
store is tied to the model settings plus a `STORE_VERSION` constant in `stages/embed.py`, so editing
that module doesn't discard hours of embeddings; bump the constant if a change alters the vectors.

### Datasets and snapshots

Documents live in `data/corpus.duckdb`, versioned. Every ingest compares incoming docs with the
dataset's current state by content hash and records a **snapshot** with only the difference:

```bash
uv run rag-lab data add my_docs ./path/to/folder          # folder adapter: .txt/.md/.jsonl (+ optional questions.jsonl)
uv run rag-lab data add my_docs ./path/to/folder          # again later: picks up only new/changed files
uv run rag-lab data add my_docs ./folder --sync           # also delete docs no longer present
uv run rag-lab data list
```

DuckDB allows one writing process at a time, so ingest while no run or `serve` is using the corpus.

Runs default to `snapshot: latest` and record the number they resolved to, so older results stay
reproducible. Comparisons (diff, heatmap) are only made between runs on the same dataset.
The folder format is documented in `backend/rag_lab/datasets/folder.py`; add a new adapter by
subclassing `Dataset` in `backend/rag_lab/datasets/`.

### Metrics

Computed for free on every run (no LLM):

- `recall@k` — share of gold docs among the top-k retrieved documents
- `context_recall` — share of gold docs that made it into the LLM prompt
- `answer_recall` / `extra_docs` — gold docs among, and non-gold docs in, the answer's `document_ids`
- `abstained` — share of questions the model declined to answer

From a judge (when answers and an API key exist): `correctness` and `completeness`, per question
and per category. EnterpriseRAG-Bench uses the benchmark's own judge, which adds `judge_doc_recall`
and `invalid_extra_docs`; other datasets use the generic judge (one Claude call per question,
comparing the answer with the gold answer; its cost is tracked). The benchmark judge runs with `--no-correction` (scores
against the original gold set), because its correction flow needs the raw `generated_data/` the
setup deliberately skips (3.9 GB). Its token costs happen inside the benchmark's code and are not
tracked; generation costs are.

`answer_recall` and `extra_docs` are, by construction, the judge's `document_recall` and
`invalid_extra_docs` under `--no-correction` (checked: identical per question), so retrieval
iterations get the benchmark's document metrics without paying for the judge.

### What a run folder holds

```
runs/<run_id>/
  run.json            status, dataset + snapshot, question ids, git, timings
  config.json         full resolved config
  code_hashes.json    source hash of every stage implementation used
  git.diff            only if the working tree was dirty
  stages.json         per stage: key, cached?, duration
  usage.json          tokens (incl. cache read/write) and cost
  retrieval.jsonl  reranked.jsonl  answers.jsonl  generations.jsonl
  per_question.jsonl  metrics.json  judge_results.json  judge.log
```

`runs/index.duckdb` indexes all folders for the UI; rebuild it with `uv run rag-lab runs reindex`.

## Architecture

```
backend/rag_lab/
  cli.py          typer CLI
  datasets/       dataset adapters (enterprise_rag_bench, folder, script)
  scripting.py    loading user-written script nodes + their templates
  workspace.py    canvas graphs (workspace/*.json) -> runnable configs
  jobs.py         background job queue; worker.py runs one job
  data.py         versioned document store, snapshots, subsets
  config.py       YAML -> validated, fully-resolved config
  registry.py     stage implementation registry
  cache.py        content-addressed stage cache
  stages/         one module per stage
  llm/            provider interface, Anthropic provider, pricing
  bench.py        benchmark judge setup + subprocess call
  pipeline.py     runs stages, records the run
  tracking.py     run folders -> runs/index.duckdb
  api.py          FastAPI backend for the UI
web/              Vite + React + TypeScript UI (React Flow canvas, CodeMirror, Recharts)
```

Anything slow or that writes to DuckDB (ingest, index builds, experiments) runs as a background
job: one at a time, each in its own subprocess, with a log you can open in the UI.

## Adding an implementation

```python
# backend/rag_lab/stages/retrieve.py (or a new module imported in stages/__init__.py)
class MyConfig(BaseModel):
    top_k: int = 20

@register("retrieve", "my_retriever", MyConfig)
def my_retriever(cfg: MyConfig, ctx: StageContext) -> dict:
    ...                              # read ctx.dirs["index"], write ctx.out / "retrieval.jsonl"
    return {"some_stat": 1}          # saved in meta.json
```

Then use `retrieve: {impl: my_retriever}` in a config.

## Notes and limits

- The dev subset is much easier than the full 500k corpus (fewer distractors). Use it for fast
  iteration, then confirm on `subset: all`.
- Vector search is exact brute force in numpy. Fine for the dev subset; for the full corpus
  (~2.5M chunks) expect several GB of RAM — an ANN index is a natural next implementation.
- Generation uses structured output (`found`, `answer`, `sources`) so the model can abstain on
  "info not found" questions and cite the documents it used.
- Claude requests opt into server-side refusal fallback (`generate.refusal_fallback`, on by default).

## License

MIT. EnterpriseRAG-Bench data and code are MIT-licensed by their authors and are downloaded, not
redistributed, by this repo.
