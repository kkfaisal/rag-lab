# rag-lab

An open-source workbench for iterating on **real RAG pipelines** and seeing exactly what each change did.

You describe a pipeline in a small YAML file, run it from the CLI, and every run is recorded with its
full resolved config, the code version (git SHA + diff, plus a hash of every stage implementation),
the dataset snapshot it saw, per-stage timings, token usage and cost, the answers, and the evaluation.
A web UI lets you browse runs, filter by any config parameter, compare two runs question by question,
and drill into a single question to see which gold documents were retrieved, missed, or cited.

It ships with an adapter for [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
(MIT): ~500k synthetic company documents (Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies,
GitHub, Jira, Confluence) with deliberate noise, and 500 questions across 10 categories, scored by
the benchmark's own LLM judge. You can also bring your own folder of documents.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, git, ~6 GB of disk for the full corpus.

```bash
git clone <this repo> rag-lab && cd rag-lab
uv sync                                   # Python deps (Python 3.12 is pinned)
cp .env.example .env                      # then put your ANTHROPIC_API_KEY in .env (optional, see below)

uv run rag-lab data download              # questions.jsonl + all_documents.zip (1.26 GB) -> data/raw/
uv run rag-lab data load                  # ingest into data/corpus.duckdb (~1 min)
uv run rag-lab data subset                # dev subset: 100 questions (10/category) + gold docs + 20k distractors
uv run rag-lab bench setup                # clone the benchmark's judge into .bench/ with its own venv

uv run rag-lab run configs/baseline.yaml --limit 20

cd web && npm install && npm run build && cd ..
uv run rag-lab serve                      # http://127.0.0.1:8000
```

The first run embeds ~105k chunks with a local model (about 9 minutes on an Apple-silicon laptop);
later runs reuse them from the cache.

**Without an API key** you can still do everything retrieval-related for free:

```bash
uv run rag-lab run configs/baseline.yaml --limit 20 --set generate.impl=none
```

This skips the LLM and the judge and reports recall@k, context recall and answer-doc recall.

**With `ANTHROPIC_API_KEY` set**, `generate` answers with Claude (`claude-sonnet-5-5` by default) and
`evaluate` runs the benchmark judge (`claude-sonnet-5-5`, with `claude-haiku-5-5` for its cheap steps).

For UI development, run `uv run rag-lab serve` and `cd web && npm run dev` (Vite proxies `/api`).

## Concepts

### The pipeline

```
load → chunk → embed → index → retrieve → rerank → generate → evaluate
```

| stage    | implementations (v1)                               | output                         |
|----------|----------------------------------------------------|--------------------------------|
| load     | `corpus` (dataset + snapshot + subset)             | docs.parquet                   |
| chunk    | `fixed`, `slack_aware` (whole messages, thread root repeated) | chunks.parquet      |
| embed    | `sentence_transformers` (default `BAAI/bge-small-en-v1.5`), `none` | embeddings.npy |
| index    | `duckdb` (BM25 via DuckDB FTS; vectors searched with numpy) | index.duckdb          |
| retrieve | `bm25`, `vector`, `hybrid` (Reciprocal Rank Fusion) | retrieval.jsonl               |
| rerank   | `none`, `cross_encoder`                            | reranked.jsonl                 |
| generate | `llm` (provider interface; `anthropic` first), `none` | answers.jsonl, generations.jsonl |
| evaluate | `standard` (free recall metrics + optional benchmark judge) | metrics.json, per_question.jsonl |

`uv run rag-lab impls` lists every implementation and its parameters.

### Configs

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

From the benchmark's judge (when it runs): `correctness`, `completeness`, `judge_doc_recall`,
`invalid_extra_docs`, per question and per category. The judge runs with `--no-correction` (scores
against the original gold set), because its correction flow needs the raw `generated_data/` the
setup deliberately skips (3.9 GB). Judge token costs happen inside the benchmark's code and are not
tracked; generation costs are.

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
  datasets/       dataset adapters (enterprise_rag_bench, folder)
  data.py         versioned document store, snapshots, subsets
  config.py       YAML -> validated, fully-resolved config
  registry.py     stage implementation registry
  cache.py        content-addressed stage cache
  stages/         one module per stage
  llm/            provider interface, Anthropic provider, pricing
  bench.py        benchmark judge setup + subprocess call
  pipeline.py     runs stages, records the run
  tracking.py     run folders -> runs/index.duckdb
  api.py          read-only FastAPI
web/              Vite + React + TypeScript + Recharts UI
```

The UI is read-only in v1: runs are launched from the CLI.

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
