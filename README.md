# rag-lab

An open-source, end-to-end **playground for RAG pipelines**. Bring data, build ingestion and
retrieval pipelines on a node canvas, run them from the browser, and benchmark every combination
against a question set.

## The model

| Thing | What it is |
|---|---|
| **Dataset** | Source documents **plus benchmark questions** (both required). Comes from a built-in data source or a Python script. Versioned: every ingest that changes something is a snapshot. |
| **Target** | A named, persistent store that chunks are written to and searched in. Built-in: local (vectors + BM25), PostgreSQL + pgvector, any SQL database, Qdrant, OpenSearch, Elasticsearch. Or a Python script for your own store. |
| **Ingestion pipeline** | `source → (transform) → chunk → (embed) → target`. Running it writes to the target, **appending** or **overwriting** as the target node says. Can branch to fill several targets. |
| **Retrieval pipeline** | `target → retrieve → rerank → generate → benchmark`. Running it answers the dataset's questions from the target and scores the result. |

Every node is either a built-in with a settings form or a **Python script you edit in the browser**.
Built-in nodes show their Python too, with an *Edit as Python script* button that swaps the node for
an equivalent editable script. Any script opens in a full-window editor with a live check and
optional AI help (see [Code editor](#code-editor)).

Every run is recorded with its full resolved config (scripts included), code version, dataset
snapshot, target version, per-stage timings, token usage and cost.

> **Local tool.** Scripts run as ordinary Python on your machine with your privileges.
> `rag-lab serve` binds to 127.0.0.1 and docker-compose publishes on 127.0.0.1 only; don't expose it
> to a network you don't trust.

## Quickstart with Docker

Requirements: Docker with Compose.

```bash
git clone <this repo> rag-lab && cd rag-lab
cp .env.example .env                      # optional: put your ANTHROPIC_API_KEY in .env
docker compose up -d                      # rag-lab + PostgreSQL/pgvector + Qdrant
docker compose --profile search up -d     # also OpenSearch and Elasticsearch (~1 GB RAM each)
docker compose exec rag-lab python scripts/get_datasets.py scifact --load   # a first benchmark (4 MB)
```

Open http://127.0.0.1:8000. The Targets page lists the database services and whether each is
running; *Add target* creates a target already pointed at it.

| Service | Image | Host port | Starts with |
|---|---|---|---|
| rag-lab | built from this repo | 8000 | `docker compose up` |
| PostgreSQL + pgvector | `pgvector/pgvector:pg17` | 5433 | `docker compose up` |
| Qdrant | `qdrant/qdrant:v1.20.0` | 6333 | `docker compose up` |
| OpenSearch | `opensearchproject/opensearch:2.19.1` | 9200 | `--profile search` |
| Elasticsearch | `docker.elastic.co/elasticsearch/elasticsearch:9.1.5` | 9201 | `--profile search` |

Ports are published on 127.0.0.1 only. Postgres is on 5433 so it doesn't clash with a Postgres you
already run. `data/`, `cache/`, `runs/`, `workspace/` and `.bench/` are bind-mounted, so the container
and a local `uv run rag-lab serve` share the same experiments (don't run both at once). Put folders
the image uses CPU-only PyTorch. Dataset folders are under `data/datasets/`, so you can drop
files there from the host.

To run only the databases and rag-lab itself with uv: `docker compose up -d postgres qdrant`.

## Quickstart without Docker

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, git.

```bash
git clone <this repo> rag-lab && cd rag-lab
uv sync                                   # Python deps (Python 3.12 is pinned)
cp .env.example .env                      # optional: put your ANTHROPIC_API_KEY in .env
cd web && npm install && npm run build && cd ..
uv run python scripts/get_datasets.py scifact --load   # a first benchmark (4 MB); see below
uv run rag-lab serve                      # http://127.0.0.1:8000
```

Then, in the browser:

1. **Datasets.** Two buttons per dataset: **Get** downloads the raw files (a Python extractor script
   you can read first), then **Load** reads them into rag-lab's internal document store. Or write your
   own extractor with **New dataset**, or create an empty folder, add files and press **Load**.
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

## Getting the data

No benchmark data is in this repository; `data/` is git-ignored. Each user downloads what they need
to their own machine with `scripts/get_datasets.py`:

```bash
uv run python scripts/get_datasets.py --list                      # what's available, with sizes
uv run python scripts/get_datasets.py scifact nfcorpus --load     # download + load into the document store
uv run python scripts/get_datasets.py enterprise_rag_bench --load # 1.3 GB, ~512k docs, builds a "dev" subset
uv run python scripts/get_datasets.py msmarco --split dev         # BEIR: pick the question split
```

| Benchmark | Download | What it is |
|---|---|---|
| `enterprise_rag_bench` | 1.3 GB | ~512k synthetic company documents (Slack, Confluence, Jira, …), 500 questions with gold answers |
| `scifact` | 4 MB | 5k scientific abstracts, 300 claims |
| `nfcorpus` | 3 MB | 3.6k nutrition and medical articles, 323 questions |
| `arguana` | 6 MB | 8.7k arguments, 1.4k counter-argument queries |
| `scidocs`, `quora`, `fiqa` | 19–28 MB | papers, duplicate questions, financial forum posts |
| `trec-covid`, `webis-touche2020` | 111–357 MB | COVID papers, controversial-topic arguments |
| `nq`, `hotpotqa`, `dbpedia-entity`, `msmarco`, `climate-fever`, `fever` | 0.8–2 GB | millions of Wikipedia or web passages |

Every dataset lands in `data/datasets/<name>/` as `docs.jsonl` + `questions.jsonl`. Any other
[BEIR](https://huggingface.co/BeIR) name works too. BEIR sets come from their Hugging Face mirror
through the extractor template, saved as `extract.py` in the folder so you can edit and re-run it from
the dataset's Extractor tab. Downloads stream to disk, so the large ones don't need much memory.
Without `--load` the script only downloads; press **Load** on the Datasets page afterwards.
`uv run rag-lab data get …` does the same as the script.

Every public dataset, EnterpriseRAG-Bench included, is fetched by a Python extractor script that is
saved in its folder as `extract.py`. EnterpriseRAG-Bench's documents stay in the release zip
(1.3 GB, rather than ~3 GB unpacked), which its own reader reads directly. Its questions can be scored
by the benchmark's LLM judge (`uv run rag-lab bench setup`), and loading it builds a 100-question `dev`
subset. Downloads skip files that are already there.

On a fresh install the Datasets page lists these public datasets, each with two buttons. **Get**
opens the dataset's extractor script, so you can read or edit it before running it; running it
downloads the raw files to `data/datasets/<name>/`. **Load** then reads that folder into the internal
document store. The two are separate so you can look at (or change) the raw files before loading. rag-lab also creates a local target and two example retrieval pipelines. The example ingestion pipeline appears once the first dataset with questions is
loaded. Each example is created only once, so deleting one doesn't bring it back.

**Licenses.** The data belongs to its authors and is downloaded from them, never redistributed here.
EnterpriseRAG-Bench is MIT; the BEIR datasets have their own licenses (see each dataset's card on
Hugging Face). Check them before you use or share the data.

## Datasets

**A dataset is a folder**: `data/datasets/<name>/` holding documents and a `questions.jsonl`.

```
data/datasets/
  enterprise_rag_bench/   extract.py, all_documents.zip, questions.jsonl   (Python extractor)
  scifact/                extract.py, docs.jsonl, questions.jsonl    (Python extractor)
  my_docs/                handbook/*.md, tickets.jsonl, questions.jsonl   (your own files)
```

Documents can be `.txt` / `.md` files (doc id = path inside the folder) or `.jsonl` lines
`{"id", "text", "title"?, "source"?, "metadata"?}`. A folder gets filled in one of three ways:

- **EnterpriseRAG-Bench** (built-in, [repo](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), MIT):
  downloads the release (1.26 GB), loads ~512k synthetic company documents with 500 questions in 10
  categories, and builds a `dev` subset (100 questions, their gold documents, 20k distractors).
- **Python extractor**: a script with `download(folder)` that fetches data, e.g. a public benchmark,
  and writes it into the folder. It is saved there as `extract.py`, so it can be edited and re-run.
  The template downloads [BEIR](https://github.com/beir-cellar/beir) SciFact (5k abstracts, 300
  questions with gold documents) from Hugging Face; change `DATASET` for other BEIR sets.
- **Your own files**: create the folder (in the UI, or just `mkdir`), then copy or upload files.

**Two steps: download, then load.** Downloading (or copying files in) only fills the folder: that is
the raw data. **Loading** reads the folder into rag-lab's internal document store
(`data/corpus.duckdb`): each document normalized to one row (id, source, title, text, metadata,
content hash), plus the questions. Each load compares the documents with the dataset's current state
by content hash and stores only the difference as a new snapshot, so older snapshots stay
reconstructable and runs record exactly which documents they saw. A folder without questions is
refused. Pipelines read from the store, not the folder, so formats are parsed once, not on every run.

The store knows nothing about chunks, vectors or indexes: those are made by ingestion pipelines. So
the same loaded dataset can feed any kind of pipeline, including future ones (a graph-building
ingestion, for example, would read the same documents and write entities and relations instead of
chunks).

Each dataset's page has:
- **Overview**: document count, approximate tokens, length percentiles and a length histogram, source
  types, question categories, how many questions have gold documents or answers, and how many gold
  document ids actually exist in the documents (missing ones cap recall).
- **Files**: browse, preview (text, and zip contents), upload and delete files.
- **Questions**, **Snapshots**, and **Extractor** (when the folder has `extract.py`).

Questions file, one JSON object per line:

```json
{"question_id": "q1", "question": "…", "expected_doc_ids": ["…"], "gold_answer": "…", "category": "…"}
```

`expected_doc_ids` drive the recall metrics; `gold_answer` enables answer judging.

From the command line: `uv run rag-lab data add <name>` loads `data/datasets/<name>/`;
`--source extract` runs its `extract.py` first.

**From folder to target.** An ingestion pipeline reads a loaded dataset, chunks it, and writes to a
target, either through an embed node (vector search) or straight from the chunk node (keyword /
BM25 only, no embedding). One pipeline can branch into both. On SciFact (300 questions, 1,200-char
chunks), vector search (`bge-small`) got recall@10 82.8 and BM25 got 81.8.

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
| `pgvector` | Postgres full-text (`ts_rank_cd`, words OR-ed) | HNSW, cosine | Postgres with the `vector` extension |
| `sql` | BM25, local index | exact, local index | any SQLAlchemy database (SQLite file by default) |
| `qdrant` | – | yes | nothing (embedded on disk), or a Qdrant URL |
| `opensearch` | BM25 | k-NN (HNSW) | an OpenSearch cluster |
| `elasticsearch` | BM25 | k-NN (HNSW) | Elasticsearch 9 |
| `python` | your `search()` | your `search()` | your code |

**`sql` is the JDBC-style connector.** It writes to any database SQLAlchemy reaches, given its
driver: PostgreSQL (`postgresql+psycopg://`) and MySQL/MariaDB (`mysql+pymysql://`) drivers are
installed; for SQL Server, Oracle, Snowflake and others, `uv add` their driver. The database is the
store of record (one table, vectors as float32 bytes). Plain SQL can't rank by BM25 or vector
similarity portably, so after each write rag-lab reads the table back and builds a local search
index from it. Changes made to the table outside rag-lab are picked up on the next write.

**Approximate vs exact vector search.** pgvector, OpenSearch and Elasticsearch use HNSW, which can
miss some true nearest neighbours; the local and sql targets search exactly. Each HNSW target has a
search-width setting (`ef_search`, `candidates`, `num_candidates`): higher is closer to exact and
slower. Changing it invalidates cached benchmark results for that target. On the EnterpriseRAG-Bench
dev subset (20 questions, `bge-small`), exact search gave recall@10 67.4. pgvector at its default
width of 40 gave 61.1 and at 100 gave 67.4. Elasticsearch at 200 candidates gave 67.4. OpenSearch at
200 gave 61.1; one question needed a width of about 1,000.

Credentials are never stored in a target. Database URLs can say `${NAME}`, filled in from the
environment (`.env`) when connecting, e.g. `postgresql://raglab:${POSTGRES_PASSWORD}@localhost:5433/raglab`.
Qdrant, OpenSearch and Elasticsearch settings take the *names* of environment variables. Every target
form has a **Test connection** button.

## Transform node

An optional node between source and chunk that changes documents before chunking:

- **clean** (built-in): strip HTML, drop lines matching a regex (boilerplate footers, signatures),
  normalise whitespace, drop documents shorter than `min_chars`, cut ones longer than `max_chars`.
- **python**: `transform(doc)` gets `doc_id`, `source`, `title`, `text` and `metadata` (a dict) and
  returns the changed document, or `None` to drop it. `doc_id` must stay the same, since questions refer to it.

Each document's content hash is recomputed from the transformed content, so the chunk and embed caches
and append mode follow the transform: change it, and only documents whose output changed are
rechunked, re-embedded and rewritten. The node panel previews the transform on a random document
(before and after), and *Edit as Python script* turns `clean` into the equivalent script.

## Script nodes

```python
def download(folder):                    # dataset extractor: write documents + questions.jsonl into folder
    ...

def transform(doc):                      # transform node: return the changed doc, or None to drop it
    return {**doc, "text": doc["text"].strip()}

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

## Code editor

Every script (data source, chunk, embed, retrieve, rerank and target scripts) has an **Open editor**
button that opens a full-window editor:

- **Check**: as you type, the script is parsed (never executed) and checked for the functions its
  stage calls, e.g. `chunk` for a chunk node or `clear` / `delete_docs` / `add` / `search` for a target.
- **Contract**: the functions rag-lab calls and the starting template with their arguments and
  return shapes.
- **AI assist** (needs `ANTHROPIC_API_KEY`): describe a change, or pick a suggestion such as *Fix the
  problem the check reports*. Claude rewrites the script within the stage's contract and the change is
  shown as a diff to accept or discard; nothing changes until you accept and save. The model is
  `claude-sonnet-5-5`, or set `RAG_LAB_ASSIST_MODEL`. Each suggestion's cost is shown.

`⌘S` saves, `Esc` closes.

## How data moves, and large datasets

Nodes don't pass data in memory. Each step writes its output to files in the cache, and the next
step reads them:

| Step | Reads | Writes |
|---|---|---|
| load | the document store | `docs.parquet` (the snapshot's documents) |
| transform | `docs.parquet` | `docs.parquet` (transformed; skipped when there's no transform node) |
| chunk | documents | `chunks.parquet` |
| embed | `chunks.parquet` | `embeddings.npy` (one vector per chunk, same row order) |
| write | `chunks.parquet` + `embeddings.npy` | the target, 2,000 chunks at a time |

Every step reads its input in batches and writes as it goes, so memory doesn't grow with the dataset.
Joins, deduplication and the append/overwrite diff run in DuckDB, which is capped at
`RAG_LAB_DUCKDB_MEMORY` per connection (default `1GB`) and spills to `cache/duckdb-tmp/` beyond that.
Raise the cap for speed if you have the RAM.

Measured on synthetic text with a fast toy embedder (32-dim, no model) and a local target, at the
default 1 GB cap. The 5 GB dataset was 800k documents, which made 5.8M chunks:

| Step | Peak memory (process) | Notes |
|---|---|---|
| load the dataset folder into the store | 3.7 GB | |
| chunk | 3.3 GB | flat across the run |
| embed | 3.1 GB | flat across the run |
| write to target | 4.2 GB | 6.1 GB including the local target's BM25 index build |

The previous version used 14.4 GB on just 1.25 GB of text, and would not have fit 5 GB on most machines.
The one part that still grows with size is the local target's BM25 index, which DuckDB builds mostly
in memory: about 2 GB for 6M chunks. It has its own cap, `RAG_LAB_FTS_MEMORY` (default `4GB`).

Memory is no longer the limit at this size; time and disk are:

- **Embedding time.** `bge-small` on a laptop embeds a few hundred chunks per second, so 5M chunks is
  hours. Use a GPU (`device` on the embed node) or an embedding API through an embed script.
- **Disk.** Text is stored in the dataset folder, the document store, `docs.parquet`, the chunk cache
  and `chunks.parquet`: plan for 3–4× the raw size.
- **The local target** searches vectors exactly in memory and rebuilds its BM25 index after every
  write. Past about a million chunks, use pgvector, Qdrant, OpenSearch or Elasticsearch.

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
  datasets/       readers for dataset folders (enterprise_rag_bench, folder)
  data.py         dataset folders, extractors, versioned document store, snapshots, subsets, metrics
  targets/        target types (local, pgvector, sql, qdrant, opensearch, elasticsearch, script) + manifest
  services.py     the docker-compose database services: URLs and reachability
  stages/         load, transform, chunk, embed, retrieve, rerank, generate, evaluate
  db.py           every DuckDB connection: memory cap, spill directory
  registry.py     stage implementation registry
  scripting.py    loading user scripts, their templates and contracts, built-in → script equivalents
  assist.py       AI help in the code editor (Claude proposes a rewritten script)
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

MIT (see `LICENSE`). Benchmark datasets are not included: `scripts/get_datasets.py` downloads them
from their authors, and each keeps its own license (EnterpriseRAG-Bench: MIT; BEIR datasets: see
their dataset cards).
