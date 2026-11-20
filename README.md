# rag-lab

**Learn to build a really good RAG pipeline, end to end, by measuring it.**

> AI can write a RAG pipeline if you ask it to. But how do you learn to write a *really good* one?
> How do you know whether hybrid search beats vector search on *your* data, whether a reranker
> earns its latency, or whether an agent actually answers better than a single prompt?
>
> **rag-lab is the answer: a hands-on lab where every step of RAG is visible, editable and scored.**

![rag-lab in 30 seconds: lessons, datasets, the pipeline canvas, the script editor, a failed step's log, the results matrix, comparing two runs and a single question](docs/demo.gif)

rag-lab is an open-source learning tool and playground for Retrieval-Augmented Generation. Load a
real benchmark, build ingestion and retrieval pipelines on a node canvas, run them, and see exactly
which questions each change fixed or broke, with a significance test that tells you whether the
difference is real or noise. Every node is either a built-in with a settings form or a Python script
you edit in the browser, so you can read, change and own every line.

**What you learn by using it**

- How chunking, embedding, keyword (BM25), vector and hybrid search, reranking and generation each
  change retrieval quality, in numbers instead of opinions.
- How to read RAG metrics: recall@k, nDCG, MRR, context recall, answer correctness.
- How to tell a real improvement from noise (paired permutation test, bootstrap intervals).
- How single-pass RAG compares with agentic RAG that searches again on its own.
- How GraphRAG works: a knowledge graph of the corpus, graph search, and when a graph helps.
- How the same pipeline behaves on different vector stores: local, pgvector, Qdrant, OpenSearch,
  Elasticsearch, any SQL database.
- How to debug a pipeline: every run keeps each step's status, timing and log.

## Walkthrough: from a dataset to a benchmark

Two minutes, end to end: load SciFact, look at a target, open the ingestion and retrieval
pipelines and their Python, run a benchmark, find a question BM25 missed, and compare BM25 with
vector search.

![Walkthrough: datasets, targets, the ingestion canvas and its Python, a BM25 retrieval pipeline and a test query, running a benchmark on 300 SciFact questions, a question BM25 missed because of a typo, comparing BM25 with vector search and the significance test, and the guided lessons](docs/walkthrough.gif)

## Try it in two minutes

No API key needed. You need [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
git clone https://github.com/kkfaisal/rag-lab.git && cd rag-lab
uv sync
uv run rag-lab quickstart --serve
```

The quickstart downloads SciFact (4 MB: 5,000 scientific abstracts and 300 questions with gold
documents), builds a BM25 index, answers all 300 questions and prints your first scores, about 15
seconds after the download. It builds the web UI the first time, then starts it at
http://127.0.0.1:8000. Open **Lessons** and carry on from there.

With Docker instead, run the quickstart before starting the server (both use the same document store):

```bash
docker compose run --rm rag-lab rag-lab quickstart
docker compose up -d
```

## Lessons

Ten small experiments, each with a goal score that rag-lab checks against your runs. The goals were
measured on SciFact, so each is reachable, but only with the lesson's idea.

| # | Lesson | You learn | Goal |
|---|---|---|---|
| 1 | Your First Benchmark | what gold documents and recall@k are | recall@10 ≥ 80 |
| 2 | Search by Meaning: Vector Search | what embeddings find, and what they miss | recall@10 ≥ 82 with vectors |
| 3 | Is It Real or Noise? | reading the paired significance test | self-check |
| 4 | Best of Both: Hybrid Search | fusing BM25 and vectors with RRF | recall@10 ≥ 84 with hybrid |
| 5 | Does Reranking Help Here? | measuring a reranker (it hurts on SciFact) | self-check |
| 6 | Write Your Own Retriever | that every node is just Python | recall@10 ≥ 84 with your script |
| 7 | Chunking Changes Everything | chunk size versus precision and context | self-check |
| 8 | GraphRAG: When Does a Graph Help? | knowledge graphs and graph search, measured | recall@10 ≥ 85.5 with the graph |
| 9 | Let Claude Answer | answers with citations and an LLM judge | correctness ≥ 60 (API key) |
| 10 | Agentic RAG | when an agent that searches again is worth it | correctness ≥ 80 (API key) |

![Lessons](docs/screenshots/00-lessons.png)

## How it works

![How rag-lab works: a dataset of raw documents, questions and gold documents and answers; raw documents go through the ingestion pipeline into a target; the retrieval pipeline searches the target; each question with its retrieved results is compared with the gold documents and gold answers to produce the benchmark](docs/rag-lab-overview.png)

1. **Dataset.** Every benchmark dataset has three parts that belong together: the raw documents (the
   corpus to search), the questions, and the gold documents and gold answers for each question.
2. **Ingestion pipeline.** The raw documents are transformed, chunked and optionally embedded...
3. **...into a target**, the index that retrieval searches: local, pgvector, Qdrant, OpenSearch,
   Elasticsearch, any SQL database or your own script.
4. **Retrieval pipeline.** Retrieves from the target (BM25, vector or hybrid), optionally reranks, and
   answers with Claude in one pass or as an agent. The search box lets you try any query live.
5. **Compare with the gold set.** Each question is joined with its retrieved results. Retrieved
   documents are scored against the gold documents (recall@k, nDCG, MRR) and generated answers against
   the gold answers (LLM judge).
6. **Benchmark.** Scores per run and per question, a results matrix across targets and pipelines, and
   run-to-run comparisons that say whether a change really helped.

Every step is cached by its settings and code, so changing one node re-runs only that node and what
follows it. Every run records its full config (scripts included), dataset snapshot, target version,
per-step timings and logs, token usage and cost.

## A walk through the tool

### 1. Get a dataset

The Datasets page lists public benchmarks (EnterpriseRAG-Bench, SciFact, NFCorpus, the BEIR sets).
**Get** downloads the raw files with a Python extractor you can read first; **Load** puts them into
the internal, versioned document store.

![Datasets](docs/screenshots/01-datasets.png)

Each dataset has an overview: document counts, token estimates, length distribution, question
categories, and how many gold documents actually exist.

![Dataset overview](docs/screenshots/02-dataset-overview.png)

### 2. Make a subset to iterate fast

Half a million documents is slow to experiment on. A **subset** picks some questions, keeps all
their gold documents and adds random distractors, so search is still hard but runs in minutes.

![Subsets](docs/screenshots/03-subsets.png)

### 3. Choose where chunks live: targets

A target is a persistent store that chunks are written to and searched in. The local store needs
nothing; PostgreSQL + pgvector, Qdrant, OpenSearch and Elasticsearch come with docker compose.

![Targets](docs/screenshots/04-targets.png)

### 4. Build an ingestion pipeline

`source → transform → chunk → embed → target` on a canvas. One pipeline can branch to fill several
targets, for example one with vectors and one BM25-only. The **Run history** below the canvas shows
every run with each step's status and time.

![Ingestion pipeline](docs/screenshots/05-ingestion.png)

### 5. Read and edit the code of any step

Every node, built-in or not, opens as Python in a full-window editor with a live contract check,
version history and optional AI help (Claude proposes a change as a diff you accept or discard).
This is where the learning happens: see how a chunker works, change it, measure the effect.

![Script editor](docs/screenshots/06-script-editor.png)

### 6. Build a retrieval pipeline

`target → retrieve → rerank → generate → benchmark`. Try BM25, vector or hybrid retrieval, a
cross-encoder reranker, and Claude answering in one pass or as an agent that searches again.

![Retrieval pipeline](docs/screenshots/07-retrieval.png)

### 7. Debug a failed run

When a run fails, the failed node turns red and its step log opens with the error. Click any run in
the history to show it on the canvas, and any node or step to read its log.

![Run history and step logs](docs/screenshots/08-run-history-logs.png)

### 8. Compare every combination

The Results matrix shows the latest score for every target × retrieval pipeline.

![Results](docs/screenshots/09-results.png)

### 9. Drill into a run and a single question

A run shows its metrics, per-category scores, timings and cost. A question shows the gold
documents, the retrieved and reranked chunks (gold in green), the generated answer with citations,
and the judge's verdict.

![A run](docs/screenshots/10-run.png)

![A single question](docs/screenshots/11-question.png)

### 10. Is B really better than A?

Compare two runs on the same questions: which were fixed, which broke, and a paired significance
test that says plainly when a gap could be noise.

![Compare two runs](docs/screenshots/12-compare.png)

The **How it works** page and the **Glossary** inside the app explain every concept and metric.

![How it works](docs/screenshots/13-how-it-works.png)

## The model

| Thing | What it is |
|---|---|
| **Dataset** | Source documents **plus benchmark questions** (both required). Comes from a built-in data source or a Python script. Versioned: every ingest that changes something is a snapshot. |
| **Target** | A named, persistent store that chunks are written to and searched in. Built-in: local (vectors + BM25), knowledge graph (local + a graph in LadybugDB), PostgreSQL + pgvector, any SQL database, Qdrant, OpenSearch, Elasticsearch. Or a Python script for your own store. |
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
git clone https://github.com/kkfaisal/rag-lab.git && cd rag-lab
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
git clone https://github.com/kkfaisal/rag-lab.git && cd rag-lab
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
the same loaded dataset can feed any kind of pipeline, including a knowledge graph (see
[GraphRAG](#graphrag-knowledge-graph-targets)).

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
| `graph` | BM25 (DuckDB FTS) | exact, in numpy | nothing; adds graph search (see GraphRAG) |
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
- `ndcg@10`: like recall@10, but gold documents ranked higher count more (binary relevance; the metric
  BEIR papers report)
- `mrr@10`: 1 / rank of the first gold document in the top 10 (0 if none)
- `context_recall`: share of gold documents that reached the LLM prompt
- `answer_recall` / `extra_docs`: gold documents among, and non-gold documents in, the answer's document ids
- `abstained`: share of questions the model declined to answer

With answers and an API key: `correctness` and `completeness`. EnterpriseRAG-Bench uses the
benchmark's own judge (run with `--no-correction`, which also yields `judge_doc_recall` and
`invalid_extra_docs`; its token costs are not tracked). Other datasets use the built-in generic
judge: one Claude call per question comparing the answer with the gold answer; its cost is tracked.

**Is B really better than A?** The Compare page runs a paired test on the questions both runs
answered: the mean difference with a 95% bootstrap confidence interval, a paired permutation test
(p-value), and how many questions each run won. It says plainly when a gap could be noise, and
warns below 30 questions. With 20 questions, differences of 10 points or more are often noise.

## GraphRAG: knowledge graph targets

A **Knowledge graph** target is a local store (BM25 + vectors) plus a graph in
[LadybugDB](https://github.com/LadybugDB/ladybug), an embedded graph database queried with Cypher
(a maintained fork of Kùzu). Nothing to install or run. After every write the graph is rebuilt from the
chunks:

```
(:Chunk)-[:Mentions]->(:Entity)              the distinctive terms of each chunk (TF-IDF, no LLM)
(:Entity)-[:Related {weight}]->(:Entity)     terms that share chunks; weight = how many
```

**Graph search** finds the question's terms in the graph, follows `Related` to each one's strongest
neighbours, and ranks chunks by the terms they mention, rarer terms counting more. Use it as the
`graph` retrieve node, or add it to hybrid search with `graph_weight`, next to `bm25_weight` and
`vector_weight`. Scripts can call `search.graph(question, k)`.

Measured on SciFact (300 questions, `bge-small`, 1,200-character chunks):

| Retrieval | recall@10 | nDCG@10 |
|---|---|---|
| graph search alone | 60.0 | 43.0 |
| hybrid (BM25 + vectors) | 85.0 | 72.8 |
| hybrid + graph (`graph_weight` 0.5) | 85.7 | 69.9 |

On SciFact the graph adds little: each question is about one claim, which BM25 and vectors already
find. Graphs are meant for questions that connect facts across documents; lesson 8 asks you to find
out where they pay off. The settings (terms per chunk, document-frequency limits, neighbours followed)
are on the target. To explore the graph yourself:

```python
import ladybug
con = ladybug.Connection(ladybug.Database("data/targets/<target id>/graph.lbdb", read_only=True))
con.execute("MATCH (a:Entity {name: 'insulin'})-[r:Related]-(b) RETURN b.name, r.weight ORDER BY r.weight DESC LIMIT 10")
```

## Generating answers: single pass or agentic

A retrieval pipeline's generate node answers each question with Claude:

- **`llm`** (single pass): the top `context_chunks` chunks (after reranking) go into one prompt; Claude
  answers once, citing the numbered documents it used, or says the answer is not in them.
- **`agent`** (agentic RAG): Claude starts from the same chunks and has a `search` tool over the same
  target (hybrid by default). It can rephrase, search for each part of a question, follow names it
  found, and then calls an `answer` tool. `max_searches` (default 5) caps the extra searches. Each
  question's page lists the searches it made. Its `context_recall` counts every document it saw.

Both write the same answer files, so the same metrics and judges apply and runs compare directly.
The default prompts are dataset-neutral; edit `system_prompt` on the node to add domain context.

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
  targets/        target types (local, graph, pgvector, sql, qdrant, opensearch, elasticsearch, script) + manifest
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests and worked examples.

## Notes and limits

- The EnterpriseRAG-Bench `dev` subset is much easier than the full corpus (fewer distractors).
- The local target's vector search is exact brute force; fine for ~100k chunks, memory-hungry for millions.
- Benchmark runs store retrieved chunk text in the run folder (self-contained, but a few MB per run).
- Claude requests opt into server-side refusal fallback (`refusal_fallback` on the generate node).

## Roadmap

- **Richer GraphRAG.** Entities and relations extracted by an LLM (the graph target now uses
  TF-IDF terms), community summaries for global questions, multi-hop benchmarks, and Neo4j as a
  server-backed graph target.
- **More rerankers and embedders.** Hosted embedding and rerank APIs as built-in nodes, alongside the
  local models.
- **Query transformation nodes.** Query rewriting, HyDE and multi-query expansion before retrieval.
- **More lessons.** Structure-aware chunking, query rewriting, metadata filters.
- **Cost and latency views.** Quality against cost and latency per pipeline, to pick the cheapest
  pipeline that is good enough.
- **More evaluation.** Faithfulness / groundedness checks and human-labelled spot checks.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT (see `LICENSE`). Benchmark datasets are not included: `scripts/get_datasets.py` downloads them
from their authors, and each keeps its own license (EnterpriseRAG-Bench: MIT; BEIR datasets: see
their dataset cards).
