# Contributing to rag-lab

Thanks for helping. rag-lab is meant to be read as well as used, so the main rule is:
**keep the code small and plain.** A change that adds a feature in 40 readable lines beats
one that adds it in 200 clever ones. Comments explain *why*, not what.

## Setup

```bash
uv sync                          # Python 3.12 deps, including pytest
cd web && npm ci && cd ..        # the React UI
uv run rag-lab serve             # http://localhost:8000 (serves web/dist; run `npm run build` in web/ first)
cd web && npm run dev            # or: the UI with hot reload on :5173, proxying /api to :8000
```

Copy `.env.example` to `.env`. `ANTHROPIC_API_KEY` is only needed for generate, the LLM
judges and the editor's AI help; everything else runs without it.

## Before you open a pull request

```bash
uv run pytest -q tests           # offline; tests needing Postgres/OpenSearch/... skip if those aren't running
cd web && npm run build          # type-checks the UI
```

CI runs the same two commands. To also test the server-backed targets, start them first:
`docker compose up -d postgres qdrant` (add `--profile search` for OpenSearch and Elasticsearch).

**Never commit data.** `data/`, `cache/`, `runs/` and `workspace/` are git-ignored; datasets
are downloaded by their extractor scripts on each user's machine.

## Where things are

```
backend/rag_lab/
  api.py            HTTP API for the UI
  pipeline.py       runs ingestion and benchmark configs, caches each step, records runs
  data.py           datasets: download (extractors), load into the document store, snapshots, subsets
  benchmarks.py     the public dataset catalog
  stages/           one module per pipeline step: load, transform, chunk, embed, retrieve, rerank, generate, evaluate
  targets/          one module per store type, plus manifest.py (versions, what was written)
  datasets/         readers for folder formats (plain files, EnterpriseRAG-Bench zips)
  scripting.py      Python-script nodes: templates, contracts, loading
  stats.py          the paired significance test behind Compare
web/src/pages/      one file per page;  web/src/components/  graph editor, code editor, jobs, ui
tests/test_core.py  the test suite
```

The UI's **How it works** page is the best overview of how the pieces fit.

## Common additions

**A stage implementation** (a new chunker, retriever, reranker, ...): add a function and a
Pydantic config to the stage's module in `stages/`:

```python
class SentenceConfig(BaseModel):
    model_config = {"extra": "forbid"}
    max_sentences: int = 5

@register("chunk", "sentences", SentenceConfig)
def sentence_chunker(cfg: SentenceConfig, ctx: StageContext) -> dict:
    """One-line description: shown in the UI."""
    ...
```

The UI form is generated from the config class, and caching follows the config and the
module's source, so nothing else needs changing. Look at a sibling implementation for the
files a stage reads and writes.

**A target type**: subclass `Target` in a new module under `targets/` (implement `clear`,
`delete_docs`, `add`, and `search_text` and/or `search_vector`), decorate it with
`@register_target("name", ConfigClass)`, import it in `targets/__init__.py`, and add it to the
parametrized target test in `tests/test_core.py`. If it needs a server, add the service to
`docker-compose.yml` and `services.py`. Put secrets in connection URLs as `${VAR}`.

**A public dataset**: add it to `BENCHMARKS` in `benchmarks.py` and give `extractor()` a script
for it. An extractor only has to define `download(folder)` and leave documents plus a
`questions.jsonl` in the folder. Check the dataset's license allows people to download it.

**A metric**: compute it per question in `stages/evaluate.py`; aggregation, per-category
breakdowns, the Compare test and the UI pick it up from there.

## Pull requests

- One topic per pull request, with a short description of why.
- Add or update a test for behaviour changes.
- Update the README or the How it works page if users would notice the change.
