"""Public benchmarks you can fetch into data/datasets/<name>/.

The data is never part of the repository: each user downloads it to their own
machine. EnterpriseRAG-Bench comes from its GitHub release; the BEIR sets come
from their Hugging Face mirror through the extractor template, which is saved in
the folder as extract.py so it can be edited and re-run from the UI.

    uv run python scripts/get_datasets.py --list
    uv run python scripts/get_datasets.py scifact nfcorpus --load
"""

import re

from . import data, paths, scripting

# name -> (what it is, approximate download size)
BENCHMARKS = {
    "enterprise_rag_bench": ("EnterpriseRAG-Bench: ~512k synthetic company docs (Slack, Confluence, Jira, ...), "
                             "500 questions with gold answers; builds a 'dev' subset", "1.3 GB"),
    "scifact": ("BEIR SciFact: 5k scientific abstracts, 300 claims to verify", "4 MB"),
    "nfcorpus": ("BEIR NFCorpus: 3.6k nutrition/medical articles, 323 questions", "3 MB"),
    "arguana": ("BEIR ArguAna: 8.7k arguments, 1.4k counter-argument queries", "6 MB"),
    "scidocs": ("BEIR SCIDOCS: 25k paper abstracts, 1k citation queries", "19 MB"),
    "quora": ("BEIR Quora: 523k questions, 10k duplicate-question queries", "25 MB"),
    "fiqa": ("BEIR FiQA: 57k financial forum posts, 648 questions", "28 MB"),
    "trec-covid": ("BEIR TREC-COVID: 171k COVID papers, 50 topics with many judged docs", "111 MB"),
    "webis-touche2020": ("BEIR Touché-2020: 382k arguments, 49 controversial topics", "357 MB"),
    "nq": ("BEIR Natural Questions: 2.7M Wikipedia passages, 3.5k questions", "764 MB"),
    "hotpotqa": ("BEIR HotpotQA: 5.2M Wikipedia passages, 7.4k multi-hop questions", "984 MB"),
    "dbpedia-entity": ("BEIR DBPedia: 4.6M entity abstracts, 400 queries", "984 MB"),
    "msmarco": ("BEIR MS MARCO: 8.8M web passages, 43 test queries (dev has 7k)", "1.6 GB"),
    "climate-fever": ("BEIR Climate-FEVER: 5.4M Wikipedia passages, 1.5k climate claims", "1.9 GB"),
    "fever": ("BEIR FEVER: 5.4M Wikipedia passages, 6.7k claims", "2 GB"),
}


def beir_extractor(name: str, split: str = "test") -> str:
    """The extractor template, set to fetch BEIR dataset `name`."""
    code = re.sub(r'^DATASET = ".*"[^\n]*$', f'DATASET = "{name}"', scripting.EXTRACT_TEMPLATE, count=1, flags=re.M)
    return re.sub(r'^SPLIT = ".*"', f'SPLIT = "{split}"', code, count=1, flags=re.M)


def extractor(name: str, split: str = "test") -> str:
    """The Python extractor script for a public dataset (EnterpriseRAG-Bench, or a BEIR set)."""
    return scripting.ERB_EXTRACT_TEMPLATE if name == data.BENCH else beir_extractor(name, split)


def _load(fn, name: str) -> None:
    import duckdb
    try:
        fn()
    except duckdb.IOException as e:
        if "lock" not in str(e):
            raise
        raise SystemExit(f"The files are in data/datasets/{name}/, but the document store is in use (another rag-lab "
                         "process has it open). Load it from the Datasets page, which queues behind running "
                         "jobs, or rerun with --load when that finishes.") from e


def fetch(name: str, load: bool = False, split: str = "test") -> None:
    """Run the dataset's extractor script into data/datasets/<name>/ (saved there as extract.py);
    with load=True also load it into the document store so pipelines can use it."""
    data.migrate_layout()
    if name not in BENCHMARKS:
        print(f"{name!r} isn't in the list; trying it as a BEIR dataset name.")
    code = extractor(name, split)
    data.run_extractor(data.dataset_dir(name), code)                     # download first, without the store
    if load:
        _load(lambda: data.ingest_source(name, "folder", {}, sync=True), name)   # EnterpriseRAG-Bench also builds "dev"
    print(f"{name}: data/datasets/{name}/" + ("" if load else "  (load it on the Datasets page, or rerun with --load)"))


def catalog() -> list[dict]:
    """The public datasets, and which of them are already on this machine."""
    return [{"name": n, "description": desc, "size": size, "downloaded": (paths.DATASETS / n).is_dir()}
            for n, (desc, size) in BENCHMARKS.items()]


def listing() -> str:
    width = max(map(len, BENCHMARKS))
    have = {d.name for d in paths.DATASETS.iterdir()} if paths.DATASETS.exists() else set()
    return "\n".join(f"  {n:<{width}}  {size:>7}  {'[downloaded] ' if n in have else ''}{desc}"
                     for n, (desc, size) in BENCHMARKS.items())
