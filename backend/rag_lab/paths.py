"""Where everything lives on disk. All paths are relative to the project root
(the current working directory, or $RAG_LAB_HOME)."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("RAG_LAB_HOME", Path.cwd())).resolve()

DATA = ROOT / "data"
RAW = DATA / "raw"                      # downloaded zips + questions.jsonl
CORPUS_DB = DATA / "corpus.duckdb"      # docs, questions, subsets
QUESTIONS_FILE = RAW / "questions.jsonl"

CACHE = ROOT / "cache"                  # cache/<stage>/<key>/ — content-addressed stage outputs
RUNS = ROOT / "runs"                    # runs/<run_id>/ — one folder per experiment
RUNS_INDEX = RUNS / "index.duckdb"      # queryable index over all run folders

BENCH_DIR = ROOT / ".bench" / "EnterpriseRAG-Bench"   # benchmark checkout used for judging
WEB_DIST = ROOT / "web" / "dist"
