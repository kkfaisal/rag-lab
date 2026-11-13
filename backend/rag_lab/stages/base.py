"""Things every stage implementation receives, plus small shared helpers."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb


@dataclass
class StageContext:
    out: Path                                  # write this stage's outputs here
    dirs: dict[str, Path]                      # output dirs of upstream stages, by stage name
    metas: dict[str, dict]                     # their meta.json contents
    corpus: duckdb.DuckDBPyConnection          # read-only connection to data/corpus.duckdb
    subset: str
    questions: list[dict] = field(default_factory=list)   # rows from the questions table

    def log(self, msg: str) -> None:
        print(f"    {msg}", flush=True)


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_docs(chunks: list[dict]) -> list[str]:
    """Doc ids in rank order of their best chunk."""
    seen, out = set(), []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            out.append(c["doc_id"])
    return out
