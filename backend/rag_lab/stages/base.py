"""Things every stage implementation receives, plus small shared helpers."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .. import db, paths


@dataclass
class StageContext:
    out: Path                                  # write this stage's outputs here
    dirs: dict[str, Path]                      # output dirs of upstream stages, by stage name
    metas: dict[str, dict]                     # their meta.json contents
    fingerprint: str                           # hash of this stage's config + code, *without* upstream
    # ingestion stages
    corpus: duckdb.DuckDBPyConnection | None = None       # read-only connection to data/corpus.duckdb
    dataset: str = ""
    snapshot: int = 0
    subset: str = "all"
    # retrieval stages
    target: object | None = None               # the open Target being searched
    embed: dict | None = None                  # how the target's vectors were made (to embed questions the same way)
    questions: list[dict] = field(default_factory=list)   # rows from the questions table

    def store(self, stage: str) -> duckdb.DuckDBPyConnection:
        """A per-item store shared by every run with the same stage config + code
        (cache/store/<stage>/<fingerprint>.duckdb). Chunk and embed use it to cache
        per document / per chunk text, so a grown dataset only processes the delta."""
        d = paths.CACHE / "store" / stage
        d.mkdir(parents=True, exist_ok=True)
        return db.connect(str(d / f"{self.fingerprint}.duckdb"))

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


def iter_parquet(path, batch_size: int, columns: list[str] | None = None):
    """Read a parquet file batch by batch with flat memory. (pyarrow's default pre-buffering
    keeps what it read, so memory would grow with the file.)"""
    import pyarrow.parquet as pq
    return pq.ParquetFile(path, pre_buffer=False).iter_batches(batch_size=batch_size, columns=columns)
