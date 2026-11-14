"""Every DuckDB connection rag-lab opens goes through here.

DuckDB uses up to 80% of RAM by default before spilling to disk. Each connection
here is capped (RAG_LAB_DUCKDB_MEMORY, default 1GB) and spills to
cache/duckdb-tmp/ instead, so large datasets use disk, not all of your memory.
"""

import os

import duckdb

from . import paths


def connect(path: str = ":memory:", read_only: bool = False) -> duckdb.DuckDBPyConnection:
    tmp = paths.CACHE / "duckdb-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    config = {"memory_limit": os.environ.get("RAG_LAB_DUCKDB_MEMORY", "1GB"), "temp_directory": str(tmp),
              "preserve_insertion_order": "false"}   # lets DuckDB stream; queries that need an order say ORDER BY
    return duckdb.connect(str(path), read_only=read_only, config=config)
