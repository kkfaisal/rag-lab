"""index: build search structures over the chunks.

`duckdb` writes index.duckdb holding a `chunks` table and, if fts=true, a BM25
full-text index (DuckDB's fts extension). Vector search uses the embed stage's
embeddings.npy directly with brute-force numpy (exact, and fast enough for
~100k chunks; swap in an ANN index here when you go to the full corpus).
"""

import duckdb
from pydantic import BaseModel

from ..registry import register
from .base import StageContext


class DuckDBIndexConfig(BaseModel):
    model_config = {"extra": "forbid"}
    fts: bool = True
    vectors: bool = True
    stemmer: str = "english"
    stopwords: str = "english"


@register("index", "duckdb", DuckDBIndexConfig)
def duckdb_index(cfg: DuckDBIndexConfig, ctx: StageContext) -> dict:
    if cfg.vectors and ctx.metas["embed"].get("n", 0) == 0:
        raise SystemExit("index.vectors=true needs embeddings; set embed.impl to something other than none")
    con = duckdb.connect(str(ctx.out / "index.duckdb"))
    con.execute(f"CREATE TABLE chunks AS SELECT chunk_id, doc_id, text FROM '{ctx.dirs['chunk'] / 'chunks.parquet'}'")
    if cfg.fts:
        ctx.log("building BM25 full-text index")
        con.execute("INSTALL fts; LOAD fts;")
        con.execute(f"PRAGMA create_fts_index('chunks', 'chunk_id', 'text', stemmer='{cfg.stemmer}', "
                    f"stopwords='{cfg.stopwords}', overwrite=1)")
    n = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    con.close()
    return {"n_chunks": n, "fts": cfg.fts, "vectors": cfg.vectors}
