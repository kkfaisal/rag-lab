"""embed: turn every chunk into a vector.

Outputs embeddings.npy (float32, L2-normalised, same row order as chunks.parquet).
Vectors are cached per chunk-text hash in a store shared by every run using the
same embedding settings, so only new or changed chunk texts are ever embedded.
The query-side settings (model, query_prefix) go in meta.json so retrieval can
embed questions the same way.
"""

from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..registry import NoConfig, register
from .base import StageContext

SAVE_EVERY = 10_000
# Bump when a code change alters the vectors themselves (e.g. normalisation), so the
# per-chunk store in cache/store/embed/ starts fresh. Other edits keep the store.
STORE_VERSION = 1


class SentenceTransformersConfig(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "   # bge's recommended query instruction
    batch_size: int = 64
    max_seq_length: int = 512
    device: str | None = None        # None = auto (cuda / mps / cpu)


@lru_cache(maxsize=2)
def load_model(name: str, max_seq_length: int, device: str | None):
    from sentence_transformers import SentenceTransformer
    try:   # use the local copy if we have one: no Hub round-trips, works offline
        m = SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:
        m = SentenceTransformer(name, device=device)
    m.max_seq_length = max_seq_length
    return m


def encode(model_name: str, texts: list[str], max_seq_length: int = 512, device=None, batch_size=64, progress=False) -> np.ndarray:
    m = load_model(model_name, max_seq_length, device)
    return m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                    show_progress_bar=progress, convert_to_numpy=True).astype(np.float32)


@register("embed", "sentence_transformers", SentenceTransformersConfig)
def sentence_transformers(cfg: SentenceTransformersConfig, ctx: StageContext) -> dict:
    """Embed only chunk texts this model hasn't seen (keyed by text hash), in
    batches that are saved as they finish — an interrupted job keeps its progress."""
    chunks = pq.read_table(ctx.dirs["chunk"] / "chunks.parquet", columns=["text_hash", "text"])
    store = ctx.store("embed")
    model = load_model(cfg.model, cfg.max_seq_length, cfg.device)
    dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
    store.execute(f"CREATE TABLE IF NOT EXISTS vecs(text_hash VARCHAR PRIMARY KEY, vec FLOAT[{dim}])")
    store.register("chunks", chunks.append_column("ord", pa.array(range(chunks.num_rows))))
    todo = store.execute("SELECT text_hash, any_value(text) FROM chunks ANTI JOIN vecs USING (text_hash) GROUP BY 1").fetchall()
    ctx.log(f"{chunks.num_rows:,} chunks: {len(todo):,} to embed with {cfg.model}, {chunks.num_rows - len(todo):,} already in store")
    for start in range(0, len(todo), SAVE_EVERY):
        part = todo[start:start + SAVE_EVERY]
        vecs = encode(cfg.model, [t for _, t in part], cfg.max_seq_length, cfg.device, cfg.batch_size, progress=True)
        store.register("new_vecs", pa.table({"text_hash": [h for h, _ in part],
                                             "vec": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), dim)}))
        store.execute("INSERT OR IGNORE INTO vecs SELECT * FROM new_vecs")
        ctx.log(f"saved {min(start + SAVE_EVERY, len(todo)):,}/{len(todo):,}")
    vec_col = store.sql("SELECT v.vec FROM chunks c JOIN vecs v USING (text_hash) ORDER BY c.ord").to_arrow_table().column("vec")
    store.close()
    emb = vec_col.combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)
    np.save(ctx.out / "embeddings.npy", emb)
    return {"n": int(emb.shape[0]), "dim": dim, "embedded": len(todo), "reused": chunks.num_rows - len(todo),
            "model": cfg.model, "query_prefix": cfg.query_prefix, "max_seq_length": cfg.max_seq_length}


@register("embed", "none", NoConfig)
def no_embeddings(cfg: NoConfig, ctx: StageContext) -> dict:
    """Skip embeddings (for BM25-only pipelines)."""
    return {"n": 0}
