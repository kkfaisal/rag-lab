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
from ..scripting import EMBED_TEMPLATE, ScriptError, load_functions
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


def _embed_with_store(ctx: StageContext, dim: int, encode_fn, label: str, meta: dict) -> dict:
    """Embed only chunk texts this embedder hasn't seen (keyed by text hash), in
    batches that are saved as they finish — an interrupted job keeps its progress."""
    chunks = pq.read_table(ctx.dirs["chunk"] / "chunks.parquet", columns=["text_hash", "text"])
    store = ctx.store("embed")
    store.execute(f"CREATE TABLE IF NOT EXISTS vecs(text_hash VARCHAR PRIMARY KEY, vec FLOAT[{dim}])")
    store.register("chunks", chunks.append_column("ord", pa.array(range(chunks.num_rows))))
    todo = store.execute("SELECT text_hash, any_value(text) FROM chunks ANTI JOIN vecs USING (text_hash) GROUP BY 1").fetchall()
    ctx.log(f"{chunks.num_rows:,} chunks: {len(todo):,} to embed with {label}, {chunks.num_rows - len(todo):,} already in store")
    for start in range(0, len(todo), SAVE_EVERY):
        part = todo[start:start + SAVE_EVERY]
        vecs = encode_fn([t for _, t in part])
        store.register("new_vecs", pa.table({"text_hash": [h for h, _ in part],
                                             "vec": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), dim)}))
        store.execute("INSERT OR IGNORE INTO vecs SELECT * FROM new_vecs")
        ctx.log(f"saved {min(start + SAVE_EVERY, len(todo)):,}/{len(todo):,}")
    vec_col = store.sql("SELECT v.vec FROM chunks c JOIN vecs v USING (text_hash) ORDER BY c.ord").to_arrow_table().column("vec")
    store.close()
    emb = vec_col.combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)
    np.save(ctx.out / "embeddings.npy", emb)
    return {"n": int(emb.shape[0]), "dim": dim, "embedded": len(todo), "reused": chunks.num_rows - len(todo), **meta}


@register("embed", "sentence_transformers", SentenceTransformersConfig)
def sentence_transformers(cfg: SentenceTransformersConfig, ctx: StageContext) -> dict:
    """A local sentence-transformers model. Vectors are cached per chunk text, so only new text is embedded."""
    model = load_model(cfg.model, cfg.max_seq_length, cfg.device)
    dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
    return _embed_with_store(
        ctx, dim, lambda texts: encode(cfg.model, texts, cfg.max_seq_length, cfg.device, cfg.batch_size, progress=True), cfg.model,
        {"model": cfg.model, "query_prefix": cfg.query_prefix, "max_seq_length": cfg.max_seq_length})


class PythonEmbedConfig(BaseModel):
    model_config = {"extra": "forbid"}
    code: str = EMBED_TEMPLATE       # must define embed(texts); may define embed_query(text)


def _normalised(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ScriptError("embed(texts) must return one vector per text (a 2-D array or list of lists)")
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def script_query_vector(code: str, text: str) -> np.ndarray:
    """Embed one question with an embed script (used by vector search)."""
    fns = _script_fns(code)
    vec = fns["embed_query"](text) if "embed_query" in fns else fns["embed"]([text])[0]
    return _normalised([np.asarray(vec, dtype=np.float32)])[0]


@lru_cache(maxsize=4)
def _script_fns(code: str) -> dict:
    return load_functions(code, "embed", ("embed_query",))


@register("embed", "python", PythonEmbedConfig)
def python_embedder(cfg: PythonEmbedConfig, ctx: StageContext) -> dict:
    """Your own embed(texts) function, edited in the UI (any model or API)."""
    fns = _script_fns(cfg.code)

    def encode_fn(texts):
        vecs = _normalised(fns["embed"](texts))
        if len(vecs) != len(texts):
            raise ScriptError(f"embed() returned {len(vecs)} vectors for {len(texts)} texts")
        return vecs
    dim = int(encode_fn(["dimension probe"]).shape[1])
    return _embed_with_store(ctx, dim, encode_fn, "your script", {"code": cfg.code})


@register("embed", "none", NoConfig)
def no_embeddings(cfg: NoConfig, ctx: StageContext) -> dict:
    """Skip embeddings (for BM25-only pipelines)."""
    return {"n": 0}
