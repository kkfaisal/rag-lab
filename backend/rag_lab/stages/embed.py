"""embed: turn every chunk into a vector.

Outputs embeddings.npy (float32, L2-normalised, same row order as chunks.parquet).
Texts are embedded in batches and the output is written through a memory map, so
memory use doesn't grow with the dataset.
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
from .base import StageContext, iter_parquet

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
    batches that are saved as they finish — an interrupted job keeps its progress.
    Memory stays flat: chunks are read SAVE_EVERY at a time, and embeddings.npy is
    filled through a memory map rather than built in RAM."""
    chunks_file = ctx.dirs["chunk"] / "chunks.parquet"
    n_chunks = pq.ParquetFile(chunks_file).metadata.num_rows
    store = ctx.store("embed")
    # No PRIMARY KEY: DuckDB keeps key indexes in memory, so they grow with the data. Only hashes not
    # yet in the store are inserted, each once, so the table stays unique without one.
    store.execute(f"CREATE TABLE IF NOT EXISTS vecs(text_hash VARCHAR, vec FLOAT[{dim}])")
    # How many distinct texts are new (hash column only, so this stays small).
    n_todo = store.execute(f"""SELECT count(DISTINCT text_hash) FROM (SELECT text_hash FROM read_parquet('{chunks_file}'))
                               ANTI JOIN vecs USING (text_hash)""").fetchall()[0][0]
    ctx.log(f"{n_chunks:,} chunks: {n_todo:,} to embed with {label}, {n_chunks - n_todo:,} already in store")
    done = 0
    for i, batch in enumerate(iter_parquet(chunks_file, SAVE_EVERY, ["text_hash", "text"])):
        store.register("batch", batch.select(["text_hash"]))
        have = {h for (h,) in store.execute("SELECT DISTINCT v.text_hash FROM vecs v SEMI JOIN batch USING (text_hash)").fetchall()}
        store.unregister("batch")
        new = {}                                   # text_hash -> text, each new text once
        for h, t in zip(batch.column("text_hash").to_pylist(), batch.column("text").to_pylist()):
            if h not in have:
                new.setdefault(h, t)
        if not new:
            continue
        vecs = encode_fn(list(new.values()))
        store.register("new_vecs", pa.table({"text_hash": list(new),
                                             "vec": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), dim)}))
        store.execute("INSERT INTO vecs SELECT * FROM new_vecs")
        store.unregister("new_vecs")
        done += len(new)
        if i % 20 == 19:
            store.execute("CHECKPOINT")             # flush inserted vectors to disk, releasing their memory
        ctx.log(f"saved {done:,}/{n_todo:,}")
    store.execute("CHECKPOINT")

    # One vector per chunk, in chunks.parquet's row order, streamed into a memory-mapped .npy.
    emb = np.lib.format.open_memmap(ctx.out / "embeddings.npy", mode="w+", dtype=np.float32, shape=(n_chunks, dim))
    reader = store.sql(f"""SELECT c.file_row_number AS row, v.vec FROM read_parquet('{chunks_file}', file_row_number = true) c
                           JOIN vecs v USING (text_hash)""").to_arrow_reader(SAVE_EVERY)
    filled = 0
    for rb in reader:
        rows = rb.column("row").to_numpy()
        emb[rows] = rb.column("vec").flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
        filled += len(rows)
    emb.flush()
    del emb
    store.close()
    if filled != n_chunks:
        raise RuntimeError(f"embedding store returned {filled:,} vectors for {n_chunks:,} chunks")
    return {"n": n_chunks, "dim": dim, "embedded": done, "reused": n_chunks - done, **meta}


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
