"""embed: turn every chunk into a vector.

Outputs embeddings.npy (float32, L2-normalised, same row order as chunks.parquet).
The query-side settings (model, query_prefix) go in meta.json so retrieval can
embed questions the same way.
"""

from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..registry import NoConfig, register
from .base import StageContext


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
    m = SentenceTransformer(name, device=device)
    m.max_seq_length = max_seq_length
    return m


def encode(model_name: str, texts: list[str], max_seq_length: int = 512, device=None, batch_size=64, progress=False) -> np.ndarray:
    m = load_model(model_name, max_seq_length, device)
    return m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                    show_progress_bar=progress, convert_to_numpy=True).astype(np.float32)


@register("embed", "sentence_transformers", SentenceTransformersConfig)
def sentence_transformers(cfg: SentenceTransformersConfig, ctx: StageContext) -> dict:
    texts = pq.read_table(ctx.dirs["chunk"] / "chunks.parquet", columns=["text"]).column("text").to_pylist()
    ctx.log(f"embedding {len(texts):,} chunks with {cfg.model}")
    vecs = encode(cfg.model, texts, cfg.max_seq_length, cfg.device, cfg.batch_size, progress=True)
    np.save(ctx.out / "embeddings.npy", vecs)
    return {"n": len(texts), "dim": int(vecs.shape[1]), "model": cfg.model,
            "query_prefix": cfg.query_prefix, "max_seq_length": cfg.max_seq_length}


@register("embed", "none", NoConfig)
def no_embeddings(cfg: NoConfig, ctx: StageContext) -> dict:
    """Skip embeddings (for BM25-only pipelines)."""
    return {"n": 0}
