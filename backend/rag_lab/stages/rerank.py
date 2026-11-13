"""rerank: re-score retrieved chunks with a slower, more accurate model.

Outputs reranked.jsonl, same shape as retrieval.jsonl.
"""

import shutil
from functools import lru_cache

import duckdb
from pydantic import BaseModel

from ..registry import NoConfig, register
from .base import StageContext, read_jsonl, write_jsonl


@register("rerank", "none", NoConfig)
def no_rerank(cfg: NoConfig, ctx: StageContext) -> dict:
    shutil.copy(ctx.dirs["retrieve"] / "retrieval.jsonl", ctx.out / "reranked.jsonl")
    return {}


class CrossEncoderConfig(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 10            # keep this many after reranking
    batch_size: int = 32


@lru_cache(maxsize=1)
def _load(name: str):
    from sentence_transformers import CrossEncoder
    try:   # local copy first, as in embed.load_model
        return CrossEncoder(name, local_files_only=True)
    except Exception:
        return CrossEncoder(name)


@register("rerank", "cross_encoder", CrossEncoderConfig)
def cross_encoder(cfg: CrossEncoderConfig, ctx: StageContext) -> dict:
    """A cross-encoder reads (question, chunk) together, so it judges relevance far
    better than comparing two independently computed vectors — but it is too slow
    to run over the whole corpus, hence retrieve-then-rerank."""
    retrieved = read_jsonl(ctx.dirs["retrieve"] / "retrieval.jsonl")
    con = duckdb.connect(str(ctx.dirs["index"] / "index.duckdb"), read_only=True)
    questions = {q["question_id"]: q["question"] for q in ctx.questions}
    model = _load(cfg.model)
    out = []
    for row in retrieved:
        ids = [c["chunk_id"] for c in row["chunks"]]
        texts = dict(con.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [ids]).fetchall())
        scores = model.predict([(questions[row["question_id"]], texts[i]) for i in ids], batch_size=cfg.batch_size) if ids else []
        ranked = sorted(zip(row["chunks"], scores), key=lambda x: -x[1])[: cfg.top_n]
        out.append({"question_id": row["question_id"],
                    "chunks": [{**c, "retrieval_score": c["score"], "score": float(s)} for c, s in ranked]})
    con.close()
    write_jsonl(ctx.out / "reranked.jsonl", out)
    ctx.log(f"reranked {len(out)} questions")
    return {"n_questions": len(out)}
