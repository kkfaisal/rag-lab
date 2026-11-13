"""rerank: re-score retrieved chunks with a slower, more accurate model.

Outputs reranked.jsonl, same shape as retrieval.jsonl.
"""

import shutil
from functools import lru_cache

import duckdb
from pydantic import BaseModel

from ..registry import NoConfig, register
from ..scripting import RERANK_TEMPLATE, ScriptError, load_functions
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


def _rerank(ctx: StageContext, rerank_one) -> dict:
    """Apply rerank_one(question, chunks_with_text) -> ranked chunks to every question."""
    retrieved = read_jsonl(ctx.dirs["retrieve"] / "retrieval.jsonl")
    con = duckdb.connect(str(ctx.dirs["index"] / "index.duckdb"), read_only=True)
    questions = {q["question_id"]: q["question"] for q in ctx.questions}
    out = []
    for row in retrieved:
        ids = [c["chunk_id"] for c in row["chunks"]]
        texts = dict(con.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [ids]).fetchall())
        chunks = [{**c, "text": texts[c["chunk_id"]]} for c in row["chunks"]]
        before = {c["chunk_id"]: c["score"] for c in row["chunks"]}
        ranked = rerank_one(questions[row["question_id"]], chunks) if chunks else []
        out.append({"question_id": row["question_id"], "chunks": [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "retrieval_score": before.get(c["chunk_id"]),
             "score": float(c.get("score", 0.0))} for c in ranked]})
    con.close()
    write_jsonl(ctx.out / "reranked.jsonl", out)
    ctx.log(f"reranked {len(out)} questions")
    return {"n_questions": len(out)}


@register("rerank", "cross_encoder", CrossEncoderConfig)
def cross_encoder(cfg: CrossEncoderConfig, ctx: StageContext) -> dict:
    """A cross-encoder reads (question, chunk) together, so it judges relevance far
    better than comparing two independently computed vectors — but it is too slow
    to run over the whole corpus, hence retrieve-then-rerank."""
    model = _load(cfg.model)

    def rerank_one(question, chunks):
        scores = model.predict([(question, c["text"]) for c in chunks], batch_size=cfg.batch_size)
        ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])[: cfg.top_n]
        return [{**c, "score": float(s)} for c, s in ranked]
    return _rerank(ctx, rerank_one)


class PythonRerankConfig(BaseModel):
    model_config = {"extra": "forbid"}
    code: str = RERANK_TEMPLATE      # must define rerank(question, chunks, top_n)
    top_n: int = 10


@register("rerank", "python", PythonRerankConfig)
def python_reranker(cfg: PythonRerankConfig, ctx: StageContext) -> dict:
    """Your own rerank(question, chunks, top_n) function, edited in the UI."""
    fn = load_functions(cfg.code, "rerank")["rerank"]

    def rerank_one(question, chunks):
        ranked = fn(question, chunks, cfg.top_n)
        if not isinstance(ranked, list) or not all(isinstance(c, dict) and "chunk_id" in c and "doc_id" in c for c in ranked):
            raise ScriptError('rerank() must return a list of chunk dicts (with "chunk_id" and "doc_id")')
        return ranked[: cfg.top_n]
    return _rerank(ctx, rerank_one)
