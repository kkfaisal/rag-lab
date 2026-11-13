"""retrieve: find candidate chunks for every question.

Outputs retrieval.jsonl: {"question_id", "chunks": [{"chunk_id", "doc_id", "score"}, ...]}
ranked best-first.
"""

import duckdb
import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..registry import register
from .base import StageContext, write_jsonl
from .embed import encode


class TopKConfig(BaseModel):
    model_config = {"extra": "forbid"}
    top_k: int = 20


class HybridConfig(TopKConfig):
    candidates: int = 100     # how many each retriever contributes before fusion
    rrf_k: int = 60           # Reciprocal Rank Fusion constant: score = Σ 1/(rrf_k + rank)
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


def _chunk_doc_ids(ctx: StageContext) -> tuple[list[str], list[str]]:
    t = pq.read_table(ctx.dirs["chunk"] / "chunks.parquet", columns=["chunk_id", "doc_id"])
    return t.column("chunk_id").to_pylist(), t.column("doc_id").to_pylist()


def bm25_search(ctx: StageContext, k: int) -> dict[str, list[dict]]:
    if not ctx.metas["index"].get("fts"):
        raise SystemExit("bm25 retrieval needs index.fts=true")
    con = duckdb.connect(str(ctx.dirs["index"] / "index.duckdb"), read_only=True)
    con.execute("LOAD fts")
    out = {}
    for q in ctx.questions:
        rows = con.execute("""
            SELECT chunk_id, doc_id, score FROM (
              SELECT chunk_id, doc_id, fts_main_chunks.match_bm25(chunk_id, ?) AS score FROM chunks
            ) WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?""", [q["question"], k]).fetchall()
        out[q["question_id"]] = [{"chunk_id": c, "doc_id": d, "score": float(s)} for c, d, s in rows]
    con.close()
    return out


def vector_search(ctx: StageContext, k: int) -> dict[str, list[dict]]:
    meta = ctx.metas["embed"]
    if not meta.get("n"):
        raise SystemExit("vector retrieval needs embeddings (embed.impl != none)")
    emb = np.load(ctx.dirs["embed"] / "embeddings.npy")
    chunk_ids, doc_ids = _chunk_doc_ids(ctx)
    qv = encode(meta["model"], [meta["query_prefix"] + q["question"] for q in ctx.questions], meta["max_seq_length"])
    scores = qv @ emb.T                                  # cosine similarity (vectors are normalised)
    out = {}
    for qi, q in enumerate(ctx.questions):
        top = np.argpartition(-scores[qi], min(k, len(chunk_ids) - 1))[:k]
        top = top[np.argsort(-scores[qi][top])]
        out[q["question_id"]] = [{"chunk_id": chunk_ids[i], "doc_id": doc_ids[i], "score": float(scores[qi][i])} for i in top]
    return out


@register("retrieve", "bm25", TopKConfig)
def bm25(cfg: TopKConfig, ctx: StageContext) -> dict:
    res = bm25_search(ctx, cfg.top_k)
    write_jsonl(ctx.out / "retrieval.jsonl", [{"question_id": q, "chunks": c} for q, c in res.items()])
    return {"n_questions": len(res)}


@register("retrieve", "vector", TopKConfig)
def vector(cfg: TopKConfig, ctx: StageContext) -> dict:
    res = vector_search(ctx, cfg.top_k)
    write_jsonl(ctx.out / "retrieval.jsonl", [{"question_id": q, "chunks": c} for q, c in res.items()])
    return {"n_questions": len(res)}


@register("retrieve", "hybrid", HybridConfig)
def hybrid(cfg: HybridConfig, ctx: StageContext) -> dict:
    """BM25 + vector, merged with Reciprocal Rank Fusion. RRF only looks at ranks,
    so the two retrievers' incomparable score scales don't matter."""
    lexical, dense = bm25_search(ctx, cfg.candidates), vector_search(ctx, cfg.candidates)
    rows = []
    for q in ctx.questions:
        qid, fused, doc_of = q["question_id"], {}, {}
        for weight, hits in [(cfg.bm25_weight, lexical[qid]), (cfg.vector_weight, dense[qid])]:
            for rank, h in enumerate(hits):
                fused[h["chunk_id"]] = fused.get(h["chunk_id"], 0.0) + weight / (cfg.rrf_k + rank + 1)
                doc_of[h["chunk_id"]] = h["doc_id"]
        best = sorted(fused.items(), key=lambda kv: -kv[1])[: cfg.top_k]
        rows.append({"question_id": qid, "chunks": [{"chunk_id": c, "doc_id": doc_of[c], "score": s} for c, s in best]})
    write_jsonl(ctx.out / "retrieval.jsonl", rows)
    return {"n_questions": len(rows)}
