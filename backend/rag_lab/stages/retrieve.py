"""retrieve: find candidate chunks for every question.

Outputs retrieval.jsonl: {"question_id", "chunks": [{"chunk_id", "doc_id", "score"}, ...]}
ranked best-first.
"""

import duckdb
import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..registry import register
from ..scripting import RETRIEVE_TEMPLATE, ScriptError, load_functions
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


class PythonRetrieveConfig(TopKConfig):
    code: str = RETRIEVE_TEMPLATE     # must define retrieve(question, search, top_k)


class Search:
    """The two primitive searches over an index. Built-in retrievers and
    script nodes are both written against this."""

    def __init__(self, ctx: StageContext):
        self.ctx = ctx
        self._con = self._emb = self._ids = None

    def bm25(self, query: str, k: int) -> list[dict]:
        if not self.ctx.metas["index"].get("fts"):
            raise SystemExit("BM25 search needs an index built with fts=true")
        if self._con is None:
            self._con = duckdb.connect(str(self.ctx.dirs["index"] / "index.duckdb"), read_only=True)
            self._con.execute("LOAD fts")
        rows = self._con.execute("""
            SELECT chunk_id, doc_id, score, text FROM (
              SELECT chunk_id, doc_id, text, fts_main_chunks.match_bm25(chunk_id, ?) AS score FROM chunks
            ) WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?""", [query, k]).fetchall()
        return [{"chunk_id": c, "doc_id": d, "score": float(s), "text": t} for c, d, s, t in rows]

    def vector(self, query: str, k: int) -> list[dict]:
        meta = self.ctx.metas["embed"]
        if not meta.get("n"):
            raise SystemExit("Vector search needs embeddings (the pipeline's embed node must not be 'none')")
        if self._emb is None:
            self._emb = np.load(self.ctx.dirs["embed"] / "embeddings.npy")
            t = pq.read_table(self.ctx.dirs["chunk"] / "chunks.parquet", columns=["chunk_id", "doc_id", "text"])
            self._ids = (t.column("chunk_id").to_pylist(), t.column("doc_id").to_pylist(), t.column("text"))
        qv = encode(meta["model"], [meta["query_prefix"] + query], meta["max_seq_length"])[0]
        scores = self._emb @ qv                              # cosine similarity (vectors are normalised)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        chunk_ids, doc_ids, texts = self._ids
        return [{"chunk_id": chunk_ids[i], "doc_id": doc_ids[i], "score": float(scores[i]), "text": texts[int(i)].as_py()}
                for i in top]

    def close(self) -> None:
        if self._con is not None:
            self._con.close()


def rrf(result_lists: list[tuple[float, list[dict]]], rrf_k: int, top_k: int) -> list[dict]:
    """Reciprocal Rank Fusion: merge ranked lists using ranks only, so the
    retrievers' incomparable score scales don't matter."""
    fused, hit_of = {}, {}
    for weight, hits in result_lists:
        for rank, h in enumerate(hits):
            fused[h["chunk_id"]] = fused.get(h["chunk_id"], 0.0) + weight / (rrf_k + rank + 1)
            hit_of[h["chunk_id"]] = h
    best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [{**hit_of[c], "score": s} for c, s in best]


def _run(ctx: StageContext, per_question) -> dict:
    search = Search(ctx)
    rows = []
    for q in ctx.questions:
        hits = per_question(q["question"], search)
        rows.append({"question_id": q["question_id"],
                     "chunks": [{"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "score": float(h.get("score", 0.0))} for h in hits]})
    search.close()
    write_jsonl(ctx.out / "retrieval.jsonl", rows)
    return {"n_questions": len(rows)}


@register("retrieve", "bm25", TopKConfig)
def bm25(cfg: TopKConfig, ctx: StageContext) -> dict:
    return _run(ctx, lambda q, s: s.bm25(q, cfg.top_k))


@register("retrieve", "vector", TopKConfig)
def vector(cfg: TopKConfig, ctx: StageContext) -> dict:
    return _run(ctx, lambda q, s: s.vector(q, cfg.top_k))


@register("retrieve", "hybrid", HybridConfig)
def hybrid(cfg: HybridConfig, ctx: StageContext) -> dict:
    """BM25 + vector, merged with Reciprocal Rank Fusion."""
    return _run(ctx, lambda q, s: rrf([(cfg.bm25_weight, s.bm25(q, cfg.candidates)),
                                       (cfg.vector_weight, s.vector(q, cfg.candidates))], cfg.rrf_k, cfg.top_k))


@register("retrieve", "python", PythonRetrieveConfig)
def python_retriever(cfg: PythonRetrieveConfig, ctx: StageContext) -> dict:
    """Your own retrieve(question, search, top_k) function, edited in the UI."""
    fn = load_functions(cfg.code, "retrieve")["retrieve"]

    def per_question(q, s):
        hits = fn(q, s, cfg.top_k)
        if not isinstance(hits, list) or not all(isinstance(h, dict) and "chunk_id" in h and "doc_id" in h for h in hits):
            raise ScriptError('retrieve() must return a list of dicts with "chunk_id" and "doc_id"')
        return hits[: cfg.top_k]
    return _run(ctx, per_question)
