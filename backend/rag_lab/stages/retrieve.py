"""retrieve: find candidate chunks for every question.

Outputs retrieval.jsonl: {"question_id", "chunks": [{"chunk_id", "doc_id", "score", "text"}, ...]}
ranked best-first. The chunk text travels with the hit, so later stages (and the
UI) never need to go back to the target.
"""

from pydantic import BaseModel

from ..registry import register
from ..scripting import RETRIEVE_TEMPLATE, ScriptError, load_functions
from .base import StageContext, write_jsonl
from .embed import encode, script_query_vector


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
    """The two primitive searches over a target. Built-in retrievers and script
    nodes are both written against this, so they work with any target type."""

    def __init__(self, ctx: StageContext):
        self.target, self.embed = ctx.target, ctx.embed

    def bm25(self, query: str, k: int) -> list[dict]:
        if not self.target.text_search:
            raise SystemExit("This target has no keyword (BM25) search. Use vector retrieval, or a target that supports it.")
        return self.target.search_text(query, k)

    def vector(self, query: str, k: int) -> list[dict]:
        if not self.target.vector_search:
            raise SystemExit("This target has no vector search. Use BM25 retrieval, or a target that stores vectors.")
        if not self.embed:
            raise SystemExit("This target was written without an embed node, so questions can't be embedded. "
                             "Add an embed node to the ingestion pipeline, or use BM25 retrieval.")
        if self.embed.get("code"):                            # embedded by a script node
            qv = script_query_vector(self.embed["code"], query)
        else:                                                 # same model + prefix as the chunks
            qv = encode(self.embed["model"], [self.embed["query_prefix"] + query], self.embed["max_seq_length"])[0]
        return self.target.search_vector(qv, k, query=query)

    def close(self) -> None:
        pass


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
                     "chunks": [{"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "score": float(h.get("score", 0.0)),
                                 "text": h.get("text", "")} for h in hits]})
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
