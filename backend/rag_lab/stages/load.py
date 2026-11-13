"""load: select the documents of the configured subset from the corpus DB."""

from pydantic import BaseModel

from ..registry import register
from .base import StageContext


class CorpusConfig(BaseModel):
    model_config = {"extra": "forbid"}
    sources: list[str] | None = None     # e.g. ["slack", "confluence"]; None = all sources


@register("load", "corpus", CorpusConfig)
def load_corpus(cfg: CorpusConfig, ctx: StageContext) -> dict:
    where = "WHERE d.doc_id IN (SELECT doc_id FROM subset_docs WHERE subset = $subset)"
    params = {"subset": ctx.subset}
    if cfg.sources:
        where += " AND d.source IN (SELECT unnest($sources))"
        params["sources"] = cfg.sources
    out = ctx.out / "docs.parquet"
    ctx.corpus.execute(
        f"COPY (SELECT doc_id, source, title, text FROM docs d {where} ORDER BY doc_id) TO '{out}' (FORMAT parquet)",
        params,
    )
    counts = ctx.corpus.execute(
        f"SELECT source, count(*) FROM docs d {where} GROUP BY 1 ORDER BY 2 DESC", params).fetchall()
    ctx.log(f"{sum(c for _, c in counts):,} docs")
    return {"n_docs": sum(c for _, c in counts), "by_source": dict(counts)}
