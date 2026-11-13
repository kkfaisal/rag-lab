"""load: select the documents of the configured dataset / snapshot / subset."""

from pydantic import BaseModel

from ..data import docs_sql
from ..registry import register
from .base import StageContext


class CorpusConfig(BaseModel):
    model_config = {"extra": "forbid"}
    sources: list[str] | None = None     # e.g. ["slack", "confluence"]; None = all sources


@register("load", "corpus", CorpusConfig)
def load_corpus(cfg: CorpusConfig, ctx: StageContext) -> dict:
    params = {"ds": ctx.dataset, "snap": ctx.snapshot, "subset": ctx.subset, "sources": cfg.sources or []}
    sql = f"""SELECT doc_id, source, title, text, content_hash FROM ({docs_sql()})
              WHERE len($sources) = 0 OR list_contains($sources, source) ORDER BY doc_id"""
    out = ctx.out / "docs.parquet"
    ctx.corpus.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet)", params)
    counts = dict(ctx.corpus.execute(f"SELECT source, count(*) FROM '{out}' GROUP BY 1 ORDER BY 2 DESC").fetchall())
    ctx.log(f"{sum(counts.values()):,} docs (dataset {ctx.dataset}, snapshot {ctx.snapshot}, subset {ctx.subset})")
    return {"n_docs": sum(counts.values()), "by_source": counts}
