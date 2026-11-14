"""load: select the documents of the configured dataset / snapshot / subset."""

from pydantic import BaseModel

from ..data import docs_sql
from ..registry import register
from .base import StageContext

ROW_GROUP = 2_048          # documents per parquet row group


class CorpusConfig(BaseModel):
    model_config = {"extra": "forbid"}
    sources: list[str] | None = None     # e.g. ["slack", "confluence"]; None = all sources


@register("load", "corpus", CorpusConfig)
def load_corpus(cfg: CorpusConfig, ctx: StageContext) -> dict:
    params = {"ds": ctx.dataset, "snap": ctx.snapshot, "subset": ctx.subset, "sources": cfg.sources or []}
    sql = f"""SELECT doc_id, source, title, text, CAST(metadata AS VARCHAR) AS metadata, content_hash
              FROM ({docs_sql('v.doc_id, v.source, v.metadata, c.title, c.text, v.content_hash')})
              WHERE len($sources) = 0 OR list_contains($sources, source)"""   # no ORDER BY: sorting all text needs memory
    out = ctx.out / "docs.parquet"
    # Small row groups: later steps read this file a row group at a time, so this bounds their memory.
    ctx.corpus.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet, ROW_GROUP_SIZE {ROW_GROUP})", params)
    counts = dict(ctx.corpus.execute(f"SELECT source, count(*) FROM '{out}' GROUP BY 1 ORDER BY 2 DESC").fetchall())
    ctx.log(f"{sum(counts.values()):,} docs (dataset {ctx.dataset}, snapshot {ctx.snapshot}, subset {ctx.subset})")
    return {"n_docs": sum(counts.values()), "by_source": counts}
