"""transform: change documents before they are chunked.

Clean text, rewrite titles, enrich or read metadata, or drop documents. Outputs
docs.parquet in the same shape as the load step's. Each document's content hash is
recomputed from the transformed content, so the chunk and embed caches, and append
mode in targets, follow what the transform produced. Documents are read and written
one batch at a time, so memory doesn't grow with the dataset.
"""

import html
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..data import content_hash
from ..registry import NoConfig, register
from ..scripting import TRANSFORM_TEMPLATE, ScriptError, load_functions
from .base import StageContext, iter_parquet

BATCH = 2_048
SCHEMA = pa.schema([("doc_id", pa.string()), ("source", pa.string()), ("title", pa.string()), ("text", pa.string()),
                    ("metadata", pa.string()), ("content_hash", pa.string())])


def docs_file(dirs: dict[str, Path]) -> Path:
    """The documents the chunk step should read: the transform's output if there is one."""
    t = dirs.get("transform")
    return t / "docs.parquet" if t is not None and (t / "docs.parquet").exists() else dirs["load"] / "docs.parquet"


class CleanConfig(BaseModel):
    model_config = {"extra": "forbid"}
    strip_html: bool = True                 # remove tags, decode entities (&amp; -> &)
    normalize_whitespace: bool = True       # collapse runs of spaces; at most one blank line in a row
    drop_lines_matching: str | None = None  # regex; matching lines are removed (e.g. boilerplate footers)
    min_chars: int = 0                      # drop documents shorter than this after cleaning
    max_chars: int | None = None            # cut documents longer than this


TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)


def make_cleaner(cfg: CleanConfig):
    drop = re.compile(cfg.drop_lines_matching) if cfg.drop_lines_matching else None

    def clean(doc: dict) -> dict | None:
        text = doc["text"]
        if cfg.strip_html:
            text = html.unescape(TAG.sub(" ", text))
        if drop:
            text = "\n".join(line for line in text.split("\n") if not drop.search(line))
        if cfg.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
        if len(text) < cfg.min_chars:
            return None
        if cfg.max_chars:
            text = text[: cfg.max_chars]
        return {**doc, "text": text}
    return clean


def _check(out, doc_id: str) -> dict | None:
    if out is None:
        return None
    if not isinstance(out, dict) or not isinstance(out.get("text"), str):
        raise ScriptError("transform(doc) must return a dict with a string 'text', or None to drop the document")
    if out.get("doc_id", doc_id) != doc_id:
        raise ScriptError("transform(doc) must not change doc_id (questions refer to documents by id)")
    return out


def make_transform(impl: str, cfg):
    """doc -> doc | None for a transform implementation (also used by the UI preview)."""
    if impl == "clean":
        return make_cleaner(cfg)
    if impl == "python":
        fn = load_functions(cfg.code, "transform")["transform"]
        return lambda doc: _check(fn(doc), doc["doc_id"])
    raise ValueError(impl)


def apply(ctx: StageContext, fn) -> dict:
    src = ctx.dirs["load"] / "docs.parquet"
    out = ctx.out / "docs.parquet"
    n_in = n_out = n_changed = chars_in = chars_out = 0
    with pq.ParquetWriter(out, SCHEMA) as writer:
        for batch in iter_parquet(src, BATCH):
            rows = {k: [] for k in SCHEMA.names}
            for d in batch.to_pylist():
                n_in += 1
                chars_in += len(d["text"])
                doc = {"doc_id": d["doc_id"], "source": d["source"], "title": d["title"] or "", "text": d["text"],
                       "metadata": json.loads(d["metadata"]) if d.get("metadata") else {}}
                new = fn(dict(doc))
                if new is None:
                    continue
                new = {**doc, **new}
                h = content_hash(new["source"], new["title"], new["text"])
                n_out += 1
                n_changed += h != d["content_hash"]
                chars_out += len(new["text"])
                for k in ("doc_id", "source", "title", "text"):
                    rows[k].append(str(new[k]))
                rows["metadata"].append(json.dumps(new["metadata"] or {}))
                rows["content_hash"].append(h)
            writer.write_table(pa.table(rows, schema=SCHEMA))
            ctx.log(f"transformed {n_in:,} docs")
    ctx.log(f"{n_in:,} docs in → {n_out:,} out ({n_in - n_out:,} dropped, {n_changed:,} changed)")
    return {"n_docs": n_out, "docs_in": n_in, "dropped": n_in - n_out, "changed": n_changed,
            "chars_in": chars_in, "chars_out": chars_out}


@register("transform", "none", NoConfig)
def passthrough(cfg: NoConfig, ctx: StageContext) -> dict:
    """No transform: the chunk step reads the loaded documents directly."""
    return {"passthrough": True}


@register("transform", "clean", CleanConfig)
def clean(cfg: CleanConfig, ctx: StageContext) -> dict:
    """Strip HTML, drop boilerplate lines, normalise whitespace, filter or cut by length."""
    return apply(ctx, make_cleaner(cfg))


class PythonTransformConfig(BaseModel):
    model_config = {"extra": "forbid"}
    code: str = TRANSFORM_TEMPLATE   # must define transform(doc) -> doc | None


@register("transform", "python", PythonTransformConfig)
def python_transform(cfg: PythonTransformConfig, ctx: StageContext) -> dict:
    """Your own transform(doc) function, edited in the UI."""
    return apply(ctx, make_transform("python", cfg))
