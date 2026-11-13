"""chunk: split documents into retrievable pieces.

Outputs chunks.parquet with columns chunk_id, doc_id, source, title, text.
chunk_id is "<doc_id>#<n>". Sizes are in characters (~4 chars per token).
"""

import re

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..registry import register
from .base import StageContext


class FixedConfig(BaseModel):
    model_config = {"extra": "forbid"}
    chunk_size: int = 1200
    overlap: int = 150
    prepend_title: bool = True      # prefix every chunk with the doc title so it carries context


class SlackAwareConfig(FixedConfig):
    root_chars: int = 300           # how much of the thread's first message to repeat in later chunks


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window of `size` chars, nudged back to the nearest newline/space so
    words aren't cut, with `overlap` chars shared between neighbours."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


SPEAKER = re.compile(r"^[A-Z][\w.'\- ]{0,40}: ")


def split_slack_thread(body: str, size: int, overlap: int, root_chars: int) -> list[str]:
    """A Slack doc is one thread: 'Name: message' lines (continuation lines such as
    code blocks belong to the message above). Pack whole messages into chunks, and
    start every chunk after the first with the thread's opening message, because
    replies like 'yes, do that' are meaningless without it."""
    messages: list[str] = []
    for line in body.split("\n"):
        if SPEAKER.match(line) or not messages:
            messages.append(line)
        else:
            messages[-1] += "\n" + line
    messages = [m.strip() for m in messages if m.strip()]
    if not messages:
        return []
    root = messages[0][:root_chars]
    chunks, current = [], ""
    for msg in messages:
        pieces = split_fixed(msg, size, overlap) if len(msg) > size else [msg]
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > size:
                chunks.append(current)
                current = f"[thread start] {root}\n…\n" if root and not piece.startswith(root) else ""
            current += piece + "\n"
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks]


def _write_chunks(ctx: StageContext, cfg: FixedConfig, split_doc) -> dict:
    docs = pq.read_table(ctx.dirs["load"] / "docs.parquet").to_pylist()
    rows = {k: [] for k in ["chunk_id", "doc_id", "source", "title", "text"]}
    for d in docs:
        title = d["title"] or ""
        body = d["text"][len(title):].lstrip("\n") if d["text"].startswith(title) else d["text"]
        for i, piece in enumerate(split_doc(d, body)):
            text = f"{title}\n{piece}" if cfg.prepend_title and title else piece
            rows["chunk_id"].append(f"{d['doc_id']}#{i}")
            rows["doc_id"].append(d["doc_id"])
            rows["source"].append(d["source"])
            rows["title"].append(title)
            rows["text"].append(text)
    pq.write_table(pa.table(rows), ctx.out / "chunks.parquet")
    n = len(rows["chunk_id"])
    ctx.log(f"{n:,} chunks from {len(docs):,} docs")
    lengths = [len(t) for t in rows["text"]] or [0]
    return {"n_chunks": n, "n_docs": len(docs), "avg_chars": round(sum(lengths) / len(lengths)), "max_chars": max(lengths)}


@register("chunk", "fixed", FixedConfig)
def fixed(cfg: FixedConfig, ctx: StageContext) -> dict:
    return _write_chunks(ctx, cfg, lambda d, body: split_fixed(body, cfg.chunk_size, cfg.overlap))


@register("chunk", "slack_aware", SlackAwareConfig)
def slack_aware(cfg: SlackAwareConfig, ctx: StageContext) -> dict:
    """Thread-aware splitting for Slack, fixed-size for every other source."""
    def split(d, body):
        if d["source"] == "slack":
            return split_slack_thread(body, cfg.chunk_size, cfg.overlap, cfg.root_chars)
        return split_fixed(body, cfg.chunk_size, cfg.overlap)
    return _write_chunks(ctx, cfg, split)
