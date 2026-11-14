"""chunk: split documents into retrievable pieces.

Outputs chunks.parquet with columns chunk_id, doc_id, source, title, text, text_hash.
chunk_id is "<doc_id>#<n>". Sizes are in characters (~4 chars per token).

Chunks are cached per document *content hash* in a store shared by every run
using the same chunker settings, so adding 1k docs to a dataset chunks only those 1k.
"""

import hashlib
import re

import pyarrow as pa
from pydantic import BaseModel

from ..registry import register
from ..scripting import CHUNK_TEMPLATE, ScriptError, load_functions
from . import transform
from .base import StageContext, iter_parquet


class FixedConfig(BaseModel):
    model_config = {"extra": "forbid"}
    chunk_size: int = 1200
    overlap: int = 150
    prepend_title: bool = True      # prefix every chunk with the doc title so it carries context
    title_max_chars: int = 200      # some Slack docs open with a long summary line; don't prepend all of it


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


BATCH = 2_048              # documents chunked at a time; memory stays flat whatever the dataset size


def _write_chunks(ctx: StageContext, cfg: FixedConfig, split_doc) -> dict:
    """Chunk only docs whose content hash isn't in this chunker's store yet, then
    assemble chunks.parquet for this run's docs from the store. Documents are read
    and chunked one batch at a time."""
    docs_file = transform.docs_file(ctx.dirs)
    store = ctx.store("chunk")
    # No PRIMARY KEY (indexes live in memory); each batch only adds hashes not already in `chunked`.
    store.execute("""CREATE TABLE IF NOT EXISTS chunked(content_hash VARCHAR);
                     CREATE TABLE IF NOT EXISTS chunks(content_hash VARCHAR, i INTEGER, text VARCHAR, text_hash VARCHAR)""")
    n_docs, n_new_docs = 0, 0
    for batch in iter_parquet(docs_file, BATCH):
        n_docs += batch.num_rows
        store.register("batch", batch.select(["content_hash"]))
        todo = {h for (h,) in store.execute("SELECT DISTINCT content_hash FROM batch ANTI JOIN chunked USING (content_hash)").fetchall()}
        store.unregister("batch")
        if not todo:
            continue
        new = {k: [] for k in ["content_hash", "i", "text", "text_hash"]}
        processed = []                             # includes docs that yield zero chunks
        for d in batch.to_pylist():
            if d["content_hash"] not in todo:
                continue
            todo.discard(d["content_hash"])        # identical content in two docs: chunk once
            processed.append(d["content_hash"])
            title = d["title"] or ""
            body = d["text"][len(title):].lstrip("\n") if d["text"].startswith(title) else d["text"]
            title = title[: cfg.title_max_chars]
            for i, piece in enumerate(split_doc(d, body)):
                text = f"{title}\n{piece}" if cfg.prepend_title and title else piece
                new["content_hash"].append(d["content_hash"])
                new["i"].append(i)
                new["text"].append(text)
                new["text_hash"].append(hashlib.sha256(text.encode()).hexdigest()[:32])
        store.register("new_chunks", pa.table(new, schema=pa.schema([("content_hash", pa.string()), ("i", pa.int32()),
                                                                     ("text", pa.string()), ("text_hash", pa.string())])))
        store.execute("INSERT INTO chunks SELECT * FROM new_chunks")
        store.register("newly_chunked", pa.table({"content_hash": pa.array(processed, pa.string())}))
        store.execute("INSERT INTO chunked SELECT content_hash FROM newly_chunked")
        store.unregister("new_chunks")
        store.unregister("newly_chunked")
        n_new_docs += len(processed)
        ctx.log(f"chunked {n_new_docs:,} new docs ({n_docs:,} read)")

    store.execute("CHECKPOINT")         # write the new chunks to disk so their memory can be released
    # Unsorted on purpose: later steps match rows by position, and sorting every chunk's text would need memory.
    out = ctx.out / "chunks.parquet"
    store.execute(f"""COPY (SELECT d.doc_id || '#' || c.i AS chunk_id, d.doc_id, d.source, d.title, c.text, c.text_hash
                            FROM read_parquet('{docs_file}') d JOIN chunks c USING (content_hash))
                      TO '{out}' (FORMAT parquet, ROW_GROUP_SIZE {BATCH * 4})""")
    n, avg, mx = store.execute(f"SELECT count(*), avg(length(text)), max(length(text)) FROM '{out}'").fetchone()
    store.close()
    ctx.log(f"{n:,} chunks from {n_docs:,} docs ({n_new_docs:,} docs newly chunked, the rest from the store)")
    return {"n_chunks": n, "n_docs": n_docs, "docs_chunked": n_new_docs,
            "docs_reused": n_docs - n_new_docs, "avg_chars": round(avg or 0), "max_chars": mx}


class PythonChunkConfig(BaseModel):
    model_config = {"extra": "forbid"}
    code: str = CHUNK_TEMPLATE      # must define chunk(doc) -> list[str]
    prepend_title: bool = False     # your script decides what goes into each chunk
    title_max_chars: int = 200


def make_splitter(impl: str, cfg):
    """(doc, body) -> list[str] for a chunk implementation. Shared by the stage
    functions below and by the UI's "preview on a sample document"."""
    if impl == "fixed":
        return lambda d, body: split_fixed(body, cfg.chunk_size, cfg.overlap)
    if impl == "slack_aware":   # thread-aware for Slack, fixed-size for every other source
        return lambda d, body: (split_slack_thread(body, cfg.chunk_size, cfg.overlap, cfg.root_chars)
                                if d["source"] == "slack" else split_fixed(body, cfg.chunk_size, cfg.overlap))
    if impl == "python":
        fn = load_functions(cfg.code, "chunk")["chunk"]

        def split(d, body):
            out = fn({**d, "body": body})
            if not isinstance(out, list) or not all(isinstance(c, str) for c in out):
                raise ScriptError("chunk(doc) must return a list of strings")
            return [c for c in out if c.strip()]
        return split
    raise ValueError(impl)


def preview(impl: str, cfg, doc: dict) -> list[str]:
    """Chunks of a single document, exactly as the stage would produce them."""
    title = doc["title"] or ""
    body = doc["text"][len(title):].lstrip("\n") if doc["text"].startswith(title) else doc["text"]
    title = title[: cfg.title_max_chars]
    return [f"{title}\n{p}" if cfg.prepend_title and title else p for p in make_splitter(impl, cfg)(doc, body)]


@register("chunk", "fixed", FixedConfig)
def fixed(cfg: FixedConfig, ctx: StageContext) -> dict:
    return _write_chunks(ctx, cfg, make_splitter("fixed", cfg))


@register("chunk", "slack_aware", SlackAwareConfig)
def slack_aware(cfg: SlackAwareConfig, ctx: StageContext) -> dict:
    return _write_chunks(ctx, cfg, make_splitter("slack_aware", cfg))


@register("chunk", "python", PythonChunkConfig)
def python_chunker(cfg: PythonChunkConfig, ctx: StageContext) -> dict:
    """Your own chunk(doc) function, edited in the UI."""
    return _write_chunks(ctx, cfg, make_splitter("python", cfg))
