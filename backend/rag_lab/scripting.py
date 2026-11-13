"""User-written Python for script nodes (extract / chunk / retrieve).

The code is plain Python, stored inside the node's config, so it is hashed for
caching and saved with every run like any other parameter. It runs with full
privileges in the worker process: rag-lab is a local tool, only run code you trust.
"""

CHUNK_TEMPLATE = '''def chunk(doc):
    """Split one document into chunks. Return a list of strings.

    doc keys: doc_id, source, title, text (full text), body (text without the title line).
    """
    paragraphs = [p.strip() for p in doc["body"].split("\\n\\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > 1200:
            chunks.append(current)
            current = ""
        current += p + "\\n\\n"
    if current.strip():
        chunks.append(current)
    return [doc["title"] + "\\n" + c.strip() for c in chunks]
'''

RETRIEVE_TEMPLATE = '''def retrieve(question, search, top_k):
    """Return ranked chunks for one question: a list of {"chunk_id", "doc_id", "score"}.

    search.bm25(query, k) and search.vector(query, k) query the index and return
    such lists (each hit also has "text"). Combine, rewrite, filter as you like.
    """
    hits = search.vector(question, top_k)
    return hits
'''

EXTRACT_TEMPLATE = '''import json
from pathlib import Path


def extract(path):
    """Yield one dict per document found under `path` (the dataset's upload folder).

    Required keys: doc_id (stable across re-ingests), text.
    Optional: title, source, metadata (dict), updated_at.
    """
    for f in sorted(Path(path).rglob("*")):
        if f.suffix in {".txt", ".md"}:
            text = f.read_text(errors="replace")
            yield {"doc_id": f.relative_to(path).as_posix(), "text": text,
                   "title": text.split("\\n", 1)[0].strip(), "source": "files"}


def questions(path):
    """Optional: yield {"question_id", "question", "expected_doc_ids", "gold_answer", "category"}."""
    q = Path(path) / "questions.jsonl"
    if q.exists():
        for line in open(q):
            if line.strip():
                yield json.loads(line)
'''


class ScriptError(Exception):
    pass


def load_functions(code: str, required: str, optional: tuple[str, ...] = ()) -> dict:
    """Execute `code` and return the named functions from it."""
    ns: dict = {"__name__": "rag_lab_script"}
    try:
        exec(compile(code, "<script node>", "exec"), ns)
    except Exception as e:
        raise ScriptError(f"Script failed to load: {type(e).__name__}: {e}") from e
    if not callable(ns.get(required)):
        raise ScriptError(f"Script must define a function named `{required}`")
    return {name: ns[name] for name in (required, *optional) if callable(ns.get(name))}
