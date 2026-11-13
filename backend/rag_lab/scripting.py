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


EMBED_TEMPLATE = '''from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = None


def _load():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model


def embed(texts):
    """Return one vector per text (list of lists or a 2-D array). Called in batches of chunks."""
    return _load().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)


def embed_query(text):
    """Optional: how a question is embedded at search time. Defaults to embed([text])[0]."""
    return embed([QUERY_PREFIX + text])[0]
'''

RERANK_TEMPLATE = '''from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def rerank(question, chunks, top_n):
    """Reorder retrieved chunks for one question and return the best top_n.

    chunks: list of {"chunk_id", "doc_id", "score", "text"}, best-first from retrieval.
    """
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL)
    scores = _model.predict([(question, c["text"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:top_n]
    return [{**c, "score": float(s)} for c, s in ranked]
'''


def equivalent_script(stage: str, impl: str, params: dict) -> dict | None:
    """Params for a `python` node that does the same as built-in `impl` with
    `params`: the starting point when a user clicks "Edit as Python script".
    Built from the real implementation source, so it can't drift from it."""
    import inspect
    p = params
    if stage == "chunk" and impl in ("fixed", "slack_aware"):
        from .stages import chunk
        head = f"SIZE = {p.get('chunk_size', 1200)}\nOVERLAP = {p.get('overlap', 150)}\n"
        title = 'doc["title"][:200] + "\\n" + c' if p.get("prepend_title", True) else "c"
        if impl == "fixed":
            code = (f"{head}\n\n{inspect.getsource(chunk.split_fixed)}\n\ndef chunk(doc):\n"
                    f"    return [{title} for c in split_fixed(doc[\"body\"], SIZE, OVERLAP)]\n")
        else:
            code = (f"import re\n\n{head}ROOT_CHARS = {p.get('root_chars', 300)}\n"
                    f"SPEAKER = re.compile(r\"{chunk.SPEAKER.pattern}\")\n\n\n"
                    f"{inspect.getsource(chunk.split_fixed)}\n\n{inspect.getsource(chunk.split_slack_thread)}\n\n"
                    "def chunk(doc):\n    if doc[\"source\"] == \"slack\":\n"
                    "        pieces = split_slack_thread(doc[\"body\"], SIZE, OVERLAP, ROOT_CHARS)\n"
                    "    else:\n        pieces = split_fixed(doc[\"body\"], SIZE, OVERLAP)\n"
                    f"    return [{title} for c in pieces]\n")
        return {"code": code, "prepend_title": False}
    if stage == "retrieve" and impl in ("bm25", "vector", "hybrid"):
        from .stages import retrieve
        k = p.get("top_k", 20)
        if impl == "hybrid":
            code = (f"CANDIDATES = {p.get('candidates', 100)}\nRRF_K = {p.get('rrf_k', 60)}\n"
                    f"BM25_WEIGHT = {p.get('bm25_weight', 1.0)}\nVECTOR_WEIGHT = {p.get('vector_weight', 1.0)}\n\n\n"
                    f"{inspect.getsource(retrieve.rrf)}\n\ndef retrieve(question, search, top_k):\n"
                    "    lexical = search.bm25(question, CANDIDATES)\n    dense = search.vector(question, CANDIDATES)\n"
                    "    return rrf([(BM25_WEIGHT, lexical), (VECTOR_WEIGHT, dense)], RRF_K, top_k)\n")
        else:
            code = f"def retrieve(question, search, top_k):\n    return search.{impl}(question, top_k)\n"
        return {"code": code, "top_k": k}
    if stage == "embed" and impl == "sentence_transformers":
        code = (EMBED_TEMPLATE.replace('"BAAI/bge-small-en-v1.5"', repr(p.get("model", "BAAI/bge-small-en-v1.5")).replace("'", '"'))
                .replace("batch_size=64", f"batch_size={p.get('batch_size', 64)}"))
        if "query_prefix" in p:
            code = code.replace('"Represent this sentence for searching relevant passages: "', repr(p["query_prefix"]))
        return {"code": code}
    if stage == "rerank" and impl == "cross_encoder":
        return {"code": RERANK_TEMPLATE.replace('"cross-encoder/ms-marco-MiniLM-L-6-v2"',
                                                '"' + p.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2") + '"'),
                "top_n": p.get("top_n", 10)}
    return None


TARGET_TEMPLATE = '''import pickle
from pathlib import Path

import numpy as np

# STATE_DIR is provided by rag-lab: a folder that belongs to this target.
STORE = Path(STATE_DIR) / "store.pkl"


def _load():
    return pickle.loads(STORE.read_bytes()) if STORE.exists() else []


def _save(rows):
    STORE.write_bytes(pickle.dumps(rows))


def clear():
    """Remove everything (called for an overwrite)."""
    _save([])


def delete_docs(doc_ids):
    """Remove all chunks of these documents (called before changed documents are re-added)."""
    gone = set(doc_ids)
    _save([r for r in _load() if r["doc_id"] not in gone])


def add(chunks):
    """Store chunks: dicts with chunk_id, doc_id, source, title, text, vector (list of floats or None)."""
    _save(_load() + chunks)


def search(query, query_vector, k):
    """Return the best k chunks as dicts with chunk_id, doc_id, text, score.

    query is the question text; query_vector is its embedding (None for keyword-only retrieval).
    """
    rows = _load()
    if query_vector is None:
        words = set(query.lower().split())
        scored = [(len(words & set(r["text"].lower().split())), r) for r in rows]
    else:
        q = np.asarray(query_vector)
        scored = [(float(np.dot(q, np.asarray(r["vector"]))), r) for r in rows if r["vector"] is not None]
    scored.sort(key=lambda pair: -pair[0])
    return [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "text": r["text"], "score": s} for s, r in scored[:k]]


def count():
    """Optional: number of chunks stored."""
    return len(_load())
'''


class ScriptError(Exception):
    pass


def load_functions(code: str, required: str, optional: tuple[str, ...] = (), extra_globals: dict | None = None) -> dict:
    """Execute `code` and return the named functions from it."""
    ns: dict = {"__name__": "rag_lab_script", **(extra_globals or {})}
    try:
        exec(compile(code, "<script node>", "exec"), ns)
    except Exception as e:
        raise ScriptError(f"Script failed to load: {type(e).__name__}: {e}") from e
    if not callable(ns.get(required)):
        raise ScriptError(f"Script must define a function named `{required}`")
    return {name: ns[name] for name in (required, *optional) if callable(ns.get(name))}
