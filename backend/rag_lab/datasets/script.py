"""A dataset whose extractor is a Python script written in the UI.

The script defines extract(path) yielding document dicts, and optionally
questions(path). See scripting.EXTRACT_TEMPLATE.
"""

from pathlib import Path

from ..scripting import ScriptError, load_functions
from .base import Dataset, Doc, register_dataset


@register_dataset("script")
class Script(Dataset):
    def __init__(self, code: str | None = None):
        if not code:
            raise SystemExit("The 'script' adapter needs extractor code")
        self.fns = load_functions(code, "extract", ("questions",))

    def iter_docs(self, path: Path):
        for r in self.fns["extract"](path):
            if not isinstance(r, dict) or "doc_id" not in r or "text" not in r:
                raise ScriptError('extract() must yield dicts with at least "doc_id" and "text"')
            text = str(r["text"])
            yield Doc(doc_id=str(r["doc_id"]), source=str(r.get("source") or "files"),
                      title=str(r.get("title") or text.split("\n", 1)[0].strip()[:300]), text=text,
                      path=r.get("path"), metadata=r.get("metadata") or {}, updated_at=r.get("updated_at"))

    def iter_questions(self, path: Path):
        for q in (self.fns["questions"](path) if "questions" in self.fns else ()):
            yield {"question_id": str(q["question_id"]), "question": q["question"],
                   "category": q.get("category", "default"), "expected_doc_ids": q.get("expected_doc_ids") or [],
                   "gold_answer": q.get("gold_answer"), "answer_facts": q.get("answer_facts") or [],
                   "source_types": q.get("source_types") or []}
