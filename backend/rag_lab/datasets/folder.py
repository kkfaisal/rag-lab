"""Bring your own data: a folder of .txt / .md files and/or .jsonl files.

  my_docs/
    handbook/onboarding.md        -> doc_id "handbook/onboarding.md", source "handbook"
    notes.txt                     -> doc_id "notes.txt", source "files"
    tickets.jsonl                 -> one doc per line: {"id", "text", "title"?, "source"?, "metadata"?, "updated_at"?}
    questions.jsonl               -> optional: {"question_id", "question", "expected_doc_ids"?, "gold_answer"?, "category"?}

Doc ids are relative paths (or the jsonl "id"), so re-ingesting the folder after
editing or adding files only picks up what changed.
"""

import json
from pathlib import Path

from .base import Dataset, Doc, register_dataset

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()[:300]
    return fallback


@register_dataset("folder")
class Folder(Dataset):
    def iter_docs(self, path: Path):
        for f in sorted(path.rglob("*")):
            if not f.is_file() or f.name == "questions.jsonl" or f.name.startswith("."):
                continue
            rel = f.relative_to(path).as_posix()
            source = rel.split("/")[0] if "/" in rel else "files"
            if f.suffix in TEXT_SUFFIXES:
                text = f.read_text(errors="replace")
                yield Doc(doc_id=rel, source=source, title=_title(text, f.stem), text=text, path=rel)
            elif f.suffix == ".jsonl":
                for n, line in enumerate(open(f)):
                    if line.strip():
                        r = json.loads(line)
                        text = r["text"]
                        yield Doc(doc_id=str(r.get("id", f"{rel}:{n}")), source=r.get("source", f.stem),
                                  title=r.get("title") or _title(text, f"{f.stem}:{n}"), text=text, path=rel,
                                  metadata=r.get("metadata") or {}, updated_at=r.get("updated_at"))

    def iter_questions(self, path: Path):
        qfile = path / "questions.jsonl"
        if not qfile.exists():
            return
        for line in open(qfile):
            if line.strip():
                q = json.loads(line)
                yield {"question_id": q["question_id"], "question": q["question"],
                       "category": q.get("category", "default"), "expected_doc_ids": q.get("expected_doc_ids") or [],
                       "gold_answer": q.get("gold_answer"), "answer_facts": q.get("answer_facts") or [],
                       "source_types": q.get("source_types") or []}
