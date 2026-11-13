"""EnterpriseRAG-Bench (https://github.com/onyx-dot-app/EnterpriseRAG-Bench).

Input: a directory holding the release zips (all_documents.zip and/or
<source>_slice_NNNN.zip) and questions.jsonl. Docs are .txt files named
dsid_<sha>__<slug>.txt, title on the first line; the source type comes from the
top-level directory (or, for flat slice zips, the zip name).
"""

import json
import re
import zipfile
from pathlib import Path

from .base import Dataset, Doc, register_dataset

SOURCES = ["slack", "gmail", "linear", "google_drive", "hubspot", "fireflies", "github", "jira", "confluence"]
DOC_ID_RE = re.compile(r"(dsid_[0-9a-f]+)")


@register_dataset("enterprise_rag_bench")
class EnterpriseRAGBench(Dataset):
    evaluator = "benchmark"

    def iter_docs(self, path: Path):
        zips = [path] if path.suffix == ".zip" else sorted(path.glob("*.zip"))
        if not zips:
            raise SystemExit(f"No .zip files in {path}. Run `rag-lab data download` first.")
        for zp in zips:
            zip_source = next((s for s in SOURCES if zp.name.startswith(s + "_slice")), None)
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    m = DOC_ID_RE.search(Path(name).name)
                    if not name.endswith(".txt") or not m:
                        continue
                    text = z.read(name).decode("utf-8", errors="replace")
                    yield Doc(doc_id=m.group(1), source=zip_source or name.split("/")[0],
                              title=text.split("\n", 1)[0].strip(), text=text, path=name)

    def iter_questions(self, path: Path):
        qfile = (path if path.is_dir() else path.parent) / "questions.jsonl"
        if not qfile.exists():
            return
        for line in open(qfile):
            q = json.loads(line)
            yield {"question_id": q["question_id"], "question": q["question"], "category": q["question_type"],
                   "expected_doc_ids": q.get("expected_doc_ids") or [], "gold_answer": q.get("gold_answer"),
                   "answer_facts": q.get("answer_facts") or [], "source_types": q.get("source_types") or []}
