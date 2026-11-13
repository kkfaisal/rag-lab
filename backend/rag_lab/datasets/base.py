"""The dataset adapter interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Doc:
    doc_id: str                 # stable id; the same doc in a later ingest must keep it
    source: str                 # e.g. "slack", "confluence", "files"
    title: str
    text: str                   # full text, title included as the first line if natural
    path: str | None = None     # where it came from (informational)
    metadata: dict = field(default_factory=dict)
    updated_at: str | None = None


class Dataset:
    """Subclasses implement iter_docs and optionally iter_questions.

    Questions are dicts: {question_id, question, category, expected_doc_ids,
    gold_answer, answer_facts, source_types}; only question_id and question are required.
    """
    # Which evaluator applies: "benchmark" = the EnterpriseRAG-Bench LLM judge is valid
    # for these questions; "recall" = only our deterministic retrieval metrics.
    evaluator = "recall"

    def iter_docs(self, path: Path) -> Iterator[Doc]:
        raise NotImplementedError

    def iter_questions(self, path: Path) -> Iterator[dict]:
        return iter(())


ADAPTERS: dict[str, type[Dataset]] = {}


def register_dataset(name: str):
    def deco(cls):
        ADAPTERS[name] = cls
        return cls
    return deco


def get_adapter(name: str, code: str | None = None) -> Dataset:
    if name not in ADAPTERS:
        raise SystemExit(f"Unknown dataset adapter {name!r}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](code) if name == "script" else ADAPTERS[name]()
