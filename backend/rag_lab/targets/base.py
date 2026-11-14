"""The target interface.

A chunk is a dict: {"chunk_id", "doc_id", "source", "title", "text", "vector"}
("vector" is a list of floats, or None when the pipeline has no embed node).
A hit is a dict: {"chunk_id", "doc_id", "score", "text"}.
"""

import os
import re
from pathlib import Path

from pydantic import BaseModel


class TargetError(Exception):
    pass


def expand_env(value: str) -> str:
    """Replace ${NAME} with the environment variable NAME, so connection URLs can
    reference secrets kept in .env instead of storing them in the target."""
    def sub(m):
        if m.group(1) not in os.environ:
            raise TargetError(f"Environment variable {m.group(1)} is not set (referenced as ${{{m.group(1)}}}). Add it to .env.")
        return os.environ[m.group(1)]
    return re.sub(r"\$\{(\w+)\}", sub, value)


def safe_identifier(name: str) -> str:
    """Table / index names are put into SQL, so only allow plain identifiers."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", name):
        raise TargetError(f"{name!r} is not a valid table name (letters, digits and _ only)")
    return name


class Target:
    """Subclasses implement the storage operations. rag-lab decides *what* to write
    (append = only new/changed documents, overwrite = everything after clear());
    the target only has to add, delete by document, and search."""
    text_search = False       # can answer search_text (keyword / BM25)
    vector_search = False     # can answer search_vector

    def __init__(self, cfg: BaseModel, state_dir: Path):
        self.cfg, self.state_dir = cfg, state_dir

    def clear(self) -> None:
        raise NotImplementedError

    def delete_docs(self, doc_ids: list[str]) -> None:
        raise NotImplementedError

    def add(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        """Called once after a write (rebuild indexes, refresh, ...)."""

    def search_text(self, query: str, k: int) -> list[dict]:
        raise TargetError(f"{type(self).__name__} does not support keyword search")

    def search_vector(self, vector, k: int, query: str = "") -> list[dict]:
        raise TargetError(f"{type(self).__name__} does not support vector search")

    def count(self) -> int | None:
        return None

    def close(self) -> None:
        pass


TARGETS: dict[str, tuple[type[Target], type[BaseModel]]] = {}


def register_target(name: str, config_cls: type[BaseModel]):
    def deco(cls):
        TARGETS[name] = (cls, config_cls)
        return cls
    return deco


def open_target(definition: dict) -> Target:
    """definition = a saved target: {"id", "name", "type", "params"}."""
    from .. import paths
    if definition["type"] not in TARGETS:
        raise TargetError(f"Unknown target type {definition['type']!r}. Available: {sorted(TARGETS)}")
    cls, config_cls = TARGETS[definition["type"]]
    state_dir = paths.TARGETS / definition["id"]
    state_dir.mkdir(parents=True, exist_ok=True)
    return cls(config_cls(**(definition.get("params") or {})), state_dir)
