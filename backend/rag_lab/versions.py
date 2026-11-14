"""Version history of every Python script you edit in the app.

Each script has a key: "<pipeline kind>/<pipeline id>/<node id>" for a script node,
"targets/<id>" for a script target, "datasets/<name>" for a dataset's extractor.
Saving records a new version only when the code changed, so the history is the list
of distinct scripts in the order they were saved. Stored as one JSON line per version
in workspace/script_versions/<key>.jsonl.
"""

import json
import re
import time
from pathlib import Path

from . import paths


def _file(key: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+){1,2}", key) or ".." in key:
        raise ValueError(f"bad script key {key!r}")
    return paths.WORKSPACE / "script_versions" / f"{key}.jsonl"


def history(key: str) -> list[dict]:
    f = _file(key)
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def record(key: str, code: str, note: str = "") -> int | None:
    """Add `code` as a new version unless it equals the latest one. Returns the new version number."""
    past = history(key)
    if past and past[-1]["code"] == code:
        return None
    f = _file(key)
    f.parent.mkdir(parents=True, exist_ok=True)
    n = (past[-1]["version"] if past else 0) + 1
    with open(f, "a") as fh:
        fh.write(json.dumps({"version": n, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note, "code": code}) + "\n")
    return n


def record_graph(kind: str, graph: dict) -> None:
    for n in graph.get("nodes", []):
        code = (n.get("params") or {}).get("code")
        if isinstance(code, str) and graph.get("id"):
            record(f"{kind}/{graph['id']}/{n['id']}", code, note=f"{n['stage']} node")


def listing(key: str) -> list[dict]:
    """Versions newest first, without the code (the UI fetches one when you pick it)."""
    return [{k: v for k, v in h.items() if k != "code"} | {"lines": h["code"].count("\n") + 1} for h in reversed(history(key))]


def get(key: str, version: int) -> dict:
    for h in history(key):
        if h["version"] == version:
            return h
    raise KeyError(version)


def record_all() -> None:
    """Make sure every script already in the workspace has at least its current version
    (scripts saved before versioning existed). No-op for scripts already recorded."""
    from . import workspace
    for kind in workspace.KINDS:
        for g in workspace.list_graphs(kind):
            record_graph(kind, g)
    for t in workspace.list_targets():
        if isinstance((t.get("params") or {}).get("code"), str):
            record(f"targets/{t['id']}", t["params"]["code"], note="target script")
    from . import paths as _paths
    if _paths.DATASETS.exists():
        for d in _paths.DATASETS.iterdir():
            f = d / "extract.py"
            if f.is_file():
                record(f"datasets/{d.name}", f.read_text(), note="extractor")
