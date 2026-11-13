"""Run folders and the run index.

runs/<run_id>/            everything about one experiment (the source of truth)
runs/index.duckdb         a queryable index over all folders; rebuildable with `rag-lab runs reindex`
"""

import json
import subprocess
import time
from pathlib import Path

import duckdb

from . import paths
from .config import flatten


def git_info(run_dir: Path) -> dict:
    """Commit SHA, and if the tree is dirty, the full diff saved as git.diff."""
    def git(*args):
        r = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    sha = (git("rev-parse", "HEAD") or "").strip() or None
    dirty = bool((git("status", "--porcelain", "--untracked-files=no") or "").strip())
    if dirty and sha:
        (run_dir / "git.diff").write_text(git("diff", "HEAD") or "")
    return {"sha": sha, "dirty": dirty, "branch": (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None}


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id VARCHAR PRIMARY KEY, name VARCHAR, description VARCHAR, created_at TIMESTAMP, status VARCHAR,
  kind VARCHAR, meta JSON, dataset VARCHAR, snapshot INTEGER, subset VARCHAR, n_questions INTEGER, "limit" INTEGER, git_sha VARCHAR, git_dirty BOOLEAN,
  duration_s DOUBLE, cost_usd DOUBLE, judged BOOLEAN, error VARCHAR,
  config JSON, params JSON, overall JSON);
CREATE TABLE IF NOT EXISTS metrics(run_id VARCHAR, category VARCHAR, metric VARCHAR, value DOUBLE);
"""


def _connect(read_only=False, retries=20):
    """DuckDB allows one writer process at a time; retry briefly if the API or
    another run has the file open."""
    paths.RUNS.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            con = duckdb.connect(str(paths.RUNS_INDEX), read_only=read_only)
            if not read_only:
                con.execute(SCHEMA)
            return con
        except duckdb.IOException:
            if i == retries - 1:
                raise
            time.sleep(0.25)


def connect_read():
    ensure_index()
    return _connect(read_only=True)


def ensure_index() -> None:
    """Create the index, or rebuild it from the run folders if it predates the current schema."""
    if not paths.RUNS_INDEX.exists():
        _connect().close()
        return
    con = _connect(read_only=True)
    cols = {r[0] for r in con.execute("DESCRIBE runs").fetchall()}
    con.close()
    if "kind" not in cols:
        reindex()


def index_run(run_dir: Path, con=None) -> None:
    ensure_index() if con is None else None
    run = json.loads((run_dir / "run.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    own = con is None
    con = con or _connect()
    rid = run["run_id"]
    con.execute("DELETE FROM runs WHERE run_id=?", [rid])
    con.execute("DELETE FROM metrics WHERE run_id=?", [rid])
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        rid, run["name"], cfg.get("description", ""), run["created_at"], run["status"],
        run.get("kind", "experiment"), json.dumps(run.get("meta") or {}),
        cfg["data"]["dataset"], cfg["data"]["snapshot"], cfg["data"]["subset"],
        run.get("n_questions"), run.get("limit"), run["git"]["sha"], run["git"]["dirty"], run.get("duration_s"),
        run.get("cost_usd"), bool(metrics.get("judged", False)), run.get("error"),
        json.dumps(cfg), json.dumps(flatten({k: v for k, v in cfg.items() if k not in ("name", "description")})),
        json.dumps(metrics.get("overall", {}))])
    rows = [(rid, "_overall", m, v) for m, v in metrics.get("overall", {}).items() if isinstance(v, (int, float))]
    for cat, ms in metrics.get("by_category", {}).items():
        rows += [(rid, cat, m, v) for m, v in ms.items() if isinstance(v, (int, float))]
    if rows:
        con.executemany("INSERT INTO metrics VALUES (?,?,?,?)", rows)
    if own:
        con.close()


def reindex() -> int:
    paths.RUNS_INDEX.unlink(missing_ok=True)
    con = _connect()
    n = 0
    for d in sorted(paths.RUNS.iterdir()):
        if (d / "run.json").exists() and (d / "config.json").exists():
            index_run(d, con)
            n += 1
    con.close()
    return n
