"""Background jobs started from the UI: index builds, experiments, dataset ingests.

Jobs run one at a time, each in its own subprocess (`python -m rag_lab.worker`),
so a crash or a cancelled job can't take the API down, model memory is released
when a job ends, and there is only ever one process writing to DuckDB.
A job is a JSON file in runs/jobs/ plus a log file next to it.
"""

import json
import queue
import subprocess
import sys
import threading
import time
import uuid

from . import paths

_queue: "queue.Queue[str]" = queue.Queue()
_procs: dict[str, subprocess.Popen] = {}
_cancelled: set[str] = set()
_lock = threading.Lock()
_started = False


def _file(job_id: str):
    return paths.JOBS / f"{job_id}.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _save(job: dict) -> None:
    with _lock:
        _file(job["id"]).write_text(json.dumps(job, indent=2))


def get(job_id: str) -> dict:
    job = json.loads(_file(job_id).read_text())
    run_file = paths.JOBS / f"{job_id}.run"
    if run_file.exists():
        job["run_id"] = run_file.read_text().strip()
    return job


def log_tail(job_id: str, max_chars: int = 6000) -> str:
    f = paths.JOBS / f"{job_id}.log"
    if not f.exists():
        return ""
    text = f.read_text(errors="replace").replace("\r", "\n")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith(("Batches:", "Loading weights"))]
    return "\n".join(lines)[-max_chars:]


def list_jobs(limit: int = 50) -> list[dict]:
    paths.JOBS.mkdir(parents=True, exist_ok=True)
    files = sorted(paths.JOBS.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
    return [get(f.stem) for f in files]


def submit(kind: str, title: str, spec: dict) -> dict:
    """kind: "build" | "experiment" | "ingest". spec is what rag_lab.worker needs."""
    start()
    job = {"id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}", "kind": kind, "title": title,
           "spec": spec, "status": "queued", "created_at": _now()}
    _save(job)
    _queue.put(job["id"])
    return job


def cancel(job_id: str) -> dict:
    job = get(job_id)
    if job["status"] in ("queued", "running"):
        _cancelled.add(job_id)
        proc = _procs.get(job_id)
        if proc:
            proc.terminate()
        else:
            job.update(status="cancelled", finished_at=_now())
            _save(job)
    return get(job_id)


def _loop() -> None:
    while True:
        job_id = _queue.get()
        job = get(job_id)
        if job_id in _cancelled or job["status"] != "queued":
            continue
        job.update(status="running", started_at=_now())
        _save(job)
        with open(paths.JOBS / f"{job_id}.log", "w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "rag_lab.worker", str(_file(job_id))],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=paths.ROOT)
            _procs[job_id] = proc
            code = proc.wait()
            _procs.pop(job_id, None)
        job = get(job_id)
        job.update(status="cancelled" if job_id in _cancelled else "ok" if code == 0 else "failed", finished_at=_now())
        if job["status"] == "failed":
            job["error"] = (log_tail(job_id, 100000).splitlines() or ["failed"])[-1]
        _save(job)


def start() -> None:
    """Start the single worker thread; mark jobs orphaned by a server restart."""
    global _started
    if _started:
        return
    _started = True
    paths.JOBS.mkdir(parents=True, exist_ok=True)
    for f in paths.JOBS.glob("*.json"):
        job = json.loads(f.read_text())
        if job["status"] in ("queued", "running"):
            job.update(status="failed", error="server restarted while this job was pending", finished_at=_now())
            f.write_text(json.dumps(job, indent=2))
    threading.Thread(target=_loop, daemon=True, name="rag-lab-jobs").start()
