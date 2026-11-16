"""HTTP API for the playground UI. Started by `rag-lab serve`.

Reads are served directly; anything slow or that writes to DuckDB (ingest, index
builds, experiments) is queued as a background job (jobs.py).
"""

import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import bench, data, jobs, paths, scripting, stats, tracking, versions, workspace
from .targets import TARGETS, Manifest, TargetError, open_target

load_dotenv(paths.ROOT / ".env")
data.migrate_layout()
versions.record_all()          # scripts saved before versioning existed get their first version
app = FastAPI(title="rag-lab")
jobs.start()


def _run_dir(run_id: str) -> Path:
    d = (paths.RUNS / run_id).resolve()
    if d.parent != paths.RUNS.resolve() or not (d / "run.json").exists():
        raise HTTPException(404, f"run {run_id} not found")
    return d


def _json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _query(sql: str, params=None) -> list[dict]:
    con = tracking.connect_read()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def _corpus():
    try:
        return data.connect(read_only=True)
    except SystemExit as e:
        raise HTTPException(409, str(e))


def _question_texts(dataset: str) -> dict[str, str]:
    con = _corpus()
    try:
        return dict(con.execute("SELECT question_id, question FROM questions WHERE dataset=?", [dataset]).fetchall())
    finally:
        con.close()


@app.get("/api/datasets")
def datasets():
    """Loaded datasets (document store), each with its folder's summary."""
    if not paths.CORPUS_DB.exists():
        return []
    con = _corpus()
    try:
        info = data.dataset_info(con)
    finally:
        con.close()
    for d in info:
        d["folder"] = data.folder_summary(paths.RAW if d["name"] == data.BENCH else paths.DATASETS / d["name"])
    return info


@app.get("/api/benchmarks")
def public_datasets():
    """Public datasets that can be downloaded with a Python extractor."""
    from . import benchmarks
    return benchmarks.catalog()


@app.get("/api/benchmarks/{name}/extractor")
def public_dataset_extractor(name: str, split: str = "test"):
    from . import benchmarks
    try:
        data.dataset_dir(name)
    except SystemExit as e:
        raise HTTPException(400, str(e))
    return {"code": benchmarks.extractor(name, split)}


@app.get("/api/folders")
def folders():
    """Every folder under data/datasets/, loaded or not."""
    return data.list_folders()


def _runs(kind: str) -> list[dict]:
    rows = _query("""SELECT run_id, name, description, created_at, status, kind, meta, dataset, snapshot, subset, n_questions,
                     "limit", git_sha, git_dirty, duration_s, cost_usd, judged, error, params, overall
                     FROM runs WHERE kind = ? ORDER BY created_at DESC""", [kind])
    for r in rows:
        r["params"], r["overall"], r["meta"] = json.loads(r["params"]), json.loads(r["overall"]), json.loads(r["meta"] or "{}")
    return rows


@app.get("/api/lessons")
def list_lessons():
    """The guided lessons, each with its best matching run and whether its goal is met."""
    from . import lessons
    return lessons.with_status()


@app.get("/api/runs")
def list_runs():
    return _runs("benchmark")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    d = _run_dir(run_id)
    return {
        "run": _json(d / "run.json"), "config": _json(d / "config.json"), "stages": _json(d / "stages.json", []),
        "usage": _json(d / "usage.json", {}), "code_hashes": _json(d / "code_hashes.json", {}),
        "metrics": _json(d / "metrics.json", {}),
        "git_diff": (d / "git.diff").read_text() if (d / "git.diff").exists() else None,
        "error": (d / "error.txt").read_text() if (d / "error.txt").exists() else None,
    }


def _clean_log(text: str, max_chars: int = 40000) -> str:
    """Progress bars rewrite their line with \r: keep only each line's final state."""
    lines = [ln.rstrip("\r").rsplit("\r", 1)[-1] for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())[-max_chars:]


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, step: str | None = None):
    """The log of one step of a run (logs/<step>.log), or the list of steps that have a log.
    Runs from before step logs existed have none: the UI then shows the job's whole log."""
    d = _run_dir(run_id) / "logs"
    steps = sorted(f.stem for f in d.glob("*.log")) if d.is_dir() else []
    if step is None:
        return {"steps": steps}
    if step not in steps:
        raise HTTPException(404, f"no log for step {step!r} in run {run_id}")
    with open(d / f"{step}.log", errors="replace", newline="") as f:      # newline="": keep \r, so progress bars collapse
        return {"step": step, "log": _clean_log(f.read())}


@app.get("/api/runs/{run_id}/questions")
def run_questions(run_id: str):
    d = _run_dir(run_id)
    per_q = _jsonl(d / "per_question.jsonl")
    answers = {a["question_id"]: a for a in _jsonl(d / "answers.jsonl")}
    texts = _question_texts(_json(d / "config.json")["data"]["dataset"])
    return [{**r, "question": texts.get(r["question_id"]), "answer": (answers.get(r["question_id"]) or {}).get("answer")}
            for r in per_q]


@app.get("/api/runs/{run_id}/questions/{qid}")
def run_question(run_id: str, qid: str):
    d = _run_dir(run_id)
    find = lambda rows: next((r for r in rows if r["question_id"] == qid), None)  # noqa: E731
    retrieval, reranked = find(_jsonl(d / "retrieval.jsonl")), find(_jsonl(d / "reranked.jsonl"))
    generation, answer, metrics = find(_jsonl(d / "generations.jsonl")), find(_jsonl(d / "answers.jsonl")), find(_jsonl(d / "per_question.jsonl"))
    if retrieval is None:
        raise HTTPException(404, f"question {qid} not in run {run_id}")
    cfg = _json(d / "config.json")["data"]
    con = _corpus()
    cur = con.execute("SELECT * EXCLUDE (dataset) FROM questions WHERE dataset=? AND question_id=?", [cfg["dataset"], qid])
    question = dict(zip([c[0] for c in cur.description], cur.fetchone()))
    doc_ids = set(question["expected_doc_ids"]) | {c["doc_id"] for c in retrieval["chunks"]}
    doc_ids |= {c["doc_id"] for c in (reranked or {}).get("chunks", [])} | set((answer or {}).get("document_ids") or [])
    cfg = {**cfg, "snapshot": cfg["snapshot"] or 1}
    sql = f"SELECT doc_id, source, title, path FROM ({data.docs_sql()}) WHERE doc_id IN (SELECT unnest($ids))"
    docs = {r[0]: {"doc_id": r[0], "source": r[1], "title": r[2], "path": r[3]} for r in con.execute(
        sql, {"ds": cfg["dataset"], "snap": cfg["snapshot"], "subset": "all", "ids": list(doc_ids)}).fetchall()}
    con.close()
    chunk_texts = {c["chunk_id"]: c.get("text", "") for c in (reranked or retrieval)["chunks"]}
    strip = lambda chunks: [{k: v for k, v in c.items() if k != "text"} for c in chunks]  # noqa: E731
    return {"dataset": cfg["dataset"], "snapshot": cfg["snapshot"], "question": question, "retrieval": strip(retrieval["chunks"]), "reranked": strip(reranked["chunks"]) if reranked else None,
            "generation": generation, "answer": answer, "metrics": metrics, "docs": docs,
            "chunk_texts": chunk_texts}


@app.get("/api/doc")
def get_doc(dataset: str, doc_id: str, snapshot: int):
    """A document as it was in the given snapshot (doc ids may contain slashes, hence query params)."""
    con = _corpus()
    r = con.execute(f"SELECT doc_id, source, path, title, text FROM ({data.docs_sql()}) WHERE doc_id = $id",
                    {"ds": dataset, "snap": snapshot, "subset": "all", "id": doc_id}).fetchone()
    con.close()
    if not r:
        raise HTTPException(404, "doc not found")
    return dict(zip(["doc_id", "source", "path", "title", "text"], r))


@app.get("/api/heatmap")
def heatmap(metric: str = "recall@10", dataset: str | None = None):
    """Runs × categories for one metric, for one dataset (categories differ across datasets)."""
    all_ds = [r["dataset"] for r in _query("SELECT DISTINCT dataset FROM runs WHERE kind = 'benchmark' ORDER BY 1")]
    dataset = dataset or (all_ds[0] if all_ds else None)
    rows = _query("""SELECT m.run_id, r.description AS name, r.created_at, r.snapshot, m.category, m.value FROM metrics m JOIN runs r USING (run_id)
                     WHERE m.metric = ? AND r.dataset = ? AND r.kind = 'benchmark' ORDER BY r.created_at DESC""", [metric, dataset])
    metrics = [r["metric"] for r in _query("SELECT DISTINCT metric FROM metrics ORDER BY 1")]
    return {"metric": metric, "metrics": metrics, "dataset": dataset, "datasets": all_ds, "cells": rows}


@app.get("/api/diff")
def diff(a: str, b: str, metric: str | None = None):
    """Per-question comparison of two runs on one metric, with a paired significance test
    (stats.py). Default metric: correctness if both runs were judged, else recall@10."""
    da, db = (_json(_run_dir(x) / "config.json")["data"] for x in (a, b))
    if da["dataset"] != db["dataset"]:
        raise HTTPException(400, f"Can't compare runs on different datasets ({da['dataset']} vs {db['dataset']})")
    pa, pb = (_jsonl(_run_dir(x) / "per_question.jsonl") for x in (a, b))
    if metric is None:
        metric = "correctness" if pa and pb and "correctness" in pa[0] and "correctness" in pb[0] else "recall@10"
    ib = {r["question_id"]: r for r in pb}
    texts = _question_texts(da["dataset"])
    rows = []
    for ra in pa:
        rb = ib.get(ra["question_id"])
        if rb is None:
            continue
        va, vb = ra.get(metric), rb.get(metric)
        status = "n/a" if va is None or vb is None else "fixed" if vb > va else "broken" if vb < va else "same"
        rows.append({"question_id": ra["question_id"], "category": ra["category"], "question": texts.get(ra["question_id"]),
                     "a": va, "b": vb, "status": status})
    counts = {s: sum(r["status"] == s for r in rows) for s in ["fixed", "broken", "same", "n/a"]}
    scored = [r for r in rows if r["a"] is not None and r["b"] is not None]
    test = stats.paired([r["a"] for r in scored], [r["b"] for r in scored])
    return {"metric": metric, "test": test, "snapshots": [da.get("snapshot"), db.get("snapshot")], "metrics": sorted({k for r in pa[:1] for k in r} - {"question_id", "category", "judge_reasoning"}),
            "counts": counts, "rows": rows}


# ----------------------------------------------------------------------------- catalogue

@app.get("/api/catalog")
def catalog():
    """Everything the UI needs to draw forms: stage implementations and target
    types with the JSON schema of their settings (generated from Pydantic classes)."""
    from . import config  # noqa: F401  (registers implementations)
    from .registry import REGISTRY
    stages = {}
    for stage in ("transform", "chunk", "embed", "retrieve", "rerank", "generate"):
        impls = REGISTRY[stage]
        stages[stage] = [{"impl": name, "doc": (impl.fn.__doc__ or "").strip().split("\n\n")[0],
                          "schema": impl.config_cls.model_json_schema(), "defaults": impl.config_cls().model_dump(),
                          "source": inspect.getsource(sys.modules[impl.fn.__module__]),
                          "fn_source": inspect.getsource(impl.fn),       # just this implementation
                          "source_file": f"backend/rag_lab/stages/{impl.fn.__module__.rsplit('.', 1)[-1]}.py",
                          "scriptable": name != "python" and "python" in impls
                                        and scripting.equivalent_script(stage, name, {}) is not None}
                         for name, impl in sorted(impls.items())]
    target_types = [{"impl": name, "doc": (sys.modules[cls.__module__].__doc__ or "").strip(),
                     "schema": cfg_cls.model_json_schema(), "defaults": cfg_cls().model_dump(),
                     "source": inspect.getsource(sys.modules[cls.__module__]),
                     "source_file": f"backend/rag_lab/targets/{cls.__module__.rsplit('.', 1)[-1]}.py"}
                    for name, (cls, cfg_cls) in TARGETS.items()]
    return {"stages": stages, "target_types": target_types, "kinds": workspace.KINDS, "sources": data.SOURCES,
            "templates": {k: c["template"] for k, c in scripting.CONTRACTS.items()},
            "contracts": {k: {"required": c["required"], "optional": c["optional"]} for k, c in scripting.CONTRACTS.items()},
            "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "bench_ready": bench.is_ready()}


@app.post("/api/script-equivalent")
def script_equivalent(body: dict = Body(...)):
    """Params for a python node equivalent to a built-in node: {"stage", "impl", "params"}."""
    out = scripting.equivalent_script(body["stage"], body["impl"], body.get("params") or {})
    if out is None:
        raise HTTPException(404, "no script equivalent for this node")
    return out


def _guard(fn):
    try:
        return fn()
    except (workspace.GraphError, TargetError, scripting.ScriptError) as e:
        raise HTTPException(400, str(e))


def _seed() -> None:
    """Example target and pipelines. On a fresh install there is no dataset yet, so the example
    ingestion pipeline is added later, once the first dataset with questions has been loaded."""
    if workspace.seeding_done():
        return
    first = max((d for d in datasets() if d["n_questions"]), key=lambda d: d["n_questions"], default=None)
    workspace.seed(first["name"] if first else None, "dev" if first and "dev" in first["subsets"] else "all")


# ----------------------------------------------------------------------------- code editor

@app.get("/api/scripts/versions")
def script_versions(key: str, code: bool = False):
    """Saved versions of one script, newest first (see versions.py for keys); code=true includes each one's code."""
    try:
        return list(reversed(versions.history(key))) if code else versions.listing(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/scripts/versions/{version}")
def script_version(key: str, version: int):
    try:
        return versions.get(key, version)
    except (ValueError, KeyError):
        raise HTTPException(404, f"no version {version} of {key}")


@app.get("/api/scripts/info")
def scripts_info():
    """What the code editor needs: each script kind's contract and template, and whether AI help is on."""
    from . import assist
    return {"contracts": scripting.CONTRACTS, "assist": assist.available(),
            "assist_model": os.environ.get("RAG_LAB_ASSIST_MODEL", "claude-sonnet-5-5")}


@app.post("/api/scripts/check")
def check_script(body: dict = Body(...)):
    """{"kind", "code"} -> {"ok", "message", "line"}. Parses only; nothing is executed."""
    return scripting.check(body.get("kind", ""), body.get("code", ""))


@app.post("/api/assist")
def assist_code(body: dict = Body(...)):
    """{"kind", "code", "instruction", "error"?} -> {"code", "explanation", "model", "cost_usd"}."""
    from . import assist
    if not assist.available():
        raise HTTPException(400, "AI help needs ANTHROPIC_API_KEY in .env (then restart rag-lab).")
    if not (body.get("instruction") or "").strip():
        raise HTTPException(400, "Say what you want changed.")
    try:
        return assist.suggest(body.get("kind", ""), body.get("code", ""), body["instruction"], body.get("error"))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- targets

@app.get("/api/services")
def service_status():
    """Database services (docker-compose) and whether each is reachable right now."""
    from . import services
    return services.status()


@app.post("/api/targets/test")
def test_target(body: dict = Body(...)):
    """Try to connect with unsaved settings: {"type", "params"} -> {"ok", "message"}."""
    t = None
    try:
        t = open_target({"id": "_connection_test", "type": body["type"], "params": body.get("params") or {}})
        n = t.count()
        return {"ok": True, "message": f"Connected. {n:,} chunks stored." if n is not None else "Connected."}
    except Exception as e:
        return {"ok": False, "message": f"{type(e).__name__}: {e}"}
    finally:
        if t is not None:
            t.close()
        shutil.rmtree(paths.TARGETS / "_connection_test", ignore_errors=True)

def _target_view(t: dict) -> dict:
    m = Manifest(paths.TARGETS / t["id"]).data
    return {**t, "version": m["version"], "n_docs": m["n_docs"], "n_chunks": m["n_chunks"], "dataset": m["dataset"],
            "subset": m["subset"], "snapshot": m["snapshot"], "embed": m["embed"], "history": m["history"][::-1][:15]}


@app.get("/api/targets")
def list_targets():
    _seed()
    return [_target_view(t) for t in workspace.list_targets()]


@app.put("/api/targets")
def save_target(body: dict = Body(...)):
    def go():
        t = workspace.save_target(body)
        if isinstance((t.get("params") or {}).get("code"), str):
            versions.record(f"targets/{t['id']}", t["params"]["code"], note="target script")
        return _target_view(t)
    return _guard(go)


@app.delete("/api/targets/{target_id}")
def delete_target(target_id: str, purge: bool = False):
    """Remove the definition. purge=true also clears the store and deletes its local files."""
    def go():
        t = workspace.get_target(target_id)
        if purge:
            store = open_target(t)
            try:
                store.clear()
            finally:
                store.close()
            shutil.rmtree(paths.TARGETS / t["id"], ignore_errors=True)
        workspace.delete_target(target_id)
        return {"ok": True}
    return _guard(go)


# ----------------------------------------------------------------------------- pipelines

@app.get("/api/graphs/{kind}")
def graphs(kind: str):
    _seed()
    return _guard(lambda: workspace.list_graphs(kind))


@app.get("/api/graphs/{kind}/{graph_id}")
def graph(kind: str, graph_id: str):
    try:
        return workspace.get_graph(kind, graph_id)
    except workspace.GraphError as e:
        raise HTTPException(404, str(e))


@app.put("/api/graphs/{kind}")
def save_graph(kind: str, body: dict = Body(...)):
    def go():
        g = workspace.save_graph(kind, body)
        versions.record_graph(kind, g)          # every changed script gets a new version
        return g
    return _guard(go)


@app.delete("/api/graphs/{kind}/{graph_id}")
def delete_graph(kind: str, graph_id: str):
    workspace.delete_graph(kind, graph_id)
    return {"ok": True}


# ----------------------------------------------------------------------------- jobs

@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if not (paths.JOBS / f"{job_id}.json").exists():
        raise HTTPException(404, "job not found")
    job = jobs.get(job_id)
    job["log"] = jobs.log_tail(job_id)
    if job.get("run_id"):
        d = paths.RUNS / job["run_id"]
        info = _json(d / "run.json") or {}
        job["run"] = {"current_stage": info.get("current_stage"), "failed_stage": info.get("failed_stage"), "stages": _json(d / "stages.json", [])}
    return job


@app.get("/api/graphs/{kind}/{graph_id}/history")
def graph_history(kind: str, graph_id: str, limit: int = 30):
    """Every run of one pipeline, newest first: the job (status, error) plus its run's steps,
    so the UI can draw each past run on the pipeline's nodes."""
    job_kind = {"ingestion": "ingest", "retrieval": "benchmark"}.get(kind)
    if not job_kind:
        raise HTTPException(404, f"unknown pipeline kind {kind!r}")
    out = []
    for f in sorted(paths.JOBS.glob("*.json"), key=lambda f: f.name, reverse=True):
        job = jobs.get(f.stem)
        if job["kind"] != job_kind or (job.get("spec") or {}).get("meta", {}).get("pipeline_id") != graph_id:
            continue
        job["spec"] = {"meta": job["spec"].get("meta", {})}          # the full config is not needed here
        if job.get("run_id"):
            info = _json(paths.RUNS / job["run_id"] / "run.json") or {}
            job["run"] = {"current_stage": info.get("current_stage"), "failed_stage": info.get("failed_stage"),
                          "stages": _json(paths.RUNS / job["run_id"] / "stages.json", [])}
        out.append(job)
        if len(out) >= limit:
            break
    return out


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str):
    if not (paths.JOBS / f"{job_id}.json").exists():
        raise HTTPException(404, "job not found")
    return {"log": jobs.log_tail(job_id, 40000)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return jobs.cancel(job_id)


@app.post("/api/jobs/ingest")
def start_ingest(body: dict = Body(...)):
    """Run an ingestion pipeline: {"pipeline_id", "node_id"?}. Without node_id, every target node runs."""
    def go():
        p = workspace.get_graph("ingestion", body["pipeline_id"])
        nodes = [n for n in workspace.target_nodes(p) if body.get("node_id") in (None, n["id"])]
        if not nodes:
            raise workspace.GraphError("this pipeline has no target node to write to")
        compiled = [workspace.compile_ingest(p, n["id"]) for n in nodes]        # validate all before queueing any
        return [jobs.submit("ingest", f"{cfg['description']}  [{cfg['target']['mode']}]", {"config": cfg, "meta": meta})
                for cfg, meta in compiled]
    return _guard(go)


@app.post("/api/jobs/benchmark")
def start_benchmark(body: dict = Body(...)):
    """Benchmark a retrieval pipeline: {"pipeline_id", "target_ids"?, "limit"?, "judge"?}.
    target_ids runs the same pipeline against several targets (default: the pipeline's own)."""
    def go():
        p = workspace.get_graph("retrieval", body["pipeline_id"])
        out = []
        for tid in body.get("target_ids") or [None]:
            cfg, meta, limit = workspace.compile_benchmark(p, tid)
            if "judge" in body:
                cfg["evaluate"]["judge"] = body["judge"]
            out.append(jobs.submit("benchmark", cfg["description"],
                                   {"config": cfg, "meta": meta, "limit": body["limit"] if "limit" in body else limit}))
        return out
    return _guard(go)


@app.get("/api/matrix")
def matrix(dataset: str | None = None):
    """Latest successful benchmark for every (target, retrieval pipeline) pair."""
    cells = {}
    for r in reversed(_runs("benchmark")):        # oldest first, so later runs overwrite
        m = r["meta"]
        if r["status"] == "ok" and m.get("pipeline_id") and (dataset is None or r["dataset"] == dataset):
            cells[(m["target_id"], m["pipeline_id"])] = {
                "run_id": r["run_id"], "target_id": m["target_id"], "target": m["target_name"],
                "pipeline_id": m["pipeline_id"], "pipeline": m["pipeline_name"], "dataset": r["dataset"],
                "target_version": r["params"].get("target.version"), "n_questions": r["n_questions"], "judged": r["judged"],
                "overall": r["overall"], "cost_usd": r["cost_usd"], "created_at": str(r["created_at"])}
    return list(cells.values())


# ----------------------------------------------------------------------------- datasets

def _dataset_dir(name: str) -> Path:
    try:
        return paths.RAW if name == data.BENCH else data.dataset_dir(name)
    except SystemExit as e:
        raise HTTPException(400, str(e))


def _inside(root: Path, rel: str) -> Path:
    """Resolve a path from the browser inside a dataset folder; refuse anything outside it."""
    target = (root / (rel or "")).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise HTTPException(400, f"path {rel!r} is outside the dataset folder")
    return target


@app.get("/api/datasets/{name}/tree")
def dataset_tree(name: str, path: str = ""):
    """One directory level of a dataset folder, for the file explorer."""
    root = _dataset_dir(name)
    d = _inside(root, path)
    if not d.is_dir():
        return {"path": path, "entries": [], "exists": False}
    entries = []
    for f in sorted(d.iterdir(), key=lambda f: (f.is_file(), f.name.lower())):
        if f.name.startswith("."):
            continue
        st = f.stat()
        entry = {"name": f.name, "path": f.relative_to(root).as_posix(), "dir": f.is_dir(),
                 "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}
        if f.is_dir():
            kids = [k for k in f.rglob("*") if k.is_file()]
            entry |= {"files": len(kids), "size": sum(k.stat().st_size for k in kids)}
        else:
            entry["size"] = st.st_size
        entries.append(entry)
    return {"path": path, "exists": True, "root": str(root), "entries": entries}


@app.get("/api/datasets/{name}/file")
def dataset_file(name: str, path: str, max_bytes: int = 65536):
    """Preview a file: the start of a text file, or the member list of a zip."""
    f = _inside(_dataset_dir(name), path)
    if not f.is_file():
        raise HTTPException(404, "file not found")
    size = f.stat().st_size
    if f.suffix == ".zip":
        with zipfile.ZipFile(f) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            members = [{"name": i.filename, "size": i.file_size} for i in infos[:300]]
            return {"kind": "zip", "size": size, "members": members, "n_members": len(infos)}
    with open(f, "rb") as fh:
        head = fh.read(max_bytes)
    if b"\x00" in head[:4096]:
        return {"kind": "binary", "size": size}
    return {"kind": "text", "size": size, "text": head.decode(errors="replace"), "truncated": size > max_bytes}


@app.post("/api/datasets/{name}/upload")
def upload(name: str, files: list[UploadFile] = File(...), path: str = ""):
    """Save uploaded files into the dataset folder (under `path`). A .zip is unpacked in place."""
    root = _dataset_dir(name)
    d = _inside(root, path)
    d.mkdir(parents=True, exist_ok=True)
    saved = 0
    for up in files:
        rel = Path(up.filename or "file").as_posix().lstrip("/")
        target = _inside(d, rel)
        if rel.endswith(".zip"):
            with zipfile.ZipFile(up.file) as z:
                for member in z.namelist():
                    if member.endswith("/") or member.startswith("__MACOSX"):
                        continue
                    out = _inside(d, member)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(member))
                    saved += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                shutil.copyfileobj(up.file, fh)
            saved += 1
    return {"saved": saved}


@app.delete("/api/datasets/{name}/file")
def delete_file(name: str, path: str):
    root = _dataset_dir(name)
    f = _inside(root, path)
    if f == root.resolve():
        raise HTTPException(400, "can't delete the dataset folder itself here")
    if f.is_dir():
        shutil.rmtree(f)
    elif f.is_file():
        f.unlink()
    return {"ok": True}


@app.post("/api/datasets/{name}/folder")
def create_folder(name: str):
    """Create an empty dataset folder to put files in."""
    d = _dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(d)}


@app.get("/api/datasets/{name}/extractor")
def get_extractor(name: str):
    f = _dataset_dir(name) / data.EXTRACTOR_FILE
    return {"code": f.read_text() if f.is_file() else scripting.EXTRACT_TEMPLATE, "saved": f.is_file()}


@app.get("/api/datasets/{name}/metrics")
def dataset_metrics(name: str):
    if not paths.CORPUS_DB.exists():
        raise HTTPException(404, "nothing loaded yet")
    con = _corpus()
    try:
        return data.dataset_metrics(con, name)
    except SystemExit as e:
        raise HTTPException(404, str(e))
    finally:
        con.close()


@app.post("/api/datasets/{name}/ingest")
def ingest_dataset(name: str, body: dict = Body(default={})):
    """Queue a dataset job: {"source": "extract" (download: run the extractor into the folder) |
    "folder" (load: read the folder into the internal document store), "params": {"code"?}, "sync"?}.
    Loading records a snapshot holding only what changed; questions are required."""
    _dataset_dir(name)
    source = body.get("source", "folder")
    if source not in data.SOURCES:
        raise HTTPException(400, f"source must be one of {sorted(data.SOURCES)}")
    if source == "extract":
        code = (body.get("params") or {}).get("code", "")
        check = scripting.check("extract", code)
        if not check["ok"]:
            raise HTTPException(400, check["message"])
        versions.record(f"datasets/{name}", code, note="extractor")
    title = {"enterprise_rag_bench": "Download + load", "extract": "Download", "folder": "Load into store"}[source]
    return jobs.submit("dataset", f"{title}: {name}",
                       {"dataset": name, "source": source, "params": body.get("params") or {}, "sync": bool(body.get("sync"))})


@app.delete("/api/datasets/{name}")
def remove_dataset(name: str):
    """Queue removal of a dataset from the document store. Its folder is kept."""
    _dataset_dir(name)
    return jobs.submit("dataset", f"Remove dataset {name}", {"dataset": name, "action": "remove"})


@app.post("/api/datasets/{name}/subsets")
def create_subset(name: str, body: dict = Body(...)):
    """Queue building a subset: {"name", "per_category"?, "distractors"?, "seed"?}: per_category questions
    from each question category (or in total, without categories), their gold documents, and
    `distractors` random other documents. A subset with the same name is replaced."""
    sub = str(body.get("name", ""))
    if sub == "all" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sub):
        raise HTTPException(400, "subset name: letters, digits, _ and - only, and not 'all'")
    try:
        params = {"name": sub, **{k: int(body[k]) for k in ("per_category", "distractors", "seed") if body.get(k) is not None}}
    except (TypeError, ValueError):
        raise HTTPException(400, "per_category, distractors and seed must be whole numbers")
    if params.get("per_category", 10) < 1 or params.get("distractors", 0) < 0:
        raise HTTPException(400, "per_category must be at least 1 and distractors at least 0")
    return jobs.submit("dataset", f"Build subset {sub}: {name}", {"dataset": name, "action": "subset", "params": params})


@app.delete("/api/datasets/{name}/subsets/{subset}")
def delete_subset(name: str, subset: str):
    if subset == "all":
        raise HTTPException(400, "'all' is not a stored subset")
    return jobs.submit("dataset", f"Remove subset {subset}: {name}", {"dataset": name, "action": "remove_subset", "params": {"name": subset}})


@app.get("/api/datasets/{name}/docs")
def sample_docs(name: str, q: str = "", limit: int = 20):
    """A few documents (optionally title-filtered) to pick from for previews."""
    con = _corpus()
    try:
        snap = data.latest_snapshot(con, name)
        rows = con.execute(f"""SELECT doc_id, source, title FROM ({data.docs_sql()})
                               WHERE $q = '' OR title ILIKE '%' || $q || '%' ORDER BY hash(doc_id) LIMIT $limit""",
                           {"ds": name, "snap": snap, "subset": "all", "q": q, "limit": limit}).fetchall()
    except SystemExit as e:
        raise HTTPException(404, str(e))
    finally:
        con.close()
    return [{"doc_id": r[0], "source": r[1], "title": r[2]} for r in rows]


@app.get("/api/datasets/{name}/questions")
def dataset_questions(name: str, limit: int = 50):
    con = _corpus()
    try:
        rows = con.execute("""SELECT question_id, category, question, gold_answer, len(expected_doc_ids) FROM questions
                              WHERE dataset=? ORDER BY question_id LIMIT ?""", [name, limit]).fetchall()
    finally:
        con.close()
    return [dict(zip(["question_id", "category", "question", "gold_answer", "n_gold_docs"], r)) for r in rows]


# ----------------------------------------------------------------------------- previews

@app.post("/api/preview/chunk")
def preview_chunk(body: dict = Body(...)):
    """Chunk one document with a node's settings, without running anything:
    {"impl", "params", "dataset", "doc_id"?}"""
    from .registry import get as get_impl
    from .stages import chunk
    con = _corpus()
    try:
        snap = data.latest_snapshot(con, body["dataset"])
        where = "WHERE doc_id = $id" if body.get("doc_id") else "ORDER BY hash(doc_id || $salt) LIMIT 1"
        params = {"ds": body["dataset"], "snap": snap, "subset": "all"} | (
            {"id": body["doc_id"]} if body.get("doc_id") else {"salt": str(time.time())})
        r = con.execute(f"SELECT doc_id, source, title, text FROM ({data.docs_sql()}) {where}", params).fetchone()
    except SystemExit as e:
        raise HTTPException(404, str(e))
    finally:
        con.close()
    if not r:
        raise HTTPException(404, "document not found")
    doc = dict(zip(["doc_id", "source", "title", "text"], r))
    try:
        cfg = get_impl("chunk", body["impl"]).config_cls(**(body.get("params") or {}))
        chunks = chunk.preview(body["impl"], cfg, doc)
    except Exception as e:      # user script errors are expected here: show them
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"doc": {**doc, "chars": len(doc["text"])}, "chunks": chunks}


@app.post("/api/preview/transform")
def preview_transform(body: dict = Body(...)):
    """Run a transform node on one document: {"impl", "params", "dataset", "doc_id"?} -> before / after."""
    from .registry import get as get_impl
    from .stages import transform
    con = _corpus()
    try:
        snap = data.latest_snapshot(con, body["dataset"])
        where = "WHERE doc_id = $id" if body.get("doc_id") else "ORDER BY hash(doc_id || $salt) LIMIT 1"
        params = {"ds": body["dataset"], "snap": snap, "subset": "all"} | (
            {"id": body["doc_id"]} if body.get("doc_id") else {"salt": str(time.time())})
        r = con.execute(f"""SELECT doc_id, source, title, text, CAST(metadata AS VARCHAR)
                            FROM ({data.docs_sql('v.doc_id, v.source, v.metadata, c.title, c.text')}) {where}""", params).fetchone()
    except SystemExit as e:
        raise HTTPException(404, str(e))
    finally:
        con.close()
    if not r:
        raise HTTPException(404, "document not found")
    doc = {"doc_id": r[0], "source": r[1], "title": r[2] or "", "text": r[3], "metadata": json.loads(r[4]) if r[4] else {}}
    if body["impl"] == "none":
        return {"before": doc, "after": doc}
    try:
        cfg = get_impl("transform", body["impl"]).config_cls(**(body.get("params") or {}))
        after = transform.make_transform(body["impl"], cfg)(dict(doc))
    except Exception as e:      # user script errors are expected here: show them
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"before": doc, "after": None if after is None else {**doc, **after}}


@app.post("/api/preview/query")
def preview_query(body: dict = Body(...)):
    """Try one query against a target with a retrieval pipeline (saved or unsaved):
    {"pipeline": <graph>, "target_id"?, "query"}. Runs retrieve and rerank only."""
    from .registry import get as get_impl
    from .stages.base import StageContext
    cfg, _, _ = _guard(lambda: workspace.compile_benchmark(body["pipeline"], body.get("target_id")))
    target_def = workspace.get_target(cfg["target"]["id"])
    manifest = Manifest(paths.TARGETS / target_def["id"]).data
    if not manifest["version"]:
        raise HTTPException(409, f"Target {target_def['name']!r} is empty. Run an ingestion pipeline that writes to it first.")
    t0, dirs = time.time(), {}
    target = _guard(lambda: open_target(target_def))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for stage in ("retrieve", "rerank"):
                out = Path(tmp) / stage
                out.mkdir()
                impl = get_impl(stage, cfg[stage]["impl"])
                ctx = StageContext(out=out, dirs=dirs, metas={}, fingerprint="preview", target=target, embed=manifest["embed"],
                                   questions=[{"question_id": "q", "question": body["query"]}])
                impl.fn(impl.config_cls(**{k: v for k, v in cfg[stage].items() if k != "impl"}), ctx)
                dirs[stage] = out
            hits = _jsonl(Path(tmp) / "rerank" / "reranked.jsonl")[0]["chunks"]
        except (Exception, SystemExit) as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        finally:
            target.close()
    return {"seconds": round(time.time() - t0, 2), "target": target_def["name"], "hits": hits[:30]}


# Serve the built React app (web/dist) if present; `npm run dev` proxies /api instead.
if paths.WEB_DIST.exists():
    @app.get("/{path:path}")
    def spa(path: str):
        f = (paths.WEB_DIST / path).resolve()
        if path and f.is_file() and paths.WEB_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(paths.WEB_DIST / "index.html")
