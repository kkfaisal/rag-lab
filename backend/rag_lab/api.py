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

import duckdb
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import bench, cache, data, jobs, paths, scripting, tracking, workspace

load_dotenv(paths.ROOT / ".env")
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
    if not paths.CORPUS_DB.exists():
        return []
    con = _corpus()
    try:
        return data.dataset_info(con)
    finally:
        con.close()


def _runs(kind: str) -> list[dict]:
    rows = _query("""SELECT run_id, name, description, created_at, status, kind, meta, dataset, snapshot, subset, n_questions,
                     "limit", git_sha, git_dirty, duration_s, cost_usd, judged, error, params, overall
                     FROM runs WHERE kind = ? ORDER BY created_at DESC""", [kind])
    for r in rows:
        r["params"], r["overall"], r["meta"] = json.loads(r["params"]), json.loads(r["overall"]), json.loads(r["meta"] or "{}")
    return rows


@app.get("/api/runs")
def list_runs():
    return _runs("experiment")


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


@app.get("/api/runs/{run_id}/questions")
def run_questions(run_id: str):
    d = _run_dir(run_id)
    per_q = _jsonl(d / "per_question.jsonl")
    answers = {a["question_id"]: a for a in _jsonl(d / "answers.jsonl")}
    texts = _question_texts(_json(d / "config.json")["data"]["dataset"])
    return [{**r, "question": texts.get(r["question_id"]), "answer": (answers.get(r["question_id"]) or {}).get("answer")}
            for r in per_q]


def _chunk_texts(run_dir: Path, chunk_ids: list[str]) -> dict[str, str]:
    """Chunk text lives in the index stage's cache entry, if it still exists."""
    stages = {s["stage"]: s for s in _json(run_dir / "stages.json", [])}
    if "index" not in stages:
        return {}
    db = cache.stage_dir("index", stages["index"]["key"]) / "index.duckdb"
    if not db.exists():
        return {}
    con = duckdb.connect(str(db), read_only=True)
    try:
        return dict(con.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [chunk_ids]).fetchall())
    finally:
        con.close()


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
    sql = f"SELECT doc_id, source, title, path FROM ({data.docs_sql()}) WHERE doc_id IN (SELECT unnest($ids))"
    docs = {r[0]: {"doc_id": r[0], "source": r[1], "title": r[2], "path": r[3]} for r in con.execute(
        sql, {"ds": cfg["dataset"], "snap": cfg["snapshot"], "subset": "all", "ids": list(doc_ids)}).fetchall()}
    con.close()
    chunk_ids = [c["chunk_id"] for c in (reranked or retrieval)["chunks"]]
    return {"dataset": cfg["dataset"], "snapshot": cfg["snapshot"], "question": question, "retrieval": retrieval["chunks"], "reranked": (reranked or {}).get("chunks"),
            "generation": generation, "answer": answer, "metrics": metrics, "docs": docs,
            "chunk_texts": _chunk_texts(d, chunk_ids)}


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
    all_ds = [r["dataset"] for r in _query("SELECT DISTINCT dataset FROM runs WHERE kind = 'experiment' ORDER BY 1")]
    dataset = dataset or (all_ds[0] if all_ds else None)
    rows = _query("""SELECT m.run_id, r.name, r.created_at, r.snapshot, m.category, m.value FROM metrics m JOIN runs r USING (run_id)
                     WHERE m.metric = ? AND r.dataset = ? AND r.kind = 'experiment' ORDER BY r.created_at DESC""", [metric, dataset])
    metrics = [r["metric"] for r in _query("SELECT DISTINCT metric FROM metrics ORDER BY 1")]
    return {"metric": metric, "metrics": metrics, "dataset": dataset, "datasets": all_ds, "cells": rows}


@app.get("/api/diff")
def diff(a: str, b: str, metric: str | None = None):
    """Per-question comparison of two runs on one metric. Default metric:
    correctness if both runs were judged, else answer_recall."""
    da, db = (_json(_run_dir(x) / "config.json")["data"] for x in (a, b))
    if da["dataset"] != db["dataset"]:
        raise HTTPException(400, f"Can't compare runs on different datasets ({da['dataset']} vs {db['dataset']})")
    pa, pb = (_jsonl(_run_dir(x) / "per_question.jsonl") for x in (a, b))
    if metric is None:
        metric = "correctness" if pa and pb and "correctness" in pa[0] and "correctness" in pb[0] else "answer_recall"
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
    return {"metric": metric, "snapshots": [da["snapshot"], db["snapshot"]], "metrics": sorted({k for r in pa[:1] for k in r} - {"question_id", "category", "judge_reasoning"}),
            "counts": counts, "rows": rows}


# ----------------------------------------------------------------------------- playground: catalogue

SOURCE_SCHEMA = {"properties": {
    "dataset": {"type": "string", "title": "Dataset"},
    "subset": {"type": "string", "title": "Subset", "default": "all"},
    "snapshot": {"title": "Snapshot", "default": "latest"},
    "sources": {"title": "Only these source types (optional)", "type": "array", "items": {"type": "string"}, "default": None}}}


@app.get("/api/catalog")
def catalog():
    """Every node type the canvases offer: implementations per stage with the JSON
    schema of their settings (generated from the Pydantic config classes)."""
    from . import config  # noqa: F401  (registers implementations)
    from .registry import REGISTRY
    stages = {"source": [{"impl": "dataset", "doc": "Documents of a dataset snapshot.", "schema": SOURCE_SCHEMA,
                          "defaults": {"dataset": "", "subset": "all", "snapshot": "latest", "sources": None}}]}
    for stage, impls in REGISTRY.items():
        if stage in ("load", "evaluate"):
            continue
        stages[stage] = [{"impl": name, "doc": (impl.fn.__doc__ or "").strip().split("\n\n")[0],
                          "schema": impl.config_cls.model_json_schema(), "defaults": impl.config_cls().model_dump(),
                          "source": inspect.getsource(sys.modules[impl.fn.__module__]),      # the module that implements it
                          "source_file": f"backend/rag_lab/stages/{impl.fn.__module__.rsplit('.', 1)[-1]}.py",
                          "scriptable": name != "python" and "python" in impls
                                        and scripting.equivalent_script(stage, name, {}) is not None}
                         for name, impl in sorted(impls.items())]
    return {"stages": stages, "kinds": workspace.KINDS, "templates": {"extract": scripting.EXTRACT_TEMPLATE},
            "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "bench_ready": bench.is_ready()}


@app.post("/api/script-equivalent")
def script_equivalent(body: dict = Body(...)):
    """Params for a python node equivalent to a built-in node: {"stage", "impl", "params"}."""
    out = scripting.equivalent_script(body["stage"], body["impl"], body.get("params") or {})
    if out is None:
        raise HTTPException(404, "no script equivalent for this node")
    return out


# ----------------------------------------------------------------------------- playground: graphs

def _graph_errors(fn):
    try:
        return fn()
    except workspace.GraphError as e:
        raise HTTPException(400, str(e))


@app.get("/api/graphs/{kind}")
def graphs(kind: str):
    if not workspace.list_graphs("retriever"):
        ds = datasets()
        first = next((d for d in ds if d["n_questions"]), None)
        workspace.seed(first["name"] if first else None, "dev" if first and "dev" in first["subsets"] else "all")
    return _graph_errors(lambda: workspace.list_graphs(kind))


@app.get("/api/graphs/{kind}/{graph_id}")
def graph(kind: str, graph_id: str):
    try:
        return workspace.get_graph(kind, graph_id)
    except workspace.GraphError as e:
        raise HTTPException(404, str(e))


@app.put("/api/graphs/{kind}")
def save_graph(kind: str, body: dict = Body(...)):
    return _graph_errors(lambda: workspace.save_graph(kind, body))


@app.delete("/api/graphs/{kind}/{graph_id}")
def delete_graph(kind: str, graph_id: str):
    workspace.delete_graph(kind, graph_id)
    return {"ok": True}


def _index_status(pipeline: dict) -> list[dict]:
    """For each index node: can it be built, and is there an up-to-date build?"""
    builds = [r for r in _runs("build") + _runs("experiment") if r["meta"].get("pipeline_id") == pipeline["id"]]
    out = []
    for node in workspace.index_nodes(pipeline):
        entry = {"pipeline_id": pipeline["id"], "pipeline_name": pipeline["name"], "node_id": node["id"],
                 "label": workspace.node_label(node), "state": "not built", "error": None, "run_id": None, "stages": []}
        try:
            cfg, meta = workspace.build_config(pipeline, node["id"])
            entry["dataset"], entry["subset"] = cfg["data"]["dataset"], cfg["data"]["subset"]
        except workspace.GraphError as e:
            out.append({**entry, "state": "invalid", "error": str(e)})
            continue
        mine = [r for r in builds if r["meta"].get("node_id") == node["id"]]
        fresh = next((r for r in mine if r["status"] == "ok" and r["meta"].get("spec_hash") == meta["spec_hash"]), None)
        if fresh:
            stages = _json(paths.RUNS / fresh["run_id"] / "stages.json", [])
            entry.update(state="built", run_id=fresh["run_id"], built_at=str(fresh["created_at"]), snapshot=fresh["snapshot"],
                         stages=[s for s in stages if s["stage"] in ("load", "chunk", "embed", "index")])
        elif any(r["status"] == "ok" for r in mine):
            entry["state"] = "stale"          # built before the pipeline was edited
        elif mine and mine[0]["status"] == "failed":
            entry.update(state="failed", error=mine[0]["error"], run_id=mine[0]["run_id"])
        out.append(entry)
    return out


@app.get("/api/indexes")
def indexes():
    """Every index node of every pipeline, with its build state."""
    return [s for p in workspace.list_graphs("pipeline") for s in _index_status(p)]


# ----------------------------------------------------------------------------- playground: jobs

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
        job["run"] = {"current_stage": (_json(d / "run.json") or {}).get("current_stage"), "stages": _json(d / "stages.json", [])}
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return jobs.cancel(job_id)


@app.post("/api/jobs/build")
def start_build(body: dict = Body(...)):
    """Build one index node: {"pipeline_id", "node_id"}."""
    def go():
        p = workspace.get_graph("pipeline", body["pipeline_id"])
        cfg, meta = workspace.build_config(p, body["node_id"])
        return jobs.submit("build", f"Build {p['name']} / {meta['node_label']}", {"config": cfg, "meta": meta})
    return _graph_errors(go)


@app.post("/api/jobs/experiments")
def start_experiments(body: dict = Body(...)):
    """Queue one experiment per (index, retriever) pair:
    {"indexes": [{"pipeline_id", "node_id"}], "retriever_ids": [...], "limit": int|null, "judge": "auto"|false|...}"""
    def go():
        out = []
        for ix in body["indexes"]:
            p = workspace.get_graph("pipeline", ix["pipeline_id"])
            for rid in body["retriever_ids"]:
                r = workspace.get_graph("retriever", rid)
                cfg, meta = workspace.experiment_config(p, ix["node_id"], r, body.get("judge", "auto"))
                out.append(jobs.submit("experiment", cfg["description"],
                                       {"config": cfg, "meta": meta, "limit": body.get("limit") or None}))
        return out
    return _graph_errors(go)


@app.get("/api/matrix")
def matrix(dataset: str | None = None):
    """Latest successful experiment for every (index, retrieval setup) pair."""
    cells = {}
    for r in reversed(_runs("experiment")):       # oldest first, so later runs overwrite
        m = r["meta"]
        if r["status"] == "ok" and m.get("retriever_id") and (dataset is None or r["dataset"] == dataset):
            cells[(m["pipeline_id"], m["node_id"], m["retriever_id"])] = {
                "run_id": r["run_id"], "index": f"{m['pipeline_name']} / {m['node_label']}", "index_key": f"{m['pipeline_id']}/{m['node_id']}",
                "retriever_id": m["retriever_id"], "retriever": m["retriever_name"], "dataset": r["dataset"], "snapshot": r["snapshot"],
                "n_questions": r["n_questions"], "judged": r["judged"], "overall": r["overall"], "cost_usd": r["cost_usd"]}
    return list(cells.values())


# ----------------------------------------------------------------------------- playground: datasets

def _upload_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name):
        raise HTTPException(400, "dataset names may only contain letters, digits, _ and -")
    return paths.UPLOADS / name


@app.get("/api/datasets/{name}/files")
def dataset_files(name: str):
    d = _upload_dir(name)
    files = [f for f in sorted(d.rglob("*")) if f.is_file()] if d.exists() else []
    return {"path": str(d), "count": len(files),
            "files": [{"path": f.relative_to(d).as_posix(), "size": f.stat().st_size} for f in files[:500]]}


@app.post("/api/datasets/{name}/upload")
def upload(name: str, files: list[UploadFile] = File(...)):
    """Save uploaded files under data/uploads/<name>/. A .zip is unpacked in place."""
    d = _upload_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    saved = 0
    for up in files:
        rel = Path(up.filename or "file").as_posix().lstrip("/")
        target = (d / rel).resolve()
        if d.resolve() not in target.parents:
            raise HTTPException(400, f"bad file name {up.filename!r}")
        if rel.endswith(".zip"):
            with zipfile.ZipFile(up.file) as z:
                for member in z.namelist():
                    out = (d / member).resolve()
                    if member.endswith("/") or d.resolve() not in out.parents:
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(member))
                    saved += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                shutil.copyfileobj(up.file, f)
            saved += 1
    return {"saved": saved, **dataset_files(name)}


@app.delete("/api/datasets/{name}/files")
def delete_file(name: str, path: str):
    d = _upload_dir(name)
    target = (d / path).resolve()
    if d.resolve() in target.parents and target.is_file():
        target.unlink()
    return dataset_files(name)


@app.post("/api/datasets/{name}/ingest")
def ingest(name: str, body: dict = Body(default={})):
    """Queue an ingest of the dataset's upload folder: {"adapter": "folder"|"script", "code"?, "sync"?}.
    Creates a new snapshot holding only what changed."""
    d = _upload_dir(name)
    if not d.exists():
        raise HTTPException(400, "upload some files first")
    adapter = body.get("adapter", "folder")
    if adapter not in ("folder", "script"):
        raise HTTPException(400, "adapter must be 'folder' or 'script'")
    return jobs.submit("ingest", f"Ingest {name}", {"dataset": name, "path": str(d), "adapter": adapter,
                                                    "code": body.get("code") if adapter == "script" else None,
                                                    "sync": bool(body.get("sync")), "note": body.get("note", "uploaded via UI")})


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


# ----------------------------------------------------------------------------- playground: previews

@app.post("/api/preview/chunk")
def preview_chunk(body: dict = Body(...)):
    """Chunk one document with a node's settings, without running anything:
    {"impl", "params", "dataset", "doc_id"?}"""
    from .registry import get as get_impl
    from .stages import chunk
    con = _corpus()
    try:
        snap = data.latest_snapshot(con, body["dataset"])
        where = "WHERE doc_id = $id" if body.get("doc_id") else "ORDER BY hash(doc_id) LIMIT 1"
        params = {"ds": body["dataset"], "snap": snap, "subset": "all"} | ({"id": body["doc_id"]} if body.get("doc_id") else {})
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


@app.post("/api/preview/query")
def preview_query(body: dict = Body(...)):
    """Try one query against a built index with a retrieval setup (saved or unsaved):
    {"pipeline_id", "node_id", "retriever": <graph>, "query"}. Runs retrieve and rerank only."""
    from .registry import get as get_impl
    from .stages.base import StageContext
    p = _graph_errors(lambda: workspace.get_graph("pipeline", body["pipeline_id"]))
    status = next((s for s in _index_status(p) if s["node_id"] == body["node_id"]), None)
    if not status or status["state"] != "built":
        raise HTTPException(409, "This index isn't built (or is stale). Build it on the Pipelines page first.")
    rcfg = _graph_errors(lambda: workspace.compile_retriever(body["retriever"]))
    dirs = {s["stage"]: cache.stage_dir(s["stage"], s["key"]) for s in status["stages"]}
    if not all((d / "meta.json").exists() for d in dirs.values()):
        raise HTTPException(409, "The index's cache files are gone. Rebuild it.")
    metas = {st: _json(d / "meta.json") for st, d in dirs.items()}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for stage in ("retrieve", "rerank"):
                out = Path(tmp) / stage
                out.mkdir()
                impl = get_impl(stage, rcfg[stage]["impl"])
                ctx = StageContext(out=out, dirs=dirs, metas=metas, corpus=None, dataset=status["dataset"], snapshot=status["snapshot"],
                                   subset=status["subset"], fingerprint="preview",
                                   questions=[{"question_id": "q", "question": body["query"]}])
                impl.fn(impl.config_cls(**{k: v for k, v in rcfg[stage].items() if k != "impl"}), ctx)
                dirs[stage] = out
            hits = _jsonl(Path(tmp) / "rerank" / "reranked.jsonl")[0]["chunks"]
        except (Exception, SystemExit) as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
    con = duckdb.connect(str(dirs["index"] / "index.duckdb"), read_only=True)
    texts = dict(con.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))",
                             [[h["chunk_id"] for h in hits]]).fetchall())
    con.close()
    return {"seconds": round(time.time() - t0, 2),
            "hits": [{**h, "text": texts.get(h["chunk_id"], "")} for h in hits[:30]]}


# Serve the built React app (web/dist) if present; `npm run dev` proxies /api instead.
if paths.WEB_DIST.exists():
    @app.get("/{path:path}")
    def spa(path: str):
        f = (paths.WEB_DIST / path).resolve()
        if path and f.is_file() and paths.WEB_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(paths.WEB_DIST / "index.html")
