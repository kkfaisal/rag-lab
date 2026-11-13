"""Read-only HTTP API over runs/ and the corpus. Started by `rag-lab serve`."""

import json
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import cache, data, paths, tracking

app = FastAPI(title="rag-lab")


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
    return duckdb.connect(str(paths.CORPUS_DB), read_only=True)


def _question_texts(dataset: str) -> dict[str, str]:
    con = _corpus()
    try:
        return dict(con.execute("SELECT question_id, question FROM questions WHERE dataset=?", [dataset]).fetchall())
    finally:
        con.close()


@app.get("/api/datasets")
def datasets():
    con = _corpus()
    try:
        return data.dataset_info(con)
    finally:
        con.close()


@app.get("/api/runs")
def list_runs():
    rows = _query("""SELECT run_id, name, description, created_at, status, dataset, snapshot, subset, n_questions, "limit", git_sha,
                     git_dirty, duration_s, cost_usd, judged, error, params, overall FROM runs ORDER BY created_at DESC""")
    for r in rows:
        r["params"], r["overall"] = json.loads(r["params"]), json.loads(r["overall"])
    return rows


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
    all_ds = [r["dataset"] for r in _query("SELECT DISTINCT dataset FROM runs ORDER BY 1")]
    dataset = dataset or (all_ds[0] if all_ds else None)
    rows = _query("""SELECT m.run_id, r.name, r.created_at, r.snapshot, m.category, m.value FROM metrics m JOIN runs r USING (run_id)
                     WHERE m.metric = ? AND r.dataset = ? ORDER BY r.created_at DESC""", [metric, dataset])
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


# Serve the built React app (web/dist) if present; `npm run dev` proxies /api instead.
if paths.WEB_DIST.exists():
    @app.get("/{path:path}")
    def spa(path: str):
        f = (paths.WEB_DIST / path).resolve()
        if path and f.is_file() and paths.WEB_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(paths.WEB_DIST / "index.html")
