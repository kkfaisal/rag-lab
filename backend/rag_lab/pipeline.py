"""Run a config end to end, reusing cached stage outputs, and record the run."""

import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime

from . import bench, cache, data, paths, tracking
from .config import apply_overrides, load_yaml, resolve
from .datasets import ADAPTERS
from .registry import STAGES, get
from .stages.base import StageContext

# Stage stats worth showing next to a node in the UI.
STAT_KEYS = {"n_docs", "n_chunks", "docs_chunked", "docs_reused", "avg_chars", "max_chars", "n", "dim",
             "embedded", "reused", "fts", "vectors", "n_questions", "abstained", "refusals", "judged"}

# Files copied from stage cache dirs into the run folder (small, human-readable).
ARTIFACTS = {
    "retrieve": ["retrieval.jsonl"],
    "rerank": ["reranked.jsonl"],
    "generate": ["answers.jsonl", "generations.jsonl"],
    "evaluate": ["per_question.jsonl", "metrics.json", "judge_results.json", "judge.log"],
}


def _resolve_judge(cfg: dict, evaluator: str, questions: list[dict]) -> None:
    """Turn evaluate.judge = "auto" into "benchmark" | "llm" | false *before*
    hashing, so the cache key and config.json say what actually happened.

    benchmark = EnterpriseRAG-Bench's own judge (only valid for that dataset);
    llm       = the built-in generic judge: Claude compares each answer with the gold answer."""
    ev = cfg["evaluate"]
    want = ev.get("judge")
    if want is True:
        want = "benchmark" if evaluator == "benchmark" else "llm"
    if want == "benchmark" and evaluator != "benchmark":
        raise SystemExit(f"The benchmark judge only applies to EnterpriseRAG-Bench, not {cfg['data']['dataset']!r}")
    kind = "benchmark" if evaluator == "benchmark" else "llm"
    reasons = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        reasons.append("ANTHROPIC_API_KEY not set")
    if cfg["generate"]["impl"] == "none":
        reasons.append("no generate step, so there are no answers to judge")
    if (want or kind) == "benchmark" and not bench.is_ready():
        reasons.append("benchmark judge not installed (rag-lab bench setup)")
    if (want or kind) == "llm" and not any(q.get("gold_answer") for q in questions):
        reasons.append("questions have no gold answers")
    if want == "auto":
        ev["judge"] = False if reasons else kind
        if reasons:
            print(f"  judge: off ({'; '.join(reasons)})")
    elif want in ("benchmark", "llm"):
        if reasons:
            raise SystemExit(f"evaluate.judge={want!r} is not possible: {'; '.join(reasons)}")
        ev["judge"] = want
    else:
        ev["judge"] = False


def run(config_path: str, overrides: list[str] | None = None, limit: int | None = None) -> str:
    """CLI entry: run a YAML config end to end."""
    return run_config(apply_overrides(load_yaml(config_path), overrides or []), limit=limit)


def run_config(raw: dict, limit: int | None = None, stop_after: str | None = None,
               meta: dict | None = None, on_start=None) -> str:
    """Run a config dict. stop_after="index" builds an index only (kind "build");
    otherwise it is a full experiment. `meta` (e.g. which canvas pipeline/node and
    retrieval setup this came from) is stored in run.json. Progress is written to
    the run folder after every stage, so the UI can follow a run live."""
    stages = STAGES[: STAGES.index(stop_after) + 1] if stop_after else STAGES
    kind = "build" if stop_after else "experiment"
    cfg = resolve(raw, stages)

    corpus = data.connect(read_only=True)
    dataset, subset = cfg["data"]["dataset"], cfg["data"]["subset"]
    row = corpus.execute("SELECT adapter FROM datasets WHERE name=?", [dataset]).fetchone()
    if not row:
        raise SystemExit(f"Unknown dataset {dataset!r}. Ingest it first (Datasets page, or `rag-lab data add`).")
    if cfg["data"]["snapshot"] == "latest":
        cfg["data"]["snapshot"] = data.latest_snapshot(corpus, dataset)
    snapshot = int(cfg["data"]["snapshot"])
    qids, questions = [], []
    if kind == "experiment":
        qids = data.subset_question_ids(corpus, dataset, subset, limit)
        questions = data.load_questions(corpus, dataset, qids)
        _resolve_judge(cfg, ADAPTERS[row[0]].evaluator, questions)
    docs_hash, n_docs = data.docs_fingerprint(corpus, dataset, snapshot, subset)

    started = time.time()
    run_id = base_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}"
    n = 1
    while (paths.RUNS / run_id).exists():       # two runs of the same config within one second
        n += 1
        run_id = f"{base_id}-{n}"
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    code_hashes = {s: {"impl": cfg[s]["impl"], "hash": get(s, cfg[s]["impl"]).code_hash()} for s in stages}
    (run_dir / "code_hashes.json").write_text(json.dumps(code_hashes, indent=2))
    run = {"run_id": run_id, "name": cfg["name"], "kind": kind, "created_at": datetime.now().isoformat(timespec="seconds"),
           "status": "running", "dataset": dataset, "snapshot": snapshot, "subset": subset, "n_docs": n_docs,
           "limit": limit, "n_questions": len(qids), "question_ids": qids, "meta": meta or {},
           "git": tracking.git_info(run_dir), "bench_commit": bench.commit_sha()}
    stage_log: list[dict] = []

    def save_progress(current: str | None) -> None:
        run["current_stage"] = current
        run["duration_s"] = round(time.time() - started, 2)
        (run_dir / "stages.json").write_text(json.dumps(stage_log, indent=2))
        (run_dir / "run.json").write_text(json.dumps(run, indent=2))

    save_progress(stages[0])
    if on_start:
        on_start(run_id)
    print(f"Run {run_id} ({kind}): {dataset} snapshot {snapshot}, subset '{subset}': {n_docs:,} docs, {len(qids)} questions", flush=True)

    upstream, dirs, metas = None, {}, {}
    try:
        for stage in stages:
            save_progress(stage)
            impl = get(stage, cfg[stage]["impl"])
            extra = {"load": {"docs": docs_hash},     # the exact (doc_id, content_hash) set
                     "retrieve": {"questions": hashlib.sha256(",".join(qids).encode()).hexdigest()}}.get(stage)
            key = cache.stage_key(stage, cfg[stage], upstream, code_hashes[stage]["hash"], extra)
            hit = cache.is_cached(stage, key)
            print(f"  [{stage:8}] {impl.name:22} key={key} {'(cached)' if hit else ''}", flush=True)
            t0 = time.time()
            if not hit:
                tmp = cache.tmp_dir(stage, key)
                ctx = StageContext(out=tmp, dirs=dirs, metas=metas, corpus=corpus, dataset=dataset, snapshot=snapshot,
                                   subset=subset, questions=questions,
                                   fingerprint=cache.stage_key(stage, cfg[stage], None, impl.store_version()))
                params = impl.config_cls(**{k: v for k, v in cfg[stage].items() if k != "impl"})
                stats = impl.fn(params, ctx) or {}
                cache.commit(stage, key, tmp, {**stats, "stage": stage, "impl": impl.name, "key": key,
                                               "config": cfg[stage], "upstream": upstream,
                                               "duration_s": round(time.time() - t0, 2),
                                               "created_at": datetime.now().isoformat(timespec="seconds")})
            dirs[stage], metas[stage] = cache.stage_dir(stage, key), cache.read_meta(stage, key)
            stats = {k: v for k, v in metas[stage].items() if isinstance(v, (int, float, str, bool)) and k in STAT_KEYS}
            stage_log.append({"stage": stage, "impl": impl.name, "key": key, "cached": hit,
                              "duration_s": round(time.time() - t0, 2),
                              "original_duration_s": metas[stage].get("duration_s"), "stats": stats})
            for f in ARTIFACTS.get(stage, []):
                if (dirs[stage] / f).exists():
                    shutil.copy(dirs[stage] / f, run_dir / f)
            upstream = key
        run["status"] = "ok"
    except BaseException as e:   # record failed/interrupted runs too
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        gen_usage = metas.get("generate", {}).get("usage", {})
        judge_usage = metas.get("evaluate", {}).get("judge_usage")
        cached = {s["stage"] for s in stage_log if s["cached"]}
        incurred = (0.0 if "generate" in cached else gen_usage.get("cost_usd", 0.0)) + \
                   (0.0 if "evaluate" in cached or not judge_usage else judge_usage.get("cost_usd", 0.0))
        usage = {"generate": gen_usage, "generate_cached": "generate" in cached, "cost_usd_incurred": round(incurred, 6),
                 "judge": judge_usage or "not tracked (the benchmark judge runs inside the benchmark's own code)"}
        run["cost_usd"] = usage["cost_usd_incurred"]
        (run_dir / "usage.json").write_text(json.dumps(usage, indent=2))
        save_progress(None)
        tracking.index_run(run_dir)
        corpus.close()
    print(f"Done in {run['duration_s']}s -> runs/{run_id}", flush=True)
    return run_id
