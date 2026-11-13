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
from .registry import STAGES, get
from .stages.base import StageContext

# Files copied from stage cache dirs into the run folder (small, human-readable).
ARTIFACTS = {
    "retrieve": ["retrieval.jsonl"],
    "rerank": ["reranked.jsonl"],
    "generate": ["answers.jsonl", "generations.jsonl"],
    "evaluate": ["per_question.jsonl", "metrics.json", "judge_results.json", "judge.log"],
}


def _resolve_judge(cfg: dict) -> None:
    """Turn evaluate.judge = "auto" into true/false *before* hashing, so the cache
    key and config.json say what actually happened."""
    ev = cfg["evaluate"]
    if ev.get("judge") == "auto":
        reasons = []
        if not os.environ.get("ANTHROPIC_API_KEY"):
            reasons.append("ANTHROPIC_API_KEY not set")
        if cfg["generate"]["impl"] == "none":
            reasons.append("generate.impl is none (no answers to judge)")
        if not bench.is_ready():
            reasons.append("benchmark not installed (rag-lab bench setup)")
        ev["judge"] = not reasons
        if reasons:
            print(f"  judge: off ({'; '.join(reasons)})")


def run(config_path: str, overrides: list[str] | None = None, limit: int | None = None) -> str:
    cfg = resolve(apply_overrides(load_yaml(config_path), overrides or []))
    _resolve_judge(cfg)

    corpus = data.connect(read_only=True)
    subset = cfg["data"]["subset"]
    qids = data.subset_question_ids(corpus, subset, limit)
    questions = data.load_questions(corpus, qids)
    docs_hash = corpus.execute(
        "SELECT md5(string_agg(doc_id, ',' ORDER BY doc_id)) FROM subset_docs WHERE subset=?", [subset]).fetchone()[0]

    started = time.time()
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}"
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    code_hashes = {s: {"impl": cfg[s]["impl"], "hash": get(s, cfg[s]["impl"]).code_hash()} for s in STAGES}
    (run_dir / "code_hashes.json").write_text(json.dumps(code_hashes, indent=2))
    run = {"run_id": run_id, "name": cfg["name"], "created_at": datetime.now().isoformat(timespec="seconds"),
           "status": "running", "subset": subset, "limit": limit, "n_questions": len(qids),
           "question_ids": qids, "git": tracking.git_info(run_dir), "bench_commit": bench.commit_sha()}
    print(f"Run {run_id}: {len(qids)} questions, subset '{subset}'")

    stage_log, upstream, dirs, metas = [], None, {}, {}
    try:
        for stage in STAGES:
            impl = get(stage, cfg[stage]["impl"])
            extra = {"load": {"subset": subset, "docs": docs_hash},
                     "retrieve": {"questions": hashlib.sha256(",".join(qids).encode()).hexdigest()}}.get(stage)
            key = cache.stage_key(stage, cfg[stage], upstream, code_hashes[stage]["hash"], extra)
            hit = cache.is_cached(stage, key)
            print(f"  [{stage:8}] {impl.name:22} key={key} {'(cached)' if hit else ''}", flush=True)
            t0 = time.time()
            if not hit:
                tmp = cache.tmp_dir(stage, key)
                ctx = StageContext(out=tmp, dirs=dirs, metas=metas, corpus=corpus, subset=subset, questions=questions)
                params = impl.config_cls(**{k: v for k, v in cfg[stage].items() if k != "impl"})
                stats = impl.fn(params, ctx) or {}
                cache.commit(stage, key, tmp, {**stats, "stage": stage, "impl": impl.name, "key": key,
                                               "config": cfg[stage], "upstream": upstream,
                                               "duration_s": round(time.time() - t0, 2),
                                               "created_at": datetime.now().isoformat(timespec="seconds")})
            dirs[stage], metas[stage] = cache.stage_dir(stage, key), cache.read_meta(stage, key)
            stage_log.append({"stage": stage, "impl": impl.name, "key": key, "cached": hit,
                              "duration_s": round(time.time() - t0, 2),
                              "original_duration_s": metas[stage].get("duration_s")})
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
        gen_cached = any(s["stage"] == "generate" and s["cached"] for s in stage_log)
        usage = {"generate": gen_usage, "generate_cached": gen_cached,
                 "cost_usd_incurred": 0.0 if gen_cached else gen_usage.get("cost_usd", 0.0),
                 "judge": "not tracked (runs inside the benchmark's own code)"}
        run["cost_usd"] = usage["cost_usd_incurred"]
        run["duration_s"] = round(time.time() - started, 2)
        (run_dir / "stages.json").write_text(json.dumps(stage_log, indent=2))
        (run_dir / "usage.json").write_text(json.dumps(usage, indent=2))
        (run_dir / "run.json").write_text(json.dumps(run, indent=2))
        tracking.index_run(run_dir)
        corpus.close()
    print(f"Done in {run['duration_s']}s -> runs/{run_id}")
    return run_id
