"""The engine: runs ingestion and benchmark configs and records every run.

run_ingest     load -> chunk -> embed (all cached) -> write to a target (append or overwrite)
run_benchmark  retrieve -> rerank -> generate -> evaluate against a target (cached per target version)

Both write runs/<run_id>/ after every step, so the UI can follow a run live.
"""

import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime

import pyarrow as pa
import numpy as np

from . import bench, cache, data, db, paths, tracking, workspace
from .config import resolve
from .datasets import ADAPTERS
from .registry import BENCH_STAGES, INGEST_STAGES, get
from .targets import Manifest, open_target
from .stages.base import StageContext, iter_parquet
from .stages.transform import docs_file as transform_docs_file

# Stage stats worth showing next to a node in the UI.
STAT_KEYS = {"n_docs", "n_chunks", "docs_chunked", "docs_reused", "avg_chars", "max_chars", "n", "dim", "embedded", "reused",
             "mode", "docs_written", "docs_skipped", "docs_deleted", "chunks_written", "target_version",
             "n_questions", "abstained", "refusals", "judged", "docs_in", "dropped", "changed", "chars_in", "chars_out", "passthrough"}

# Files copied from stage cache dirs into the run folder (small, human-readable).
ARTIFACTS = {
    "retrieve": ["retrieval.jsonl"],
    "rerank": ["reranked.jsonl"],
    "generate": ["answers.jsonl", "generations.jsonl"],
    "evaluate": ["per_question.jsonl", "metrics.json", "judge_results.json", "judge.log"],
}


def _resolve_judge(cfg: dict, dataset: str, evaluator: str, questions: list[dict]) -> None:
    """Turn evaluate.judge = "auto" into "benchmark" | "llm" | false *before*
    hashing, so the cache key and config.json say what actually happened.

    benchmark = EnterpriseRAG-Bench's own judge (only valid for that dataset);
    llm       = the built-in generic judge: Claude compares each answer with the gold answer."""
    ev = cfg["evaluate"]
    want = ev.get("judge")
    if want is True:
        want = "benchmark" if evaluator == "benchmark" else "llm"
    if want == "benchmark" and evaluator != "benchmark":
        raise SystemExit(f"The benchmark judge only applies to EnterpriseRAG-Bench, not {dataset!r}")
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


class _Run:
    """Bookkeeping shared by both run kinds: the run folder, progress files, cached stages."""

    def __init__(self, kind: str, cfg: dict, stages: list[str], meta: dict | None, on_start, **fields):
        self.kind, self.cfg, self.started = kind, cfg, time.time()
        self.id = base = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}"
        n = 1
        while (paths.RUNS / self.id).exists():       # two runs of the same config within one second
            n += 1
            self.id = f"{base}-{n}"
        self.dir = paths.RUNS / self.id
        self.dir.mkdir(parents=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2))
        self.code = {s: {"impl": cfg[s]["impl"], "hash": get(s, cfg[s]["impl"]).code_hash()} for s in stages}
        (self.dir / "code_hashes.json").write_text(json.dumps(self.code, indent=2))
        self.info = {"run_id": self.id, "name": cfg["name"], "kind": kind, "created_at": datetime.now().isoformat(timespec="seconds"),
                     "status": "running", "meta": meta or {}, "git": tracking.git_info(self.dir),
                     "bench_commit": bench.commit_sha(), **fields}
        self.log: list[dict] = []
        self.dirs: dict = {}
        self.metas: dict = {}
        self.upstream: str | None = None
        self.save(None)
        if on_start:
            on_start(self.id)

    def save(self, current: str | None) -> None:
        self.info["current_stage"] = current
        self.info["duration_s"] = round(time.time() - self.started, 2)
        (self.dir / "stages.json").write_text(json.dumps(self.log, indent=2))
        (self.dir / "run.json").write_text(json.dumps(self.info, indent=2))

    def stage(self, stage: str, extra, **ctx_fields) -> None:
        """Run one registered stage, or reuse its cached output."""
        self.save(stage)
        block = self.cfg[stage]
        impl = get(stage, block["impl"])
        key = cache.stage_key(stage, block, self.upstream, self.code[stage]["hash"], extra)
        hit = cache.is_cached(stage, key)
        print(f"  [{stage:8}] {impl.name:22} key={key} {'(cached)' if hit else ''}", flush=True)
        t0 = time.time()
        if not hit:
            tmp = cache.tmp_dir(stage, key)
            ctx = StageContext(out=tmp, dirs=self.dirs, metas=self.metas,
                               fingerprint=cache.stage_key(stage, block, None, impl.store_version()), **ctx_fields)
            stats = impl.fn(impl.config_cls(**{k: v for k, v in block.items() if k != "impl"}), ctx) or {}
            cache.commit(stage, key, tmp, {**stats, "stage": stage, "impl": impl.name, "key": key, "config": block,
                                           "upstream": self.upstream, "duration_s": round(time.time() - t0, 2),
                                           "created_at": datetime.now().isoformat(timespec="seconds")})
        self.dirs[stage], self.metas[stage] = cache.stage_dir(stage, key), cache.read_meta(stage, key)
        self._logged(stage, impl.name, key, hit, t0, self.metas[stage])
        for f in ARTIFACTS.get(stage, []):
            if (self.dirs[stage] / f).exists():
                shutil.copy(self.dirs[stage] / f, self.dir / f)
        self.upstream = key

    def _logged(self, stage, impl, key, cached, t0, stats: dict) -> None:
        self.log.append({"stage": stage, "impl": impl, "key": key, "cached": cached, "duration_s": round(time.time() - t0, 2),
                         "original_duration_s": stats.get("duration_s"),
                         "stats": {k: v for k, v in stats.items() if k in STAT_KEYS and isinstance(v, (int, float, str, bool))}})

    def step(self, name: str, impl: str, fn) -> dict:
        """Run an uncached step (the target write)."""
        self.save(name)
        print(f"  [{name:8}] {impl}", flush=True)
        t0 = time.time()
        stats = fn()
        self._logged(name, impl, None, False, t0, stats)
        return stats

    def close(self, error: BaseException | None, usage: dict | None = None) -> None:
        self.info["status"] = "failed" if error else "ok"
        if error:
            self.info["error"] = f"{type(error).__name__}: {error}"
            (self.dir / "error.txt").write_text("".join(traceback.format_exception(error)))
        if usage is not None:
            self.info["cost_usd"] = usage["cost_usd_incurred"]
            (self.dir / "usage.json").write_text(json.dumps(usage, indent=2))
        self.save(None)
        tracking.index_run(self.dir)
        print(f"{'Failed' if error else 'Done'} in {self.info['duration_s']}s -> runs/{self.id}", flush=True)


# ----------------------------------------------------------------------------- ingestion

def run_ingest(raw: dict, meta: dict | None = None, on_start=None) -> str:
    """Config: {name, data, load, chunk, embed, target: {id, mode}}."""
    cfg = {**resolve(raw, INGEST_STAGES), "target": dict(raw["target"])}
    target_def = workspace.get_target(cfg["target"]["id"])
    mode = cfg["target"].setdefault("mode", "append")
    if mode not in ("append", "overwrite"):
        raise SystemExit(f"target mode must be 'append' or 'overwrite', not {mode!r}")
    cfg["target"].update(name=target_def["name"], type=target_def["type"])

    corpus = data.connect(read_only=True)
    dataset, subset = cfg["data"]["dataset"], cfg["data"]["subset"]
    data.require_ready(corpus, dataset)
    if cfg["data"]["snapshot"] == "latest":
        cfg["data"]["snapshot"] = data.latest_snapshot(corpus, dataset)
    snapshot = int(cfg["data"]["snapshot"])
    docs_hash, n_docs = data.docs_fingerprint(corpus, dataset, snapshot, subset)

    run = _Run("ingest", cfg, INGEST_STAGES, meta, on_start, dataset=dataset, snapshot=snapshot, subset=subset, n_docs=n_docs)
    print(f"Run {run.id} (ingest): {dataset} snapshot {snapshot}, subset '{subset}': {n_docs:,} docs -> {target_def['name']} [{mode}]", flush=True)
    error = None
    try:
        for stage in INGEST_STAGES:
            run.stage(stage, {"docs": docs_hash} if stage == "load" else None,
                      corpus=corpus, dataset=dataset, snapshot=snapshot, subset=subset)
        run.step("write", f"{target_def['type']} [{mode}]", lambda: _write_target(run, target_def, mode, dataset, subset, snapshot))
    except BaseException as e:
        error = e
        raise
    finally:
        corpus.close()
        run.close(error)
    return run.id


def _write_target(run: _Run, target_def: dict, mode: str, dataset: str, subset: str, snapshot: int) -> dict:
    """Write this run's chunks to the target.

    overwrite: clear the target, write everything.
    append:    write only documents that are new or whose content / chunking /
               embedding changed since they were last written (their old chunks
               are deleted first). Documents no longer in the source are left alone.
    """
    cfg = run.cfg
    embed_meta = {k: run.metas["embed"][k] for k in ("model", "query_prefix", "max_seq_length", "code") if k in run.metas["embed"]} or None
    settings = cache.stage_key("doc", {"chunk": cfg["chunk"], "embed": cfg["embed"]}, None, run.code["chunk"]["hash"])
    docs_file, chunks_file = transform_docs_file(run.dirs), run.dirs["chunk"] / "chunks.parquet"

    target, manifest = open_target(target_def), Manifest(paths.TARGETS / target_def["id"])
    work = db.connect()                 # in-memory, capped, spills to disk: the diff never sits in Python
    try:
        if mode == "overwrite":
            target.clear()
            manifest.reset()
        elif manifest.data["n_docs"] and manifest.data["embed"] != embed_meta:
            raise SystemExit(f"Target {target_def['name']!r} was written with a different embedder "
                             f"({manifest.data['embed']}). Use mode 'overwrite' to rebuild it, or a different target.")
        # What each document should look like in the target: hash of its content + chunk/embed settings.
        work.execute("CREATE TABLE wanted(doc_id VARCHAR, doc_hash VARCHAR)")
        for b in iter_parquet(docs_file, 50_000, ["doc_id", "content_hash"]):
            ids, hashes = b.column("doc_id").to_pylist(), b.column("content_hash").to_pylist()
            work.register("w", pa.table({"doc_id": ids, "doc_hash": [cache.stage_key("doc", h, settings, "") for h in hashes]}))
            work.execute("INSERT INTO wanted SELECT * FROM w")
            work.unregister("w")
        # todo = new documents + documents whose hash changed (those were written before: changed = true).
        work.execute(f"ATTACH '{manifest.db_file()}' AS m (READ_ONLY)")
        work.execute("""CREATE TABLE todo AS SELECT w.doc_id, w.doc_hash, k.doc_id IS NOT NULL AS changed
                        FROM wanted w LEFT JOIN m.docs k USING (doc_id) WHERE k.doc_hash IS DISTINCT FROM w.doc_hash""")
        work.execute("DETACH m")
        n_wanted, n_todo, n_changed = work.execute(
            "SELECT (SELECT count(*) FROM wanted), count(*), count(*) FILTER (WHERE changed) FROM todo").fetchall()[0]
        print(f"    {n_todo:,} documents to write ({n_changed:,} changed), {n_wanted - n_todo:,} already up to date", flush=True)
        if not n_todo and manifest.data["n_docs"]:     # nothing to do: leave the target and its version untouched
            return {"mode": mode, "docs_written": 0, "docs_skipped": n_wanted, "chunks_written": 0,
                    "target_version": manifest.data["version"]}

        for b in work.sql("SELECT doc_id FROM todo WHERE changed").to_arrow_reader(10_000):
            target.delete_docs(b.column("doc_id").to_pylist())       # their old chunks go first

        emb = np.load(run.dirs["embed"] / "embeddings.npy", mmap_mode="r") if run.metas["embed"].get("n") else None
        reader = work.sql(f"""SELECT c.file_row_number AS row, c.chunk_id, c.doc_id, c.source, c.title, c.text
                              FROM read_parquet('{chunks_file}', file_row_number = true) c SEMI JOIN todo USING (doc_id)
                           """).to_arrow_reader(2_000)    # no ORDER BY: sorting chunk text needs memory; vectors are looked up by row
        written = 0
        for b in reader:
            rows = b.column("row").to_numpy()
            vecs = emb[rows] if emb is not None else None
            target.add([{**c, "vector": vecs[i].tolist() if vecs is not None else None}
                        for i, c in enumerate(b.drop_columns(["row"]).to_pylist())])
            written += b.num_rows
            if written % 100_000 < b.num_rows:
                print(f"    wrote {written:,} chunks", flush=True)
        del emb
        target.finish()
        # Remember what was written: per document its hash and chunk count.
        work.execute(f"""CREATE TABLE written AS SELECT t.doc_id, t.doc_hash, count(c.doc_id)::INTEGER AS n_chunks
                         FROM todo t LEFT JOIN read_parquet('{chunks_file}') c USING (doc_id) GROUP BY 1, 2""")
        manifest.record_from(work, "written")
        version = manifest.commit({"mode": mode, "run_id": run.id, "pipeline": (run.info["meta"] or {}).get("pipeline_name"),
                                   "docs_written": n_todo, "docs_skipped": n_wanted - n_todo, "chunks_written": written},
                                  embed=embed_meta, dataset=dataset, subset=subset, snapshot=snapshot)
    finally:
        work.close()
        target.close()
    run.info["target_version"] = version
    return {"mode": mode, "docs_written": n_todo, "docs_skipped": n_wanted - n_todo, "chunks_written": written,
            "target_version": version}


# ----------------------------------------------------------------------------- benchmark

def _target_code_hash(target_type: str) -> str:
    import inspect
    import sys
    from .targets import TARGETS
    src = inspect.getsource(sys.modules[TARGETS[target_type][0].__module__])
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def run_benchmark(raw: dict, limit: int | None = None, meta: dict | None = None, on_start=None) -> str:
    """Config: {name, target: {id}, retrieve, rerank, generate, evaluate}. The
    questions are those of the dataset the target was last written from."""
    cfg = {**resolve(raw, BENCH_STAGES), "target": dict(raw["target"])}
    target_def = workspace.get_target(cfg["target"]["id"])
    manifest = Manifest(paths.TARGETS / target_def["id"]).data
    if not manifest["version"]:
        raise SystemExit(f"Target {target_def['name']!r} is empty. Run an ingestion pipeline that writes to it first.")
    dataset, subset, snapshot = manifest["dataset"], manifest["subset"], manifest["snapshot"]
    cfg["target"].update(name=target_def["name"], type=target_def["type"], version=manifest["version"],
                         params=target_def.get("params") or {}, n_chunks=manifest["n_chunks"], embed=manifest["embed"])
    cfg["data"] = {"dataset": dataset, "subset": subset, "snapshot": snapshot}

    corpus = data.connect(read_only=True)
    adapter = data.require_ready(corpus, dataset)
    qids = data.subset_question_ids(corpus, dataset, subset, limit)
    questions = data.load_questions(corpus, dataset, qids)
    corpus.close()
    _resolve_judge(cfg, dataset, ADAPTERS[adapter].evaluator, questions)

    run = _Run("benchmark", cfg, BENCH_STAGES, meta, on_start, dataset=dataset, snapshot=snapshot, subset=subset,
               limit=limit, n_questions=len(qids), question_ids=qids, target_version=manifest["version"])
    print(f"Run {run.id} (benchmark): {target_def['name']} v{manifest['version']} ({dataset}), {len(qids)} questions", flush=True)
    # Cache keys start from the target's identity + version, its settings and the source of its
    # target type (search-time code): a re-run on an unchanged target is free.
    run.upstream = cache.stage_key("target", {"id": target_def["id"], "version": manifest["version"],
                                              "params": target_def.get("params")}, None, _target_code_hash(target_def["type"]))
    target, error = open_target(target_def), None
    try:
        for stage in BENCH_STAGES:
            extra = {"questions": hashlib.sha256(",".join(qids).encode()).hexdigest()} if stage == "retrieve" else None
            run.stage(stage, extra, target=target, embed=manifest["embed"], questions=questions, dataset=dataset,
                      snapshot=snapshot, subset=subset)
    except BaseException as e:
        error = e
        raise
    finally:
        target.close()
        gen, judge = run.metas.get("generate", {}).get("usage", {}), run.metas.get("evaluate", {}).get("judge_usage")
        cached = {s["stage"] for s in run.log if s["cached"]}
        incurred = (0.0 if "generate" in cached else gen.get("cost_usd", 0.0)) + \
                   (0.0 if "evaluate" in cached or not judge else judge.get("cost_usd", 0.0))
        run.close(error, {"generate": gen, "generate_cached": "generate" in cached, "cost_usd_incurred": round(incurred, 6),
                          "judge": judge or "not tracked (the benchmark judge runs inside the benchmark's own code)"})
    return run.id
