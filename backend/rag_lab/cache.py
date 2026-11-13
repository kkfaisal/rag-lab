"""Content-addressed cache for stage outputs.

key(stage) = sha256(stage config + upstream stage's key + implementation source hash + extra inputs)

Because each key includes the upstream key, a change anywhere invalidates
everything downstream of it — and nothing upstream. Change only the reranker and
load/chunk/embed/index/retrieve are reused from cache/<stage>/<key>/.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

from . import paths


def stage_key(stage: str, stage_cfg: dict, upstream_key: str | None, code_hash: str, extra=None) -> str:
    payload = json.dumps(
        {"stage": stage, "config": stage_cfg, "upstream": upstream_key, "code": code_hash, "extra": extra},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def stage_dir(stage: str, key: str) -> Path:
    return paths.CACHE / stage / key


def is_cached(stage: str, key: str) -> bool:
    return (stage_dir(stage, key) / "meta.json").exists()


def tmp_dir(stage: str, key: str) -> Path:
    """Stages write into a temp dir that is renamed into place only on success,
    so a crashed stage never leaves a half-written cache entry."""
    d = paths.CACHE / stage / f".{key}.tmp-{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def commit(stage: str, key: str, tmp: Path, meta: dict) -> Path:
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    final = stage_dir(stage, key)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def read_meta(stage: str, key: str) -> dict:
    return json.loads((stage_dir(stage, key) / "meta.json").read_text())
