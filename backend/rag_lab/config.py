"""Run configs: a dict in, a fully-resolved dict (all defaults filled in) out.

    {"name": "baseline", "data": {"dataset": ..., "subset": "all", "snapshot": "latest"},
     "chunk": {"impl": "fixed", "chunk_size": 1200}, ...}

Each stage block names an `impl` from the registry; the remaining keys are
validated against that implementation's Pydantic config class. The resolved
config is what gets hashed for caching and saved in runs/<id>/config.json.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from . import stages  # noqa: F401  (importing registers all implementations)
from .registry import STAGES, get


class DataConfig(BaseModel):
    model_config = {"extra": "forbid"}
    dataset: str
    subset: str = "all"                  # a named subset, or "all"
    snapshot: int | str = "latest"       # resolved to a number when the run starts


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """`--set retrieve.top_k=50` style overrides. Values are parsed as YAML."""
    for ov in overrides:
        key, _, value = ov.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        new = yaml.safe_load(value)
        if leaf == "impl" and node.get("impl") != new:
            node.clear()          # switching implementation: the old impl's params don't apply
        node[leaf] = new
    return raw


def resolve(raw: dict, stages: list[str] = STAGES) -> dict:
    """Validate every stage block and fill in defaults. `stages` lets an index
    build resolve only load..index."""
    out = {"name": raw.get("name", "unnamed"), "description": raw.get("description", "")}
    if "data" in raw:
        out["data"] = DataConfig(**raw["data"]).model_dump()
    unknown = set(raw) - {"name", "description", "data", "target", *STAGES}
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {sorted(unknown)}")
    for stage in stages:
        block = dict(raw.get(stage) or {})
        impl_name = block.pop("impl", "none" if stage in ("rerank", "transform") else None)
        if impl_name is None:
            raise SystemExit(f"Config is missing `{stage}: {{impl: ...}}`")
        impl = get(stage, impl_name)
        params = impl.config_cls(**block).model_dump()
        out[stage] = {"impl": impl_name, **params}
    return out


def flatten(cfg: dict, prefix: str = "") -> dict:
    """{"chunk": {"impl": "fixed"}} -> {"chunk.impl": "fixed"}; used for filtering runs."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat
