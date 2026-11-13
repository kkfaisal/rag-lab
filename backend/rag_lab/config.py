"""Run configs: YAML in, fully-resolved (all defaults filled in) dict out.

    name: baseline
    data: {subset: dev}
    chunk: {impl: fixed, chunk_size: 1200}
    ...

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
    subset: str = "dev"


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
        node[leaf] = yaml.safe_load(value)
    return raw


def resolve(raw: dict) -> dict:
    """Validate every stage block and fill in defaults."""
    out = {
        "name": raw.get("name", "unnamed"),
        "description": raw.get("description", ""),
        "data": DataConfig(**raw.get("data", {})).model_dump(),
    }
    unknown = set(raw) - {"name", "description", "data", *STAGES}
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {sorted(unknown)}")
    for stage in STAGES:
        block = dict(raw.get(stage) or {})
        impl_name = block.pop("impl", "none" if stage == "rerank" else None)
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
