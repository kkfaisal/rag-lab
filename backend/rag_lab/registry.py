"""Registry of stage implementations.

Every pipeline stage (chunk, embed, retrieve, ...) can have several implementations.
(Targets — where chunks are stored and searched — have their own registry in targets/.)
An implementation is a function plus a Pydantic config class:

    class FixedConfig(BaseModel):
        chunk_size: int = 1200

    @register("chunk", "fixed", FixedConfig)
    def fixed_chunker(cfg: FixedConfig, ctx: StageContext) -> dict: ...

A run config then selects it with `chunk: {impl: fixed, chunk_size: 800}`.
To add your own, write a function like the above in any module imported by
`rag_lab.stages` and it will show up automatically.
"""

import hashlib
import inspect
import sys
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

INGEST_STAGES = ["load", "chunk", "embed"]                       # then the chunks are written to a target
BENCH_STAGES = ["retrieve", "rerank", "generate", "evaluate"]    # run against a target
STAGES = INGEST_STAGES + BENCH_STAGES


@dataclass
class Impl:
    stage: str
    name: str
    fn: Callable
    config_cls: type[BaseModel]

    def store_version(self) -> str:
        """Identity of this implementation's per-item store (cache/store/). A module
        can pin it with STORE_VERSION so harmless edits don't throw away hours of
        embeddings; bump the constant when the stored items' meaning changes.
        Without the constant, any edit to the module starts a fresh store."""
        return str(getattr(sys.modules[self.fn.__module__], "STORE_VERSION", self.code_hash()))

    def code_hash(self) -> str:
        """Hash of the source of the module that defines this implementation.
        Whole module (not just the function) so edits to helper functions count too."""
        src = inspect.getsource(sys.modules[self.fn.__module__])
        return hashlib.sha256(src.encode()).hexdigest()[:16]


REGISTRY: dict[str, dict[str, Impl]] = {s: {} for s in STAGES}


class NoConfig(BaseModel):
    model_config = {"extra": "forbid"}


def register(stage: str, name: str, config_cls: type[BaseModel] = NoConfig):
    if stage not in REGISTRY:
        raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")

    def deco(fn):
        REGISTRY[stage][name] = Impl(stage, name, fn, config_cls)
        return fn
    return deco


def get(stage: str, name: str) -> Impl:
    try:
        return REGISTRY[stage][name]
    except KeyError:
        raise SystemExit(f"No {stage} implementation named {name!r}. Available: {sorted(REGISTRY[stage])}")
