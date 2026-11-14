"""Everything a user defines in the UI, stored as JSON under workspace/:

  targets/     named stores: {"id", "name", "type", "params"}
  ingestion/   pipelines   source -> chunk -> embed -> target      (may branch to several targets)
  retrieval/   pipelines   target -> retrieve -> rerank -> generate -> benchmark

A pipeline is a graph: {"id", "name", "description", "nodes": [{"id", "stage",
"impl", "params", "position"}], "edges": [{"source", "target"}]}. `compile_*`
walks a graph and produces the plain config dict the engine (pipeline.py) runs.
"""

import json
import re
import time
from pathlib import Path

from . import paths

KINDS = {"ingestion": ["source", "transform", "chunk", "embed", "target"],
         "retrieval": ["target", "retrieve", "rerank", "generate", "benchmark"]}


class GraphError(ValueError):
    pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def _dir(kind: str) -> Path:
    if kind not in (*KINDS, "targets"):
        raise GraphError(f"unknown kind {kind!r}")
    d = paths.WORKSPACE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list(kind: str) -> list[dict]:
    return sorted((json.loads(f.read_text()) for f in _dir(kind).glob("*.json")), key=lambda g: g["name"].lower())


def _get(kind: str, item_id: str, what: str) -> dict:
    f = _dir(kind) / f"{slug(item_id)}.json"
    if not f.exists():
        raise GraphError(f"{what} {item_id!r} not found")
    return json.loads(f.read_text())


def _save(kind: str, item: dict) -> dict:
    if not item.get("name", "").strip():
        raise GraphError("name is required")
    item["id"] = item.get("id") or slug(item["name"])
    item["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (_dir(kind) / f"{slug(item['id'])}.json").write_text(json.dumps(item, indent=2))
    return item


# ----------------------------------------------------------------------------- targets

def list_targets() -> list[dict]:
    return _list("targets")


def get_target(target_id: str) -> dict:
    return _get("targets", target_id, "target")


def save_target(target: dict) -> dict:
    from .targets import TARGETS
    if target.get("type") not in TARGETS:
        raise GraphError(f"unknown target type {target.get('type')!r}")
    try:
        TARGETS[target["type"]][1](**(target.get("params") or {}))      # validate settings
    except Exception as e:
        raise GraphError(f"invalid target settings: {e}")
    return _save("targets", {"description": "", "params": {}, **target})


def delete_target(target_id: str) -> None:
    (_dir("targets") / f"{slug(target_id)}.json").unlink(missing_ok=True)


# ----------------------------------------------------------------------------- pipelines

def list_graphs(kind: str) -> list[dict]:
    return _list(kind)


def get_graph(kind: str, graph_id: str) -> dict:
    return _get(kind, graph_id, f"{kind} pipeline")


def save_graph(kind: str, graph: dict) -> dict:
    graph = {"description": "", "nodes": [], "edges": [], **graph}
    for n in graph["nodes"]:
        if n["stage"] not in KINDS[kind]:
            raise GraphError(f"a {kind} pipeline can't contain a {n['stage']!r} node")
    return _save(kind, graph)


def delete_graph(kind: str, graph_id: str) -> None:
    (_dir(kind) / f"{slug(graph_id)}.json").unlink(missing_ok=True)


def _block(node: dict) -> dict:
    return {"impl": node["impl"], **(node.get("params") or {})}


def _path_to(pipeline: dict, node_id: str) -> list[dict]:
    """Nodes from the root down to `node_id`, following edges backwards."""
    nodes = {n["id"]: n for n in pipeline["nodes"]}
    parent = {}
    for e in pipeline["edges"]:
        if e["target"] in parent:
            raise GraphError(f"node {nodes[e['target']]['stage']} has more than one input")
        parent[e["target"]] = e["source"]
    path, cur, seen = [], node_id, set()
    while cur is not None:
        if cur in seen:
            raise GraphError("the pipeline has a cycle")
        seen.add(cur)
        path.append(nodes[cur])
        cur = parent.get(cur)
    return path[::-1]


def target_nodes(pipeline: dict) -> list[dict]:
    return [n for n in pipeline["nodes"] if n["stage"] == "target"]


def compile_ingest(pipeline: dict, node_id: str) -> tuple[dict, dict]:
    """(config, meta) for the path that ends at target node `node_id`."""
    node = next((n for n in pipeline["nodes"] if n["id"] == node_id and n["stage"] == "target"), None)
    if node is None:
        raise GraphError(f"{node_id!r} is not a target node of {pipeline['name']!r}")
    path = _path_to(pipeline, node_id)
    stages = [n["stage"] for n in path]
    core = [s for s in stages if s not in ("transform", "embed")]
    if core != ["source", "chunk", "target"] or stages.count("transform") > 1 or stages.count("embed") > 1 \
            or ("transform" in stages and stages.index("transform") != 1) or ("embed" in stages and stages[-2] != "embed"):
        raise GraphError("a target must be fed by source → (transform) → chunk → (embed) → target; "
                         f"this path is {' → '.join(stages)}")
    by = {n["stage"]: n for n in path}
    src, tp = by["source"].get("params") or {}, node.get("params") or {}
    if not src.get("dataset"):
        raise GraphError("the source node has no dataset selected")
    if not tp.get("target_id"):
        raise GraphError("the target node has no target selected")
    target = get_target(tp["target_id"])
    cfg = {"name": f"{slug(pipeline['name'])}.{slug(target['name'])}",
           "description": f"{pipeline['name']} → {target['name']}",
           "data": {"dataset": src["dataset"], "subset": src.get("subset") or "all", "snapshot": src.get("snapshot") or "latest"},
           "load": {"impl": "corpus", "sources": src.get("sources") or None},
           "transform": _block(by["transform"]) if "transform" in by else {"impl": "none"},
           "chunk": _block(by["chunk"]),
           "embed": _block(by["embed"]) if "embed" in by else {"impl": "none"},
           "target": {"id": target["id"], "mode": tp.get("mode") or "append"}}
    return cfg, {"pipeline_id": pipeline["id"], "pipeline_name": pipeline["name"], "node_id": node_id,
                 "target_id": target["id"], "target_name": target["name"]}


def compile_benchmark(pipeline: dict, target_id: str | None = None) -> tuple[dict, dict, int | None]:
    """(config, meta, question limit). `target_id` overrides the pipeline's own
    target node, so one retrieval pipeline can be benchmarked against several targets."""
    by = {}
    for n in pipeline["nodes"]:
        if n["stage"] in by:
            raise GraphError(f"a retrieval pipeline can have only one {n['stage']} node")
        by[n["stage"]] = n
    if "retrieve" not in by:
        raise GraphError("a retrieval pipeline needs a retrieve node")
    target_id = target_id or (by.get("target", {}).get("params") or {}).get("target_id")
    if not target_id:
        raise GraphError("the target node has no target selected")
    target = get_target(target_id)
    bench = (by.get("benchmark", {}).get("params") or {})
    cfg = {"name": f"{slug(pipeline['name'])}--{slug(target['name'])}",
           "description": f"{pipeline['name']} on {target['name']}",
           "target": {"id": target["id"]},
           "retrieve": _block(by["retrieve"]),
           "rerank": _block(by["rerank"]) if "rerank" in by else {"impl": "none"},
           "generate": _block(by["generate"]) if "generate" in by else {"impl": "none"},
           "evaluate": {"impl": "standard", "judge": bench.get("judge", "auto"),
                        **({"recall_k": bench["recall_k"]} if bench.get("recall_k") else {})}}
    meta = {"pipeline_id": pipeline["id"], "pipeline_name": pipeline["name"], "target_id": target["id"], "target_name": target["name"]}
    return cfg, meta, bench.get("limit") or None


def seed(default_dataset: str | None, subset: str = "all") -> None:
    """First start: one local target, one ingestion pipeline, two retrieval pipelines."""
    if not list_targets():
        save_target({"name": "Local store", "type": "local", "description": "Vectors + BM25 in a file on disk.",
                     "params": {"vectors": True, "bm25": True}})
    tid = list_targets()[0]["id"]
    if default_dataset and not list_graphs("ingestion"):
        save_graph("ingestion", {"name": "Baseline ingestion", "description": "Fixed-size chunks, local embeddings.", "nodes": [
            {"id": "source", "stage": "source", "impl": "dataset", "params": {"dataset": default_dataset, "subset": subset, "snapshot": "latest"}, "position": {"x": 0, "y": 60}},
            {"id": "chunk", "stage": "chunk", "impl": "fixed", "params": {"chunk_size": 1200, "overlap": 150}, "position": {"x": 290, "y": 60}},
            {"id": "embed", "stage": "embed", "impl": "sentence_transformers", "params": {"model": "BAAI/bge-small-en-v1.5"}, "position": {"x": 580, "y": 60}},
            {"id": "target", "stage": "target", "impl": "target", "params": {"target_id": tid, "mode": "append"}, "position": {"x": 870, "y": 60}}],
            "edges": [{"source": "source", "target": "chunk"}, {"source": "chunk", "target": "embed"}, {"source": "embed", "target": "target"}]})
    if not list_graphs("retrieval"):
        def chain(name, desc, steps):
            nodes = [{"id": "target", "stage": "target", "impl": "target", "params": {"target_id": tid}, "position": {"x": 0, "y": 60}}]
            for i, (stage, impl, params) in enumerate(steps, 1):
                nodes.append({"id": stage, "stage": stage, "impl": impl, "params": params, "position": {"x": 290 * i, "y": 60}})
            edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
            save_graph("retrieval", {"name": name, "description": desc, "nodes": nodes, "edges": edges})
        chain("Vector search", "Dense retrieval only; retrieval metrics, no LLM.",
              [("retrieve", "vector", {"top_k": 20}), ("benchmark", "benchmark", {"limit": 20, "judge": "auto"})])
        chain("Hybrid + rerank", "BM25 + vector fused with RRF, then a cross-encoder.",
              [("retrieve", "hybrid", {"top_k": 50}), ("rerank", "cross_encoder", {"top_n": 10}),
               ("benchmark", "benchmark", {"limit": 20, "judge": "auto"})])
