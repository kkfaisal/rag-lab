"""Canvas definitions and how they turn into runnable configs.

Two kinds of graph, both stored as JSON under workspace/:

  pipeline   source -> chunk -> embed -> index      (may branch: one source, many indexes)
  retriever  retrieve -> rerank -> generate          (a chain; rerank and generate optional)

A graph is {"id", "name", "description", "nodes": [{"id", "stage", "impl", "params",
"position"}], "edges": [{"source", "target"}]}.

An experiment = one index node of a pipeline + one retriever. `compile_*` walks the
graph and produces the same linear config the CLI runs, so the engine (cache, run
folders, metrics) is identical for canvas-built and YAML-built pipelines.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from . import paths

KINDS = {"pipeline": ["source", "chunk", "embed", "index"], "retriever": ["retrieve", "rerank", "generate"]}


class GraphError(ValueError):
    pass


def _dir(kind: str) -> Path:
    if kind not in KINDS:
        raise GraphError(f"unknown kind {kind!r}")
    d = paths.WORKSPACE / f"{kind}s"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def list_graphs(kind: str) -> list[dict]:
    return sorted((json.loads(f.read_text()) for f in _dir(kind).glob("*.json")), key=lambda g: g["name"].lower())


def get_graph(kind: str, graph_id: str) -> dict:
    f = _dir(kind) / f"{slug(graph_id)}.json"
    if not f.exists():
        raise GraphError(f"{kind} {graph_id!r} not found")
    return json.loads(f.read_text())


def save_graph(kind: str, graph: dict) -> dict:
    graph = {"description": "", "nodes": [], "edges": [], **graph}
    if not graph.get("name", "").strip():
        raise GraphError("name is required")
    graph["id"] = graph.get("id") or slug(graph["name"])
    for n in graph["nodes"]:
        if n["stage"] not in KINDS[kind]:
            raise GraphError(f"a {kind} can't contain a {n['stage']!r} node")
    graph["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (_dir(kind) / f"{slug(graph['id'])}.json").write_text(json.dumps(graph, indent=2))
    return graph


def delete_graph(kind: str, graph_id: str) -> None:
    (_dir(kind) / f"{slug(graph_id)}.json").unlink(missing_ok=True)


def _block(node: dict) -> dict:
    return {"impl": node["impl"], **(node.get("params") or {})}


def node_label(node: dict) -> str:
    return node.get("label") or f"{node['stage']}:{node['impl']}"


def compile_index(pipeline: dict, node_id: str) -> dict:
    """Config for load..index along the path that ends at index node `node_id`."""
    nodes = {n["id"]: n for n in pipeline["nodes"]}
    if node_id not in nodes or nodes[node_id]["stage"] != "index":
        raise GraphError(f"{node_id!r} is not an index node of pipeline {pipeline['name']!r}")
    parent = {}
    for e in pipeline["edges"]:
        if e["target"] in parent:
            raise GraphError(f"node {node_label(nodes[e['target']])} has more than one input")
        parent[e["target"]] = e["source"]
    path, cur = [], node_id
    while cur is not None:
        path.append(nodes[cur])
        cur = parent.get(cur)
        if cur is not None and nodes[cur] in path:
            raise GraphError("the pipeline has a cycle")
    path.reverse()
    stages = [n["stage"] for n in path]
    if stages not in (["source", "chunk", "embed", "index"], ["source", "chunk", "index"]):
        raise GraphError(f"an index must be fed by source -> chunk -> (embed) -> index; this path is {' -> '.join(stages)}")
    by = {n["stage"]: n for n in path}
    src = by["source"].get("params") or {}
    if not src.get("dataset"):
        raise GraphError("the source node has no dataset selected")
    return {
        "data": {"dataset": src["dataset"], "subset": src.get("subset") or "all", "snapshot": src.get("snapshot") or "latest"},
        "load": {"impl": "corpus", "sources": src.get("sources") or None},
        "chunk": _block(by["chunk"]),
        "embed": _block(by["embed"]) if "embed" in by else {"impl": "none"},
        "index": _block(by["index"]),
    }


def compile_retriever(retriever: dict) -> dict:
    """Config for retrieve..generate. Missing rerank/generate nodes mean 'none'."""
    by = {}
    for n in retriever["nodes"]:
        if n["stage"] in by:
            raise GraphError(f"a retrieval setup can have only one {n['stage']} node")
        by[n["stage"]] = n
    if "retrieve" not in by:
        raise GraphError("a retrieval setup needs a retrieve node")
    return {"retrieve": _block(by["retrieve"]),
            "rerank": _block(by["rerank"]) if "rerank" in by else {"impl": "none"},
            "generate": _block(by["generate"]) if "generate" in by else {"impl": "none"}}


def index_nodes(pipeline: dict) -> list[dict]:
    return [n for n in pipeline["nodes"] if n["stage"] == "index"]


def build_config(pipeline: dict, node_id: str) -> tuple[dict, dict]:
    node = next(n for n in pipeline["nodes"] if n["id"] == node_id)
    cfg = {"name": f"{slug(pipeline['name'])}.{slug(node_label(node))}", "description": f"Index build: {pipeline['name']} / {node_label(node)}",
           **compile_index(pipeline, node_id)}
    spec_hash = hashlib.sha256(json.dumps(compile_index(pipeline, node_id), sort_keys=True).encode()).hexdigest()[:12]
    return cfg, {"pipeline_id": pipeline["id"], "pipeline_name": pipeline["name"], "node_id": node_id,
                 "node_label": node_label(node), "spec_hash": spec_hash}


def experiment_config(pipeline: dict, node_id: str, retriever: dict, judge="auto") -> tuple[dict, dict]:
    cfg, meta = build_config(pipeline, node_id)
    cfg.update(compile_retriever(retriever))
    cfg["name"] = f"{cfg['name']}--{slug(retriever['name'])}"
    cfg["description"] = f"{pipeline['name']} / {meta['node_label']}  ×  {retriever['name']}"
    cfg["evaluate"] = {"impl": "standard", "judge": judge}
    return cfg, {**meta, "retriever_id": retriever["id"], "retriever_name": retriever["name"]}


def seed(default_dataset: str | None, subset: str = "all") -> None:
    """First start: one example pipeline (two chunkers sharing a source) and three retrieval setups."""
    if not list_graphs("retriever"):
        save_graph("retriever", {"name": "Vector top-20", "description": "Dense retrieval only, no LLM.", "nodes": [
            {"id": "retrieve", "stage": "retrieve", "impl": "vector", "params": {"top_k": 20}, "position": {"x": 40, "y": 80}}], "edges": []})
        save_graph("retriever", {"name": "Hybrid + rerank", "description": "BM25 + vector fused with RRF, then a cross-encoder.", "nodes": [
            {"id": "retrieve", "stage": "retrieve", "impl": "hybrid", "params": {"top_k": 50}, "position": {"x": 40, "y": 80}},
            {"id": "rerank", "stage": "rerank", "impl": "cross_encoder", "params": {"top_n": 10}, "position": {"x": 340, "y": 80}}],
            "edges": [{"source": "retrieve", "target": "rerank"}]})
        save_graph("retriever", {"name": "Hybrid + rerank + Claude", "description": "As above, then Claude answers (needs ANTHROPIC_API_KEY).", "nodes": [
            {"id": "retrieve", "stage": "retrieve", "impl": "hybrid", "params": {"top_k": 50}, "position": {"x": 40, "y": 80}},
            {"id": "rerank", "stage": "rerank", "impl": "cross_encoder", "params": {"top_n": 10}, "position": {"x": 340, "y": 80}},
            {"id": "generate", "stage": "generate", "impl": "llm", "params": {"model": "claude-sonnet-5-5"}, "position": {"x": 640, "y": 80}}],
            "edges": [{"source": "retrieve", "target": "rerank"}, {"source": "rerank", "target": "generate"}]})
    if default_dataset and not list_graphs("pipeline"):
        embed = {"impl": "sentence_transformers", "params": {"model": "BAAI/bge-small-en-v1.5"}}
        save_graph("pipeline", {"name": "Baseline", "description": "One source, two chunking strategies, an index for each.", "nodes": [
            {"id": "source", "stage": "source", "impl": "dataset", "params": {"dataset": default_dataset, "subset": subset, "snapshot": "latest"}, "position": {"x": 20, "y": 140}},
            {"id": "chunk-fixed", "stage": "chunk", "impl": "fixed", "params": {"chunk_size": 1200, "overlap": 150}, "position": {"x": 300, "y": 40}},
            {"id": "embed-fixed", "stage": "embed", **embed, "position": {"x": 580, "y": 40}},
            {"id": "index-fixed", "stage": "index", "impl": "duckdb", "params": {}, "label": "fixed-1200", "position": {"x": 860, "y": 40}},
            {"id": "chunk-slack", "stage": "chunk", "impl": "slack_aware", "params": {"chunk_size": 1200, "overlap": 150}, "position": {"x": 300, "y": 240}},
            {"id": "embed-slack", "stage": "embed", **embed, "position": {"x": 580, "y": 240}},
            {"id": "index-slack", "stage": "index", "impl": "duckdb", "params": {}, "label": "slack-aware", "position": {"x": 860, "y": 240}}],
            "edges": [{"source": "source", "target": "chunk-fixed"}, {"source": "chunk-fixed", "target": "embed-fixed"},
                      {"source": "embed-fixed", "target": "index-fixed"}, {"source": "source", "target": "chunk-slack"},
                      {"source": "chunk-slack", "target": "embed-slack"}, {"source": "embed-slack", "target": "index-slack"}]})
