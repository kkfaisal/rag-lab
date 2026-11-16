"""From a fresh clone to a first benchmark result in a couple of minutes, with no API key:

    uv run rag-lab quickstart [--serve]

Downloads SciFact (4 MB: 5k abstracts, 300 questions with gold documents), indexes it for BM25
keyword search, answers the 300 questions and scores them. Then the Lessons page takes over.
Everything it creates is ordinary, editable: a target, an ingestion and a retrieval pipeline.
"""

import json
import shutil
import subprocess

from . import data, paths, workspace

DATASET = "scifact"
TARGET = {"id": "scifact-bm25", "name": "SciFact BM25", "type": "local",
          "description": "Keyword (BM25) only, no embedding: fast and free.", "params": {"vectors": False, "bm25": True}}
INGESTION = {"id": "scifact-ingestion", "name": "SciFact ingestion",
             "description": "Fixed-size chunks into a BM25-only target. Add an embed node for vector search.",
             "nodes": [{"id": "source", "stage": "source", "impl": "dataset", "params": {"dataset": DATASET, "subset": "all", "snapshot": "latest"}, "position": {"x": 0, "y": 60}},
                       {"id": "chunk", "stage": "chunk", "impl": "fixed", "params": {"chunk_size": 1200, "overlap": 150}, "position": {"x": 290, "y": 60}},
                       {"id": "target", "stage": "target", "impl": "target", "params": {"target_id": TARGET["id"], "mode": "overwrite"}, "position": {"x": 580, "y": 60}}],
             "edges": [{"source": "source", "target": "chunk"}, {"source": "chunk", "target": "target"}]}
RETRIEVAL = {"id": "bm25", "name": "BM25", "description": "Keyword search only; all questions, no LLM.",
             "nodes": [{"id": "target", "stage": "target", "impl": "target", "params": {"target_id": TARGET["id"]}, "position": {"x": 0, "y": 60}},
                       {"id": "retrieve", "stage": "retrieve", "impl": "bm25", "params": {"top_k": 20}, "position": {"x": 290, "y": 60}},
                       {"id": "benchmark", "stage": "benchmark", "impl": "benchmark", "params": {"judge": "auto"}, "position": {"x": 580, "y": 60}}],
             "edges": [{"source": "target", "target": "retrieve"}, {"source": "retrieve", "target": "benchmark"}]}


def _step(n: int, text: str) -> None:
    print(f"\n[{n}/4] {text}", flush=True)


def _ensure(kind: str, item: dict) -> None:
    """Create an example unless something with that id exists (never overwrite the user's edits)."""
    try:
        workspace.get_target(item["id"]) if kind == "targets" else workspace.get_graph(kind, item["id"])
    except workspace.GraphError:
        workspace.save_target(dict(item)) if kind == "targets" else workspace.save_graph(kind, dict(item))


def build_ui() -> bool:
    """Build web/ once if it isn't built and npm is available."""
    if paths.WEB_DIST.exists():
        return True
    if not (paths.ROOT / "web" / "package.json").exists():
        print(f"No web UI sources under {paths.ROOT}; run this from the rag-lab folder to use the UI.")
        return False
    if not shutil.which("npm"):
        print("The web UI isn't built and npm isn't installed. Install Node 20+, then: cd web && npm install && npm run build")
        return False
    print("Building the web UI (once)...", flush=True)
    web = paths.ROOT / "web"
    subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=web, check=True)
    subprocess.run(["npm", "run", "build"], cwd=web, check=True)
    return True


def run() -> str:
    from . import benchmarks, pipeline
    data.migrate_layout()
    _step(1, "Dataset: SciFact (5k scientific abstracts, 300 questions with gold documents)")
    con = data.connect(read_only=True) if paths.CORPUS_DB.exists() else None
    have = con is not None and any(d["name"] == DATASET and d["n_questions"] for d in data.dataset_info(con))
    if con is not None:
        con.close()
    if have:
        print("  already loaded")
    else:
        benchmarks.fetch(DATASET, load=True)

    _step(2, "Examples: a BM25 target, an ingestion and a retrieval pipeline (plus the usual examples)")
    if not workspace.seeding_done():
        workspace.seed(DATASET)
    _ensure("targets", TARGET)
    _ensure("ingestion", INGESTION)
    _ensure("retrieval", RETRIEVAL)

    _step(3, "Ingestion: chunk the documents and build the BM25 index")
    graph = workspace.get_graph("ingestion", INGESTION["id"])
    for node in workspace.target_nodes(graph):
        cfg, meta = workspace.compile_ingest(graph, node["id"])
        pipeline.run_ingest(cfg, meta=meta)

    _step(4, "Retrieval: answer all 300 questions with BM25 and score them against the gold documents")
    cfg, meta, limit = workspace.compile_benchmark(workspace.get_graph("retrieval", RETRIEVAL["id"]))
    run_id = pipeline.run_benchmark(cfg, limit=limit, meta=meta)
    overall = json.loads((paths.RUNS / run_id / "metrics.json").read_text()).get("overall", {})

    print("\nYour first benchmark (BM25 on SciFact, 300 questions):")
    for m, what in [("recall@1", "gold documents ranked first"), ("recall@10", "gold documents in the top 10"),
                    ("ndcg@10", "like recall@10, but higher ranks count more"), ("mrr@10", "how high the first gold document is")]:
        if m in overall:
            print(f"  {m:<10} {overall[m]:5.1f}   {what}")
    print(f"\nRun: runs/{run_id}/")
    return run_id
