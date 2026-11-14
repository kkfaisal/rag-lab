"""rag-lab command line. The UI (`rag-lab serve`) is the main way to use the tool;
these commands cover setup and running saved pipelines without a browser."""

from typing import Optional

import typer
from dotenv import load_dotenv

from . import paths

load_dotenv(paths.ROOT / ".env")

app = typer.Typer(no_args_is_help=True, add_completion=False, help="A playground for RAG ingestion and retrieval pipelines.")
data_app = typer.Typer(no_args_is_help=True, help="Datasets: documents + benchmark questions.")
bench_app = typer.Typer(no_args_is_help=True, help="The EnterpriseRAG-Bench judge.")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect and index runs.")
app.add_typer(data_app, name="data")
app.add_typer(bench_app, name="bench")
app.add_typer(runs_app, name="runs")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the web UI and API."""
    import uvicorn
    uvicorn.run("rag_lab.api:app", host=host, port=port, reload=reload)


@data_app.command("add")
def data_add(dataset: str = typer.Argument(..., help="Dataset name = its folder under data/datasets/"),
             source: str = typer.Option("folder", help="folder | extract | enterprise_rag_bench"),
             sync: bool = typer.Option(False, help="Also delete docs that are no longer in the folder")):
    """Load a dataset folder (data/datasets/<name>/). `extract` first runs the folder's extract.py.
    Only new and changed documents are stored (a new snapshot)."""
    from . import data
    data.migrate_layout()
    params = {}
    if source == "extract":
        params["code"] = (data.dataset_dir(dataset) / data.EXTRACTOR_FILE).read_text()
    data.ingest_source(dataset, source, params, sync=sync)


@data_app.command("list")
def data_list():
    """Datasets, their snapshots and doc counts."""
    from . import data
    con = data.connect(read_only=True)
    for d in data.dataset_info(con):
        typer.echo(f"{d['name']} ({d['adapter']}): snapshot {d['latest_snapshot']}, {sum(d['by_source'].values()):,} docs, "
                   f"{d['n_questions']} questions, subsets {d['subsets'] or '-'}")
        for s in d["snapshots"]:
            typer.echo(f"  #{s['snapshot']} {s['created_at'][:19]}  +{s['added']} ~{s['changed']} -{s['deleted']}  -> {s['n_docs']:,}  {s['note']}")


@data_app.command("subset")
def data_subset(dataset: str, name: str = "dev", per_category: int = 10, distractors: int = 20_000, seed: int = 42):
    """Build a smaller subset: N questions per category + their gold docs + random distractors."""
    from . import data
    data.build_subset(dataset, name, per_category, distractors, seed)


@bench_app.command("setup")
def bench_setup():
    """Clone EnterpriseRAG-Bench into .bench/ and install its judge's dependencies."""
    from . import bench
    bench.setup()


@app.command()
def ingest(pipeline: str = typer.Argument(..., help="Id of a saved ingestion pipeline (see workspace/ingestion/)")):
    """Run an ingestion pipeline: every path that ends in a target node."""
    from . import pipeline as engine, workspace
    p = workspace.get_graph("ingestion", pipeline)
    for node in workspace.target_nodes(p):
        cfg, meta = workspace.compile_ingest(p, node["id"])
        engine.run_ingest(cfg, meta=meta)


@app.command()
def benchmark(pipeline: str = typer.Argument(..., help="Id of a saved retrieval pipeline (see workspace/retrieval/)"),
              target: Optional[str] = typer.Option(None, help="Benchmark against this target instead of the pipeline's own"),
              limit: Optional[int] = typer.Option(None, help="Only the first N questions")):
    """Run a retrieval pipeline against a target and score it on the dataset's questions."""
    from . import pipeline as engine, workspace
    cfg, meta, default_limit = workspace.compile_benchmark(workspace.get_graph("retrieval", pipeline), target)
    engine.run_benchmark(cfg, limit=limit or default_limit, meta=meta)


@runs_app.command("list")
def runs_list():
    from . import tracking
    con = tracking.connect_read()
    for r in con.execute("SELECT run_id, kind, status, n_questions, overall FROM runs ORDER BY created_at DESC").fetchall():
        typer.echo(" ".join(str(x) for x in r))


@runs_app.command("reindex")
def runs_reindex():
    """Rebuild runs/index.duckdb from the run folders."""
    from . import tracking
    typer.echo(f"Indexed {tracking.reindex()} runs")
