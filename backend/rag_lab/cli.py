"""rag-lab command line. Run `rag-lab --help`."""

from typing import Optional

import typer
from dotenv import load_dotenv

from . import paths

load_dotenv(paths.ROOT / ".env")

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Iterate on RAG pipelines against EnterpriseRAG-Bench.")
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare benchmark data.")
bench_app = typer.Typer(no_args_is_help=True, help="The benchmark's own judge.")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect and index runs.")
app.add_typer(data_app, name="data")
app.add_typer(bench_app, name="bench")
app.add_typer(runs_app, name="runs")


@data_app.command("download")
def data_download(slices: Optional[list[str]] = typer.Option(None, "--slice", help="e.g. slack_slice_0001 (repeatable). Default: all_documents.zip")):
    """Download questions.jsonl and the documents into data/raw/."""
    from . import data
    data.download(slices)


@data_app.command("load")
def data_load():
    """Load every zip in data/raw/ into data/corpus.duckdb."""
    from . import data
    data.load()


@data_app.command("subset")
def data_subset(name: str = "dev", per_category: int = 10, distractors: int = 20_000, seed: int = 42):
    """Build a dev subset: N questions per category + their gold docs + random distractors."""
    from . import data
    data.build_subset(name, per_category, distractors, seed)


@bench_app.command("setup")
def bench_setup():
    """Clone EnterpriseRAG-Bench into .bench/ and install its judge's dependencies."""
    from . import bench
    bench.setup()


@app.command()
def run(config: str = typer.Argument(..., help="configs/<name>.yaml"),
        limit: Optional[int] = typer.Option(None, help="Only the first N questions of the subset"),
        set_: Optional[list[str]] = typer.Option(None, "--set", help="Override, e.g. --set retrieve.top_k=50")):
    """Run a pipeline config end to end and record it under runs/."""
    from . import pipeline
    pipeline.run(config, set_ or [], limit)


@app.command("impls")
def list_impls():
    """List every registered stage implementation and its config fields."""
    from . import config  # noqa: F401  (registers implementations)
    from .registry import REGISTRY
    for stage, impls in REGISTRY.items():
        typer.echo(stage)
        for name, impl in sorted(impls.items()):
            fields = ", ".join(f"{k}={v.default!r}" for k, v in impl.config_cls.model_fields.items() if k != "system_prompt")
            typer.echo(f"  {name:22} {fields}")


@runs_app.command("list")
def runs_list():
    from . import tracking
    con = tracking.connect_read()
    for r in con.execute("SELECT run_id, status, n_questions, cost_usd, overall FROM runs ORDER BY created_at DESC").fetchall():
        typer.echo(" ".join(str(x) for x in r))


@runs_app.command("reindex")
def runs_reindex():
    """Rebuild runs/index.duckdb from the run folders."""
    from . import tracking
    typer.echo(f"Indexed {tracking.reindex()} runs")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the API (and the built web UI, if web/dist exists)."""
    import uvicorn
    uvicorn.run("rag_lab.api:app", host=host, port=port, reload=reload)
