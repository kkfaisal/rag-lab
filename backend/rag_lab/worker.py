"""Runs one background job (see jobs.py): python -m rag_lab.worker <job.json>

Job kinds:
  dataset    fill and/or load a dataset folder into the document store (or remove a dataset)
  ingest     run an ingestion pipeline path and write to its target
  benchmark  run a retrieval pipeline against a target and score it
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import paths


def main(job_file: str) -> None:
    load_dotenv(paths.ROOT / ".env")
    job = json.loads(Path(job_file).read_text())
    spec = job["spec"]
    run_file = Path(job_file).with_suffix(".run")
    on_start = lambda run_id: run_file.write_text(run_id)  # noqa: E731
    if job["kind"] == "dataset":
        from . import data
        if spec.get("action") == "remove":
            data.remove_dataset(spec["dataset"])
        else:
            data.ingest_source(spec["dataset"], spec["source"], spec.get("params") or {}, sync=spec.get("sync", False))
    elif job["kind"] == "ingest":
        from . import pipeline
        pipeline.run_ingest(spec["config"], meta=spec.get("meta"), on_start=on_start)
    elif job["kind"] == "benchmark":
        from . import pipeline
        pipeline.run_benchmark(spec["config"], limit=spec.get("limit"), meta=spec.get("meta"), on_start=on_start)
    else:
        raise SystemExit(f"unknown job kind {job['kind']!r}")


if __name__ == "__main__":
    main(sys.argv[1])
