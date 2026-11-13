"""Runs one background job (see jobs.py): python -m rag_lab.worker <job.json>"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import paths


def main(job_file: str) -> None:
    load_dotenv(paths.ROOT / ".env")
    job = json.loads(Path(job_file).read_text())
    spec = job["spec"]
    if job["kind"] == "ingest":
        from . import data
        data.ingest(spec["dataset"], spec["path"], adapter=spec.get("adapter"), sync=spec.get("sync", False),
                    note=spec.get("note", ""), code=spec.get("code"))
        return
    from . import pipeline
    run_file = Path(job_file).with_suffix(".run")
    pipeline.run_config(spec["config"], limit=spec.get("limit"), stop_after="index" if job["kind"] == "build" else None,
                        meta=spec.get("meta"), on_start=lambda run_id: run_file.write_text(run_id))


if __name__ == "__main__":
    main(sys.argv[1])
