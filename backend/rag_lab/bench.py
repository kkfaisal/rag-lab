"""The benchmark's own judge, run from a separate checkout with its own venv.

We deliberately call their code as a subprocess instead of copying it, so our
scores are exactly the benchmark's scores.
"""

import json
import os
import subprocess
from pathlib import Path

from . import paths

REPO = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench"
PYTHON = paths.BENCH_DIR / ".venv" / "bin" / "python"


def setup() -> None:
    """Clone the benchmark (code only: a sparse checkout skips the 3.9 GB
    generated_data/ folder) and install its requirements into .bench/.../.venv."""
    d = paths.BENCH_DIR
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(d)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "src", "answer_evaluation"], cwd=d, check=True)
    else:
        print(f"  exists: {d}")
    if not PYTHON.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(d / ".venv")], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(PYTHON), "-r", str(d / "requirements.txt")], check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
    print(f"Benchmark judge ready at {d} (commit {sha[:10]})")


def is_ready() -> bool:
    return PYTHON.exists()


def commit_sha() -> str | None:
    if not paths.BENCH_DIR.exists():
        return None
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.BENCH_DIR, capture_output=True, text=True)
    return r.stdout.strip() or None


def run_judge(answers_file: Path, out_dir: Path, model: str, cheap_model: str, parallelism: int) -> dict:
    """Run metrics_based_eval. --no-correction: score against the original gold
    set. (The correction flow re-judges gold docs and needs the raw generated_data/
    sources we skip in setup.)"""
    if not is_ready():
        raise SystemExit("Benchmark judge not installed. Run `rag-lab bench setup`.")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env)")
    env = {**os.environ, "LLM_PROVIDER": "anthropic", "LLM_API_KEY": api_key,
           "LLM_MODEL_NAME": model, "CHEAP_LLM_MODEL_NAME": cheap_model, "PYTHONUNBUFFERED": "1"}
    results = out_dir / "judge_results.json"
    cmd = [str(PYTHON), "-m", "src.scripts.answer_evaluation.metrics_based_eval",
           "--answers-file", str(answers_file.resolve()),
           "--questions-file", str(paths.QUESTIONS_FILE.resolve()),
           "--results-file", str(results.resolve()),
           "--updated-questions-file", str((out_dir / "judge_questions_updated.jsonl").resolve()),
           "--parallelism", str(parallelism), "--no-correction"]
    with open(out_dir / "judge.log", "w") as log:
        proc = subprocess.run(cmd, cwd=paths.BENCH_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not results.exists():
        tail = (out_dir / "judge.log").read_text()[-2000:]
        raise RuntimeError(f"Judge failed (exit {proc.returncode}). Last log lines:\n{tail}")
    return json.loads(results.read_text())
