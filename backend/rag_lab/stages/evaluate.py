"""evaluate: score the run.

Two layers:
  1. Free, deterministic retrieval metrics computed here (no LLM):
       recall@k        share of gold docs among the top-k retrieved docs
       context_recall  share of gold docs that made it into the LLM prompt
       answer_recall   share of gold docs in the answer's document_ids
       extra_docs      answer document_ids that are not gold
  2. The benchmark's own LLM judge (optional; costs money): correctness,
     completeness, document recall, invalid extra docs.

Outputs per_question.jsonl and metrics.json ({"overall": {...}, "by_category": {...}}).
"""

import json

from pydantic import BaseModel

from .. import bench
from ..registry import register
from .base import StageContext, read_jsonl, unique_docs, write_jsonl


class StandardEvalConfig(BaseModel):
    model_config = {"extra": "forbid"}
    judge: bool | str = "auto"              # auto = run the judge if an API key is set and answers exist
    judge_model: str = "claude-sonnet-5-5"
    judge_cheap_model: str = "claude-haiku-5-5"
    judge_parallelism: int = 4
    recall_k: list[int] = [1, 5, 10, 20]


def _recall(found: list[str], gold: set[str]) -> float | None:
    return round(100 * len(gold & set(found)) / len(gold), 2) if gold else None


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def aggregate(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("question_id", "category")] if rows else []
    return {"n": len(rows), **{k: _mean(r.get(k) for r in rows) for k in keys}}


@register("evaluate", "standard", StandardEvalConfig)
def standard_eval(cfg: StandardEvalConfig, ctx: StageContext) -> dict:
    retrieved = {r["question_id"]: unique_docs(r["chunks"]) for r in read_jsonl(ctx.dirs["retrieve"] / "retrieval.jsonl")}
    gens = {g["question_id"]: g for g in read_jsonl(ctx.dirs["generate"] / "generations.jsonl")}
    answers_file = ctx.dirs["generate"] / "answers.jsonl"
    answers = {a["question_id"]: a for a in read_jsonl(answers_file)}

    rows = []
    for q in ctx.questions:
        qid, gold = q["question_id"], set(q["expected_doc_ids"])
        ans_docs = answers[qid]["document_ids"]
        row = {"question_id": qid, "category": q["category"]}
        for k in cfg.recall_k:
            row[f"recall@{k}"] = _recall(retrieved[qid][:k], gold)
        row["context_recall"] = _recall(gens[qid]["context_doc_ids"], gold)
        row["answer_recall"] = _recall(ans_docs, gold)
        row["extra_docs"] = len(set(ans_docs) - gold) if gold else None
        row["abstained"] = 100.0 if gens[qid]["abstained"] else 0.0
        rows.append(row)

    if cfg.judge is True:
        ctx.log(f"running benchmark judge ({cfg.judge_model} / {cfg.judge_cheap_model}) on {len(rows)} answers")
        judged = {r["question_id"]: r for r in bench.run_judge(
            answers_file, ctx.out, cfg.judge_model, cfg.judge_cheap_model, cfg.judge_parallelism)["questions"]}
        for row in rows:
            j = judged.get(row["question_id"])
            if j:
                row["correctness"] = 100.0 if j["answer_correct"] else 0.0
                row["completeness"] = j["completeness_pct"]
                row["judge_doc_recall"] = j["document_recall_pct"]
                row["invalid_extra_docs"] = j["invalid_extra_docs"]
                row["judge_reasoning"] = j.get("correctness_reasoning")
    write_jsonl(ctx.out / "per_question.jsonl", rows)

    numeric = [{k: v for k, v in r.items() if k != "judge_reasoning"} for r in rows]
    by_cat: dict[str, list] = {}
    for r in numeric:
        by_cat.setdefault(r["category"], []).append(r)
    metrics = {"overall": aggregate(numeric), "by_category": {c: aggregate(v) for c, v in sorted(by_cat.items())},
               "judged": cfg.judge is True}
    ctx.log("overall: " + ", ".join(f"{k}={v}" for k, v in metrics["overall"].items()))
    (ctx.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"judged": cfg.judge is True, "overall": metrics["overall"]}
