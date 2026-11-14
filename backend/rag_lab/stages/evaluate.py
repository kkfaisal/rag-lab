"""evaluate: score the run.

Two layers:
  1. Free, deterministic retrieval metrics computed here (no LLM):
       recall@k        share of gold docs among the top-k retrieved docs
       ndcg@10         like recall@10, but a gold doc ranked higher counts more (the BEIR leaderboard metric)
       mrr@10          1 / rank of the first gold doc in the top 10 (0 if none): how soon the first hit comes
       context_recall  share of gold docs that made it into the LLM prompt
       answer_recall   share of gold docs in the answer's document_ids
       extra_docs      answer document_ids that are not gold
  2. An LLM judge (optional; costs money), chosen by `judge`:
       "benchmark"  EnterpriseRAG-Bench's own judge: correctness, completeness,
                    document recall, invalid extra docs.
       "llm"        the generic judge for any dataset with gold answers: Claude
                    compares each answer with the gold answer -> correctness, completeness.

Outputs per_question.jsonl and metrics.json ({"overall": {...}, "by_category": {...}}).
"""

import json
import math

from pydantic import BaseModel

from concurrent.futures import ThreadPoolExecutor

from .. import bench
from ..llm import Usage, get_provider
from ..registry import register
from .base import StageContext, read_jsonl, unique_docs, write_jsonl


class StandardEvalConfig(BaseModel):
    model_config = {"extra": "forbid"}
    judge: bool | str = "auto"              # auto | benchmark | llm | false (auto is resolved when the run starts)
    judge_model: str = "claude-sonnet-5-5"
    judge_cheap_model: str = "claude-haiku-5-5"
    judge_parallelism: int = 4
    recall_k: list[int] = [1, 5, 10, 20]


JUDGE_SYSTEM = """You grade an answer to a question against a reference (gold) answer.

- "correct": true if the answer conveys the same key facts as the gold answer and contains no \
contradicting or fabricated claims. Wording differences and extra correct detail are fine. If the gold \
answer says the information is not available, the answer is correct only if it also declines.
- "completeness": integer 0-100, the share of the gold answer's distinct facts that the answer states.
- "reasoning": one or two sentences."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "completeness": {"type": "integer"}, "reasoning": {"type": "string"}},
    "required": ["correct", "completeness", "reasoning"],
    "additionalProperties": False,
}


def llm_judge(cfg, ctx: StageContext, rows: list[dict], answers: dict) -> dict:
    """Generic judge: one Claude call per question that has a gold answer."""
    provider = get_provider("anthropic")
    gold = {q["question_id"]: q for q in ctx.questions if q.get("gold_answer")}
    todo = [r for r in rows if r["question_id"] in gold]
    ctx.log(f"judging {len(todo)} answers with {cfg.judge_model}")

    def judge_one(row):
        q = gold[row["question_id"]]
        user = (f"Question: {q['question']}\n\nGold answer: {q['gold_answer']}\n\n"
                f"Answer to grade: {answers[row['question_id']].get('answer') or '(no answer)'}")
        res = provider.complete(model=cfg.judge_model, system=JUDGE_SYSTEM, user=user, max_tokens=2000,
                                json_schema=JUDGE_SCHEMA, effort="low")
        try:
            verdict = json.loads(res.text)
            row["correctness"] = 100.0 if verdict["correct"] else 0.0
            row["completeness"] = float(max(0, min(100, verdict["completeness"])))
            row["judge_reasoning"] = verdict["reasoning"]
        except (json.JSONDecodeError, KeyError):
            row["judge_reasoning"] = f"judge output unusable (stop_reason={res.stop_reason})"
        return res.usage

    total = Usage()
    with ThreadPoolExecutor(cfg.judge_parallelism) as pool:
        for u in pool.map(judge_one, todo):
            total.merge(u)
    return total.to_dict()


def _ndcg(ranked: list[str], gold: set[str], k: int = 10) -> float | None:
    """Binary-relevance nDCG@k, as a percentage: DCG of the ranking / DCG of a perfect ranking."""
    if not gold:
        return None
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return round(100 * dcg / ideal, 2)


def _mrr(ranked: list[str], gold: set[str], k: int = 10) -> float | None:
    if not gold:
        return None
    return round(100 * next((1 / (i + 1) for i, d in enumerate(ranked[:k]) if d in gold), 0.0), 2)


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
        row["ndcg@10"] = _ndcg(retrieved[qid], gold)
        row["mrr@10"] = _mrr(retrieved[qid], gold)
        row["context_recall"] = _recall(gens[qid]["context_doc_ids"], gold)
        row["answer_recall"] = _recall(ans_docs, gold)
        row["extra_docs"] = len(set(ans_docs) - gold) if gold else None
        row["abstained"] = 100.0 if gens[qid]["abstained"] else 0.0
        rows.append(row)

    judge_usage = None
    if cfg.judge == "llm":
        judge_usage = llm_judge(cfg, ctx, rows, answers)
    if cfg.judge == "benchmark":
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
               "judged": cfg.judge if cfg.judge in ("benchmark", "llm") else False}
    ctx.log("overall: " + ", ".join(f"{k}={v}" for k, v in metrics["overall"].items()))
    (ctx.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"judged": bool(metrics["judged"]), "overall": metrics["overall"], "judge_usage": judge_usage}
