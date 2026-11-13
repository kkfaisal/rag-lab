"""generate: answer each question from its top chunks.

Outputs
  answers.jsonl      benchmark format: {"question_id", "answer", "document_ids"}
  generations.jsonl  everything else: context docs, raw output, abstained, usage, stop_reason
"""

import json
from concurrent.futures import ThreadPoolExecutor

import duckdb
from pydantic import BaseModel

from ..llm import Usage, get_provider
from ..registry import register
from .base import StageContext, read_jsonl, unique_docs, write_jsonl

ABSTAIN_TEXT = "I could not find this information in the available documents."

DEFAULT_SYSTEM_PROMPT = """You answer questions about the internal documents of a company called Redwood Inference \
(Slack threads, emails, tickets, docs, CRM records, meeting transcripts, pull requests).

Rules:
- Use only the numbered documents provided. Do not use outside knowledge.
- If the documents do not contain the answer, set "found" to false and say plainly that the information \
is not in the documents. Do not guess.
- If documents disagree, say so, give each version with its source, and prefer the most recent or most \
authoritative one when it is clear which that is.
- When the question asks for a list or "all" of something, include every item the documents support.
- Be concise and specific: names, numbers, dates.
- In "sources", list the numbers of the documents your answer relies on (empty if not found)."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "sources"],
    "additionalProperties": False,
}


class LLMGenerateConfig(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}
    provider: str = "anthropic"
    model: str = "claude-sonnet-5-5"
    effort: str | None = "medium"      # low | medium | high | xhigh | max (None = model default)
    max_tokens: int = 8000
    context_chunks: int = 10           # how many top chunks go into the prompt
    concurrency: int = 4
    doc_ids_from: str = "cited"        # "cited" = docs the model cites; "context" = every doc in the prompt
    refusal_fallback: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class NoGenerateConfig(BaseModel):
    model_config = {"extra": "forbid"}
    context_chunks: int = 10


def build_context(chunks: list[dict], texts: dict[str, str]) -> tuple[str, list[str]]:
    """Group the chunks by document (in rank order) and number the documents."""
    doc_order = unique_docs(chunks)
    parts = []
    for n, doc_id in enumerate(doc_order, 1):
        body = "\n...\n".join(texts[c["chunk_id"]] for c in chunks if c["doc_id"] == doc_id)
        parts.append(f"<document number=\"{n}\">\n{body}\n</document>")
    return "\n\n".join(parts), doc_order


def _top_chunks(ctx: StageContext, n: int) -> tuple[list[dict], dict[str, str]]:
    rows = read_jsonl(ctx.dirs["rerank"] / "reranked.jsonl")
    for r in rows:
        r["chunks"] = r["chunks"][:n]
    ids = [c["chunk_id"] for r in rows for c in r["chunks"]]
    con = duckdb.connect(str(ctx.dirs["index"] / "index.duckdb"), read_only=True)
    texts = dict(con.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [ids]).fetchall())
    con.close()
    return rows, texts


@register("generate", "llm", LLMGenerateConfig)
def llm_generate(cfg: LLMGenerateConfig, ctx: StageContext) -> dict:
    provider = get_provider(cfg.provider, refusal_fallback=cfg.refusal_fallback)
    rows, texts = _top_chunks(ctx, cfg.context_chunks)
    questions = {q["question_id"]: q["question"] for q in ctx.questions}

    def answer_one(row):
        context, doc_order = build_context(row["chunks"], texts)
        user = f"<documents>\n{context}\n</documents>\n\nQuestion: {questions[row['question_id']]}"
        res = provider.complete(model=cfg.model, system=cfg.system_prompt, user=user,
                                max_tokens=cfg.max_tokens, json_schema=ANSWER_SCHEMA, effort=cfg.effort)
        try:
            parsed = json.loads(res.text)
        except json.JSONDecodeError:      # e.g. stop_reason == "refusal" or "max_tokens"
            parsed = {"found": False, "answer": res.text or "", "sources": []}
        found = bool(parsed.get("found"))
        cited = [doc_order[i - 1] for i in parsed.get("sources", []) if isinstance(i, int) and 1 <= i <= len(doc_order)]
        doc_ids = (cited if cfg.doc_ids_from == "cited" else doc_order) if found else []
        answer = parsed.get("answer") or ""
        if not found and not answer:
            answer = ABSTAIN_TEXT
        return {
            "answer_row": {"question_id": row["question_id"], "answer": answer, "document_ids": list(dict.fromkeys(doc_ids))},
            "detail": {"question_id": row["question_id"], "context_doc_ids": doc_order, "cited_doc_ids": cited,
                       "abstained": not found, "stop_reason": res.stop_reason, "model": res.model,
                       "raw": res.text, "usage": res.usage.to_dict()},
            "usage": res.usage,
        }

    ctx.log(f"generating {len(rows)} answers with {cfg.provider}:{cfg.model}")
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        results = list(pool.map(answer_one, rows))
    total = Usage()
    for r in results:
        total.merge(r["usage"])
    write_jsonl(ctx.out / "answers.jsonl", [r["answer_row"] for r in results])
    write_jsonl(ctx.out / "generations.jsonl", [r["detail"] for r in results])
    ctx.log(f"tokens in/out {total.input_tokens:,}/{total.output_tokens:,}  cost ${total.cost_usd:.4f}")
    return {"n": len(results), "abstained": sum(r["detail"]["abstained"] for r in results),
            "refusals": sum(r["detail"]["stop_reason"] == "refusal" for r in results), "usage": total.to_dict()}


@register("generate", "none", NoGenerateConfig)
def no_generate(cfg: NoGenerateConfig, ctx: StageContext) -> dict:
    """Retrieval-only: no LLM call, no answer text; document_ids = docs of the top
    chunks. Free, and enough for recall metrics."""
    rows, _ = _top_chunks(ctx, cfg.context_chunks)
    answers = [{"question_id": r["question_id"], "answer": None, "document_ids": unique_docs(r["chunks"])} for r in rows]
    write_jsonl(ctx.out / "answers.jsonl", answers)
    write_jsonl(ctx.out / "generations.jsonl", [
        {"question_id": a["question_id"], "context_doc_ids": a["document_ids"], "cited_doc_ids": [],
         "abstained": False, "stop_reason": None, "model": None, "raw": None} for a in answers])
    return {"n": len(answers), "usage": Usage().to_dict()}
