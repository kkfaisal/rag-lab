"""generate: answer each question from its top chunks.

  llm    single pass: the top chunks go into one prompt, Claude answers once
  agent  agentic: Claude starts from the same chunks, can search the target again with
         its own queries (rephrased, split into parts), then answers
  none   no LLM; the answer's documents are the top chunks' documents

Outputs
  answers.jsonl      benchmark format: {"question_id", "answer", "document_ids"}
  generations.jsonl  everything else: context docs, raw output, abstained, usage, stop_reason
"""

import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from ..llm import Usage, get_provider
from ..registry import register
from .base import StageContext, read_jsonl, unique_docs, write_jsonl

ABSTAIN_TEXT = "I could not find this information in the available documents."

DEFAULT_SYSTEM_PROMPT = """You answer questions using only the documents provided with each question.

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
    texts = {c["chunk_id"]: c["text"] for r in rows for c in r["chunks"]}
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


# ----------------------------------------------------------------------------- agentic RAG

AGENT_SYSTEM_PROMPT = """You answer questions using only documents from a search engine.

You get the results of a first search for the question. If they are not enough, call the search tool \
with your own queries: rephrase, use names or terms from documents you have seen, or search for each part \
of a multi-part question separately. Stop searching when you have what you need or further searches stop \
turning up anything new. Then call the answer tool exactly once.

Rules for the answer:
- Use only the numbered documents you were shown. Do not use outside knowledge.
- If the documents do not contain the answer, set "found" to false and say plainly that the information \
is not in the documents. Do not guess.
- If documents disagree, say so, give each version with its source, and prefer the most recent or most \
authoritative one when it is clear which that is.
- When the question asks for a list or "all" of something, include every item the documents support.
- Be concise and specific: names, numbers, dates.
- In "sources", list the numbers of the documents your answer relies on (empty if not found)."""

SEARCH_TOOL = {
    "name": "search",
    "description": "Search the document collection. Returns the best matching passages, grouped by numbered "
                   "document. Documents keep their number across searches.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}},
                     "required": ["query"], "additionalProperties": False},
}
ANSWER_TOOL = {
    "name": "answer",
    "description": "Give the final answer. Call once, after searching.",
    "input_schema": ANSWER_SCHEMA,
}


class AgentGenerateConfig(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}
    model: str = "claude-sonnet-5-5"
    effort: str | None = "medium"      # low | medium | high | xhigh | max (None = model default)
    max_tokens: int = 8000             # per model turn
    context_chunks: int = 10           # chunks from the pipeline's own retrieval, shown first
    max_searches: int = 5              # extra searches the agent may run per question
    search: str = "hybrid"             # how the search tool searches the target: hybrid | vector | bm25
    search_k: int = 8                  # chunks per search
    concurrency: int = 4
    doc_ids_from: str = "cited"        # "cited" = docs the model cites; "context" = every doc it saw
    system_prompt: str = AGENT_SYSTEM_PROMPT


class _Docs:
    """Numbers documents as the agent first sees them and renders search hits grouped by document."""

    def __init__(self):
        self.order: list[str] = []

    def render(self, hits: list[dict]) -> str:
        if not hits:
            return "No results."
        by_doc: dict[str, list[str]] = {}
        for h in hits:
            by_doc.setdefault(h["doc_id"], []).append(h.get("text", ""))
            if h["doc_id"] not in self.order:
                self.order.append(h["doc_id"])
        return "\n\n".join(f'<document number="{self.order.index(d) + 1}">\n' + "\n...\n".join(t) + "\n</document>"
                            for d, t in by_doc.items())


@register("generate", "agent", AgentGenerateConfig)
def agent_generate(cfg: AgentGenerateConfig, ctx: StageContext) -> dict:
    """Agentic RAG: Claude can search the target again with its own queries before answering."""
    import threading

    import anthropic

    from .retrieve import Search, rrf

    if cfg.search not in ("hybrid", "vector", "bm25"):
        raise SystemExit("agent search must be hybrid, vector or bm25")
    client = anthropic.Anthropic(max_retries=5)
    search, lock = Search(ctx), threading.Lock()       # targets and embedders aren't thread-safe: one search at a time
    mode = cfg.search
    if mode == "hybrid" and not (ctx.target.text_search and ctx.target.vector_search and ctx.embed):
        mode = "bm25" if ctx.target.text_search else "vector"

    def run_search(query: str) -> list[dict]:
        with lock:
            if mode == "bm25":
                return search.bm25(query, cfg.search_k)
            if mode == "vector":
                return search.vector(query, cfg.search_k)
            return rrf([(1.0, search.bm25(query, 50)), (1.0, search.vector(query, 50))], 60, cfg.search_k)

    rows, texts = _top_chunks(ctx, cfg.context_chunks)
    questions = {q["question_id"]: q["question"] for q in ctx.questions}

    def answer_one(row):
        docs, usage, searches = _Docs(), Usage(), []
        first = docs.render([{**c, "text": texts[c["chunk_id"]]} for c in row["chunks"]])
        messages = [{"role": "user", "content": f"Question: {questions[row['question_id']]}\n\n"
                                                f"<first_search_results>\n{first}\n</first_search_results>"}]
        final, stop_reason, model = None, None, cfg.model
        for turn in range(cfg.max_searches + 2):
            force = len(searches) >= cfg.max_searches or turn == cfg.max_searches + 1
            kwargs = dict(model=cfg.model, max_tokens=cfg.max_tokens, system=cfg.system_prompt, messages=messages,
                          tools=[SEARCH_TOOL, ANSWER_TOOL],
                          tool_choice={"type": "tool", "name": "answer"} if force else {"type": "auto"})
            if cfg.effort:
                kwargs["output_config"] = {"effort": cfg.effort}
            resp = client.messages.create(**kwargs)
            u = resp.usage
            usage.add(resp.model, u.input_tokens, u.output_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0)
            stop_reason, model = resp.stop_reason, resp.model
            calls = [b for b in resp.content if b.type == "tool_use"]
            final = next((b.input for b in calls if b.name == "answer"), None)
            if final is not None or not calls:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in calls:
                if b.name == "search" and len(searches) < cfg.max_searches:
                    query = str(b.input.get("query", ""))
                    hits = run_search(query)
                    new = len({h["doc_id"] for h in hits} - set(docs.order))
                    searches.append({"query": query, "hits": len(hits), "new_docs": new})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": docs.render(hits)})
                else:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                    "content": "Search limit reached: call the answer tool now."})
            messages.append({"role": "user", "content": results})

        parsed = final if isinstance(final, dict) else {"found": False, "answer": "", "sources": []}
        found = bool(parsed.get("found"))
        cited = [docs.order[i - 1] for i in parsed.get("sources") or [] if isinstance(i, int) and 1 <= i <= len(docs.order)]
        doc_ids = (cited if cfg.doc_ids_from == "cited" else docs.order) if found else []
        answer = parsed.get("answer") or ("" if found else ABSTAIN_TEXT)
        return {
            "answer_row": {"question_id": row["question_id"], "answer": answer, "document_ids": list(dict.fromkeys(doc_ids))},
            "detail": {"question_id": row["question_id"], "context_doc_ids": docs.order, "cited_doc_ids": cited,
                       "abstained": not found, "stop_reason": stop_reason, "model": model, "searches": searches,
                       "raw": json.dumps(final) if final is not None else None, "usage": usage.to_dict()},
            "usage": usage,
        }

    ctx.log(f"answering {len(rows)} questions with an agent ({cfg.model}, up to {cfg.max_searches} searches each, {mode} search)")
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        results = list(pool.map(answer_one, rows))
    search.close()
    total = Usage()
    for r in results:
        total.merge(r["usage"])
    write_jsonl(ctx.out / "answers.jsonl", [r["answer_row"] for r in results])
    write_jsonl(ctx.out / "generations.jsonl", [r["detail"] for r in results])
    n_searches = sum(len(r["detail"]["searches"]) for r in results)
    ctx.log(f"{n_searches} extra searches; tokens in/out {total.input_tokens:,}/{total.output_tokens:,}  cost ${total.cost_usd:.4f}")
    return {"n": len(results), "abstained": sum(r["detail"]["abstained"] for r in results),
            "searches": n_searches, "refusals": sum(r["detail"]["stop_reason"] == "refusal" for r in results),
            "usage": total.to_dict()}
