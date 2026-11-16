"""Guided lessons: each one is a small experiment with a goal score to reach.

A lesson's goal is checked against your benchmark runs: the best successful run on the lesson's
dataset whose settings match `where` (e.g. {"retrieve.impl": "hybrid"}) and that answered at least
`min_questions` questions. Lessons without a goal are checked off by hand in the UI.

Steps are plain text; [label](/path) becomes a link to that page of the app and **bold** is bold.
Goal scores were measured on SciFact with the default settings, so each one is reachable but
needs the lesson's idea (on SciFact, recall@10: BM25 81.8, vector 82.8, hybrid 85.0).
"""

import json

LESSONS = [
    {"id": "first-benchmark", "level": "Start here", "title": "Your First Benchmark",
     "learn": "What a benchmark is: questions whose right documents are known (the gold documents), "
              "and how recall@k measures whether search finds them.",
     "goal": {"dataset": "scifact", "where": {}, "metric": "recall@10", "at_least": 80, "min_questions": 300},
     "steps": ["Run `uv run rag-lab quickstart`, or open [Retrieval → BM25](/retrieval/bm25) and press **Run benchmark**.",
               "Open the run from [Runs](/runs) and read recall@1, recall@10 and nDCG@10. Every metric name explains itself when you hover it.",
               "In the run's question list, open a question with recall@10 = 0. Compare its words with the gold document's: why did keyword search miss it?"],
     "takeaway": "BM25 finds documents that share words with the question. It is fast, free and a strong baseline: "
                 "a new idea has to beat it before it earns its complexity."},

    {"id": "vector-search", "level": "Core", "title": "Search by Meaning: Vector Search",
     "learn": "How embeddings find documents with the same meaning but different words, and what they miss.",
     "goal": {"dataset": "scifact", "where": {"retrieve.impl": "vector"}, "metric": "recall@10", "at_least": 82, "min_questions": 300},
     "steps": ["Open [Ingestion → Baseline ingestion](/ingestion/baseline-ingestion): source → chunk → embed (bge-small) → Local store. "
               "Check the source is SciFact and press **Run ingestion**. The first run downloads the embedding model (about 130 MB).",
               "Open [Retrieval → Vector search](/retrieval/vector-search). Click the **Benchmark** node and clear its question limit, "
               "so all 300 questions count. Tick **Local store** and press **Run benchmark**.",
               "Open [Compare](/diff) with the BM25 run as A and the vector run as B. Read the questions vector search fixed, and the ones it broke."],
     "takeaway": "Embeddings match meaning, so they find paraphrases BM25 misses, but they blur exact terms such as names, "
                 "numbers and codes, which BM25 finds easily. Neither is best everywhere."},

    {"id": "noise", "level": "Core", "title": "Is It Real or Noise?",
     "learn": "Why a higher average is not proof, and how the paired test on the Compare page tells a real gain from luck.",
     "steps": ["Open [Compare](/diff) with your BM25 run as A and your vector run as B, metric recall@10.",
               "Read the verdict: the p-value, the 95% interval and how many questions each run won. Is the gap real?",
               "Run both pipelines again with the Benchmark node's limit set to 20 questions, and compare those two runs. "
               "Watch how much wider the interval gets, and how the verdict changes."],
     "takeaway": "With few questions, big-looking gaps are often noise. Use as many questions as you can, and trust the "
                 "paired test over the averages."},

    {"id": "hybrid", "level": "Core", "title": "Best of Both: Hybrid Search",
     "learn": "How Reciprocal Rank Fusion merges keyword and vector results, so each covers the other's blind spots.",
     "goal": {"dataset": "scifact", "where": {"retrieve.impl": "hybrid", "rerank.impl": "none"}, "metric": "recall@10", "at_least": 84, "min_questions": 300},
     "steps": ["Open [Retrieval → Hybrid + rerank](/retrieval/hybrid-rerank) and press **Duplicate**.",
               "In the copy, select the **rerank** node and press Backspace, then connect retrieve to benchmark. "
               "Clear the Benchmark node's question limit and run it against **Local store**.",
               "Try the retrieve node's **bm25_weight** and **vector_weight**: which balance works best on SciFact?"],
     "takeaway": "Hybrid search is the usual production default: it rarely loses to either retriever alone, and it is still cheap."},

    {"id": "rerank", "level": "Core", "title": "Does Reranking Help Here?",
     "learn": "What a cross-encoder reranker does, how to measure it (context_recall: gold documents that reach the prompt), "
              "and why a popular component can make things worse on your data.",
     "steps": ["Open [Retrieval → Hybrid + rerank](/retrieval/hybrid-rerank): hybrid search fetches 50 chunks, a cross-encoder "
               "rescores them and keeps the best 10 for the prompt.",
               "Clear the Benchmark node's question limit and run it against **Local store**. It is slower: the cross-encoder reads "
               "every question with every candidate.",
               "Compare it with your hybrid-only run on **context_recall**. recall@k and nDCG score what retrieval found, before "
               "reranking; context_recall scores what reached the prompt, after it.",
               "Try the rerank node's **top_n** (keep more chunks) or another cross-encoder model, and measure again."],
     "takeaway": "On SciFact the default reranker, trained on web search queries, lowers context_recall (82.7 → 79.8): scientific "
                 "claims are not web queries. Components that help on one dataset can hurt on another, so measure every one on yours."},

    {"id": "own-retriever", "level": "Core", "title": "Write Your Own Retriever",
     "learn": "That every node is just Python: write a retriever yourself and measure it like any built-in.",
     "goal": {"dataset": "scifact", "where": {"retrieve.impl": "python"}, "metric": "recall@10", "at_least": 84, "min_questions": 300},
     "steps": ["Open [Retrieval → Vector search](/retrieval/vector-search), press **Duplicate**, select the retrieve node and press **✎ Edit script**.",
               "Write `retrieve(question, search, top_k)` using `search.bm25(question, k)` and `search.vector(question, k)`: "
               "for example, your own fusion of the two lists. The check tells you if the function is missing; AI assist can explain or fix it.",
               "Save, clear the Benchmark node's question limit and run it against **Local store**."],
     "hints": ["Reciprocal Rank Fusion in five lines: for each list, add 1 / (60 + rank) to each chunk's score, then sort by score.",
               "Fetch more candidates than you return (say 100 from each search), and return the top `top_k`."],
     "takeaway": "Built-ins are a starting point, not a limit. Any idea (query rewriting, filters, your own fusion) is one script away, "
                 "and it is measured exactly like the built-ins."},

    {"id": "chunking", "level": "Core", "title": "Chunking Changes Everything",
     "learn": "How chunk size trades precision for context, and why ingestion choices show up in retrieval scores.",
     "steps": ["In [Targets](/targets), add a local target named “SciFact small chunks” with vectors and BM25.",
               "Duplicate [Baseline ingestion](/ingestion/baseline-ingestion). In the copy, set the chunk node to chunk_size 300, overlap 50, "
               "and point the target node at the new target. Run it.",
               "Benchmark [Vector search](/retrieval/vector-search) against both targets (tick both) with all questions, then compare the two runs."],
     "takeaway": "Smaller chunks match a question more precisely but lose surrounding context; bigger ones keep context but dilute "
                 "the match. The best size depends on your documents, so measure it."},

    {"id": "generate", "level": "Advanced", "title": "Let Claude Answer",
     "needs": "An ANTHROPIC_API_KEY in .env and EnterpriseRAG-Bench (1.3 GB) loaded, with its dev subset.",
     "learn": "How retrieved chunks become an answer with citations, and how an LLM judge scores it against the gold answer.",
     "goal": {"dataset": "enterprise_rag_bench", "where": {"generate.impl": "llm"}, "metric": "correctness", "at_least": 60, "min_questions": 20},
     "steps": ["Load EnterpriseRAG-Bench from [Datasets](/datasets), and ingest its dev subset into a target with vectors and BM25.",
               "Open [Retrieval → Hybrid + rerank](/retrieval/hybrid-rerank), duplicate it, and add a **generate** node (llm) before the benchmark node.",
               "Run 20 questions. Open a question to read the answer, the documents it cited and the judge's verdict."],
     "takeaway": "Good answers need good retrieval first: when the gold document isn't in the prompt, the model can only refuse or guess. "
                 "Check context_recall before blaming the prompt."},

    {"id": "agent", "level": "Advanced", "title": "Agentic RAG",
     "needs": "Lesson “Let Claude Answer” done.",
     "learn": "How an agent that can search again (rephrase, split the question, follow names) compares with a single pass.",
     "goal": {"dataset": "enterprise_rag_bench", "where": {"generate.impl": "agent"}, "metric": "correctness", "at_least": 80, "min_questions": 20},
     "steps": ["In your pipeline from the last lesson, set the generate node to **agent** (max_searches 5).",
               "Run the same 20 questions. Open a question to see the searches the agent made.",
               "Compare the agent run with the single-pass run on correctness, then on cost and time. Is the gain real, and worth it?"],
     "takeaway": "Agents help most on questions that need several facts or a follow-up search, and cost several times more. "
                 "Measure both quality and cost before choosing."},
]


def _matches(params: dict, where: dict) -> bool:
    return all(params.get(k, "none" if k.endswith(".impl") else None) == v for k, v in where.items())


def with_status() -> list[dict]:
    """Every lesson, with its best matching run so far and whether its goal is met."""
    from . import tracking
    try:
        con = tracking.connect_read()
        rows = con.execute("SELECT run_id, dataset, n_questions, params, overall FROM runs "
                           "WHERE kind = 'benchmark' AND status = 'ok'").fetchall()
        con.close()
    except Exception:
        rows = []
    runs = [{"run_id": r[0], "dataset": r[1], "n": r[2] or 0, "params": json.loads(r[3] or "{}"), "overall": json.loads(r[4] or "{}")}
            for r in rows]
    out = []
    for lesson in LESSONS:
        g, status = lesson.get("goal"), {}
        if g:
            tried = [r for r in runs if r["dataset"] == g["dataset"] and _matches(r["params"], g["where"])
                     and r["n"] >= g["min_questions"] and isinstance(r["overall"].get(g["metric"]), (int, float))]
            best = max(tried, key=lambda r: r["overall"][g["metric"]], default=None)
            status = {"tried": len(tried), "best": best and round(best["overall"][g["metric"]], 1),
                      "best_run": best and best["run_id"], "passed": bool(best) and best["overall"][g["metric"]] >= g["at_least"]}
        out.append({**lesson, "status": status})
    return out
