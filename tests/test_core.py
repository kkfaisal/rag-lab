from rag_lab import cache, data
from rag_lab.config import apply_overrides, resolve
from rag_lab.stages.chunk import split_fixed, split_slack_thread


def test_split_fixed_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_fixed(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]          # neighbours overlap
    assert chunks[-1].endswith("word499")


def test_slack_chunks_keep_whole_messages_and_root():
    msgs = [f"Person{i}: message number {i} " + "x" * 60 for i in range(20)]
    chunks = split_slack_thread("\n".join(msgs), 300, 0, 40)
    assert len(chunks) > 1
    assert all(c.startswith("[thread start] Person0") for c in chunks[1:])
    for m in msgs:                                       # no message is split across chunks
        assert any(m in c for c in chunks)


def test_cache_key_changes_only_downstream():
    a = cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 1200}, "up", "code")
    b = cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 800}, "up", "code")
    c = cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 1200}, "up", "code2")
    assert len({a, b, c}) == 3
    assert a == cache.stage_key("chunk", {"chunk_size": 1200, "impl": "fixed"}, "up", "code")


def test_override_switching_impl_drops_old_params():
    raw = {"generate": {"impl": "llm", "model": "x"}}
    assert apply_overrides(raw, ["generate.impl=none"])["generate"] == {"impl": "none"}


def test_resolve_fills_defaults():
    cfg = resolve({"load": {"impl": "corpus"}, "chunk": {"impl": "fixed"}, "embed": {"impl": "none"},
                   "index": {"impl": "duckdb", "vectors": False}, "retrieve": {"impl": "bm25"},
                   "generate": {"impl": "none"}, "evaluate": {"impl": "standard"}})
    assert cfg["chunk"]["chunk_size"] == 1200 and cfg["rerank"] == {"impl": "none"}
    assert cfg["data"]["snapshot"] == "latest"


def test_incremental_ingest(tmp_path):
    (tmp_path / "a.txt").write_text("Alpha\nfirst doc")
    (tmp_path / "b.md").write_text("# Beta\nsecond doc")
    assert data.ingest("t", tmp_path) == 1
    assert data.ingest("t", tmp_path) is None                        # nothing changed
    (tmp_path / "b.md").write_text("# Beta\nsecond doc, edited")
    (tmp_path / "c.txt").write_text("Gamma\nthird doc")
    assert data.ingest("t", tmp_path) == 2
    (tmp_path / "a.txt").unlink()
    assert data.ingest("t", tmp_path) is None                        # upsert: deletions ignored
    assert data.ingest("t", tmp_path, sync=True) == 3                # sync: a.txt deleted
    con = data.connect(read_only=True)
    snaps = con.execute("SELECT snapshot, added, changed, deleted, n_docs FROM snapshots WHERE dataset='t' ORDER BY 1").fetchall()
    assert snaps == [(1, 2, 0, 0, 2), (2, 1, 1, 0, 3), (3, 0, 0, 1, 2)]
    at = lambda s: sorted(r[0] for r in con.execute(f"SELECT doc_id, text FROM ({data.docs_sql()})",  # noqa: E731
                                                     {"ds": "t", "snap": s, "subset": "all"}).fetchall())
    assert at(1) == ["a.txt", "b.md"] and at(2) == ["a.txt", "b.md", "c.txt"] and at(3) == ["b.md", "c.txt"]
    old = con.execute(f"SELECT text FROM ({data.docs_sql()}) WHERE doc_id='b.md'", {"ds": "t", "snap": 1, "subset": "all"}).fetchone()[0]
    assert old.endswith("second doc")                                # old snapshot still reconstructable


# ---- playground: canvas graphs, script nodes, script datasets

def _pipeline():
    return {"id": "p", "name": "P", "nodes": [
        {"id": "s", "stage": "source", "impl": "dataset", "params": {"dataset": "t", "subset": "all"}, "position": {"x": 0, "y": 0}},
        {"id": "c1", "stage": "chunk", "impl": "fixed", "params": {"chunk_size": 500}, "position": {"x": 0, "y": 0}},
        {"id": "c2", "stage": "chunk", "impl": "python", "params": {}, "position": {"x": 0, "y": 0}},
        {"id": "e1", "stage": "embed", "impl": "sentence_transformers", "params": {}, "position": {"x": 0, "y": 0}},
        {"id": "i1", "stage": "index", "impl": "duckdb", "params": {}, "label": "A", "position": {"x": 0, "y": 0}},
        {"id": "i2", "stage": "index", "impl": "duckdb", "params": {"vectors": False}, "label": "B", "position": {"x": 0, "y": 0}},
        {"id": "i3", "stage": "index", "impl": "duckdb", "params": {}, "position": {"x": 0, "y": 0}}],
        "edges": [{"source": "s", "target": "c1"}, {"source": "c1", "target": "e1"}, {"source": "e1", "target": "i1"},
                  {"source": "s", "target": "c2"}, {"source": "c2", "target": "i2"}]}


def test_branching_pipeline_compiles_each_index_path():
    import pytest
    from rag_lab import workspace as w
    p = _pipeline()
    a, b = w.compile_index(p, "i1"), w.compile_index(p, "i2")
    assert a["chunk"] == {"impl": "fixed", "chunk_size": 500} and a["embed"]["impl"] == "sentence_transformers"
    assert b["chunk"]["impl"] == "python" and b["embed"] == {"impl": "none"}     # chunk -> index: BM25 only
    assert a["data"] == b["data"] == {"dataset": "t", "subset": "all", "snapshot": "latest"}
    with pytest.raises(w.GraphError):
        w.compile_index(p, "i3")                                                  # not connected to a source
    r = {"id": "r", "name": "R", "nodes": [{"id": "x", "stage": "retrieve", "impl": "bm25", "params": {"top_k": 5}}], "edges": []}
    cfg, meta = w.experiment_config(p, "i2", r, judge=False)
    assert resolve(cfg)["rerank"] == {"impl": "none"} and meta["retriever_id"] == "r"
    edited = {**p, "nodes": [{**n, "params": {"chunk_size": 900}} if n["id"] == "c1" else n for n in p["nodes"]]}
    assert w.build_config(edited, "i1")[1]["spec_hash"] != w.build_config(p, "i1")[1]["spec_hash"]   # edit -> stale
    assert w.build_config(edited, "i2")[1]["spec_hash"] == w.build_config(p, "i2")[1]["spec_hash"]   # other branch untouched


def test_python_chunk_script_and_errors():
    import pytest
    from rag_lab.scripting import ScriptError
    from rag_lab.stages import chunk
    doc = {"doc_id": "d", "source": "x", "title": "T", "text": "T\nalpha\n\nbeta"}
    cfg = chunk.PythonChunkConfig(code="def chunk(doc):\n    return [p.upper() for p in doc['body'].split('\\n\\n')]")
    assert chunk.preview("python", cfg, doc) == ["ALPHA", "BETA"]
    with pytest.raises(ScriptError):
        chunk.preview("python", chunk.PythonChunkConfig(code="def chunk(doc):\n    return 'not a list'"), doc)
    with pytest.raises(ScriptError):
        chunk.preview("python", chunk.PythonChunkConfig(code="x = 1"), doc)


def test_script_dataset_end_to_end_bm25(tmp_path):
    """Extractor script -> ingest -> script chunker -> BM25 index -> experiment, no models needed."""
    import json
    from rag_lab import pipeline
    (tmp_path / "rows.csv").write_text("id,body\na,Refunds are issued within 14 days\nb,Orders ship from the Leeds warehouse\n")
    (tmp_path / "questions.jsonl").write_text(json.dumps({"question_id": "q1", "question": "where do orders ship from", "expected_doc_ids": ["b"]}) + "\n")
    code = ("import csv, json\nfrom pathlib import Path\n"
            "def extract(path):\n    for r in csv.DictReader(open(Path(path) / 'rows.csv')):\n        yield {'doc_id': r['id'], 'text': r['body']}\n"
            "def questions(path):\n    for line in open(Path(path) / 'questions.jsonl'):\n        yield json.loads(line)\n")
    assert data.ingest("csv", tmp_path, adapter="script", code=code) == 1
    cfg = {"name": "csv-test", "data": {"dataset": "csv", "subset": "all"}, "load": {"impl": "corpus"},
           "chunk": {"impl": "python", "code": "def chunk(doc):\n    return [doc['text']]"}, "embed": {"impl": "none"},
           "index": {"impl": "duckdb", "vectors": False}, "retrieve": {"impl": "bm25", "top_k": 1},
           "generate": {"impl": "none"}, "evaluate": {"impl": "standard", "judge": "auto"}}
    build_id = pipeline.run_config({k: cfg[k] for k in ["name", "data", "load", "chunk", "embed", "index"]}, stop_after="index", meta={"node_id": "n"})
    run_id = pipeline.run_config(cfg)
    from rag_lab import paths
    assert json.loads((paths.RUNS / build_id / "run.json").read_text())["kind"] == "build"
    stages = json.loads((paths.RUNS / run_id / "stages.json").read_text())
    assert [s["cached"] for s in stages[:4]] == [True] * 4                       # the experiment reused the build
    assert json.loads((paths.RUNS / run_id / "metrics.json").read_text())["overall"]["recall@1"] == 100.0


def test_script_embed_and_rerank_nodes(tmp_path):
    """Embed + rerank script nodes end to end with a toy bag-of-letters embedding (no model download)."""
    import json
    from rag_lab import paths, pipeline
    (tmp_path / "a.txt").write_text("Refunds\nrefunds are issued within fourteen days")
    (tmp_path / "b.txt").write_text("Shipping\norders ship from the leeds warehouse")
    (tmp_path / "questions.jsonl").write_text(json.dumps({"question_id": "q1", "question": "leeds warehouse shipping", "expected_doc_ids": ["b.txt"]}) + "\n")
    data.ingest("toy", tmp_path)
    embed = ("def embed(texts):\n    return [[t.lower().count(c) + 0.01 for c in 'abcdefghijklmnopqrstuvwxyz'] for t in texts]\n")
    rerank = "def rerank(question, chunks, top_n):\n    return list(reversed(chunks))[:top_n]\n"
    cfg = {"name": "toy", "data": {"dataset": "toy", "subset": "all"}, "load": {"impl": "corpus"}, "chunk": {"impl": "fixed"},
           "embed": {"impl": "python", "code": embed}, "index": {"impl": "duckdb", "fts": False},
           "retrieve": {"impl": "vector", "top_k": 2}, "rerank": {"impl": "python", "code": rerank, "top_n": 2},
           "generate": {"impl": "none"}, "evaluate": {"impl": "standard", "judge": False}}
    run_dir = paths.RUNS / pipeline.run_config(cfg)
    retrieved = json.loads((run_dir / "retrieval.jsonl").read_text())["chunks"]
    reranked = json.loads((run_dir / "reranked.jsonl").read_text())["chunks"]
    assert retrieved[0]["doc_id"] == "b.txt"                                   # the script's vectors drive the search
    assert [c["chunk_id"] for c in reranked] == [c["chunk_id"] for c in reversed(retrieved)]


def test_script_equivalents_match_builtins():
    from rag_lab import scripting
    from rag_lab.stages import chunk
    doc = {"doc_id": "d", "source": "slack", "title": "eng", "text": "eng\n" + "\n".join(f"Sam{i}: message {i} " + "x" * 80 for i in range(30))}
    for impl, cls in (("fixed", chunk.FixedConfig), ("slack_aware", chunk.SlackAwareConfig)):
        p = {"chunk_size": 400, "overlap": 50}
        eq = scripting.equivalent_script("chunk", impl, p)
        assert chunk.preview("python", chunk.PythonChunkConfig(**eq), doc) == chunk.preview(impl, cls(**p), doc)
