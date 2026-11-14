"""Fast tests: no model downloads, no network. Everything runs in a temp project root (see conftest)."""

import json
import uuid

import pytest

from rag_lab import cache, data, paths, pipeline, scripting, services, workspace
from rag_lab.config import resolve
from rag_lab.registry import INGEST_STAGES
from rag_lab.stages import chunk
from rag_lab.stages.chunk import split_fixed, split_slack_thread
from rag_lab.targets import Manifest, open_target

# A toy embedder for script nodes: letter counts. Deterministic, no model needed.
EMBED = "def embed(texts):\n    return [[t.lower().count(c) + 0.01 for c in 'abcdefghijklmnopqrstuvwxyz'] for t in texts]\n"


def _dataset(tmp_path, name, docs: dict[str, str]):
    for fname, text in docs.items():
        (tmp_path / fname).write_text(text)
    return data.ingest(name, tmp_path)


def _questions(tmp_path, rows):
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _ingest_cfg(dataset, target_id, mode="append", embed=True):
    return {"name": f"{dataset}-ingest", "data": {"dataset": dataset}, "load": {"impl": "corpus"}, "chunk": {"impl": "fixed"},
            "embed": {"impl": "python", "code": EMBED} if embed else {"impl": "none"}, "target": {"id": target_id, "mode": mode}}


def _last_write(run_id):
    return json.loads((paths.RUNS / run_id / "stages.json").read_text())[-1]["stats"]


# ---- chunking, caching, config

def test_split_fixed_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_fixed(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]
    assert chunks[-1].endswith("word499")


def test_slack_chunks_keep_whole_messages_and_root():
    msgs = [f"Person{i}: message number {i} " + "x" * 60 for i in range(20)]
    chunks = split_slack_thread("\n".join(msgs), 300, 0, 40)
    assert len(chunks) > 1 and all(c.startswith("[thread start] Person0") for c in chunks[1:])
    assert all(any(m in c for c in chunks) for m in msgs)


def test_cache_key_depends_on_config_and_code():
    a = cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 1200}, "up", "code")
    assert a != cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 800}, "up", "code")
    assert a != cache.stage_key("chunk", {"impl": "fixed", "chunk_size": 1200}, "up", "code2")
    assert a == cache.stage_key("chunk", {"chunk_size": 1200, "impl": "fixed"}, "up", "code")


def test_resolve_fills_defaults():
    cfg = resolve({"data": {"dataset": "d"}, "load": {"impl": "corpus"}, "chunk": {"impl": "fixed"}, "embed": {"impl": "none"}}, INGEST_STAGES)
    assert cfg["chunk"]["chunk_size"] == 1200 and cfg["data"] == {"dataset": "d", "subset": "all", "snapshot": "latest"}


def test_python_chunk_script_and_errors():
    doc = {"doc_id": "d", "source": "x", "title": "T", "text": "T\nalpha\n\nbeta"}
    cfg = chunk.PythonChunkConfig(code="def chunk(doc):\n    return [p.upper() for p in doc['body'].split('\\n\\n')]")
    assert chunk.preview("python", cfg, doc) == ["ALPHA", "BETA"]
    with pytest.raises(scripting.ScriptError):
        chunk.preview("python", chunk.PythonChunkConfig(code="def chunk(doc):\n    return 'not a list'"), doc)


def test_script_equivalents_match_builtins():
    doc = {"doc_id": "d", "source": "slack", "title": "eng", "text": "eng\n" + "\n".join(f"Sam{i}: message {i} " + "x" * 80 for i in range(30))}
    for impl, cls in (("fixed", chunk.FixedConfig), ("slack_aware", chunk.SlackAwareConfig)):
        p = {"chunk_size": 400, "overlap": 50}
        eq = scripting.equivalent_script("chunk", impl, p)
        assert chunk.preview("python", chunk.PythonChunkConfig(**eq), doc) == chunk.preview(impl, cls(**p), doc)


# ---- datasets

def test_dataset_requires_questions(tmp_path):
    (tmp_path / "a.txt").write_text("Alpha\nfirst doc")
    with pytest.raises(SystemExit, match="questions"):
        data.ingest("noq", tmp_path)
    con = data.connect(read_only=True)
    assert con.execute("SELECT count(*) FROM datasets WHERE name='noq'").fetchone()[0] == 0     # nothing half-created


def test_incremental_dataset_snapshots(tmp_path):
    _questions(tmp_path, [{"question_id": "q1", "question": "first?", "expected_doc_ids": ["a.txt"]}])
    assert _dataset(tmp_path, "t", {"a.txt": "Alpha\nfirst doc", "b.md": "# Beta\nsecond doc"}) == 1
    assert data.ingest("t", tmp_path) is None                                 # nothing changed
    assert _dataset(tmp_path, "t", {"b.md": "# Beta\nsecond doc, edited", "c.txt": "Gamma\nthird doc"}) == 2
    (tmp_path / "a.txt").unlink()
    assert data.ingest("t", tmp_path) is None                                 # upsert: deletions ignored
    assert data.ingest("t", tmp_path, sync=True) == 3                         # sync: a.txt deleted
    con = data.connect(read_only=True)
    snaps = con.execute("SELECT snapshot, added, changed, deleted, n_docs FROM snapshots WHERE dataset='t' ORDER BY 1").fetchall()
    assert snaps == [(1, 2, 0, 0, 2), (2, 1, 1, 0, 3), (3, 0, 0, 1, 2)]
    old = con.execute(f"SELECT text FROM ({data.docs_sql()}) WHERE doc_id='b.md'", {"ds": "t", "snap": 1, "subset": "all"}).fetchone()[0]
    assert old.endswith("second doc")                                         # old snapshot still reconstructable


def test_extractor_fills_folder_then_loads_it():
    # An extractor writes files into data/datasets/<name>/; the folder is then loaded.
    code = ("import json\nfrom pathlib import Path\n"
            "def download(folder):\n"
            "    f = Path(folder)\n"
            "    (f / 'docs.jsonl').write_text(json.dumps({'id': 'a', 'text': 'Refunds take 14 days'}) + '\\n'"
            " + json.dumps({'id': 'b', 'text': 'Orders ship from Leeds'}) + '\\n')\n"
            "    (f / 'questions.jsonl').write_text(json.dumps({'question_id': 'q1', 'question': 'where do orders ship from',"
            " 'expected_doc_ids': ['b', 'missing']}) + '\\n')\n")
    assert data.ingest_source("ex", "extract", {"code": code}) == 1
    folder = data.dataset_dir("ex")
    assert (folder / data.EXTRACTOR_FILE).read_text() == code                  # the script is kept with its data
    summary = data.folder_summary(folder)
    assert summary["has_questions"] and summary["has_extractor"] and summary["n_files"] == 3

    (folder / "notes.md").write_text("# Support\nopen on weekdays")           # files added by hand ...
    assert data.ingest_source("ex", "folder", {}) == 2                          # ... are picked up on reload
    con = data.connect(read_only=True)
    m = data.dataset_metrics(con, "ex")
    con.close()
    assert m["docs"]["n"] == 3 and m["questions"]["gold_doc_ids"] == 2 and m["questions"]["gold_docs_found"] == 1

    data.remove_dataset("ex")
    con = data.connect(read_only=True)
    assert con.execute("SELECT count(*) FROM doc_versions WHERE dataset='ex'").fetchone()[0] == 0
    con.close()
    assert folder.is_dir()                                                     # removing a dataset keeps its folder


# ---- targets: append / overwrite, and every testable type behaves the same

SERVER_PARAMS = {   # server-backed target types; each test run uses its own table / index
    "pgvector": lambda n: {"table": n}, "opensearch": lambda n: {"index": n},
    "elasticsearch": lambda n: {"index": n}, "qdrant-server": lambda n: {"url": services.url("qdrant"), "collection": n},
    "sql-postgres": lambda n: {"url": services.url("pgvector").replace("postgresql://", "postgresql+psycopg://"), "table": n},
}
SERVER_OF = {"qdrant-server": "qdrant", "sql-postgres": "pgvector"}


def _server_up(ttype):
    from rag_lab.services import _reachable
    return _reachable(services.url(SERVER_OF.get(ttype, ttype)))


@pytest.mark.parametrize("ttype", ["local", "qdrant", "python", "sql",
                                   *[pytest.param(t, marks=pytest.mark.skipif(not _server_up(t), reason=f"{t} not running"))
                                     for t in SERVER_PARAMS]])
def test_ingest_append_overwrite_and_benchmark(tmp_path, ttype):
    ds = f"shop_{ttype}".replace("-", "_")
    _questions(tmp_path, [{"question_id": "q1", "question": "leeds warehouse shipping", "expected_doc_ids": ["b.txt"], "gold_answer": "Leeds"}])
    _dataset(tmp_path, ds, {"a.txt": "Refunds\nrefunds are issued within fourteen days",
                            "b.txt": "Shipping\norders ship from the leeds warehouse"})
    params = SERVER_PARAMS[ttype](f"raglab_test_{uuid.uuid4().hex[:8]}") if ttype in SERVER_PARAMS else {}
    target = workspace.save_target({"name": f"T {ttype}", "type": ttype.split("-")[0], "params": params})
    try:
        _check_target(tmp_path, ds, target)
    finally:
        t = open_target(target)
        t.clear()           # don't leave test tables / indexes in a shared server
        t.close()


def _check_target(tmp_path, ds, target):
    cfg = _ingest_cfg(ds, target["id"])

    assert _last_write(pipeline.run_ingest(cfg))["docs_written"] == 2
    again = _last_write(pipeline.run_ingest(cfg))                              # append, nothing changed
    assert (again["docs_written"], again["docs_skipped"]) == (0, 2)
    assert Manifest(paths.TARGETS / target["id"]).data["version"] == 1          # a no-op append doesn't bump the version

    _dataset(tmp_path, ds, {"a.txt": "Refunds\nrefunds are issued within thirty days", "c.txt": "Support\nsupport is open on weekdays"})
    delta = _last_write(pipeline.run_ingest(cfg))                              # append: 1 changed + 1 new, 1 untouched
    assert (delta["docs_written"], delta["docs_skipped"]) == (2, 1)
    t = open_target(target)
    assert t.count() == 3
    t.close()

    run_id = pipeline.run_benchmark({"name": "bench", "target": {"id": target["id"]}, "retrieve": {"impl": "vector", "top_k": 2},
                                     "generate": {"impl": "none"}, "evaluate": {"impl": "standard", "judge": "auto"}})
    run_dir = paths.RUNS / run_id
    assert json.loads((run_dir / "metrics.json").read_text())["overall"]["recall@1"] == 100.0
    hit = json.loads((run_dir / "retrieval.jsonl").read_text())["chunks"][0]
    assert hit["doc_id"] == "b.txt" and "leeds" in hit["text"]                 # chunk text travels with the hit
    assert json.loads((run_dir / "config.json").read_text())["target"]["version"] == 2
    if open_target(target).text_search:
        run_id = pipeline.run_benchmark({"name": "bench-kw", "target": {"id": target["id"]}, "retrieve": {"impl": "bm25", "top_k": 2},
                                         "generate": {"impl": "none"}, "evaluate": {"impl": "standard", "judge": "auto"}})
        assert json.loads((paths.RUNS / run_id / "metrics.json").read_text())["overall"]["recall@1"] == 100.0

    over = _last_write(pipeline.run_ingest({**cfg, "target": {"id": target["id"], "mode": "overwrite"}}))
    assert (over["docs_written"], over["docs_skipped"]) == (3, 0)
    m = Manifest(paths.TARGETS / target["id"]).data
    assert (m["version"], m["n_docs"], [h["mode"] for h in m["history"]]) == (3, 3, ["append", "append", "overwrite"])


def test_append_with_different_embedder_is_refused(tmp_path):
    _questions(tmp_path, [{"question_id": "q1", "question": "x", "expected_doc_ids": ["a.txt"]}])
    _dataset(tmp_path, "emb", {"a.txt": "Alpha\nsome text"})
    target = workspace.save_target({"name": "T emb", "type": "local", "params": {}})
    pipeline.run_ingest(_ingest_cfg("emb", target["id"]))
    other = {**_ingest_cfg("emb", target["id"]), "embed": {"impl": "python", "code": EMBED + "# v2\n"}}
    with pytest.raises(SystemExit, match="different embedder"):
        pipeline.run_ingest(other)
    pipeline.run_ingest({**other, "target": {"id": target["id"], "mode": "overwrite"}})      # overwrite is allowed


def test_bm25_only_target_and_script_nodes(tmp_path):
    _questions(tmp_path, [{"question_id": "q1", "question": "where do orders ship from", "expected_doc_ids": ["b.txt"]}])
    _dataset(tmp_path, "kw", {"a.txt": "Refunds\nrefunds within fourteen days", "b.txt": "Shipping\norders ship from leeds"})
    target = workspace.save_target({"name": "T kw", "type": "local", "params": {"vectors": False}})
    pipeline.run_ingest(_ingest_cfg("kw", target["id"], embed=False))
    base = {"name": "kw", "target": {"id": target["id"]}, "generate": {"impl": "none"}, "evaluate": {"impl": "standard", "judge": False}}
    rerank = "def rerank(question, chunks, top_n):\n    return list(reversed(chunks))[:top_n]\n"
    retrieve = "def retrieve(question, search, top_k):\n    return search.bm25(question, top_k)\n"
    run_dir = paths.RUNS / pipeline.run_benchmark({**base, "retrieve": {"impl": "python", "code": retrieve, "top_k": 2},
                                                   "rerank": {"impl": "python", "code": rerank, "top_n": 2}})
    retrieved = json.loads((run_dir / "retrieval.jsonl").read_text())["chunks"]
    reranked = json.loads((run_dir / "reranked.jsonl").read_text())["chunks"]
    assert retrieved[0]["doc_id"] == "b.txt"
    assert [c["chunk_id"] for c in reranked] == [c["chunk_id"] for c in reversed(retrieved)]
    with pytest.raises(SystemExit, match="no vector search"):                  # vector retrieval on a keyword-only target
        pipeline.run_benchmark({**base, "retrieve": {"impl": "vector"}})


def test_transform_node_rewrites_and_drops_documents(tmp_path):
    _questions(tmp_path, [{"question_id": "q1", "question": "warehouse", "expected_doc_ids": ["b.txt"]}])
    _dataset(tmp_path, "tr", {"a.txt": "Refunds\nrefunds within fourteen days", "b.txt": "Shipping\n<b>orders</b> ship from the leeds depot",
                              "c.txt": "Draft\nDRAFT do not index"})
    target = workspace.save_target({"name": "T tr", "type": "local", "params": {"vectors": False}})
    code = ("def transform(doc):\n"
            "    if 'DRAFT' in doc['text']:\n        return None\n"
            "    return {**doc, 'text': doc['text'].replace('<b>', '').replace('</b>', '').replace('depot', 'warehouse')}\n")
    cfg = {**_ingest_cfg("tr", target["id"], embed=False), "transform": {"impl": "python", "code": code}}
    run_id = pipeline.run_ingest(cfg)
    stats = {s["stage"]: s["stats"] for s in json.loads((paths.RUNS / run_id / "stages.json").read_text())}
    assert (stats["transform"]["docs_in"], stats["transform"]["dropped"], stats["transform"]["changed"]) == (3, 1, 1)
    assert stats["write"]["docs_written"] == 2                                 # the dropped document never reaches the target
    t = open_target(target)
    hits = t.search_text("warehouse", 5)
    t.close()
    assert [h["doc_id"] for h in hits] == ["b.txt"] and "<b>" not in hits[0]["text"]

    # Changing the transform changes the documents' content hashes, so an append rewrites them.
    cfg["transform"]["code"] = code.replace("'warehouse'", "'warehouse building'")
    assert _last_write(pipeline.run_ingest(cfg))["docs_written"] == 1


def test_clean_transform_script_equivalent_matches():
    from rag_lab.stages import transform
    params = {"drop_lines_matching": "^Sent from", "min_chars": 3, "max_chars": 40}
    fn = scripting.load_functions(scripting.equivalent_script("transform", "clean", params)["code"], "transform")["transform"]
    builtin = transform.make_cleaner(transform.CleanConfig(**params))
    for text in ["<p>Hi &amp;  there</p>\nSent from phone\n\n\n\nbye", "x", "<script>var a</script>long " * 10]:
        doc = {"doc_id": "a", "source": "s", "title": "", "text": text, "metadata": {}}
        assert fn(dict(doc)) == builtin(dict(doc))


# ---- canvas graphs

def test_graphs_compile_to_configs():
    t1 = workspace.save_target({"name": "Store A", "type": "local", "params": {}})
    t2 = workspace.save_target({"name": "Store B", "type": "local", "params": {"vectors": False}})
    ing = {"id": "p", "name": "P", "nodes": [
        {"id": "s", "stage": "source", "impl": "dataset", "params": {"dataset": "t"}},
        {"id": "c1", "stage": "chunk", "impl": "fixed", "params": {"chunk_size": 500}},
        {"id": "c2", "stage": "chunk", "impl": "python", "params": {}},
        {"id": "e1", "stage": "embed", "impl": "sentence_transformers", "params": {}},
        {"id": "ta", "stage": "target", "impl": "target", "params": {"target_id": t1["id"], "mode": "overwrite"}},
        {"id": "tb", "stage": "target", "impl": "target", "params": {"target_id": t2["id"]}},
        {"id": "tc", "stage": "target", "impl": "target", "params": {"target_id": t1["id"]}}],
        "edges": [{"source": "s", "target": "c1"}, {"source": "c1", "target": "e1"}, {"source": "e1", "target": "ta"},
                  {"source": "s", "target": "c2"}, {"source": "c2", "target": "tb"}]}
    a, meta = workspace.compile_ingest(ing, "ta")
    b, _ = workspace.compile_ingest(ing, "tb")
    assert a["chunk"] == {"impl": "fixed", "chunk_size": 500} and a["target"] == {"id": t1["id"], "mode": "overwrite"}
    assert b["embed"] == {"impl": "none"} and b["target"]["mode"] == "append" and meta["target_name"] == "Store A"
    with pytest.raises(workspace.GraphError):
        workspace.compile_ingest(ing, "tc")                                    # not connected to a source
    ret = {"id": "r", "name": "R", "nodes": [
        {"id": "t", "stage": "target", "impl": "target", "params": {"target_id": t1["id"]}},
        {"id": "x", "stage": "retrieve", "impl": "bm25", "params": {"top_k": 5}},
        {"id": "b", "stage": "benchmark", "impl": "benchmark", "params": {"limit": 7, "judge": False}}], "edges": []}
    cfg, meta, limit = workspace.compile_benchmark(ret)
    assert cfg["target"] == {"id": t1["id"]} and cfg["rerank"] == {"impl": "none"} and limit == 7 and cfg["evaluate"]["judge"] is False
    assert workspace.compile_benchmark(ret, t2["id"])[1]["target_name"] == "Store B"       # run-time target override
