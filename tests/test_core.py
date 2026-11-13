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
