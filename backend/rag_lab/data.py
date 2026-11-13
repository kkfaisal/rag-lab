"""Benchmark data: download, load into DuckDB, and build dev subsets.

Tables in data/corpus.duckdb:
  docs(doc_id, source, path, title, text)       one row per benchmark document
  questions(question_id, category, question, gold_answer, expected_doc_ids, answer_facts, source_types)
  subset_docs(subset, doc_id)                   which docs belong to a named subset
  subset_questions(subset, question_id, ord)    which questions belong to it (ord = run order)
"""

import json
import random
import re
import zipfile
from pathlib import Path

import duckdb
import httpx

from . import paths

RELEASE_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
SOURCES = ["slack", "gmail", "linear", "google_drive", "hubspot", "fireflies", "github", "jira", "confluence"]
DOC_ID_RE = re.compile(r"(dsid_[0-9a-f]+)")


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def download(slices: list[str] | None = None) -> None:
    """Fetch questions.jsonl plus either all_documents.zip (default) or the named
    per-source slice zips, e.g. ["slack_slice_0001", "confluence_slice_0001"]."""
    paths.RAW.mkdir(parents=True, exist_ok=True)
    _download(f"{RELEASE_URL}/questions.jsonl", paths.QUESTIONS_FILE)
    for name in slices or ["all_documents"]:
        _download(f"{RELEASE_URL}/{name}.zip", paths.RAW / f"{name}.zip")


def _iter_zip_docs(zip_path: Path):
    """Yield (doc_id, source, path, title, text) for every .txt in a release zip.
    all_documents.zip keeps <source>/<sub>/... dirs; slice zips are flat and the
    source comes from the zip name."""
    zip_source = next((s for s in SOURCES if zip_path.name.startswith(s + "_slice")), None)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            m = DOC_ID_RE.search(Path(name).name)
            if not m:
                continue
            source = zip_source or name.split("/")[0]
            text = z.read(name).decode("utf-8", errors="replace")
            title = text.split("\n", 1)[0].strip()
            yield m.group(1), source, name, title, text


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(paths.CORPUS_DB), read_only=read_only)


def load() -> None:
    """(Re)build data/corpus.duckdb from every zip in data/raw/ plus questions.jsonl."""
    import pyarrow as pa

    zips = sorted(paths.RAW.glob("*.zip"))
    if not zips:
        raise SystemExit("No zips in data/raw/. Run `rag-lab data download` first.")
    paths.CORPUS_DB.unlink(missing_ok=True)
    con = connect()
    con.execute("CREATE TABLE docs(doc_id VARCHAR, source VARCHAR, path VARCHAR, title VARCHAR, text VARCHAR)")
    cols = ["doc_id", "source", "path", "title", "text"]
    for zp in zips:
        batch, n = [], 0
        for row in _iter_zip_docs(zp):
            batch.append(row)
            if len(batch) == 20_000:
                n += _flush(con, batch, cols, pa)
                print(f"\r  {zp.name}: {n:,} docs", end="", flush=True)
        n += _flush(con, batch, cols, pa)
        print(f"\r  {zp.name}: {n:,} docs")
    # A doc can appear in both all_documents.zip and a slice zip: keep one copy.
    con.execute("CREATE TABLE docs_dedup AS SELECT * FROM docs QUALIFY row_number() OVER (PARTITION BY doc_id ORDER BY path) = 1")
    con.execute("DROP TABLE docs; ALTER TABLE docs_dedup RENAME TO docs")
    con.execute("CREATE UNIQUE INDEX docs_pk ON docs(doc_id)")

    qs = [json.loads(line) for line in open(paths.QUESTIONS_FILE)]
    con.execute("""CREATE TABLE questions(question_id VARCHAR PRIMARY KEY, category VARCHAR, question VARCHAR,
                   gold_answer VARCHAR, expected_doc_ids VARCHAR[], answer_facts VARCHAR[], source_types VARCHAR[])""")
    con.executemany("INSERT INTO questions VALUES (?,?,?,?,?,?,?)", [
        (q["question_id"], q["question_type"], q["question"], q.get("gold_answer"),
         q.get("expected_doc_ids") or [], q.get("answer_facts") or [], q.get("source_types") or []) for q in qs])
    con.execute("CREATE TABLE subset_docs(subset VARCHAR, doc_id VARCHAR)")
    con.execute("CREATE TABLE subset_questions(subset VARCHAR, question_id VARCHAR, ord INTEGER)")
    _add_full_subset(con)
    print(con.execute("SELECT source, count(*) FROM docs GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print(f"Loaded {con.execute('SELECT count(*) FROM docs').fetchone()[0]:,} docs, {len(qs)} questions")
    con.close()


def _flush(con, batch, cols, pa) -> int:
    if not batch:
        return 0
    tbl = pa.table({c: [r[i] for r in batch] for i, c in enumerate(cols)})
    con.register("batch_tbl", tbl)
    con.execute("INSERT INTO docs SELECT * FROM batch_tbl")
    con.unregister("batch_tbl")
    n = len(batch)
    batch.clear()
    return n


def _add_full_subset(con) -> None:
    """The 'full' subset = every doc and every question (round-robin by category)."""
    con.execute("DELETE FROM subset_docs WHERE subset='full'; DELETE FROM subset_questions WHERE subset='full'")
    con.execute("INSERT INTO subset_docs SELECT 'full', doc_id FROM docs")
    qids = _round_robin(con.execute("SELECT question_id, category FROM questions ORDER BY question_id").fetchall())
    con.executemany("INSERT INTO subset_questions VALUES ('full', ?, ?)", [(q, i) for i, q in enumerate(qids)])


def _round_robin(rows: list[tuple[str, str]]) -> list[str]:
    """Order question ids so categories interleave: with --limit 20 you get ~2 per category."""
    by_cat: dict[str, list[str]] = {}
    for qid, cat in rows:
        by_cat.setdefault(cat, []).append(qid)
    out, i = [], 0
    while any(i < len(v) for v in by_cat.values()):
        out += [v[i] for _, v in sorted(by_cat.items()) if i < len(v)]
        i += 1
    return out


def build_subset(name: str = "dev", per_category: int = 10, distractors: int = 20_000, seed: int = 42) -> None:
    """Dev subset: `per_category` questions from each of the 10 categories, all their
    gold docs, plus `distractors` random other docs. Seeded, so everyone gets the same subset."""
    rng = random.Random(seed)
    con = connect()
    rows = con.execute("SELECT question_id, category, expected_doc_ids FROM questions ORDER BY question_id").fetchall()
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r[1], []).append(r)
    picked = [r for cat in sorted(by_cat) for r in rng.sample(by_cat[cat], min(per_category, len(by_cat[cat])))]
    gold = sorted({d for r in picked for d in r[2]})
    others = [d for (d,) in con.execute("SELECT doc_id FROM docs ORDER BY doc_id").fetchall()]
    gold_set = set(gold)
    pool = [d for d in others if d not in gold_set]
    doc_ids = gold + rng.sample(pool, min(distractors, len(pool)))
    present = {d for (d,) in con.execute("SELECT doc_id FROM docs").fetchall()}
    missing = [d for d in gold if d not in present]

    con.execute("DELETE FROM subset_docs WHERE subset=?; ", [name])
    con.execute("DELETE FROM subset_questions WHERE subset=?", [name])
    con.executemany("INSERT INTO subset_docs VALUES (?, ?)", [(name, d) for d in doc_ids if d in present])
    qids = _round_robin([(r[0], r[1]) for r in picked])
    con.executemany("INSERT INTO subset_questions VALUES (?, ?, ?)", [(name, q, i) for i, q in enumerate(qids)])
    con.close()
    print(f"Subset '{name}': {len(qids)} questions, {len(gold)} gold docs, {len(doc_ids) - len(gold)} distractors")
    if missing:
        print(f"  WARNING: {len(missing)} gold docs not in corpus (did you download all slices?)")


def subset_question_ids(con, subset: str, limit: int | None = None) -> list[str]:
    ids = [q for (q,) in con.execute(
        "SELECT question_id FROM subset_questions WHERE subset=? ORDER BY ord", [subset]).fetchall()]
    if not ids:
        raise SystemExit(f"Unknown or empty subset '{subset}'. Run `rag-lab data subset --name {subset}`.")
    return ids[:limit] if limit else ids


def load_questions(con, question_ids: list[str]) -> list[dict]:
    rows = con.execute("SELECT * FROM questions WHERE question_id IN (SELECT unnest(?))", [question_ids]).fetchall()
    cols = [d[0] for d in con.description]
    by_id = {r[0]: dict(zip(cols, r)) for r in rows}
    return [by_id[q] for q in question_ids]
