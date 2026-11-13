"""The document store: every dataset, every version of every doc, in data/corpus.duckdb.

Ingesting is incremental. Each `rag-lab data add` compares the incoming docs with
the dataset's current state by content hash and records a new *snapshot* holding
only the difference (added / changed / deleted docs). Old snapshots stay
reconstructable, so a run can always say exactly which documents it saw.

Tables
  datasets(name, adapter, created_at)
  snapshots(dataset, snapshot, created_at, added, changed, deleted, n_docs, note)
  doc_contents(content_hash, title, text)            text stored once per distinct content
  doc_versions(dataset, doc_id, source, path, metadata, updated_at, content_hash, valid_from, valid_to)
      a doc is in snapshot S if valid_from <= S and (valid_to is null or valid_to > S)
  questions(dataset, question_id, category, question, gold_answer, expected_doc_ids, answer_facts, source_types)
  subset_docs(dataset, subset, doc_id)               named doc subsets (the subset "all" needs no rows)
  subset_questions(dataset, subset, question_id, ord)
"""

import hashlib
import json
import random
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa

from . import paths
from .datasets import get_adapter

BENCH = "enterprise_rag_bench"
RELEASE_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(name VARCHAR PRIMARY KEY, adapter VARCHAR, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS snapshots(dataset VARCHAR, snapshot INTEGER, created_at TIMESTAMP,
  added INTEGER, changed INTEGER, deleted INTEGER, n_docs INTEGER, note VARCHAR);
CREATE TABLE IF NOT EXISTS doc_contents(content_hash VARCHAR, title VARCHAR, text VARCHAR);
CREATE TABLE IF NOT EXISTS doc_versions(dataset VARCHAR, doc_id VARCHAR, source VARCHAR, path VARCHAR,
  metadata JSON, updated_at VARCHAR, content_hash VARCHAR, valid_from INTEGER, valid_to INTEGER);
CREATE TABLE IF NOT EXISTS questions(dataset VARCHAR, question_id VARCHAR, category VARCHAR, question VARCHAR,
  gold_answer VARCHAR, expected_doc_ids VARCHAR[], answer_facts VARCHAR[], source_types VARCHAR[]);
CREATE TABLE IF NOT EXISTS subset_docs(dataset VARCHAR, subset VARCHAR, doc_id VARCHAR);
CREATE TABLE IF NOT EXISTS subset_questions(dataset VARCHAR, subset VARCHAR, question_id VARCHAR, ord INTEGER);
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    paths.DATA.mkdir(parents=True, exist_ok=True)
    if read_only and not paths.CORPUS_DB.exists():
        raise SystemExit("No data/corpus.duckdb yet. See the README quickstart (rag-lab data download / load).")
    con = duckdb.connect(str(paths.CORPUS_DB), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def content_hash(source: str, title: str, text: str) -> str:
    return hashlib.sha256(f"{source}\x00{title}\x00{text}".encode()).hexdigest()[:32]


# ----------------------------------------------------------------------------- download

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def download(slices: list[str] | None = None) -> None:
    """Fetch EnterpriseRAG-Bench questions.jsonl plus all_documents.zip (default) or
    the named per-source slice zips, e.g. ["slack_slice_0001"], into data/raw/."""
    paths.RAW.mkdir(parents=True, exist_ok=True)
    _download(f"{RELEASE_URL}/questions.jsonl", paths.QUESTIONS_FILE)
    for name in slices or ["all_documents"]:
        _download(f"{RELEASE_URL}/{name}.zip", paths.RAW / f"{name}.zip")


# ----------------------------------------------------------------------------- ingest

def ingest(dataset: str, path: str | Path, adapter: str | None = None, sync: bool = False, note: str = "") -> int | None:
    """Add docs from `path` to `dataset` and record a new snapshot.

    sync=False (default): upsert — new docs are added, changed docs get a new
    version, docs absent from `path` are left alone (incremental addition).
    sync=True: `path` is the complete dataset — docs absent from it are deleted.
    Returns the new snapshot number, or None if nothing changed.
    """
    path = Path(path)
    con = connect()
    row = con.execute("SELECT adapter FROM datasets WHERE name=?", [dataset]).fetchone()
    adapter = adapter or (row[0] if row else (BENCH if dataset == BENCH else "folder"))
    if row and row[0] != adapter:
        raise SystemExit(f"Dataset {dataset!r} uses adapter {row[0]!r}, not {adapter!r}")
    ds = get_adapter(adapter)
    if not row:
        con.execute("INSERT INTO datasets VALUES (?, ?, now())", [dataset, adapter])

    # 1. Stage every incoming doc with its content hash.
    con.execute("""CREATE OR REPLACE TEMP TABLE staged(doc_id VARCHAR, source VARCHAR, path VARCHAR, title VARCHAR,
                   text VARCHAR, metadata JSON, updated_at VARCHAR, content_hash VARCHAR)""")
    batch, n = [], 0
    for d in ds.iter_docs(path):
        batch.append((d.doc_id, d.source, d.path, d.title, d.text, json.dumps(d.metadata), d.updated_at,
                      content_hash(d.source, d.title, d.text)))
        if len(batch) == 20_000:
            n += _insert_staged(con, batch)
            print(f"\r  read {n:,} docs", end="", flush=True)
    n += _insert_staged(con, batch)
    print(f"\r  read {n:,} docs")
    con.execute("""CREATE OR REPLACE TEMP TABLE incoming AS SELECT * FROM staged
                   QUALIFY row_number() OVER (PARTITION BY doc_id ORDER BY path) = 1""")

    # 2. Diff against the current state (open versions) by content hash.
    snap = (con.execute("SELECT max(snapshot) FROM snapshots WHERE dataset=?", [dataset]).fetchone()[0] or 0) + 1
    con.execute("""CREATE OR REPLACE TEMP TABLE cur AS
                   SELECT doc_id, content_hash FROM doc_versions WHERE dataset=? AND valid_to IS NULL""", [dataset])
    added = con.execute("SELECT count(*) FROM incoming i ANTI JOIN cur USING (doc_id)").fetchone()[0]
    changed = con.execute("SELECT count(*) FROM incoming i JOIN cur c USING (doc_id) WHERE i.content_hash <> c.content_hash").fetchone()[0]
    deleted = con.execute("SELECT count(*) FROM cur ANTI JOIN incoming USING (doc_id)").fetchone()[0] if sync else 0
    if added + changed + deleted == 0:
        print(f"No changes for '{dataset}'.")
        _upsert_questions(con, dataset, ds, path)
        con.close()
        return None

    # 3. Store new contents once, close replaced/deleted versions, open new ones.
    con.execute("""INSERT INTO doc_contents SELECT content_hash, any_value(title), any_value(text) FROM incoming
                   WHERE content_hash NOT IN (SELECT content_hash FROM doc_contents) GROUP BY content_hash""")
    con.execute("""UPDATE doc_versions SET valid_to = $snap WHERE dataset = $ds AND valid_to IS NULL AND doc_id IN (
                     SELECT i.doc_id FROM incoming i JOIN cur c USING (doc_id) WHERE i.content_hash <> c.content_hash)""",
                {"snap": snap, "ds": dataset})
    if sync:
        con.execute("""UPDATE doc_versions SET valid_to = $snap WHERE dataset = $ds AND valid_to IS NULL
                       AND doc_id NOT IN (SELECT doc_id FROM incoming)""", {"snap": snap, "ds": dataset})
    con.execute("""INSERT INTO doc_versions
                   SELECT $ds, i.doc_id, i.source, i.path, i.metadata, i.updated_at, i.content_hash, $snap, NULL
                   FROM incoming i LEFT JOIN cur c USING (doc_id)
                   WHERE c.doc_id IS NULL OR i.content_hash <> c.content_hash""", {"snap": snap, "ds": dataset})
    n_docs = con.execute(f"SELECT count(*) FROM ({docs_sql()})", {"ds": dataset, "snap": snap, "subset": "all"}).fetchone()[0]
    con.execute("INSERT INTO snapshots VALUES (?, ?, now(), ?, ?, ?, ?, ?)",
                [dataset, snap, added, changed, deleted, n_docs, note or str(path)])
    _upsert_questions(con, dataset, ds, path)
    print(f"Dataset '{dataset}' snapshot {snap}: +{added:,} added, ~{changed:,} changed, -{deleted:,} deleted -> {n_docs:,} docs")
    con.close()
    return snap


def _insert_staged(con, batch) -> int:
    if not batch:
        return 0
    cols = ["doc_id", "source", "path", "title", "text", "metadata", "updated_at", "content_hash"]
    con.register("batch_tbl", pa.table({c: [r[i] for r in batch] for i, c in enumerate(cols)}))
    con.execute("INSERT INTO staged SELECT * FROM batch_tbl")
    con.unregister("batch_tbl")
    n = len(batch)
    batch.clear()
    return n


def _upsert_questions(con, dataset: str, ds, path: Path) -> None:
    qs = list(ds.iter_questions(path))
    if not qs:
        return
    con.execute("DELETE FROM questions WHERE dataset=? AND question_id IN (SELECT unnest(?))",
                [dataset, [q["question_id"] for q in qs]])
    con.executemany("INSERT INTO questions VALUES (?,?,?,?,?,?,?,?)", [
        (dataset, q["question_id"], q.get("category", "default"), q["question"], q.get("gold_answer"),
         q.get("expected_doc_ids") or [], q.get("answer_facts") or [], q.get("source_types") or []) for q in qs])
    print(f"  {len(qs)} questions")


# ----------------------------------------------------------------------------- reading

def docs_sql(columns: str = "v.doc_id, v.source, v.path, c.title, c.text, v.content_hash") -> str:
    """Docs of dataset $ds as of snapshot $snap, restricted to subset $subset
    ("all" = no restriction). The single definition used by runs and the API."""
    return f"""
      SELECT {columns} FROM doc_versions v JOIN doc_contents c USING (content_hash)
      WHERE v.dataset = $ds AND v.valid_from <= $snap AND (v.valid_to IS NULL OR v.valid_to > $snap)
        AND ($subset = 'all' OR v.doc_id IN (SELECT doc_id FROM subset_docs WHERE dataset = $ds AND subset = $subset))"""


def latest_snapshot(con, dataset: str) -> int:
    snap = con.execute("SELECT max(snapshot) FROM snapshots WHERE dataset=?", [dataset]).fetchone()[0]
    if snap is None:
        raise SystemExit(f"Dataset {dataset!r} has no data. Run `rag-lab data add {dataset} <path>` (or `data load`).")
    return snap


def docs_fingerprint(con, dataset: str, snapshot: int, subset: str) -> tuple[str, int]:
    """Hash of the (doc_id, content_hash) pairs a run will see. Two snapshots that
    give a subset identical content share cache entries."""
    fp, n = con.execute(f"SELECT md5(string_agg(doc_id || ':' || content_hash, ',' ORDER BY doc_id)), count(*) "
                        f"FROM ({docs_sql('v.doc_id, v.content_hash')})",
                        {"ds": dataset, "snap": snapshot, "subset": subset}).fetchone()
    return fp, n


def subset_question_ids(con, dataset: str, subset: str, limit: int | None = None) -> list[str]:
    if subset == "all":
        rows = con.execute("SELECT question_id, category FROM questions WHERE dataset=? ORDER BY question_id", [dataset]).fetchall()
        ids = _round_robin(rows)
    else:
        ids = [q for (q,) in con.execute("SELECT question_id FROM subset_questions WHERE dataset=? AND subset=? ORDER BY ord",
                                         [dataset, subset]).fetchall()]
    if not ids:
        raise SystemExit(f"No questions for dataset {dataset!r} subset {subset!r}. "
                         "Add a questions.jsonl, or build the subset with `rag-lab data subset`.")
    return ids[:limit] if limit else ids


def load_questions(con, dataset: str, question_ids: list[str]) -> list[dict]:
    cur = con.execute("SELECT * EXCLUDE (dataset) FROM questions WHERE dataset=? AND question_id IN (SELECT unnest(?))",
                      [dataset, question_ids])
    cols = [d[0] for d in cur.description]
    by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    return [by_id[q] for q in question_ids]


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


def build_subset(dataset: str = BENCH, name: str = "dev", per_category: int = 10,
                 distractors: int = 20_000, seed: int = 42) -> None:
    """`per_category` questions from each category, all their gold docs, plus
    `distractors` random other docs from the latest snapshot. Seeded, so everyone
    who runs this gets the same subset."""
    if name == "all":
        raise SystemExit("'all' is reserved (it means every doc and question)")
    rng = random.Random(seed)
    con = connect()
    snap = latest_snapshot(con, dataset)
    rows = con.execute("SELECT question_id, category, expected_doc_ids FROM questions WHERE dataset=? ORDER BY question_id",
                       [dataset]).fetchall()
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r[1], []).append(r)
    picked = [r for cat in sorted(by_cat) for r in rng.sample(by_cat[cat], min(per_category, len(by_cat[cat])))]
    present = [d for (d,) in con.execute(f"SELECT doc_id FROM ({docs_sql('v.doc_id')}) ORDER BY doc_id",
                                         {"ds": dataset, "snap": snap, "subset": "all"}).fetchall()]
    present_set = set(present)
    gold = sorted({d for r in picked for d in r[2]})
    missing = [d for d in gold if d not in present_set]
    gold_set = set(gold)
    pool = [d for d in present if d not in gold_set]
    doc_ids = [d for d in gold if d in present_set] + rng.sample(pool, min(distractors, len(pool)))
    con.execute("DELETE FROM subset_docs WHERE dataset=? AND subset=?", [dataset, name])
    con.execute("DELETE FROM subset_questions WHERE dataset=? AND subset=?", [dataset, name])
    con.executemany("INSERT INTO subset_docs VALUES (?, ?, ?)", [(dataset, name, d) for d in doc_ids])
    qids = _round_robin([(r[0], r[1]) for r in picked])
    con.executemany("INSERT INTO subset_questions VALUES (?, ?, ?, ?)", [(dataset, name, q, i) for i, q in enumerate(qids)])
    con.close()
    print(f"Subset '{name}' of '{dataset}': {len(qids)} questions, {len(gold)} gold docs, "
          f"{len(doc_ids) - len(gold) + len(missing)} distractors (from snapshot {snap})")
    if missing:
        print(f"  WARNING: {len(missing)} gold docs not in the dataset (did you download all slices?)")


def dataset_info(con) -> list[dict]:
    out = []
    for name, adapter, created in con.execute("SELECT name, adapter, created_at FROM datasets ORDER BY name").fetchall():
        snaps = con.execute("""SELECT snapshot, created_at, added, changed, deleted, n_docs, note FROM snapshots
                               WHERE dataset=? ORDER BY snapshot""", [name]).fetchall()
        latest = snaps[-1][0] if snaps else 0
        by_source = con.execute(f"SELECT source, count(*) FROM ({docs_sql('v.source')}) GROUP BY 1 ORDER BY 2 DESC",
                                {"ds": name, "snap": latest, "subset": "all"}).fetchall()
        subsets = con.execute("SELECT subset, count(*) FROM subset_questions WHERE dataset=? GROUP BY 1", [name]).fetchall()
        n_q = con.execute("SELECT count(*) FROM questions WHERE dataset=?", [name]).fetchone()[0]
        out.append({"name": name, "adapter": adapter, "created_at": str(created), "latest_snapshot": latest,
                    "n_questions": n_q, "by_source": dict(by_source), "subsets": dict(subsets),
                    "snapshots": [dict(zip(["snapshot", "created_at", "added", "changed", "deleted", "n_docs", "note"],
                                           [*s[:1], str(s[1]), *s[2:]])) for s in snaps]})
    return out

