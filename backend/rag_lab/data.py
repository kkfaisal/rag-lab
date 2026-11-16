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
import re
import time
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa

from . import db, paths
from .datasets import get_adapter

BENCH = "enterprise_rag_bench"
RELEASE_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(name VARCHAR PRIMARY KEY, adapter VARCHAR, created_at TIMESTAMP);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS code VARCHAR;   -- extractor script, for the 'script' adapter
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
    for attempt in range(40):      # one writer process at a time: wait briefly if the file is busy
        try:
            con = db.connect(str(paths.CORPUS_DB), read_only=read_only)
            break
        except duckdb.IOException:
            if attempt == 39:
                raise
            time.sleep(0.25)
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
    the named per-source slice zips, e.g. ["slack_slice_0001"], into data/datasets/enterprise_rag_bench/."""
    paths.RAW.mkdir(parents=True, exist_ok=True)
    _download(f"{RELEASE_URL}/questions.jsonl", paths.QUESTIONS_FILE)
    for name in slices or ["all_documents"]:
        _download(f"{RELEASE_URL}/{name}.zip", paths.RAW / f"{name}.zip")


# ----------------------------------------------------------------------------- ingest

def ingest(dataset: str, path: str | Path, adapter: str | None = None, sync: bool = False, note: str = "",
           code: str | None = None) -> int | None:
    """Add docs from `path` to `dataset` and record a new snapshot.

    sync=False (default): upsert — new docs are added, changed docs get a new
    version, docs absent from `path` are left alone (incremental addition).
    sync=True: `path` is the complete dataset — docs absent from it are deleted.
    Returns the new snapshot number, or None if nothing changed.
    """
    path = Path(path)
    con = connect()
    row = con.execute("SELECT adapter, code FROM datasets WHERE name=?", [dataset]).fetchone()
    adapter = adapter or (row[0] if row else (BENCH if dataset == BENCH else "folder"))
    if row and row[0] == BENCH and adapter != BENCH:
        raise SystemExit(f"Dataset {dataset!r} is the benchmark dataset; its adapter can't be changed")
    code = code if code is not None else (row[1] if row else None)
    ds = get_adapter(adapter, code)
    if row:
        con.execute("UPDATE datasets SET adapter=?, code=? WHERE name=?", [adapter, code, dataset])
    else:
        con.execute("INSERT INTO datasets(name, adapter, created_at, code) VALUES (?, ?, now(), ?)", [dataset, adapter, code])

    # A dataset must come with benchmark questions: check before storing anything.
    questions = list(ds.iter_questions(path))
    if not questions and not con.execute("SELECT count(*) FROM questions WHERE dataset=?", [dataset]).fetchone()[0]:
        con.execute("DELETE FROM datasets WHERE name=? AND name NOT IN (SELECT dataset FROM snapshots)", [dataset])
        con.close()
        raise SystemExit("No benchmark questions found. A dataset needs both documents and questions: add a "
                         "questions.jsonl (or return them from your script's questions() function) and ingest again.")

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
    # A doc id seen twice keeps its first occurrence. Only ids and row ids are compared: a window or
    # GROUP BY over rows that carry the text would need memory proportional to the dataset.
    if con.execute("SELECT count(*) - count(DISTINCT doc_id) FROM staged").fetchall()[0][0]:
        con.execute("DELETE FROM staged WHERE rowid NOT IN (SELECT min(rowid) FROM staged GROUP BY doc_id)")
    con.execute("CREATE OR REPLACE TEMP VIEW incoming AS SELECT * FROM staged")

    # 2. Diff against the current state (open versions) by content hash.
    snap = (con.execute("SELECT max(snapshot) FROM snapshots WHERE dataset=?", [dataset]).fetchone()[0] or 0) + 1
    con.execute("""CREATE OR REPLACE TEMP TABLE cur AS
                   SELECT doc_id, content_hash FROM doc_versions WHERE dataset=? AND valid_to IS NULL""", [dataset])
    added = con.execute("SELECT count(*) FROM incoming i ANTI JOIN cur USING (doc_id)").fetchone()[0]
    changed = con.execute("SELECT count(*) FROM incoming i JOIN cur c USING (doc_id) WHERE i.content_hash <> c.content_hash").fetchone()[0]
    deleted = con.execute("SELECT count(*) FROM cur ANTI JOIN incoming USING (doc_id)").fetchone()[0] if sync else 0
    if added + changed + deleted == 0:
        print(f"No document changes for '{dataset}'.")
        _upsert_questions(con, dataset, questions)
        con.close()
        return None

    # 3. Store new contents once, close replaced/deleted versions, open new ones.
    con.execute("""INSERT INTO doc_contents SELECT content_hash, title, text FROM staged WHERE rowid IN (
                     SELECT min(rowid) FROM staged WHERE content_hash NOT IN (SELECT content_hash FROM doc_contents)
                     GROUP BY content_hash)""")
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
    _upsert_questions(con, dataset, questions)
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


def _upsert_questions(con, dataset: str, qs: list[dict]) -> None:
    if not qs:
        return
    con.execute("DELETE FROM questions WHERE dataset=? AND question_id IN (SELECT unnest(?))",
                [dataset, [q["question_id"] for q in qs]])
    con.executemany("INSERT INTO questions VALUES (?,?,?,?,?,?,?,?)", [
        (dataset, q["question_id"], q.get("category", "default"), q["question"], q.get("gold_answer"),
         q.get("expected_doc_ids") or [], q.get("answer_facts") or [], q.get("source_types") or []) for q in qs])
    print(f"  {len(qs)} questions")


# A dataset has two steps, each its own button on the Datasets page:
#   Download  a Python extractor puts the raw files in data/datasets/<name>/ (or you copy them there).
#   Load      parse that folder into rag-lab's internal document store (data/corpus.duckdb) as a
#             snapshot: normalized documents + questions. Pipelines read from the store, never the folder.
SOURCES = {
    "enterprise_rag_bench": "EnterpriseRAG-Bench: downloads the release, loads ~500k docs and 500 questions, builds a dev subset",
    "extract": "Download: run the dataset's Python extractor, which writes raw files into its folder (no loading)",
    "folder": "Load: read the dataset's folder into the internal document store as a new snapshot",
}
EXTRACTOR_FILE = "extract.py"           # an extractor script lives in its dataset's folder
NAME_RE = r"[A-Za-z0-9_-]{1,64}"


def dataset_dir(name: str) -> Path:
    import re
    if not re.fullmatch(NAME_RE, name):
        raise SystemExit(f"Invalid dataset name {name!r}: letters, digits, _ and - only")
    return paths.DATASETS / name


def migrate_layout() -> None:
    """Older layouts kept the benchmark in data/raw and uploads in data/uploads/<name>."""
    old_raw, old_uploads = paths.DATA / "raw", paths.DATA / "uploads"
    if old_raw.is_dir() and not paths.RAW.exists():
        paths.DATASETS.mkdir(parents=True, exist_ok=True)
        old_raw.rename(paths.RAW)
    if old_uploads.is_dir():
        for d in old_uploads.iterdir():
            if d.is_dir() and not (paths.DATASETS / d.name).exists():
                paths.DATASETS.mkdir(parents=True, exist_ok=True)
                d.rename(paths.DATASETS / d.name)
        if not any(old_uploads.iterdir()):
            old_uploads.rmdir()


def run_extractor(folder: Path, code: str) -> None:
    """Run an extractor script: its download(folder) writes files into the folder."""
    from . import scripting
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EXTRACTOR_FILE).write_text(code)
    fns = scripting.load_functions(code, "download")
    print(f"Running extractor into {folder}", flush=True)
    fns["download"](str(folder))


def ingest_source(dataset: str, source: str, params: dict, sync: bool = False) -> int | None:
    """Fill and/or load a dataset folder (what the UI's Datasets page runs as a job)."""
    if source == "enterprise_rag_bench" or (source == "folder" and dataset == BENCH):
        if source == "enterprise_rag_bench":
            download(params.get("slices") or None)
        snap = ingest(dataset, paths.RAW, adapter=BENCH, sync=True, note="EnterpriseRAG-Bench release")
        if source == "enterprise_rag_bench":
            build_subset(dataset, "dev", distractors=int(params.get("distractors", 20_000)))
        return snap
    folder = dataset_dir(dataset)
    if source == "extract":             # download only; loading is its own step
        run_extractor(folder, params.get("code") or "")
        print(f"Downloaded into {folder}. Press Load to read it into the document store.")
        return None
    if source == "folder":
        if not folder.is_dir():
            raise SystemExit(f"No folder {folder}. Download it or put the dataset's files there first.")
        if dataset == BENCH:            # its documents stay zipped; read with its own reader, then build "dev"
            snap = ingest(dataset, folder, adapter=BENCH, sync=True, note="EnterpriseRAG-Bench release")
            build_subset(dataset, "dev")
            return snap
        return ingest(dataset, folder, adapter="folder", sync=sync, note="folder")
    raise SystemExit(f"Unknown data source {source!r}. Available: {sorted(SOURCES)}")


def remove_dataset(dataset: str) -> None:
    """Remove a dataset from the document store (its folder is left alone)."""
    con = connect()
    for table in ("snapshots", "doc_versions", "questions", "subset_docs", "subset_questions"):
        con.execute(f"DELETE FROM {table} WHERE dataset=?", [dataset])
    con.execute("DELETE FROM datasets WHERE name=?", [dataset])
    con.execute("DELETE FROM doc_contents WHERE content_hash NOT IN (SELECT content_hash FROM doc_versions)")
    con.close()
    print(f"Removed dataset {dataset!r} from the document store")


def folder_summary(folder: Path) -> dict:
    files = [f for f in folder.rglob("*") if f.is_file() and not f.name.startswith(".")] if folder.is_dir() else []
    by_ext: dict[str, list] = {}
    for f in files:
        e = by_ext.setdefault(f.suffix.lower() or "(none)", [0, 0])
        e[0] += 1
        e[1] += f.stat().st_size
    return {"exists": folder.is_dir(), "n_files": len(files), "size": sum(v[1] for v in by_ext.values()),
            "by_ext": {k: {"files": v[0], "size": v[1]} for k, v in sorted(by_ext.items(), key=lambda kv: -kv[1][1])},
            "has_questions": (folder / "questions.jsonl").is_file(), "has_extractor": (folder / EXTRACTOR_FILE).is_file()}


def list_folders() -> list[dict]:
    paths.DATASETS.mkdir(parents=True, exist_ok=True)
    return [{"name": d.name, **folder_summary(d)} for d in sorted(paths.DATASETS.iterdir())
            if d.is_dir() and not d.name.startswith(".")]


def dataset_metrics(con, dataset: str) -> dict:
    """Numbers for the dataset page: document lengths, sources, question coverage."""
    snap = latest_snapshot(con, dataset)
    p = {"ds": dataset, "snap": snap, "subset": "all"}
    docs = f"SELECT doc_id, source, length(text) AS n FROM ({docs_sql('v.doc_id, v.source, c.text')})"
    n, chars, avg, p50, p90, mx = con.execute(
        f"SELECT count(*), sum(n), avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.9), max(n) FROM ({docs})", p).fetchone()
    # Log-scale length histogram: <100, 100-300, 300-1k, ... characters.
    edges = [0, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
    hist = con.execute(f"""SELECT bucket, count(*) FROM (SELECT CASE {' '.join(f"WHEN n < {b} THEN {i}" for i, b in enumerate(edges[1:]))}
                           ELSE {len(edges) - 1} END AS bucket FROM ({docs})) GROUP BY 1 ORDER BY 1""", p).fetchall()
    labels = [f"<{edges[1]:,}"] + [f"{a:,}–{b:,}" for a, b in zip(edges[1:], edges[2:])] + [f"≥{edges[-1]:,}"]
    by_source = con.execute(f"SELECT source, count(*), avg(n) FROM ({docs}) GROUP BY 1 ORDER BY 2 DESC LIMIT 20", p).fetchall()
    q = con.execute("""SELECT count(*), count(*) FILTER (WHERE len(expected_doc_ids) > 0),
                       count(*) FILTER (WHERE gold_answer IS NOT NULL AND gold_answer <> ''), avg(len(expected_doc_ids)),
                       avg(length(question)) FROM questions WHERE dataset=?""", [dataset]).fetchone()
    cats = con.execute("SELECT category, count(*) FROM questions WHERE dataset=? GROUP BY 1 ORDER BY 2 DESC LIMIT 20", [dataset]).fetchall()
    # Coverage: how many gold doc ids exist in the store (a gold id that is missing can never be retrieved).
    gold, found = con.execute(f"""WITH g AS (SELECT DISTINCT unnest(expected_doc_ids) AS doc_id FROM questions WHERE dataset=$ds)
                                  SELECT count(*), count(*) FILTER (WHERE doc_id IN (SELECT doc_id FROM ({docs})))
                                  FROM g""", p).fetchone()
    return {"snapshot": snap,
            "docs": {"n": n, "chars": chars or 0, "avg_chars": round(avg or 0), "p50_chars": round(p50 or 0),
                     "p90_chars": round(p90 or 0), "max_chars": mx or 0, "approx_tokens": round((chars or 0) / 4)},
            "length_hist": [{"bucket": labels[b], "docs": c} for b, c in hist],
            "by_source": [{"source": s_, "docs": c, "avg_chars": round(a)} for s_, c, a in by_source],
            "questions": {"n": q[0], "with_gold_docs": q[1], "with_gold_answer": q[2], "avg_gold_docs": round(q[3] or 0, 2),
                          "avg_question_chars": round(q[4] or 0), "gold_doc_ids": gold, "gold_docs_found": found},
            "categories": [{"category": c, "questions": k} for c, k in cats]}


# ----------------------------------------------------------------------------- reading

def docs_sql(columns: str = "v.doc_id, v.source, v.path, c.title, c.text, v.content_hash") -> str:
    """Docs of dataset $ds as of snapshot $snap, restricted to subset $subset
    ("all" = no restriction). The single definition used by runs and the API."""
    return f"""
      SELECT {columns} FROM doc_versions v JOIN doc_contents c USING (content_hash)
      WHERE v.dataset = $ds AND v.valid_from <= $snap AND (v.valid_to IS NULL OR v.valid_to > $snap)
        AND ($subset = 'all' OR v.doc_id IN (SELECT doc_id FROM subset_docs WHERE dataset = $ds AND subset = $subset))"""


def require_ready(con, dataset: str) -> str:
    """A dataset can be used once it has documents *and* benchmark questions. Returns its adapter."""
    row = con.execute("SELECT adapter FROM datasets WHERE name=?", [dataset]).fetchone()
    if not row:
        raise SystemExit(f"Unknown dataset {dataset!r}. Create it on the Datasets page first.")
    if not con.execute("SELECT count(*) FROM questions WHERE dataset=?", [dataset]).fetchone()[0]:
        raise SystemExit(f"Dataset {dataset!r} has no benchmark questions. Add a questions file and ingest again.")
    return row[0]


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
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise SystemExit(f"Bad subset name {name!r}: letters, digits, _ and - only")
    rng = random.Random(seed)
    con = connect()
    snap = latest_snapshot(con, dataset)
    rows = con.execute("SELECT question_id, category, expected_doc_ids FROM questions WHERE dataset=? ORDER BY question_id",
                       [dataset]).fetchall()
    if not rows:
        raise SystemExit(f"Dataset {dataset!r} has no questions loaded, so there is nothing to build a subset from")
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


def remove_subset(dataset: str, name: str) -> None:
    con = connect()
    con.execute("DELETE FROM subset_docs WHERE dataset=? AND subset=?", [dataset, name])
    con.execute("DELETE FROM subset_questions WHERE dataset=? AND subset=?", [dataset, name])
    con.close()
    print(f"Removed subset '{name}' of '{dataset}'")


def dataset_info(con) -> list[dict]:
    out = []
    cols = {r[0] for r in con.execute("DESCRIBE datasets").fetchall()}
    code_col = "code" if "code" in cols else "NULL"
    for name, adapter, created, code in con.execute(f"SELECT name, adapter, created_at, {code_col} FROM datasets ORDER BY name").fetchall():
        snaps = con.execute("""SELECT snapshot, created_at, added, changed, deleted, n_docs, note FROM snapshots
                               WHERE dataset=? ORDER BY snapshot""", [name]).fetchall()
        latest = snaps[-1][0] if snaps else 0
        by_source = con.execute(f"SELECT source, count(*) FROM ({docs_sql('v.source')}) GROUP BY 1 ORDER BY 2 DESC",
                                {"ds": name, "snap": latest, "subset": "all"}).fetchall()
        subsets = con.execute("SELECT subset, count(*) FROM subset_questions WHERE dataset=? GROUP BY 1", [name]).fetchall()
        subset_docs = con.execute("SELECT subset, count(*) FROM subset_docs WHERE dataset=? GROUP BY 1", [name]).fetchall()
        n_q = con.execute("SELECT count(*) FROM questions WHERE dataset=?", [name]).fetchone()[0]
        out.append({"name": name, "adapter": adapter, "code": code, "created_at": str(created), "latest_snapshot": latest,
                    "n_questions": n_q, "by_source": dict(by_source), "subsets": dict(subsets), "subset_docs": dict(subset_docs),
                    "snapshots": [dict(zip(["snapshot", "created_at", "added", "changed", "deleted", "n_docs", "note"],
                                           [*s[:1], str(s[1]), *s[2:]])) for s in snaps]})
    return out

