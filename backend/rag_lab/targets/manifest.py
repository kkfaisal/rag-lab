"""What rag-lab remembers about a target, independent of the store behind it.

data/targets/<id>/manifest.json   version counter, embedder used, source dataset, write history
data/targets/<id>/docs.duckdb     doc_id -> hash of (document content + chunk + embed settings)

The doc hashes are what make "append" incremental: a document whose hash is
already recorded is skipped, so re-running an ingestion pipeline only writes
new or changed documents.
"""

import json
import time
from pathlib import Path

from .. import db


class Manifest:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.file = state_dir / "manifest.json"
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {
            "version": 0, "embed": None, "dataset": None, "subset": None, "snapshot": None,
            "n_docs": 0, "n_chunks": 0, "history": []}

    def db_file(self) -> Path:
        self._db().close()          # make sure the file and table exist
        return self.dir / "docs.duckdb"

    def _db(self):
        con = db.connect(str(self.dir / "docs.duckdb"))
        # No PRIMARY KEY (DuckDB keeps key indexes in memory); writes replace rows with delete + insert.
        con.execute("CREATE TABLE IF NOT EXISTS docs(doc_id VARCHAR, doc_hash VARCHAR, n_chunks INTEGER)")
        return con

    def known(self) -> dict[str, str]:
        con = self._db()
        out = dict(con.execute("SELECT doc_id, doc_hash FROM docs").fetchall())
        con.close()
        return out

    def reset(self) -> None:
        con = self._db()
        con.execute("DELETE FROM docs")
        con.close()
        self.data.update(embed=None, n_docs=0, n_chunks=0)

    def record(self, rows: list[tuple[str, str, int]]) -> None:
        """rows = (doc_id, doc_hash, n_chunks) for every document just written."""
        if not rows:
            return
        import pyarrow as pa
        con = self._db()
        con.register("rows", pa.table({"doc_id": [r[0] for r in rows], "doc_hash": [r[1] for r in rows],
                                       "n_chunks": pa.array([r[2] for r in rows], pa.int32())}))
        con.execute("DELETE FROM docs WHERE doc_id IN (SELECT doc_id FROM rows)")
        con.execute("INSERT INTO docs SELECT * FROM rows")
        con.close()

    def record_from(self, con, table: str) -> None:
        """Like record(), from a table (doc_id, doc_hash, n_chunks) in another DuckDB connection,
        without loading it into Python."""
        con.execute(f"ATTACH '{self.db_file()}' AS m")
        con.execute(f"DELETE FROM m.docs WHERE doc_id IN (SELECT doc_id FROM {table})")
        con.execute(f"INSERT INTO m.docs SELECT doc_id, doc_hash, n_chunks FROM {table}")
        con.execute("DETACH m")

    def commit(self, entry: dict, **fields) -> int:
        """Bump the version after a write and add a history entry."""
        con = self._db()
        n_docs, n_chunks = con.execute("SELECT count(*), coalesce(sum(n_chunks), 0) FROM docs").fetchone()
        con.close()
        self.data["version"] += 1
        self.data.update(fields, n_docs=n_docs, n_chunks=int(n_chunks))
        self.data["history"] = (self.data["history"] + [{"version": self.data["version"],
                                                          "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}])[-50:]
        self.file.write_text(json.dumps(self.data, indent=2))
        return self.data["version"]
