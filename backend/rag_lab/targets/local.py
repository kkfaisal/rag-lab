"""Local target: one DuckDB file on disk. Vector search is exact (numpy over all
vectors), keyword search is BM25 via DuckDB's full-text extension. Zero setup."""

import duckdb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from .base import Target, TargetError, register_target


class LocalConfig(BaseModel):
    model_config = {"extra": "forbid"}
    vectors: bool = True        # store vectors and allow vector search
    bm25: bool = True           # build a BM25 full-text index and allow keyword search
    stemmer: str = "english"
    stopwords: str = "english"


_vector_cache: dict = {}        # (file, mtime) -> (matrix, chunk_ids, doc_ids); avoids reloading per query


@register_target("local", LocalConfig)
class LocalTarget(Target):
    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.text_search, self.vector_search = cfg.bm25, cfg.vectors
        self.file = state_dir / "store.duckdb"
        self._con, self._read_only = None, False

    def con(self, read_only=False):
        if self._con is not None and self._read_only and not read_only:
            self.close()                    # a search left a read-only connection open; writes need a new one
        if self._con is None:
            self._read_only = read_only
            if read_only and not self.file.exists():
                raise TargetError("This target is empty. Run an ingestion pipeline that writes to it.")
            self._con = duckdb.connect(str(self.file), read_only=read_only)
            if not read_only:
                self._con.execute("""CREATE TABLE IF NOT EXISTS chunks(chunk_id VARCHAR, doc_id VARCHAR, source VARCHAR,
                                     title VARCHAR, text VARCHAR, vec FLOAT[])""")
        return self._con

    def clear(self):
        _vector_cache.clear()
        self.con().execute("DROP TABLE IF EXISTS chunks")
        self.close()
        self.con()

    def delete_docs(self, doc_ids):
        _vector_cache.clear()
        self.con().execute("DELETE FROM chunks WHERE doc_id IN (SELECT unnest(?))", [doc_ids])

    def add(self, chunks):
        _vector_cache.clear()
        con = self.con()
        con.register("new_chunks", pa.table({
            "chunk_id": [c["chunk_id"] for c in chunks], "doc_id": [c["doc_id"] for c in chunks],
            "source": [c["source"] for c in chunks], "title": [c["title"] for c in chunks],
            "text": [c["text"] for c in chunks],
            "vec": pa.array([c["vector"] if self.cfg.vectors else None for c in chunks], pa.list_(pa.float32()))}))
        con.execute("INSERT INTO chunks SELECT * FROM new_chunks")
        con.unregister("new_chunks")

    def finish(self):
        if self.cfg.bm25:       # DuckDB's FTS index is not incremental: rebuild it after every write
            con = self.con()
            con.execute("INSTALL fts; LOAD fts;")
            con.execute(f"PRAGMA create_fts_index('chunks', 'chunk_id', 'text', stemmer='{self.cfg.stemmer}', "
                        f"stopwords='{self.cfg.stopwords}', overwrite=1)")
        self.close()            # checkpoint to disk so other processes (and the vector cache) see the new state

    def search_text(self, query, k):
        if not self.cfg.bm25:
            raise TargetError("This local target was created without BM25 (bm25=false)")
        con = self.con(read_only=True)
        con.execute("LOAD fts")
        rows = con.execute("""
            SELECT chunk_id, doc_id, score, text FROM (
              SELECT chunk_id, doc_id, text, fts_main_chunks.match_bm25(chunk_id, ?) AS score FROM chunks
            ) WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?""", [query, k]).fetchall()
        return [{"chunk_id": c, "doc_id": d, "score": float(s), "text": t} for c, d, s, t in rows]

    def _matrix(self):
        key = (str(self.file), self.file.stat().st_mtime)
        if key not in _vector_cache:
            t = self.con(read_only=True).sql("SELECT chunk_id, doc_id, vec FROM chunks WHERE vec IS NOT NULL").to_arrow_table()
            if t.num_rows == 0:
                raise TargetError("This target holds no vectors. Does the ingestion pipeline have an embed node?")
            flat = t.column("vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            _vector_cache.clear()
            _vector_cache[key] = (flat.reshape(t.num_rows, -1), t.column("chunk_id").to_pylist(), t.column("doc_id").to_pylist())
        return _vector_cache[key]

    def search_vector(self, vector, k, query=""):
        if not self.cfg.vectors:
            raise TargetError("This local target was created without vectors (vectors=false)")
        emb, chunk_ids, doc_ids = self._matrix()
        scores = emb @ np.asarray(vector, dtype=np.float32)       # cosine similarity: vectors are normalised
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        ids = [chunk_ids[i] for i in top]
        texts = dict(self.con(read_only=True).execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [ids]).fetchall())
        return [{"chunk_id": chunk_ids[i], "doc_id": doc_ids[i], "score": float(scores[i]), "text": texts[chunk_ids[i]]} for i in top]

    def count(self):
        return self.con(read_only=True).execute("SELECT count(*) FROM chunks").fetchone()[0] if self.file.exists() else 0

    def close(self):
        if self._con is not None:
            self._con.close()
            self._con = None
