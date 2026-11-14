"""SQL database target: any database SQLAlchemy can reach, the Python counterpart
of a JDBC connection (PostgreSQL, MySQL/MariaDB, SQL Server, Oracle, SQLite,
Snowflake, ... given the driver is installed).

The database is the store of record: one table of chunks, vectors as float32
bytes. Plain SQL can't rank by BM25 or vector similarity portably, so after each
write rag-lab reads the table back and builds a local search index (BM25 +
exact vectors, the same as the `local` target). If you change the table outside
rag-lab, run an ingestion again (an append with nothing new is enough) to refresh it.
"""

import numpy as np
from pydantic import BaseModel

from .base import Target, TargetError, expand_env, register_target, safe_identifier
from .local import LocalConfig, LocalTarget


class SQLConfig(BaseModel):
    model_config = {"extra": "forbid"}
    # SQLAlchemy URL, e.g. postgresql+psycopg://user:${PG_PASSWORD}@host/db or mysql+pymysql://...
    # ${VAR} is read from .env. Empty = a SQLite file in the target's folder.
    url: str = ""
    table: str = "rag_lab_chunks"
    vectors: bool = True
    bm25: bool = True


def _clean(s):
    return s.replace("\x00", "") if isinstance(s, str) else s     # several databases reject NUL bytes in text


@register_target("sql", SQLConfig)
class SQLTarget(Target):
    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.text_search, self.vector_search = cfg.bm25, cfg.vectors
        self.name = safe_identifier(cfg.table)
        self._engine = self._table = None
        # The search index rebuilt from the database after every write.
        (state_dir / "search-index").mkdir(parents=True, exist_ok=True)
        self.index = LocalTarget(LocalConfig(vectors=cfg.vectors, bm25=cfg.bm25), state_dir / "search-index")

    def engine(self):
        if self._engine is None:
            import sqlalchemy as sa
            try:
                url = expand_env(self.cfg.url) if self.cfg.url else f"sqlite:///{self.state_dir / 'chunks.sqlite'}"
                self._engine = sa.create_engine(url)
                with self._engine.connect():
                    pass
            except Exception as e:
                raise TargetError(f"Cannot connect to the database: {type(e).__name__}: {e}") from e
            md = sa.MetaData()
            self._table = sa.Table(self.name, md,
                                   sa.Column("chunk_id", sa.String(255), primary_key=True),
                                   sa.Column("doc_id", sa.String(255), index=True, nullable=False),
                                   sa.Column("source", sa.String(255)), sa.Column("title", sa.Text),
                                   sa.Column("text", sa.Text), sa.Column("vec", sa.LargeBinary, nullable=True))
        return self._engine

    def _create(self):
        self._table.metadata.create_all(self.engine(), checkfirst=True)

    def clear(self):
        self.engine()
        self._table.drop(self.engine(), checkfirst=True)
        self.index.clear()

    def delete_docs(self, doc_ids):
        self.engine()
        self._create()
        with self.engine().begin() as con:
            for start in range(0, len(doc_ids), 500):
                con.execute(self._table.delete().where(self._table.c.doc_id.in_(doc_ids[start:start + 500])))

    def add(self, chunks):
        if self.cfg.vectors and chunks[0]["vector"] is None:
            raise TargetError("vectors=true but the pipeline has no embed node. Add one, or set vectors=false.")
        self.engine()
        self._create()
        rows = [{**{k: _clean(c[k]) for k in ("chunk_id", "doc_id", "source", "title", "text")},
                 "vec": np.asarray(c["vector"], dtype=np.float32).tobytes() if self.cfg.vectors else None} for c in chunks]
        with self.engine().begin() as con:
            for start in range(0, len(rows), 1000):
                con.execute(self._table.insert(), rows[start:start + 1000])

    def finish(self):
        """Rebuild the local search index from what is in the database now."""
        self.engine()
        self._create()
        self.index.clear()
        cols = self._table.c
        with self.engine().connect() as con:
            result = con.execution_options(stream_results=True).execute(
                self._table.select().with_only_columns(cols.chunk_id, cols.doc_id, cols.source, cols.title, cols.text, cols.vec))
            while batch := result.fetchmany(5000):
                self.index.add([{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "source": r.source, "title": r.title,
                                 "text": r.text,
                                 "vector": np.frombuffer(r.vec, dtype=np.float32).tolist() if r.vec is not None else None}
                                for r in batch])
        self.index.finish()

    def search_text(self, query, k):
        return self.index.search_text(query, k)

    def search_vector(self, vector, k, query=""):
        return self.index.search_vector(vector, k, query)

    def count(self):
        import sqlalchemy as sa
        self.engine()
        if not sa.inspect(self.engine()).has_table(self.name):
            return 0
        with self.engine().connect() as con:
            return con.execute(sa.select(sa.func.count()).select_from(self._table)).scalar()

    def close(self):
        self.index.close()
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
