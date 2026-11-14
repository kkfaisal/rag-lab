"""PostgreSQL target: vectors in a pgvector column (HNSW index, cosine distance)
and keyword search with Postgres full-text search.

Keyword ranking is Postgres's ts_rank_cd, not true BM25; the query's words are
OR-ed together so a long question still matches documents that share some words.
Needs a Postgres server with the `vector` extension (docker-compose starts one).
"""

from pydantic import BaseModel, Field

from .. import services

from .base import Target, TargetError, expand_env, register_target, safe_identifier


class PgvectorConfig(BaseModel):
    model_config = {"extra": "forbid"}
    url: str = Field(default_factory=lambda: services.url("pgvector"))   # ${VAR} is read from .env
    table: str = "rag_lab_chunks"
    vectors: bool = True             # store vectors and allow vector search
    keyword: bool = True             # full-text index and keyword search
    text_config: str = "english"     # Postgres text search configuration (stemming, stop words)
    ef_search: int = 100             # HNSW search width: higher = closer to exact search, slower (Postgres default 40)


def _clean(s):
    return s.replace("\x00", "") if isinstance(s, str) else s     # Postgres text can't hold NUL bytes


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"     # pgvector's text format


@register_target("pgvector", PgvectorConfig)
class PgvectorTarget(Target):
    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.text_search, self.vector_search = cfg.keyword, cfg.vectors
        self.table = safe_identifier(cfg.table)
        self._con = None

    def con(self):
        if self._con is None:
            import psycopg
            try:
                self._con = psycopg.connect(expand_env(self.cfg.url), autocommit=True)
            except psycopg.OperationalError as e:
                msg = str(e).strip()
                fatal = [line.split("FATAL:", 1)[1].strip() for line in msg.splitlines() if "FATAL:" in line]
                raise TargetError(f"Cannot connect to Postgres: {fatal[0] if fatal else msg.splitlines()[0]}") from e
        return self._con

    def _exists(self) -> bool:
        return self.con().execute("SELECT to_regclass(%s)", [self.table]).fetchone()[0] is not None

    def clear(self):
        self.con().execute(f"DROP TABLE IF EXISTS {self.table}")

    def delete_docs(self, doc_ids):
        if self._exists():
            self.con().execute(f"DELETE FROM {self.table} WHERE doc_id = ANY(%s)", [doc_ids])

    def add(self, chunks):
        use_vec = self.cfg.vectors
        if use_vec and chunks[0]["vector"] is None:
            raise TargetError("vectors=true but the pipeline has no embed node. Add one, or set vectors=false.")
        con = self.con()
        if not self._exists():
            con.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vec_col = f", vec vector({len(chunks[0]['vector'])})" if use_vec else ""
            tsv_col = (f", tsv tsvector GENERATED ALWAYS AS (to_tsvector('{self.cfg.text_config}', text)) STORED"
                       if self.cfg.keyword else "")
            con.execute(f"""CREATE TABLE {self.table} (chunk_id text PRIMARY KEY, doc_id text NOT NULL,
                            source text, title text, text text{vec_col}{tsv_col})""")
            con.execute(f"CREATE INDEX ON {self.table} (doc_id)")
        cols = "chunk_id, doc_id, source, title, text" + (", vec" if use_vec else "")
        with con.cursor() as cur, cur.copy(f"COPY {self.table} ({cols}) FROM STDIN") as copy:
            for c in chunks:
                row = [_clean(c[k]) for k in ("chunk_id", "doc_id", "source", "title", "text")]
                copy.write_row(row + [_vec(c["vector"])] if use_vec else row)

    def finish(self):
        # Indexes are created once and then kept up to date by Postgres on every insert.
        if not self._exists():
            return
        if self.cfg.vectors:
            self.con().execute(f"CREATE INDEX IF NOT EXISTS {self.table}_vec ON {self.table} USING hnsw (vec vector_cosine_ops)")
        if self.cfg.keyword:
            self.con().execute(f"CREATE INDEX IF NOT EXISTS {self.table}_tsv ON {self.table} USING gin (tsv)")
        self.con().execute(f"ANALYZE {self.table}")

    def _rows(self, sql, params):
        if not self._exists():
            raise TargetError("This target is empty. Run an ingestion pipeline that writes to it.")
        return [{"chunk_id": c, "doc_id": d, "score": float(s), "text": t} for c, d, s, t in self.con().execute(sql, params)]

    def search_text(self, query, k):
        if not self.cfg.keyword:
            raise TargetError("This Postgres target was created with keyword=false")
        cfg = self.cfg.text_config
        return self._rows(f"""
            SELECT chunk_id, doc_id, ts_rank_cd(tsv, q) AS score, text
            FROM {self.table}, to_tsquery('{cfg}', replace(plainto_tsquery('{cfg}', %s)::text, '&', '|')) AS q
            WHERE tsv @@ q ORDER BY score DESC LIMIT %s""", [query, k])

    def search_vector(self, vector, k, query=""):
        if not self.cfg.vectors:
            raise TargetError("This Postgres target was created with vectors=false")
        self.con().execute(f"SET hnsw.ef_search = {max(int(self.cfg.ef_search), int(k))}")   # HNSW returns at most ef_search rows
        return self._rows(f"""SELECT chunk_id, doc_id, 1 - (vec <=> %s::vector) AS score, text
                              FROM {self.table} ORDER BY vec <=> %s::vector LIMIT %s""", [_vec(vector), _vec(vector), k])

    def count(self):
        return self.con().execute(f"SELECT count(*) FROM {self.table}").fetchone()[0] if self._exists() else 0

    def close(self):
        if self._con is not None:
            self._con.close()
            self._con = None
