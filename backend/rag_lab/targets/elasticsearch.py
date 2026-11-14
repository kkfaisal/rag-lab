"""Elasticsearch target: BM25 keyword search, plus approximate k-NN vector search
(HNSW on a dense_vector field) when vectors are stored. Needs Elasticsearch 9."""

import os

from pydantic import BaseModel, Field

from .. import services

from .base import Target, TargetError, register_target


class ElasticsearchConfig(BaseModel):
    model_config = {"extra": "forbid"}
    url: str = Field(default_factory=lambda: services.url("elasticsearch"))
    index: str = "rag_lab"
    vectors: bool = True
    num_candidates: int = 200           # k-NN search width: higher = closer to exact search, slower
    api_key_env: str | None = None      # names of environment variables holding the credentials
    username_env: str | None = None
    password_env: str | None = None
    verify_certs: bool = True


@register_target("elasticsearch", ElasticsearchConfig)
class ElasticsearchTarget(Target):
    text_search = True

    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.vector_search = cfg.vectors
        self._client = None

    def client(self):
        if self._client is None:
            from elasticsearch import Elasticsearch
            kw = {"verify_certs": self.cfg.verify_certs, "request_timeout": 120}
            if self.cfg.api_key_env:
                kw["api_key"] = os.environ.get(self.cfg.api_key_env, "")
            elif self.cfg.username_env:
                kw["basic_auth"] = (os.environ.get(self.cfg.username_env, ""), os.environ.get(self.cfg.password_env or "", ""))
            self._client = Elasticsearch(self.cfg.url, **kw)
        return self._client

    def _exists(self) -> bool:
        try:
            return bool(self.client().indices.exists(index=self.cfg.index))
        except Exception as e:
            if type(e).__name__ == "ConnectionError":
                raise TargetError(f"Cannot reach Elasticsearch at {self.cfg.url}") from e
            raise

    def clear(self):
        if self._exists():
            self.client().indices.delete(index=self.cfg.index)

    def delete_docs(self, doc_ids):
        if self._exists():
            for start in range(0, len(doc_ids), 1000):
                self.client().delete_by_query(index=self.cfg.index, refresh=True,
                                              query={"terms": {"doc_id": doc_ids[start:start + 1000]}})

    def add(self, chunks):
        from elasticsearch import helpers
        if self.cfg.vectors and chunks[0]["vector"] is None:
            raise TargetError("vectors=true but the pipeline has no embed node. Add one, or set vectors=false.")
        if not self._exists():
            props = {"chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "source": {"type": "keyword"},
                     "title": {"type": "text"}, "text": {"type": "text"}}
            if self.cfg.vectors:
                props["vec"] = {"type": "dense_vector", "dims": len(chunks[0]["vector"]), "index": True, "similarity": "cosine"}
            self.client().indices.create(index=self.cfg.index, mappings={"properties": props})
        actions = ({"_index": self.cfg.index, "_id": c["chunk_id"],
                    "_source": {**{k: c[k] for k in ("chunk_id", "doc_id", "source", "title", "text")},
                                **({"vec": [float(x) for x in c["vector"]]} if self.cfg.vectors else {})}} for c in chunks)
        helpers.bulk(self.client(), actions, chunk_size=500)

    def finish(self):
        if self._exists():
            self.client().indices.refresh(index=self.cfg.index)

    def _hits(self, res):
        return [{"chunk_id": h["_source"]["chunk_id"], "doc_id": h["_source"]["doc_id"], "score": float(h["_score"]),
                 "text": h["_source"]["text"]} for h in res["hits"]["hits"]]

    def _check(self):
        if not self._exists():
            raise TargetError("This target is empty. Run an ingestion pipeline that writes to it.")

    def search_text(self, query, k):
        self._check()
        return self._hits(self.client().search(index=self.cfg.index, size=k, query={"match": {"text": query}},
                                               source=["chunk_id", "doc_id", "text"]))

    def search_vector(self, vector, k, query=""):
        if not self.cfg.vectors:
            raise TargetError("This Elasticsearch target was created with vectors=false")
        self._check()
        knn = {"field": "vec", "query_vector": [float(x) for x in vector], "k": k, "num_candidates": max(self.cfg.num_candidates, k)}
        return self._hits(self.client().search(index=self.cfg.index, size=k, knn=knn, source=["chunk_id", "doc_id", "text"]))

    def count(self):
        return int(self.client().count(index=self.cfg.index)["count"]) if self._exists() else 0

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
