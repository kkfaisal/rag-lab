"""OpenSearch target: BM25 keyword search, plus k-NN vector search when vectors are stored.
Needs a reachable OpenSearch cluster (with the k-NN plugin for vectors)."""

import os

from pydantic import BaseModel

from .base import Target, TargetError, register_target


class OpenSearchConfig(BaseModel):
    model_config = {"extra": "forbid"}
    url: str = "http://localhost:9200"
    index: str = "rag_lab"
    vectors: bool = True                # also store vectors in a knn_vector field
    username_env: str | None = None     # names of environment variables holding the credentials
    password_env: str | None = None
    verify_certs: bool = True


@register_target("opensearch", OpenSearchConfig)
class OpenSearchTarget(Target):
    text_search = True

    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.vector_search = cfg.vectors
        self._client = None

    def client(self):
        if self._client is None:
            from opensearchpy import OpenSearch
            auth = None
            if self.cfg.username_env:
                auth = (os.environ.get(self.cfg.username_env, ""), os.environ.get(self.cfg.password_env or "", ""))
            self._client = OpenSearch(hosts=[self.cfg.url], http_auth=auth, verify_certs=self.cfg.verify_certs, timeout=60)
        return self._client

    def _exists(self) -> bool:
        return bool(self.client().indices.exists(index=self.cfg.index))

    def clear(self):
        if self._exists():
            self.client().indices.delete(index=self.cfg.index)

    def delete_docs(self, doc_ids):
        if self._exists():
            for start in range(0, len(doc_ids), 1000):
                self.client().delete_by_query(index=self.cfg.index, refresh=True,
                                              body={"query": {"terms": {"doc_id": doc_ids[start:start + 1000]}}})

    def add(self, chunks):
        from opensearchpy import helpers
        use_vec = self.cfg.vectors and chunks[0]["vector"] is not None
        if self.cfg.vectors and not use_vec:
            raise TargetError("vectors=true but the pipeline has no embed node. Add one, or set vectors=false.")
        if not self._exists():
            props = {"chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "source": {"type": "keyword"},
                     "title": {"type": "text"}, "text": {"type": "text"}}
            if use_vec:
                props["vec"] = {"type": "knn_vector", "dimension": len(chunks[0]["vector"]),
                                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
            self.client().indices.create(index=self.cfg.index, body={
                "settings": {"index": {"knn": use_vec}}, "mappings": {"properties": props}})
        actions = ({"_index": self.cfg.index, "_id": c["chunk_id"],
                    "_source": {**{k: c[k] for k in ("chunk_id", "doc_id", "source", "title", "text")},
                                **({"vec": [float(x) for x in c["vector"]]} if use_vec else {})}} for c in chunks)
        helpers.bulk(self.client(), actions, chunk_size=500, request_timeout=120)

    def finish(self):
        if self._exists():
            self.client().indices.refresh(index=self.cfg.index)

    def _search(self, query: dict, k: int):
        if not self._exists():
            raise TargetError("This target is empty. Run an ingestion pipeline that writes to it.")
        res = self.client().search(index=self.cfg.index, body={"size": k, "query": query, "_source": ["chunk_id", "doc_id", "text"]})
        return [{"chunk_id": h["_source"]["chunk_id"], "doc_id": h["_source"]["doc_id"], "score": float(h["_score"]),
                 "text": h["_source"]["text"]} for h in res["hits"]["hits"]]

    def search_text(self, query, k):
        return self._search({"match": {"text": query}}, k)

    def search_vector(self, vector, k, query=""):
        if not self.cfg.vectors:
            raise TargetError("This OpenSearch target was created with vectors=false")
        return self._search({"knn": {"vec": {"vector": [float(x) for x in vector], "k": k}}}, k)

    def count(self):
        return int(self.client().count(index=self.cfg.index)["count"]) if self._exists() else 0

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
