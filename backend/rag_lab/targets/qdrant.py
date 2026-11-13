"""Qdrant target (vector search). With no URL it runs embedded, storing its
files in the target's folder — no server needed. With a URL it talks to a Qdrant server."""

import os
import uuid

from pydantic import BaseModel

from .base import Target, TargetError, register_target


class QdrantConfig(BaseModel):
    model_config = {"extra": "forbid"}
    url: str | None = None              # e.g. http://localhost:6333 ; empty = embedded on disk
    collection: str = "rag_lab"
    api_key_env: str | None = None      # name of an environment variable holding the API key (never the key itself)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))     # Qdrant ids must be ints or UUIDs


@register_target("qdrant", QdrantConfig)
class QdrantTarget(Target):
    vector_search = True

    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self._client = None

    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self.cfg.url:
                key = os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else None
                self._client = QdrantClient(url=self.cfg.url, api_key=key)
            else:
                self._client = QdrantClient(path=str(self.state_dir / "qdrant"))
        return self._client

    def clear(self):
        if self.client().collection_exists(self.cfg.collection):
            self.client().delete_collection(self.cfg.collection)

    def delete_docs(self, doc_ids):
        from qdrant_client import models
        if self.client().collection_exists(self.cfg.collection):
            self.client().delete(self.cfg.collection, points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))]))

    def add(self, chunks):
        from qdrant_client import models
        if any(c["vector"] is None for c in chunks):
            raise TargetError("Qdrant needs vectors: add an embed node before this target")
        c = self.client()
        if not c.collection_exists(self.cfg.collection):
            c.create_collection(self.cfg.collection, vectors_config=models.VectorParams(
                size=len(chunks[0]["vector"]), distance=models.Distance.COSINE))
        for start in range(0, len(chunks), 500):
            c.upsert(self.cfg.collection, points=[models.PointStruct(
                id=_point_id(ch["chunk_id"]), vector=[float(x) for x in ch["vector"]],
                payload={k: ch[k] for k in ("chunk_id", "doc_id", "source", "title", "text")}) for ch in chunks[start:start + 500]])

    def search_vector(self, vector, k, query=""):
        if not self.client().collection_exists(self.cfg.collection):
            raise TargetError("This target is empty. Run an ingestion pipeline that writes to it.")
        points = self.client().query_points(self.cfg.collection, query=[float(x) for x in vector], limit=k, with_payload=True).points
        return [{"chunk_id": p.payload["chunk_id"], "doc_id": p.payload["doc_id"], "score": float(p.score), "text": p.payload["text"]}
                for p in points]

    def count(self):
        c = self.client()
        return c.count(self.cfg.collection).count if c.collection_exists(self.cfg.collection) else 0

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
