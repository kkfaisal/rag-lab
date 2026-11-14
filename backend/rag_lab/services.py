"""The database services that docker-compose can start next to rag-lab, and where
to find them. Outside Docker they are on localhost; inside docker-compose the
RAG_LAB_*_URL variables point at the service hostnames instead. Target settings
use these as defaults, and the Targets page shows which ones are reachable."""

import os
import socket
from urllib.parse import urlparse

SERVICES = [
    # (target type, label, env var, default URL)
    ("pgvector", "PostgreSQL + pgvector", "RAG_LAB_POSTGRES_URL", "postgresql://raglab:${POSTGRES_PASSWORD}@localhost:5433/raglab"),
    ("qdrant", "Qdrant", "RAG_LAB_QDRANT_URL", "http://localhost:6333"),
    ("opensearch", "OpenSearch", "RAG_LAB_OPENSEARCH_URL", "http://localhost:9200"),
    ("elasticsearch", "Elasticsearch", "RAG_LAB_ELASTICSEARCH_URL", "http://localhost:9201"),
]


def url(target_type: str) -> str:
    for t, _, env, default in SERVICES:
        if t == target_type:
            return os.environ.get(env, default)
    raise KeyError(target_type)


def _reachable(u: str) -> bool:
    p = urlparse(u.replace("${", "").replace("}", ""))     # placeholders don't matter for host:port
    port = p.port or {"postgresql": 5432, "http": 80, "https": 443}.get(p.scheme.split("+")[0], 80)
    try:
        with socket.create_connection((p.hostname or "localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def status() -> list[dict]:
    return [{"type": t, "label": label, "url": url(t), "reachable": _reachable(url(t))} for t, label, _, _ in SERVICES]
