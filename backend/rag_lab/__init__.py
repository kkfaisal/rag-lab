"""rag-lab: build, run and benchmark RAG ingestion and retrieval pipelines."""

# Verify HTTPS with the operating system's certificate store (not only Python's bundled
# list), so downloads work behind company proxies that re-sign TLS traffic.
import truststore as _truststore

_truststore.inject_into_ssl()
