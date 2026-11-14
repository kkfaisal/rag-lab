"""rag-lab: iterate on RAG pipelines against EnterpriseRAG-Bench."""

# Verify HTTPS with the operating system's certificate store (not only Python's bundled
# list), so downloads work behind company proxies that re-sign TLS traffic.
import truststore as _truststore

_truststore.inject_into_ssl()
