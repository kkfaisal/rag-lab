"""Targets: where an ingestion pipeline writes chunks and a retrieval pipeline searches.

Built-ins: local (vector + BM25 on disk), graph (local + a knowledge graph in LadybugDB), pgvector (Postgres), sql (any SQLAlchemy/JDBC-style
database), qdrant, opensearch, elasticsearch. `python` is a script
target for any store of your own. Add a built-in by subclassing `Target` and
decorating it with `@register_target("name", ConfigClass)`.
"""

from .base import TARGETS, Target, TargetError, open_target, register_target  # noqa: F401
from . import elasticsearch, graph, local, opensearch, pgvector, qdrant, script, sql  # noqa: F401  (registers target types)
from .manifest import Manifest  # noqa: F401
