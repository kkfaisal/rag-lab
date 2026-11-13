"""Pipeline stages. Importing this package registers every implementation.

load → chunk → embed → index → retrieve → rerank → generate → evaluate

Each implementation is `fn(cfg, ctx) -> dict` that writes its outputs into
`ctx.out` and returns a small dict of stats (saved as meta.json).
"""

from . import load, chunk, embed, index, retrieve, rerank, generate, evaluate  # noqa: F401
