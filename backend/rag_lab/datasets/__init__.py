"""Dataset adapters: turn some input (a benchmark release, a folder of files, ...)
into normalized documents and, optionally, questions with gold doc ids.

Add one by subclassing `Dataset` and decorating it with `@register_dataset("name")`.
"""

from .base import ADAPTERS, Dataset, Doc, get_adapter, register_dataset  # noqa: F401
from . import enterprise_rag_bench, folder  # noqa: F401  (registers adapters)
