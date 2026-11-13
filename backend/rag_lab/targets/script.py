"""A target written as a Python script in the UI: your own vector store, a
database, a search service — anything you can add to, delete from and search."""

from pydantic import BaseModel

from ..scripting import TARGET_TEMPLATE, ScriptError, load_functions
from .base import Target, register_target


class PythonTargetConfig(BaseModel):
    model_config = {"extra": "forbid"}
    code: str = TARGET_TEMPLATE     # must define clear(), delete_docs(doc_ids), add(chunks), search(query, query_vector, k)


@register_target("python", PythonTargetConfig)
class PythonTarget(Target):
    text_search = vector_search = True      # the script's search() receives both the text and the vector

    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.fns = load_functions(cfg.code, "search", ("clear", "delete_docs", "add", "finish", "count"),
                                  extra_globals={"STATE_DIR": str(state_dir)})
        missing = [f for f in ("clear", "delete_docs", "add") if f not in self.fns]
        if missing:
            raise ScriptError(f"Target script must also define: {', '.join(missing)}")
        self._query_vector = None

    def clear(self):
        self.fns["clear"]()

    def delete_docs(self, doc_ids):
        self.fns["delete_docs"](doc_ids)

    def add(self, chunks):
        self.fns["add"](chunks)

    def finish(self):
        if "finish" in self.fns:
            self.fns["finish"]()

    def _search(self, query, vector, k):
        hits = self.fns["search"](query, vector, k)
        if not isinstance(hits, list) or not all(isinstance(h, dict) and {"chunk_id", "doc_id", "text"} <= set(h) for h in hits):
            raise ScriptError('search() must return a list of dicts with "chunk_id", "doc_id", "text" (and "score")')
        return [{**h, "score": float(h.get("score", 0.0))} for h in hits[:k]]

    def search_text(self, query, k):
        return self._search(query, None, k)

    def search_vector(self, vector, k, query: str = ""):
        return self._search(query, vector, k)

    def count(self):
        return int(self.fns["count"]()) if "count" in self.fns else None
