"""Graph target: the local store (BM25 + vectors) plus a knowledge graph in LadybugDB, an embedded
graph database queried with Cypher (a maintained fork of Kùzu). No server, no API key.

After every write the graph is rebuilt from all chunks:

    (:Chunk)-[:Mentions]->(:Entity)                 the key terms of each chunk
    (:Entity)-[:Related {weight}]->(:Entity)        terms that appear in the same chunks; weight = how often

Entities are found without an LLM: the words and two-word phrases that are distinctive for a chunk
(TF-IDF), ignoring terms that are too rare (fewer than `min_df` chunks) or too common (more than
`max_df` of all chunks). This makes a term co-occurrence graph, the simplest kind of knowledge graph.

Graph search (`search.graph(question, k)`): find the question's terms in the graph (the seeds), follow
Related one hop to each seed's strongest neighbours, and rank chunks by the seeds and neighbours they
mention, rarer terms counting more. So it can find chunks that share few words with the question but
talk about closely related terms. Combine it with BM25 and vectors in a hybrid retrieve node.

Open the graph yourself:  import ladybug; ladybug.Connection(ladybug.Database("data/targets/<id>/graph.lbdb",
read_only=True)).execute("MATCH (e:Entity)-[r:Related]-(f) RETURN e.name, f.name, r.weight LIMIT 20")
"""

import json
import math
import re
import shutil
from collections import Counter

import pyarrow as pa

from .base import TargetError, register_target
from .local import LocalConfig, LocalTarget

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either et few for from further had has have having he
her here hers him his how however i if in into is it its itself just may me might more most must my no nor not of off on
once only or other our ours out over own per same she should so some such than that the their them then there these they
this those through thus to too under until up upon very via was we were what when where which while who whom why will
with within without would you your among based found shown showed show using used use study studies results result
associated increased decreased significantly significant compared effect effects role level levels patients cells cell
data analysis may however two one three new high low""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z]")
_BREAK = re.compile(r"[\n.;:,!?()\[\]]+")


def terms(text: str) -> Counter:
    """Words (3+ letters, not stopwords) and two-word phrases of adjacent such words. Phrases never
    cross a line break or punctuation, so a title and the text below it don't merge."""
    out = Counter()
    for part in _BREAK.split(text.lower()):
        words = [w if len(w) >= 3 and w not in STOP and not w.replace("-", "").isdigit() else None for w in _TOKEN.findall(part)]
        out.update(w for w in words if w)
        out.update(f"{a} {b}" for a, b in zip(words, words[1:]) if a and b)
    return out


class GraphConfig(LocalConfig):
    terms_per_chunk: int = 20       # entities kept per chunk (the most distinctive terms)
    min_df: int = 2                 # a term must appear in at least this many chunks
    max_df: float = 0.05            # ... and in at most this share of all chunks
    min_cooccurrence: int = 2       # two terms are Related if they share at least this many chunks
    neighbours: int = 5             # graph search: strongest neighbours followed per seed term
    neighbour_weight: float = 0.5   # how much a neighbour counts compared with a seed


_graph_cache: dict = {}             # graph folder -> (mtime, Database); avoids reopening per query


@register_target("graph", GraphConfig)
class GraphTarget(LocalTarget):
    graph_search = True

    def __init__(self, cfg, state_dir):
        super().__init__(cfg, state_dir)
        self.graph_file, self.stats_file = state_dir / "graph.lbdb", state_dir / "graph.json"

    def clear(self):
        super().clear()
        self._remove_graph()
        self.stats_file.unlink(missing_ok=True)

    def _remove_graph(self):
        """The database is one file plus side files (write-ahead log, ...): graph.lbdb*."""
        cached = _graph_cache.pop(str(self.graph_file), None)
        if cached:
            cached[2].close()
            cached[1].close()
        for f in self.state_dir.glob(self.graph_file.name + "*"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()

    def finish(self):
        super().finish()                # BM25 index, and the chunks are on disk
        self.build_graph()

    def build_graph(self) -> dict:
        import ladybug
        rows = self.con(read_only=True).execute("SELECT chunk_id, doc_id, text FROM chunks").fetchall()
        self.close()
        per_chunk = [terms(t) for _, _, t in rows]
        df = Counter(t for c in per_chunk for t in c)
        n, cfg = len(rows), self.cfg
        keep = lambda t: cfg.min_df <= df[t] <= max(cfg.min_df, cfg.max_df * n)  # noqa: E731
        mentions, pairs = [], Counter()
        for (chunk_id, _, _), tf in zip(rows, per_chunk):
            top = sorted((t for t in tf if keep(t)), key=lambda t: -tf[t] * math.log(n / df[t]))[:cfg.terms_per_chunk]
            mentions += [(chunk_id, t) for t in top]
            pairs.update((a, b) for i, a in enumerate(sorted(top)) for b in sorted(top)[i + 1:])
        related = [(a, b, w) for (a, b), w in pairs.items() if w >= cfg.min_cooccurrence]
        entities = sorted({t for _, t in mentions})

        self._remove_graph()
        db = ladybug.Database(str(self.graph_file))
        con = ladybug.Connection(db)
        for ddl in ["CREATE NODE TABLE Chunk(id STRING, doc_id STRING, PRIMARY KEY(id))",
                    "CREATE NODE TABLE Entity(name STRING, df INT64, PRIMARY KEY(name))",
                    "CREATE REL TABLE Mentions(FROM Chunk TO Entity)",
                    "CREATE REL TABLE Related(FROM Entity TO Entity, weight DOUBLE)"]:
            con.execute(ddl)
        chunk_t = pa.table({"id": [r[0] for r in rows], "doc_id": [r[1] for r in rows]})  # noqa: F841 (read by COPY)
        entity_t = pa.table({"name": entities, "df": [df[e] for e in entities]})  # noqa: F841
        mention_t = pa.table({"from": [m[0] for m in mentions], "to": [m[1] for m in mentions]})  # noqa: F841
        related_t = pa.table({"from": [r[0] for r in related], "to": [r[1] for r in related],  # noqa: F841
                              "weight": [float(r[2]) for r in related]})
        for table, var in [("Chunk", "chunk_t"), ("Entity", "entity_t"), ("Mentions", "mention_t"), ("Related", "related_t")]:
            if len(locals()[var]):
                con.execute(f"COPY {table} FROM {var}")
        con.close()
        db.close()
        stats = {"chunks": n, "entities": len(entities), "mentions": len(mentions), "relations": len(related)}
        self.stats_file.write_text(json.dumps(stats))
        print(f"    graph: {stats['entities']:,} entities, {stats['relations']:,} relations, "
              f"{stats['mentions']:,} chunk mentions", flush=True)
        return stats

    def _graph(self):
        import ladybug
        if not self.stats_file.exists():
            raise TargetError("This graph target has no graph yet. Run an ingestion pipeline that writes to it.")
        key, mtime = str(self.graph_file), self.stats_file.stat().st_mtime
        if key not in _graph_cache or _graph_cache[key][0] != mtime:
            db = ladybug.Database(key, read_only=True)
            _graph_cache[key] = (mtime, db, ladybug.Connection(db), json.loads(self.stats_file.read_text()))
        return _graph_cache[key][2], _graph_cache[key][3]

    def search_graph(self, query, k):
        con, stats = self._graph()
        n = max(stats["chunks"], 1)

        def rows(cypher, params):
            r, out = con.execute(cypher, params), []
            while r.has_next():
                out.append(r.get_next())
            return out

        seeds = {name: math.log(n / d) for name, d in rows(
            "MATCH (e:Entity) WHERE e.name IN $t RETURN e.name, e.df", {"t": list(terms(query))})}
        if not seeds:
            return []
        weight, best, strongest = dict(seeds), {}, {}
        for a, b, d, w in rows("MATCH (a:Entity)-[r:Related]-(b:Entity) WHERE a.name IN $s "
                               "RETURN a.name, b.name, b.df, r.weight ORDER BY r.weight DESC", {"s": list(seeds)}):
            strongest.setdefault(a, w)                  # rows come strongest first
            best.setdefault(a, [])
            if b not in seeds and len(best[a]) < self.cfg.neighbours:
                best[a].append(b)
                score = self.cfg.neighbour_weight * math.log(n / d) * w / strongest[a]
                weight[b] = max(weight.get(b, 0.0), score)
        scores, doc_of = Counter(), {}
        for chunk_id, doc_id, name in rows("MATCH (c:Chunk)-[:Mentions]->(e:Entity) WHERE e.name IN $names "
                                           "RETURN c.id, c.doc_id, e.name", {"names": list(weight)}):
            scores[chunk_id] += weight[name]
            doc_of[chunk_id] = doc_id
        top = [c for c, _ in scores.most_common(k)]
        texts = dict(self.con(read_only=True).execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id IN (SELECT unnest(?))", [top]).fetchall())
        return [{"chunk_id": c, "doc_id": doc_of[c], "score": float(scores[c]), "text": texts.get(c, "")} for c in top]
