"""Draws docs/rag-lab-flow.svg, the README's "How it works" diagram.

    python docs/make_flow_diagram.py
    uv run --no-project --with playwright python docs/make_flow_diagram.py --png   # also docs/rag-lab-flow.png

Layout: the data source on the left holds raw documents and the gold set (questions, gold documents,
gold answers). Raw documents flow through the ingestion pipeline into a target; the retrieval
pipeline answers the questions from that target; its results are compared with the gold set, which
gives the benchmark.
"""

import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
W, H = 1700, 915
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#0f172a", "#64748b", "#475569"
DATA, ING, TGT, RET, GOLD, CMP, BENCH = "#7c3aed", "#2563eb", "#059669", "#0891b2", "#d97706", "#db2777", "#4338ca"
TINT = {DATA: "#f5f3ff", ING: "#eff6ff", TGT: "#ecfdf5", RET: "#ecfeff", GOLD: "#fffbeb", CMP: "#fdf2f8"}
COLOURS = [DATA, ING, TGT, RET, GOLD, CMP, LINE]

# rows: (top, bottom) of the ingestion, retrieval and evaluation bands
R1, R2, R3 = (100, 300), (380, 590), (650, 840)
LEFT, MAIN, RIGHT = (30, 330), 390, W - 30
NODE_W, NODE_GAP, PAD = 196, 40, 24


def text(x, y, s, size=15, weight=400, fill=INK, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def box(x, y, w, h, colour, title, sub="", fill="#fff", dash=False, size=18):
    d = ' stroke-dasharray="7 5"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{colour}" stroke-width="1.8"{d}/>']
    lines = sub if isinstance(sub, list) else ([sub] if sub else [])
    y0 = y + h / 2 - 11 * len(lines) + 6
    out.append(text(x + w / 2, y0, title, size, 700))
    for k, s in enumerate(lines):
        out.append(text(x + w / 2, y0 + 24 + 21 * k, s, 14.5, 400, MUTED))
    return out


def panel(x, y, w, h, colour, n, title, dash=False):
    d = ' stroke-dasharray="8 6"' if dash else ""
    return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{TINT[colour]}" stroke="{colour}" stroke-opacity=".45" stroke-width="1.6"{d}/>',
            f'<circle cx="{x + 30}" cy="{y + 30}" r="14" fill="{colour}"/>',
            text(x + 30, y + 35.5, str(n), 15, 700, "#fff"),
            text(x + 52, y + 36.5, title, 19, 700, colour, "start")]


def arrow(points, colour=LINE, dash=False, width=2.4):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    s = ' stroke-dasharray="7 6"' if dash else ""
    return [f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{s} marker-end="url(#m{COLOURS.index(colour)})"/>']


def chain(y, top, colour, nodes):
    out = []
    for i, (name, sub) in enumerate(nodes):
        x = MAIN + PAD + i * (NODE_W + NODE_GAP)
        out += box(x, y, NODE_W, top, colour, name, sub)
        if i < len(nodes) - 1:
            out += arrow([(x + NODE_W + 4, y + top / 2), (x + NODE_W + NODE_GAP - 5, y + top / 2)], colour)
    return out


def pill(cx, cy, w, label, fill=INK):
    return [f'<rect x="{cx - w / 2}" y="{cy - 17}" width="{w}" height="34" rx="17" fill="{fill}"/>',
            text(cx, cy + 5.5, label, 14.5, 600, "#fff")]


def svg():
    defs = "".join(f'<marker id="m{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for i, c in enumerate(COLOURS))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f"<defs>{defs}</defs>",
         f'<rect width="{W}" height="{H}" rx="20" fill="#ffffff"/>',
         text(30, 56, "rag-lab · how a RAG pipeline is built and benchmarked", 26, 750, INK, "start")]

    # 1 · data source: raw documents + the gold set
    lx, lw = LEFT[0], LEFT[1] - LEFT[0]
    o += panel(lx, R1[0], lw, R3[1] - R1[0], DATA, 1, "Data source")
    o += box(lx + 20, 160, lw - 40, 110, DATA, "Raw documents", ["a public benchmark", "or your own files"])
    o += [f'<rect x="{lx + 12}" y="{R2[0] + 6}" width="{lw - 24}" height="{R3[1] - R2[0] - 18}" rx="14" fill="{TINT[GOLD]}" stroke="{GOLD}" stroke-width="1.6" stroke-dasharray="8 6"/>',
          text(lx + lw / 2, R2[0] + 36, "Gold set: the answer key", 16, 700, GOLD)]
    o += box(lx + 26, 445, lw - 52, 115, GOLD, "Questions", ["what users would ask"])
    o += box(lx + 26, 685, lw - 52, 125, GOLD, "Gold documents", ["+ gold answers:", "what a perfect system", "finds and says"])

    # 2 · ingestion pipeline → 3 · target
    ing_w = 4 * NODE_W + 3 * NODE_GAP + 2 * PAD
    o += panel(MAIN, R1[0], ing_w, R1[1] - R1[0], ING, 2, "Ingestion pipeline")
    o += chain(160, 110, ING, [("source", "dataset or subset"), ("transform", "optional: clean"),
                               ("chunk", "fixed, Slack-aware"), ("embed", "optional: vectors")])
    o += arrow([(LEFT[1] - 18, 215), (MAIN - 5, 215)], DATA)
    tx, tw = MAIN + ing_w + 48, RIGHT - (MAIN + ing_w + 48)
    o += arrow([(MAIN + ing_w + 3, 215), (tx - 5, 215)], ING)
    top, bot = R1[0] + 22, R1[1]
    o += [f'<path d="M{tx},{top} v{bot - top} a{tw / 2},18 0 0 0 {tw},0 v-{bot - top}" fill="{TINT[TGT]}" stroke="{TGT}" stroke-width="1.8"/>',
          f'<ellipse cx="{tx + tw / 2}" cy="{top}" rx="{tw / 2}" ry="18" fill="#fff" stroke="{TGT}" stroke-width="1.8"/>',
          f'<circle cx="{tx + 30}" cy="{top + 50}" r="14" fill="{TGT}"/>', text(tx + 30, top + 55.5, "3", 15, 700, "#fff"),
          text(tx + 52, top + 56.5, "Target (the index)", 19, 700, TGT, "start"),
          text(tx + tw / 2, top + 94, "chunks, vectors, BM25", 14.5, 600, INK),
          text(tx + tw / 2, top + 122, "local · pgvector · Qdrant", 14.5, 400, MUTED),
          text(tx + tw / 2, top + 144, "OpenSearch · Elasticsearch", 14.5, 400, MUTED),
          text(tx + tw / 2, top + 166, "any SQL DB · your script", 14.5, 400, MUTED)]

    # 4 · retrieval pipeline, fed by the questions (or any query typed in the search box)
    ret_w = 3 * NODE_W + 2 * NODE_GAP + 2 * PAD
    o += panel(MAIN, R2[0], ret_w, R2[1] - R2[0], RET, 4, "Retrieval pipeline")
    sx, sw = MAIN + ret_w - PAD - 330, 330          # the search box
    o += [f'<rect x="{sx}" y="{R2[0] + 13}" width="{sw}" height="36" rx="18" fill="#fff" stroke="{RET}" stroke-width="1.6"/>',
          f'<circle cx="{sx + 22}" cy="{R2[0] + 29}" r="7" fill="none" stroke="{RET}" stroke-width="2.2"/>',
          f'<line x1="{sx + 27}" y1="{R2[0] + 34}" x2="{sx + 33}" y2="{R2[0] + 40}" stroke="{RET}" stroke-width="2.4" stroke-linecap="round"/>',
          text(sx + 44, R2[0] + 36, "Search: try any query live", 15, 600, RET, "start")]
    o += chain(450, 115, RET, [("retrieve", ["BM25, vector", "or hybrid"]), ("rerank", ["optional:", "cross-encoder"]),
                               ("generate", ["Claude: one pass,", "or an agent"])])
    o += arrow([(LEFT[1] - 14, 502), (MAIN - 5, 502)], GOLD)
    rx0 = sx - 76                                   # between the panel title and the search box
    o += arrow([(tx + tw / 2, R1[1] + 20), (tx + tw / 2, 340), (rx0, 340), (rx0, R2[0] - 4)], TGT)
    o += pill((MAIN + ing_w) / 2 + 240, 340, 250, "retrieve searches the index", TGT)

    # retrieved results
    qx = MAIN + ret_w + 48
    o += arrow([(MAIN + ret_w + 3, 507), (qx - 5, 507)], RET)
    o += box(qx, R2[0] + 20, RIGHT - qx, R2[1] - R2[0] - 20, RET, "Retrieved results",
             ["top-k chunks and their documents", "+ the generated answer, citing them"], fill=TINT[RET])

    # 5 · compare with the gold set → 6 · benchmark
    o += panel(MAIN, R3[0], ret_w, R3[1] - R3[0], CMP, 5, "Compare with the gold set")
    for k, (a, b, m) in enumerate([("retrieved docs", "gold docs", "recall@k · nDCG · MRR"),
                                   ("generated answer", "gold answer", "LLM judge: correct? complete?")]):
        y = R3[0] + 60 + k * 62
        o += [f'<rect x="{MAIN + PAD}" y="{y}" width="{ret_w - 2 * PAD}" height="50" rx="10" fill="#fff" stroke="{CMP}" stroke-width="1.6"/>',
              text(MAIN + PAD + 16, y + 31, a, 16, 700, RET, "start"),
              text(MAIN + PAD + 168, y + 31, "vs", 15, 400, MUTED, "start"),
              text(MAIN + PAD + 196, y + 31, b, 16, 700, GOLD, "start"),
              text(MAIN + ret_w - PAD - 16, y + 31, "→  " + m, 15, 600, CMP, "end")]
    o += arrow([(LEFT[1] - 14, 747), (MAIN - 5, 747)], GOLD)
    o += arrow([(qx + 120, R2[1] + 3), (qx + 120, 620), (MAIN + ret_w - 140, 620), (MAIN + ret_w - 140, R3[0] - 5)], RET)
    o += arrow([(MAIN + ret_w + 3, 745), (qx - 5, 745)], CMP)
    bw = RIGHT - qx
    o += [f'<rect x="{qx}" y="{R3[0]}" width="{bw}" height="{R3[1] - R3[0]}" rx="16" fill="{BENCH}"/>',
          f'<circle cx="{qx + 30}" cy="{R3[0] + 30}" r="14" fill="#fff"/>', text(qx + 30, R3[0] + 35.5, "6", 15, 700, BENCH),
          text(qx + 52, R3[0] + 36.5, "Benchmark", 19, 700, "#fff", "start")]
    for k, s in enumerate(["scores for every run and every question",
                           "results matrix: every target × pipeline",
                           "compare two runs: fixed, broken, or noise?"]):
        o.append(text(qx + 30, R3[0] + 82 + k * 32, "•  " + s, 15.5, 500, "#e0e7ff", "start"))

    o.append(text(W / 2, H - 26, "Change any node (or its Python), run again, and see which questions it fixed or broke. "
                                 "Only what changed re-runs.", 15.5, 400, MUTED))
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    f = HERE / "rag-lab-flow.svg"
    f.write_text(svg())
    print("wrote", f)
    if "--png" in sys.argv:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch(channel="chrome")
            page = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            page.set_content(f'<html><body style="margin:0;background:transparent">{svg()}</body></html>')
            page.locator("svg").screenshot(path=str(HERE / "rag-lab-flow.png"), omit_background=True)
            br.close()
        print("wrote", HERE / "rag-lab-flow.png")
