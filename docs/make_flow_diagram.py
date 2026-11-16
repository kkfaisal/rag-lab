"""Draws docs/rag-lab-flow.svg, the README's "How it works" diagram.

    python docs/make_flow_diagram.py
    uv run --no-project --with playwright python docs/make_flow_diagram.py --png   # also docs/rag-lab-flow.png
"""

import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
W, H = 1670, 735
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
COL_W, GAP, X0, TOP, BOTTOM = 290, 40, 30, 96, 612

# (title, colour, light tint, boxes as (title, subtitle))
COLUMNS = [
    ("Data", "#7c3aed", "#f5f3ff", [
        ("Benchmark or your files", "extractor script you can read"),
        ("Document store", "docs + questions, versioned"),
        ("Subsets", "dev, small: iterate fast"),
    ]),
    ("Ingestion pipeline", "#2563eb", "#eff6ff", [
        ("source", "dataset · subset · snapshot"),
        ("transform", "optional: clean, filter"),
        ("chunk", "fixed, Slack-aware, or a script"),
        ("embed", "optional: local model or a script"),
    ]),
    ("Targets", "#059669", "#ecfdf5", [
        ("Local store", "BM25 + exact vectors"),
        ("PostgreSQL + pgvector", ""),
        ("Qdrant", ""),
        ("OpenSearch", ""),
        ("Elasticsearch", ""),
        ("Any SQL database", "or your own Python script"),
    ]),
    ("Retrieval pipeline", "#d97706", "#fffbeb", [
        ("retrieve", "BM25 · vector · hybrid"),
        ("rerank", "cross-encoder"),
        ("generate", "Claude: single pass or agent"),
        ("benchmark", "recall, nDCG, MRR, judge"),
    ]),
    ("Evaluate & learn", "#db2777", "#fdf2f8", [
        ("Results matrix", "every target × pipeline"),
        ("Compare two runs", "fixed / broken, is it noise?"),
        ("One question", "gold vs retrieved, answer"),
    ]),
]


def text(x, y, s, size=15, weight=400, fill="#0f172a", anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def column(i, title, colour, tint, boxes):
    x = X0 + i * (COL_W + GAP)
    out = [f'<rect x="{x}" y="{TOP}" width="{COL_W}" height="{BOTTOM - TOP}" rx="16" fill="{tint}" stroke="{colour}" stroke-opacity=".35" stroke-width="1.5"/>',
           f'<circle cx="{x + 32}" cy="{TOP + 34}" r="15" fill="{colour}"/>',
           text(x + 32, TOP + 39.5, str(i + 1), 15, 700, "#fff"),
           text(x + 56, TOP + 40, title, 18, 700, colour, "start")]
    chain = title != "Targets"           # targets are alternatives, the rest is a sequence
    top, bottom = TOP + 70, BOTTOM - 22
    gap = 40 if chain else 12
    h = (bottom - top - gap * (len(boxes) - 1)) / len(boxes)
    for j, (name, sub) in enumerate(boxes):
        y = top + j * (h + gap)
        out.append(f'<rect x="{x + 18}" y="{y}" width="{COL_W - 36}" height="{h}" rx="10" fill="#fff" stroke="{colour}" stroke-width="1.5"/>')
        if sub:
            out += [text(x + COL_W / 2, y + h / 2 - 3, name, 16, 650), text(x + COL_W / 2, y + h / 2 + 17, sub, 13, 400, "#64748b")]
        else:
            out.append(text(x + COL_W / 2, y + h / 2 + 5.5, name, 16, 650))
        if chain and j < len(boxes) - 1:
            cx = x + COL_W / 2
            out.append(f'<line x1="{cx}" y1="{y + h + 4}" x2="{cx}" y2="{y + h + gap - 6}" stroke="{colour}" stroke-width="2" marker-end="url(#a-{i})"/>')
    return out


def svg():
    defs = "".join(f'<marker id="a-{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c[1]}"/></marker>' for i, c in enumerate(COLUMNS))
    defs += ('<marker id="a-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f"<defs>{defs}</defs>",
           f'<rect width="{W}" height="{H}" rx="20" fill="#ffffff"/>',
           text(X0, 54, "rag-lab · how a RAG experiment flows", 26, 750, "#0f172a", "start"),
           text(W - X0, 54, "every node: a built-in with settings, or Python you edit in the browser", 15, 400, "#64748b", "end")]
    for i, c in enumerate(COLUMNS):
        out += column(i, *c)
    mid = (TOP + BOTTOM) / 2 + 20
    for i in range(len(COLUMNS) - 1):         # stage → stage
        x1 = X0 + (i + 1) * COL_W + i * GAP + 3
        out.append(f'<line x1="{x1}" y1="{mid}" x2="{x1 + GAP - 6}" y2="{mid}" stroke="#475569" stroke-width="2.5" marker-end="url(#a-g)"/>')
    # the learning loop: from evaluation back to both pipelines
    cx = lambda i: X0 + i * (COL_W + GAP) + COL_W / 2  # noqa: E731
    y = BOTTOM + 48
    out += [f'<path d="M{cx(4)},{BOTTOM + 4} V{y} H{cx(1)} V{BOTTOM + 8}" fill="none" stroke="#475569" stroke-width="2.5" stroke-dasharray="7 6" marker-end="url(#a-g)"/>',
            f'<path d="M{cx(3)},{y} V{BOTTOM + 8}" fill="none" stroke="#475569" stroke-width="2.5" stroke-dasharray="7 6" marker-end="url(#a-g)"/>',
            f'<rect x="{cx(2) - 250}" y="{y - 18}" width="500" height="36" rx="18" fill="#0f172a"/>',
            text(cx(2), y + 5.5, "change one node, run again: only what changed re-runs", 15, 600, "#fff"),
            text(W / 2, H - 28, "Every run keeps its full config, scripts, dataset snapshot, target version, per-step logs and timings, tokens and cost.", 15, 400, "#64748b")]
    out.append("</svg>")
    return "\n".join(out)


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
