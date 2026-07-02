"""Shared color palettes and small plotting helpers for the working-note figures.

Addresses the reviewer comment that curve colors were not consistent across
figures. Colors are organized into three groups so a given hue never means two
different things in the recurring pipeline story:

1. ``STAGE`` — the recurring pipeline components (BM25, dense, BM25+Dense fusion,
   cross-encoder rerank, post-rerank fusion). These use the tab10 family and mean
   *exactly the same thing in every figure they appear in*. BM25 / dense /
   BM25+Dense fusion keep their original blue / orange / green; CE rerank and
   post-rerank fusion get their own reserved purple / red so they are never
   confused with a first-stage retriever.

2. ``ABLATION`` — a single shared palette for the one-off ablation figures
   (reranker comparison, question-length bins, query rewriting). It is built from
   the four tab10 colors the pipeline stages do *not* use (cyan / pink / olive /
   brown), so it is distinct from the STAGE colors by construction. These three
   figures are unrelated topics with their own legends, so reusing one small
   palette across them keeps the total number of distinct colors low without
   creating confusion.

3. Monochrome — the snippet (docs vs snippets) and HyDE (HyDE vs original)
   figures use dark grey for both curves and label them with text directly on the
   panel (see ``annotate_curves_end``), so they add no extra colors at all.
"""

from __future__ import annotations

# --- Pipeline stages: same meaning in every figure --------------------------
# tab10 family. bm25 / dense / hybrid_fusion are unchanged from the original
# figures; ce_rerank and post_fusion are reserved so they never collide with a
# first-stage retriever (previously blue / orange).
STAGE: dict[str, str] = {
    "bm25": "#1f77b4",           # blue   (unchanged)
    "dense": "#ff7f0e",          # orange (unchanged)
    "hybrid_fusion": "#2ca02c",  # green  (unchanged)
    "ce_rerank": "#9467bd",      # purple (reserved)
    "post_fusion": "#d62728",    # red    (reserved)
}

# --- One shared palette for the ablation figures (05, 08, 11) ----------------
# The four tab10 colors NOT used by STAGE (which takes blue/orange/green/red/
# purple), so it is guaranteed distinct from every pipeline-stage color. Grey is
# excluded because it is reserved for the monochrome snippet / HyDE figures.
ABLATION: list[str] = [
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#8c564b",  # brown (only the 4-series reranker figure uses this one)
]

# Fig 05 — reranker comparison (4 models).
RERANKER: dict[str, str] = dict(zip(
    [
        "ms-marco-MiniLM-L-12-v2, 33.4M",
        "bge-reranker-v2-m3, 568M, tok_len=200",
        "bge-reranker-v2-m3, 568M",
        "bge-reranker-v2-gemma, 2.5B",
    ],
    ABLATION,
))

# Fig 08 — MAP/Recall by question-length bin.
LENGTH_BIN: dict[str, str] = dict(zip(
    ["short (\u22647)", "mid (8\u201310)", "long (\u226511)"],
    ABLATION,
))

# Fig 11 — query rewriting ablation.
REWRITE: dict[str, str] = dict(zip(
    [
        "CE Reranker (no rewriting)",
        "query rewriting A: only typo fixing and minimal grammatical edits",
        "query rewriting B: questions generic enrichment",
    ],
    ABLATION,
))

# --- Monochrome figures: curves labeled with on-panel text, no color legend --
_DARK_GREY = "#444444"

# Fig 09 — snippet route (docs vs snippets).
SNIPPET: dict[str, str] = {
    "docs (full abstracts)": _DARK_GREY,
    "snippets": _DARK_GREY,
}

# Fig 10 — HyDE vs original dense query.
HYDE: dict[str, str] = {
    "HyDE": _DARK_GREY,
    "Original": _DARK_GREY,
}


def annotate_curves_end(
    ax,
    series,
    y_offsets=None,
    *,
    x_offset: float = -6.0,
    fontsize: float = 12.0,
    fontweight: str = "bold",
):
    """Label each curve with in-panel text near its right end (no color legend).

    Parameters
    ----------
    ax
        Matplotlib axes to annotate.
    series
        Iterable of ``(xs, ys, label, color)`` tuples. ``xs``/``ys`` are the
        plotted data; the label is anchored at the curve's last point.
    y_offsets
        Per-series vertical offset in *points* (positive = above the point,
        negative = below). Defaults to alternating +10 / -10 so two nearby
        curves are separated. Extend the list for more curves.
    x_offset
        Horizontal offset in points from the last point. Negative pushes the
        text left (into the panel) so long labels stay inside the axes.
    fontsize, fontweight
        Text styling.

    Notes
    -----
    Text is right-aligned and anchored on the last data point, so it grows
    leftward into the panel and does not run off the right edge. Callers should
    still leave a little head/tail room (e.g. widen the x-axis and add top
    y-headroom) so labels sit inside the axes box.
    """
    series = list(series)
    if y_offsets is None:
        y_offsets = [10.0 if i % 2 == 0 else -10.0 for i in range(len(series))]

    for (xs, ys, label, color), dy in zip(series, y_offsets):
        if xs is None or ys is None or len(xs) == 0 or len(ys) == 0:
            continue
        x_end = xs[-1]
        y_end = ys[-1]
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            label,
            xy=(x_end, y_end),
            xytext=(x_offset, dy),
            textcoords="offset points",
            ha="right",
            va=va,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            clip_on=False,
        )
