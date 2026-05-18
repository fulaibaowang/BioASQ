# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dicty (Python 3.14 venv)
#     language: python
#     name: dicty-py314
# ---

# %% [markdown]
# # Rerank cutoff 02 — Hard global score threshold
#
# **Scope:** rerank-only analysis. Gold is evaluated only within the top-N
# candidate pool `C_q^N` (not full retrieval recall).
#
# For each reranker family (`bgem3`, `bge_gemma`), sweep a global score threshold
# `t` and measure **conditional residual recall loss** on pool-reachable gold
# `G_q^(N) = G_q ∩ C_q^N`:
#
# `residual_loss_q(t) = |G_q^(N) \\ K_q(t)| / |G_q^(N)|`, with `K_q(t) = {d in C_q^N : score(d) >= t}`.
#
# Select the **largest** `t` such that macro mean loss `<= alpha` and p90 loss `<= beta`
# (defaults `alpha=0.10`, `beta=0.30`). See **Discussion** for caveats.

# %%
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore[assignment]

try:
    _NB_DIR = Path(__file__).resolve().parent
except NameError:
    _NB_DIR = Path(".").resolve()
    if not (_NB_DIR / "scripts").exists():
        _NB_DIR = _NB_DIR.parent

REPO_ROOT = _NB_DIR if (_NB_DIR / "scripts").exists() else _NB_DIR.parent
assert (REPO_ROOT / "scripts").exists(), f"Cannot locate repo root (tried {REPO_ROOT})"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "public" / "shared_scripts"))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from retrieval_eval.common import build_topics_and_gold, load_questions

from _rerank_plot_sample import DEFAULT_N_GOLD_BINS, sample_queries_for_diagnostic_plots

# %% [markdown]
# ## Configuration

# %%
N_CAP = 200
ALPHA_DEFAULT = 0.10
BETA_DEFAULT = 0.30
ALPHA_GRID = [0.05, 0.10, 0.15]
BETA_GRID = [0.20, 0.30, 0.40]

RNG_SEED = 42
SAMPLE_SIZE = 48
MAX_UNIQUE_T = 8000

RERANK_RUN_SOURCES: list[tuple[str, Path]] = [
    (
        "bgem3",
        REPO_ROOT / "output" / "workflow_baseline_full_run_both_routes" / "rerank" / "runs",
    ),
    (
        "bge_gemma",
        REPO_ROOT / "output" / "workflow_baseline_full_run_both_routes_gemma" / "rerank" / "runs",
    ),
]

GOLD_JSON_MAP: dict[str, Path] = {
    "13B1_golden": REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B1_golden.json",
    "13B2_golden": REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B2_golden.json",
    "13B3_golden": REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B3_golden.json",
    "13B4_golden": REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B4_golden.json",
    "training14b_10pct_sample": REPO_ROOT / "example" / "training14b_10pct_sample.json",
}

# %% [markdown]
# ## Helpers

# %%
def load_run_tsv(path: Path, n_cap: int = 200) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["score"] = df["score"].astype(float)
    df["docno"] = df["docno"].astype(str)
    df = df.sort_values(["qid", "score"], ascending=[True, False])
    return df.groupby("qid").head(n_cap).reset_index(drop=True)


def load_gold_set_map(json_path: Path) -> dict[str, set[str]]:
    qs = load_questions(json_path)
    _, gold_lists = build_topics_and_gold(qs)
    return {qid: set(pmids) for qid, pmids in gold_lists.items()}


def _split_from_stem(stem: str) -> str | None:
    m = re.match(r"best_rrf_(.+?)_top\d+", stem)
    return m.group(1) if m else None


def build_t_candidates(all_scores: list[np.ndarray], max_unique: int = MAX_UNIQUE_T) -> np.ndarray:
    if not all_scores:
        return np.array([0.0])
    u = np.unique(np.concatenate([np.asarray(s, dtype=np.float64) for s in all_scores]))
    u.sort()
    if len(u) > max_unique:
        idx = np.round(np.linspace(0, len(u) - 1, max_unique)).astype(int)
        u = np.unique(u[idx])
    t0 = float(u[0] - 1e-12) if len(u) else 0.0
    return np.concatenate([[t0], u])


def sweep_thresholds_for_source(
    queries: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """*queries*: each dict has scores (np desc), gold_mask (bool), R (int >0)."""
    if not queries:
        return pd.DataFrame(), dict(n_queries=0)
    all_scores = [q["scores"] for q in queries]
    t_candidates = build_t_candidates(all_scores)
    rows = []
    for t in t_candidates:
        losses = []
        kepts = []
        for q in queries:
            sc = q["scores"]
            gm = q["gold_mask"]
            R = int(gm.sum())
            lost = int(np.sum((sc < t) & gm))
            losses.append(lost / R)
            kepts.append(int(np.sum(sc >= t)))
        lo = np.asarray(losses, dtype=np.float64)
        rows.append(
            dict(
                t=float(t),
                macro_mean_loss=float(lo.mean()),
                macro_median_loss=float(np.median(lo)),
                p90_loss=float(np.percentile(lo, 90)),
                mean_kept=float(np.mean(kepts)),
                median_kept=float(np.median(kepts)),
                n_queries=len(queries),
            )
        )
    sweep = pd.DataFrame(rows)
    meta = dict(n_queries=len(queries))
    return sweep, meta


def select_t_star(sweep: pd.DataFrame, alpha: float, beta: float) -> float | None:
    ok = (sweep["macro_mean_loss"] <= alpha) & (sweep["p90_loss"] <= beta)
    idx = np.where(ok.values)[0]
    if len(idx) == 0:
        return None
    return float(sweep["t"].values[idx[-1]])


# %% [markdown]
# ## 1. Load runs and per-query pool statistics

# %%
meta_rows: list[dict] = []
query_by_key: dict[tuple[str, str, str], dict] = {}

for src, runs_dir in RERANK_RUN_SOURCES:
    if not runs_dir.is_dir():
        print(f"[skip] {src}: missing {runs_dir}")
        continue
    for rp in sorted(runs_dir.glob("best_rrf_*_top*.tsv")):
        split = _split_from_stem(rp.stem)
        if split is None or split not in GOLD_JSON_MAP:
            continue
        gp = GOLD_JSON_MAP[split]
        if not gp.exists():
            continue
        rdf = load_run_tsv(rp, N_CAP)
        gmap = load_gold_set_map(gp)
        common = sorted(set(rdf["qid"].unique()) & set(gmap))
        for qid in common:
            qdf = rdf[rdf["qid"] == qid]
            scores = qdf["score"].values.astype(np.float64)
            docnos = qdf["docno"].values.astype(str)
            gold = gmap[qid]
            pool = set(docnos)
            gold_in_pool = gold & pool
            R = len(gold_in_pool)
            gold_mask = np.array([d in gold_in_pool for d in docnos], dtype=bool)
            meta_rows.append(
                dict(
                    rerank_source=src,
                    split=split,
                    qid=qid,
                    n_gold_full=len(gold),
                    n_pool_gold=R,
                    n_docs=len(scores),
                )
            )
            query_by_key[(src, split, qid)] = dict(
                scores=scores,
                docnos=docnos,
                gold_mask=gold_mask,
                gold_in_pool=gold_in_pool,
                R=R,
            )

df_meta = pd.DataFrame(meta_rows)
print(df_meta.groupby("rerank_source").size())
print("skipped no pool gold (per row):", int((df_meta["n_pool_gold"] == 0).sum()))

# %% [markdown]
# ## 2. Threshold sweep and default selection (per rerank_source, pooled splits)

# %%
sweep_by_src: dict[str, pd.DataFrame] = {}
t_star_by_src: dict[str, float | None] = {}
skip_report: list[dict] = []

for src in [s for s, _ in RERANK_RUN_SOURCES]:
    keys = [k for k in query_by_key if k[0] == src]
    queries = [query_by_key[k] for k in keys if query_by_key[k]["R"] > 0]
    n_skip = sum(1 for k in keys if query_by_key[k]["R"] == 0)
    skip_report.append(dict(rerank_source=src, n_queries_R_pos=len(queries), n_queries_R_zero=n_skip))
    if not queries:
        print(f"{src}: no queries with pool gold; skip sweep")
        continue
    sw, _ = sweep_thresholds_for_source(queries)
    sweep_by_src[src] = sw
    t_star_by_src[src] = select_t_star(sw, ALPHA_DEFAULT, BETA_DEFAULT)
    print(f"{src}: selected t* = {t_star_by_src[src]}  (alpha={ALPHA_DEFAULT}, beta={BETA_DEFAULT})")

df_skip = pd.DataFrame(skip_report)
display(df_skip)

# %% [markdown]
# ## 1b. Aligned pool-conditional summary at `t*` (trilogy table schema)

# %%
def _metrics_at_t(
    queries: list[tuple[tuple[str, str, str], dict]], t: float | None,
) -> dict:
    if t is None or not queries:
        return dict(
            n_eval=0,
            n_skipped_pool_gold=0,
            mean_res_loss=np.nan,
            median_res_loss=np.nan,
            p90_res_loss=np.nan,
            mean_kept=np.nan,
            median_kept=np.nan,
        )
    losses, kepts = [], []
    for _k, q in queries:
        sc, gm = q["scores"], q["gold_mask"]
        R = int(gm.sum())
        if R == 0:
            continue
        losses.append(float(np.sum((sc < t) & gm) / R))
        kepts.append(int(np.sum(sc >= t)))
    if not losses:
        return dict(
            n_eval=0,
            n_skipped_pool_gold=0,
            mean_res_loss=np.nan,
            median_res_loss=np.nan,
            p90_res_loss=np.nan,
            mean_kept=np.nan,
            median_kept=np.nan,
        )
    lo = np.asarray(losses, dtype=np.float64)
    kk = np.asarray(kepts, dtype=np.float64)
    return dict(
        n_eval=len(losses),
        n_skipped_pool_gold=0,
        mean_res_loss=float(lo.mean()),
        median_res_loss=float(np.median(lo)),
        p90_res_loss=float(np.percentile(lo, 90)),
        mean_kept=float(kk.mean()),
        median_kept=float(np.median(kk)),
    )


_align_fmt = dict(
    mean_res_loss="{:.4f}",
    median_res_loss="{:.4f}",
    p90_res_loss="{:.4f}",
    mean_kept="{:.1f}",
    median_kept="{:.0f}",
)

for src in [s for s, _ in RERANK_RUN_SOURCES]:
    t_star = t_star_by_src.get(src)
    align_rows = []
    for split in sorted(df_meta[df_meta["rerank_source"] == src]["split"].unique()):
        keys = [k for k in query_by_key if k[0] == src and k[1] == split]
        all_k = [(k, query_by_key[k]) for k in keys]
        n_sk = sum(1 for k, q in all_k if q["R"] == 0)
        qpos = [(k, q) for k, q in all_k if q["R"] > 0]
        m = _metrics_at_t(qpos, t_star)
        m["n_skipped_pool_gold"] = n_sk
        align_rows.append(
            dict(
                rerank_source=src,
                split=split,
                method="hard_t_star",
                t_star=t_star,
                **m,
            )
        )
    # pooled
    keys = [k for k in query_by_key if k[0] == src]
    all_k = [(k, query_by_key[k]) for k in keys]
    n_sk = sum(1 for k, q in all_k if q["R"] == 0)
    qpos = [(k, q) for k, q in all_k if q["R"] > 0]
    m = _metrics_at_t(qpos, t_star)
    m["n_skipped_pool_gold"] = n_sk
    align_rows.append(
        dict(
            rerank_source=src,
            split="__pooled__",
            method="hard_t_star",
            t_star=t_star,
            **m,
        )
    )
    print(f"\n=== Aligned at t*  ({src}) ===")
    display(pd.DataFrame(align_rows).style.format(_align_fmt))

# %% [markdown]
# ### Sweep summary (decimated for display if long)

# %%
_disp_n = 400
for src, sw in sweep_by_src.items():
    print(f"\n=== {src}  (showing up to {_disp_n} rows) ===")
    sub = sw if len(sw) <= _disp_n else sw.iloc[:: max(1, len(sw) // _disp_n)]
    display(sub.style.format({c: "{:.4f}" for c in sub.columns if c != "n_queries"}))

# %% [markdown]
# ## 3. Aggregate plots and selected threshold

# %%
for src, sw in sweep_by_src.items():
    t_star = t_star_by_src.get(src)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(sw["t"], sw["macro_mean_loss"], label="macro mean", color="tab:blue")
    ax.plot(sw["t"], sw["p90_loss"], label="p90", color="tab:orange", alpha=0.85)
    ax.axhline(ALPHA_DEFAULT, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(BETA_DEFAULT, color="tab:orange", ls="--", lw=0.8, alpha=0.6)
    if t_star is not None:
        ax.axvline(t_star, color="tab:green", lw=1.5, label=f"t*={t_star:.4g}")
    ax.set_xlabel("threshold t")
    ax.set_ylabel("residual loss")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{src}: loss vs t")

    ax2 = axes[1]
    ax2.plot(sw["t"], sw["mean_kept"], label="mean |K|", color="tab:purple")
    ax2.plot(sw["t"], sw["median_kept"], label="median |K|", color="tab:brown", alpha=0.8)
    if t_star is not None:
        ax2.axvline(t_star, color="tab:green", lw=1.5, label=f"t*={t_star:.4g}")
    ax2.set_xlabel("threshold t")
    ax2.set_ylabel("kept docs")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_title(f"{src}: kept vs t")

    fig.suptitle(f"Hard score cutoff  |  {src}  (N={N_CAP})", fontsize=12)
    plt.show()

# %% [markdown]
# ## 4. Alpha / beta sensitivity

# %%
def sensitivity_table(sweep: pd.DataFrame, src: str) -> pd.DataFrame:
    rows = []
    for a in ALPHA_GRID:
        t_ = select_t_star(sweep, a, BETA_DEFAULT)
        row: dict = dict(rerank_source=src, mode="vary_alpha", alpha=a, beta=BETA_DEFAULT, t_star=t_)
        if t_ is not None:
            m = sweep[np.isclose(sweep["t"], t_, rtol=0.0, atol=1e-9)]
            if m.empty:
                m = sweep.iloc[(sweep["t"] - t_).abs().argsort()[:1]]
            row["macro_mean_at_t"] = float(m["macro_mean_loss"].iloc[0])
            row["p90_at_t"] = float(m["p90_loss"].iloc[0])
            row["mean_kept_at_t"] = float(m["mean_kept"].iloc[0])
        rows.append(row)
    for b in BETA_GRID:
        t_ = select_t_star(sweep, ALPHA_DEFAULT, b)
        row = dict(rerank_source=src, mode="vary_beta", alpha=ALPHA_DEFAULT, beta=b, t_star=t_)
        if t_ is not None:
            m = sweep[np.isclose(sweep["t"], t_, rtol=0.0, atol=1e-9)]
            if m.empty:
                m = sweep.iloc[(sweep["t"] - t_).abs().argsort()[:1]]
            row["macro_mean_at_t"] = float(m["macro_mean_loss"].iloc[0])
            row["p90_at_t"] = float(m["p90_loss"].iloc[0])
            row["mean_kept_at_t"] = float(m["mean_kept"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


for src, sw in sweep_by_src.items():
    print(f"\n=== sensitivity  {src} ===")
    display(sensitivity_table(sw, src))

# %% [markdown]
# ## 5. Canonical 48 diagnostic plots (same `(split, qid)` as notebook 01)

# %%
canonical_pool = (
    df_meta[df_meta["n_gold_full"] > 0][["split", "qid", "n_gold_full"]]
    .rename(columns={"n_gold_full": "n_gold"})
    .drop_duplicates(subset=["split", "qid"], keep="first")
)
canonical_48 = sample_queries_for_diagnostic_plots(
    canonical_pool,
    sample_size=SAMPLE_SIZE,
    rng_seed=RNG_SEED,
    n_gold_bins=DEFAULT_N_GOLD_BINS,
)
print(f"canonical_48: {len(canonical_48)} queries")

# %%
for src in sweep_by_src:
    t_star = t_star_by_src.get(src)
    ncols = 6
    nrows = (len(canonical_48) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6), constrained_layout=True)
    axf = np.asarray(axes).flatten()
    for idx, (sp, qid) in enumerate(canonical_48):
        ax = axf[idx]
        key = (src, sp, qid)
        if key not in query_by_key or t_star is None:
            ax.set_visible(False)
            continue
        qd = query_by_key[key]
        sc, docnos, gm = qd["scores"], qd["docnos"], qd["gold_mask"]
        ranks = np.arange(1, len(sc) + 1)
        ax.plot(ranks, sc, color="tab:blue", lw=0.7)
        ax.axhline(t_star, color="tab:green", lw=1.2, ls="-")
        hit_ranks = ranks[gm]
        hit_scores = sc[gm]
        ax.scatter(hit_ranks, hit_scores, color="tab:red", s=8, zorder=3, label="pool gold")
        ax.set_title(f"{sp[:12]} {qid[:8]}.. |G^N|={qd['R']}", fontsize=5)
        ax.tick_params(labelsize=4)
    for j in range(len(canonical_48), len(axf)):
        axf[j].set_visible(False)
    fig.suptitle(
        f"Scores + global t*  |  {src}  t*={t_star:.4g}  (alpha={ALPHA_DEFAULT}, beta={BETA_DEFAULT})",
        fontsize=11,
    )
    h = [plt.Line2D([0], [0], color="tab:blue", lw=1, label="score"),
         plt.Line2D([0], [0], color="tab:green", lw=1.2, label="t*"),
         plt.Line2D([0], [0], marker="o", color="tab:red", ls="", markersize=5, label="gold in pool")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.show()

# %% [markdown]
# ## Conclusion (fill after run)
#
# - If **loss curves** are flat near `t*`, a stable hard cutoff exists for that family.
# - If curves are **sharp**, small calibration drift could violate alpha/beta.
# - Compare **bgem3** vs **bge_gemma** `t*` and typical **kept** counts.

# %% [markdown]
# ## Discussion (limitations)
#
# 1. **Small `|G_q^(N)|`**: residual loss is coarse (multiples of `1/R`); a few queries dominate macro mean.
# 2. **One global `t`**: ignores per-query score shift; good as a family default only.
# 3. **Score ties**: `|K_q(t)|` jumps at tied scores; stability depends on tie structure.
# 4. **p90**: at each `t`, p90 is over **per-query** residual losses (not document-level).
# 5. **Alpha / beta**: policy knobs; use the sensitivity tables above.
# 6. **Scope**: gold **outside** top-N is ignored (retrieval errors not measured).
# 7. **Threshold grid**: if `MAX_UNIQUE_T` subsamples unique scores, `t*` is optimal on that grid only.
