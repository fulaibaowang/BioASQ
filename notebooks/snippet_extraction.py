# ---
# jupyter:
#   jupytext:
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

# %%
import sys
from pathlib import Path

REPO_ROOT = Path("..").resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "public" / "shared_scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retrieval_eval.common import (
    ap_at_k,
    recall_at_k,
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    run_df_to_run_map,
)
from compare_result_dirs import compute_map_at_ks, _load_run_tsv

# %matplotlib inline

DOC_FUSION_DIR = REPO_ROOT / "output" / "workflow_local_10pct_hpc_bge" / "rerank" / "post_rerank_fusion"
SNIPPET_DIRS = [
    ("windows2", REPO_ROOT / "output" / "workflow_local_10pct_hpc_bge" / "snippet_rerank_windows2"),
    ("windows3", REPO_ROOT / "output" / "workflow_local_10pct_hpc_bge" / "snippet_rerank_windows3"),
]
SNIPPET_LABELS = [label for label, _ in SNIPPET_DIRS]

TRAIN_JSON = REPO_ROOT / "example" / "training14b_10pct_sample.json"
TEST_JSONS = [
    REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B1_golden.json",
    REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B2_golden.json",
    REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B3_golden.json",
    REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched" / "13B4_golden.json",
]

SPLITS = [
    ("training14b_10pct_sample", "best_rrf_training14b_10pct_sample_top5000_rrf_pool50_k60"),
    ("13B1_golden", "best_rrf_13B1_golden_top5000_rrf_pool50_k60"),
    ("13B2_golden", "best_rrf_13B2_golden_top5000_rrf_pool50_k60"),
    ("13B3_golden", "best_rrf_13B3_golden_top5000_rrf_pool50_k60"),
    ("13B4_golden", "best_rrf_13B4_golden_top5000_rrf_pool50_k60"),
]

KS = list(range(10, 101, 10))

# RRF fusion: post_rerank_fusion + snippet_rerank (top-50 union -> weighted RRF -> top-10)
RUN_TOP = 100
OUTPUT_TOP = 10
RRF_KS = [(60)]
RRF_WEIGHTS = [(1, 0),(0.9, 0.1),(0.8, 0.2),  (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]  # (w_doc_fusion, w_snippet_rerank)

# %%
# Load gold maps per batch, with training dedup
train_qs = load_questions(TRAIN_JSON)
_, train_gold_raw = build_topics_and_gold(train_qs, query_field="body")

test_golds = {}  # batch_stem -> gold_map
test_qids_all = set()
for p in TEST_JSONS:
    qs = load_questions(p)
    _, gm = build_topics_and_gold(qs, query_field="body")
    test_golds[p.stem] = gm
    test_qids_all.update(gm.keys())
    print(f"  {p.stem}: {len(gm)} queries")

overlap = set(train_gold_raw.keys()) & test_qids_all
train_only_gold = {q: d for q, d in train_gold_raw.items() if q not in test_qids_all}

print(f"\nTraining raw: {len(train_gold_raw)}, test qids: {len(test_qids_all)}, overlap removed: {len(overlap)}")
print(f"Training deduped: {len(train_only_gold)}")

# Unified gold lookup: split_label -> gold_map
split_gold = {"training14b_10pct_sample": train_only_gold}
split_gold.update(test_golds)


# %%
# Load run maps for both dirs; filter training split to train-only qids
def load_run_maps(result_dir):
    """Return {split_label: run_map} for a result dir."""
    out = {}
    for split_label, run_stem in SPLITS:
        tsv = result_dir / "runs" / f"{run_stem}.tsv"
        if not tsv.exists():
            print(f"  SKIP (missing): {tsv}")
            continue
        df = _load_run_tsv(tsv)
        rm = run_df_to_run_map(df, qid_col="qid", docno_col="docno")
        if split_label == "training14b_10pct_sample":
            rm = {q: d for q, d in rm.items() if q not in test_qids_all}
        out[split_label] = rm
    return out

hybrid_runs = load_run_maps(DOC_FUSION_DIR)
snippet_runs = {label: load_run_maps(path) for label, path in SNIPPET_DIRS}

for split_label in hybrid_runs:
    parts = [f"hybrid={len(hybrid_runs[split_label])}"]
    for sl in SNIPPET_LABELS:
        parts.append(f"{sl}={len(snippet_runs.get(sl, {}).get(split_label, {}))}")
    print(f"  {split_label}: " + ", ".join(parts) + " queries")

# Effective query counts per split (used in labels: tables, plot titles)
split_n = {}
for split_label, _ in SPLITS:
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    qids = [q for q in gold if q in h_rm and gold[q]]
    split_n[split_label] = len(qids)

print("\nEffective query counts per split used in evals:")
for k, v in split_n.items():
    print(f"  {k}: n={v}")

# %%
# Summary metrics table: MAP@10, MRR@10 per split — hybrid, windows2, windows3
rows = []
for split_label, _ in SPLITS:
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    h_metrics, _ = evaluate_run(gold, h_rm)
    n_q = split_n.get(split_label, 0)
    row = {
        "split": f"{split_label} (n={n_q})",
        "hybrid MAP@10": h_metrics["MAP@10"],
        "hybrid MRR@10": h_metrics["MRR@10"],
    }
    for sl in SNIPPET_LABELS:
        s_rm = snippet_runs.get(sl, {}).get(split_label, {})
        s_metrics, _ = evaluate_run(gold, s_rm)
        row[f"{sl} MAP@10"] = s_metrics["MAP@10"]
        row[f"{sl} MRR@10"] = s_metrics["MRR@10"]
        row[f"d_MAP@10 ({sl})"] = s_metrics["MAP@10"] - h_metrics["MAP@10"]
        row[f"d_MRR@10 ({sl})"] = s_metrics["MRR@10"] - h_metrics["MRR@10"]
    rows.append(row)

summary_df = pd.DataFrame(rows)
delta_cols = [c for c in summary_df.columns if c.startswith("d_")]
summary_df.style.format({
    c: "{:.4f}" for c in summary_df.columns if c != "split"
}).bar(subset=delta_cols, color="#5fba7d", align="zero")

# %%
# MAP@k curves (k = 10..100 step 10), one subplot per split — hybrid, windows2, windows3
fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 4), sharey=True)
if len(SPLITS) == 1:
    axes = [axes]

markers = {"hybrid": "o", "windows2": "s", "windows3": "^"}
for ax, (split_label, _) in zip(axes, SPLITS):
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    h_map = compute_map_at_ks(gold, h_rm, KS)
    ax.plot(KS, [h_map[k] for k in KS], "o-", label="hybrid", markersize=4)
    for sl in SNIPPET_LABELS:
        s_rm = snippet_runs.get(sl, {}).get(split_label, {})
        s_map = compute_map_at_ks(gold, s_rm, KS)
        ax.plot(KS, [s_map[k] for k in KS], f"{markers.get(sl, 'x')}-", label=sl, markersize=4)
    n_q = split_n.get(split_label, 0)
    ax.set_title(f"{split_label} (n={n_q})", fontsize=10)
    ax.set_xlabel("k")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("MAP@k")
axes[-1].legend(loc="lower right", fontsize=8)
fig.suptitle("MAP@k: post_rerank_fusion vs snippet_rerank (windows2, windows3)", fontsize=12)
fig.tight_layout()
plt.show()


# %%
# Recall@k curves (k = 10..100 step 10), one subplot per split — hybrid, windows2, windows3
def mean_recall_at_ks(gold_map, run_map, ks):
    qids = [q for q in gold_map if q in run_map and gold_map[q]]
    if not qids:
        return {k: 0.0 for k in ks}
    return {
        k: float(np.mean([recall_at_k(set(gold_map[q]), run_map[q], k) for q in qids]))
        for k in ks
    }

fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 4), sharey=True)
if len(SPLITS) == 1:
    axes = [axes]

markers = {"hybrid": "o", "windows2": "s", "windows3": "^"}
for ax, (split_label, _) in zip(axes, SPLITS):
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    h_rec = mean_recall_at_ks(gold, h_rm, KS)
    ax.plot(KS, [h_rec[k] for k in KS], "o-", label="hybrid", markersize=4)
    for sl in SNIPPET_LABELS:
        s_rm = snippet_runs.get(sl, {}).get(split_label, {})
        s_rec = mean_recall_at_ks(gold, s_rm, KS)
        ax.plot(KS, [s_rec[k] for k in KS], f"{markers.get(sl, 'x')}-", label=sl, markersize=4)
    n_q = split_n.get(split_label, 0)
    ax.set_title(f"{split_label} (n={n_q})", fontsize=10)
    ax.set_xlabel("k")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Mean Recall@k")
axes[-1].legend(loc="lower right", fontsize=8)
fig.suptitle("Recall@k: post_rerank_fusion vs snippet_rerank (windows2, windows3)", fontsize=12)
fig.tight_layout()
plt.show()

# %%
# Per-query win/loss: better/worse/tied AP@10 with snippet (windows2, windows3) vs hybrid
wl_rows = []
for split_label, _ in SPLITS:
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    for sl in SNIPPET_LABELS:
        s_rm = snippet_runs.get(sl, {}).get(split_label, {})
        qids = [q for q in gold if q in h_rm and q in s_rm and gold[q]]
        deltas = []
        for q in qids:
            g = set(gold[q])
            h_ap = ap_at_k(g, h_rm[q], k=10)
            s_ap = ap_at_k(g, s_rm[q], k=10)
            deltas.append(s_ap - h_ap)

        deltas = np.array(deltas)
        n = len(deltas)
        better = deltas > 0
        worse = deltas < 0
        tied = deltas == 0

        wl_rows.append({
            "split": split_label,
            "snippet": sl,
            "n_queries": n,
            "better": int(better.sum()),
            "worse": int(worse.sum()),
            "tied": int(tied.sum()),
            "better %": f"{100 * better.mean():.1f}%",
            "worse %": f"{100 * worse.mean():.1f}%",
            "tied %": f"{100 * tied.mean():.1f}%",
            "mean d(better)": f"+{deltas[better].mean():.4f}" if better.any() else "-",
            "mean d(worse)": f"{deltas[worse].mean():.4f}" if worse.any() else "-",
            "mean d(all)": f"{deltas.mean():+.4f}",
        })

pd.DataFrame(wl_rows)


# %%
# RRF fusion: union of top RUN_TOP from hybrid + snippet; weighted RRF; return top OUTPUT_TOP
def rrf_fuse_top10(hybrid_docs: list, snippet_docs: list, k: int, w_hybrid: float, w_snippet: float) -> list:
    hybrid_top = hybrid_docs[:RUN_TOP]
    snippet_top = snippet_docs[:RUN_TOP]
    rank_hybrid = {str(d): i + 1 for i, d in enumerate(hybrid_top)}
    rank_snippet = {str(d): i + 1 for i, d in enumerate(snippet_top)}
    union = list(dict.fromkeys(hybrid_top + snippet_top))
    scores = []
    for d in union:
        ds = str(d)
        s = 0.0
        if ds in rank_hybrid:
            s += w_hybrid / (k + rank_hybrid[ds])
        if ds in rank_snippet:
            s += w_snippet / (k + rank_snippet[ds])
        scores.append((ds, s))
    # Primary: RRF score desc. Secondary: docno asc so tie-break is neutral (not hybrid-first from union order)
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scores[:OUTPUT_TOP]]

rrf_rows = []
for split_label, _ in SPLITS:
    gold = split_gold.get(split_label, {})
    h_rm = hybrid_runs.get(split_label, {})
    if not gold:
        continue
    for sl in SNIPPET_LABELS:
        s_rm = snippet_runs.get(sl, {}).get(split_label, {})
        qids = [q for q in gold if q in h_rm and q in s_rm and gold[q]]
        if not qids:
            continue
        for k in RRF_KS:
            for w_hybrid, w_snippet in RRF_WEIGHTS:
                ap10_list = [
                    ap_at_k(set(gold[q]), rrf_fuse_top10(h_rm[q], s_rm[q], k, w_hybrid, w_snippet), k=OUTPUT_TOP)
                    for q in qids
                ]
                rrf_rows.append({
                    "split": split_label,
                    "snippet_label": sl,
                    "k": k,
                    "w_hybrid": w_hybrid,
                    "w_snippet": w_snippet,
                    "MAP@10": float(np.mean(ap10_list)),
                    "n_queries": len(qids),
                })

# Sweep once with all four test batches merged, for each snippet source
TEST_SPLITS = ["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"]
gold_merged = {}
h_merged = {}
for spl in TEST_SPLITS:
    gold_merged.update(split_gold.get(spl, {}))
    h_merged.update(hybrid_runs.get(spl, {}))
for sl in SNIPPET_LABELS:
    s_merged = {}
    for spl in TEST_SPLITS:
        s_merged.update(snippet_runs.get(sl, {}).get(spl, {}))
    qids_merged = [q for q in gold_merged if q in h_merged and q in s_merged and gold_merged[q]]
    for k in RRF_KS:
        for w_hybrid, w_snippet in RRF_WEIGHTS:
            ap10_list = [
                ap_at_k(set(gold_merged[q]), rrf_fuse_top10(h_merged[q], s_merged[q], k, w_hybrid, w_snippet), k=OUTPUT_TOP)
                for q in qids_merged
            ]
            rrf_rows.append({
                "split": "test_merged",
                "snippet_label": sl,
                "k": k,
                "w_hybrid": w_hybrid,
                "w_snippet": w_snippet,
                "MAP@10": float(np.mean(ap10_list)),
                "n_queries": len(qids_merged),
            })

rrf_results = pd.DataFrame(rrf_rows)
rrf_results["weight_label"] = rrf_results.apply(
    lambda r: f"({r['w_hybrid']:.3f},{r['w_snippet']:.3f})", axis=1
)
weight_labels = [f"({w[0]:.3f},{w[1]:.3f})" for w in RRF_WEIGHTS]
assert len(weight_labels) == len(set(weight_labels)), "Duplicate weight labels would break pivot/plot"
print("RRF fusion: MAP@10 per (split, snippet_label, k, weights). test_merged = all four test batches (13B1–13B4) combined.")
display(rrf_results)

# Pivot per split: rows = (k, snippet_label), columns = weight config (windows2 and windows3 together)
for split_label in rrf_results["split"].unique():
    grp = rrf_results[rrf_results["split"] == split_label]
    pivot = grp.pivot_table(index=["k", "snippet_label"], columns="weight_label", values="MAP@10", aggfunc="first")
    pivot = pivot.reindex(weight_labels, axis=1)
    # n per snippet within this split (for display)
    n_by_snip = grp.groupby("snippet_label")["n_queries"].max().to_dict()
    n_info = ", ".join(f"{sl}: n={n_by_snip.get(sl, 0)}" for sl in SNIPPET_LABELS if sl in n_by_snip)
    print(f"\n{split_label} [{n_info}]")
    display(pivot)

# %%
# Line plot: MAP@10 vs weight config — one line per (k, snippet_label), one panel per split
weight_order = [f"({w[0]:.3f},{w[1]:.3f})" for w in RRF_WEIGHTS]
n_splits = rrf_results.groupby("split").ngroups
n_cols = min(3, n_splits)
n_rows = (n_splits + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
if n_splits == 1:
    axes = np.array([axes])
axes = axes.flat
for ax, (split_label, grp) in zip(axes, rrf_results.groupby("split", sort=False)):
    for k in RRF_KS:
        for sl in SNIPPET_LABELS:
            sub_grp = grp[(grp["k"] == k) & (grp["snippet_label"] == sl)]
            sub = sub_grp.set_index("weight_label").reindex(weight_order)
            vals = sub["MAP@10"].values
            ax.plot(range(len(weight_order)), vals, marker="o", label=f"k={k} {sl}", markersize=6)
    ax.set_xticks(range(len(weight_order)))
    ax.set_xticklabels(weight_order, rotation=45, ha="right")
    # n per snippet within this split for title
    n_by_snip = grp.groupby("snippet_label")["n_queries"].max().to_dict()
    n_info = ", ".join(f"{sl}: n={n_by_snip.get(sl, 0)}" for sl in SNIPPET_LABELS if sl in n_by_snip)
    ax.set_ylabel("MAP@10")
    ax.set_xlabel("(w_hybrid, w_snippet)")
    ax.set_title(f"{split_label} ({n_info})")
    ax.legend()
    ax.grid(True, alpha=0.3)
for j in range(n_splits, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("RRF fusion: MAP@10 vs weight (post_rerank_fusion vs snippet windows2/windows3), by k", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Heatmap per (split, snippet_label): rows = k, columns = (w_hybrid, w_snippet), color = MAP@10
weight_order = [f"({w[0]:.3f},{w[1]:.3f})" for w in RRF_WEIGHTS]
for (split_label, sl), grp in rrf_results.groupby(["split", "snippet_label"], sort=False):
    pivot = grp.pivot_table(index="k", columns="weight_label", values="MAP@10", aggfunc="first")
    pivot = pivot.reindex(index=RRF_KS, columns=weight_order)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(weight_order)))
    ax.set_xticklabels(weight_order, rotation=45, ha="right")
    ax.set_yticks(range(len(RRF_KS)))
    ax.set_yticklabels(RRF_KS)
    ax.set_xlabel("(w_hybrid, w_snippet)")
    ax.set_ylabel("k")
    n_q = grp["n_queries"].iloc[0] if not grp.empty else 0
    ax.set_title(f"MAP@10 heatmap — {split_label} ({sl}, n={n_q})")
    plt.colorbar(im, ax=ax, label="MAP@10")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# window3, 0.8,0.2

# %%
