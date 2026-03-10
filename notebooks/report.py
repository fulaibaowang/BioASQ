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

# %% [markdown]
# # Workflow Baseline Full Run – Both Routes
# Plots built from `output/workflow_baseline_full_run_both_routes/{bm25,dense,hybrid}`.

# %% [markdown]
# https://github.com/fulaibaowang/BioASQ/issues/4

# %% [markdown]
# ## 1. Imports and Setup

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# %% [markdown]
# ## 2. Load Data

# %%
base_dir = Path.cwd().resolve()
if not (base_dir / "output").exists() and (base_dir.parent / "output").exists():
    base_dir = base_dir.parent

print("Base dir:", base_dir)

workflow_dir = base_dir / "output" / "workflow_baseline_full_run_both_routes"

bm25_metrics = pd.read_csv(workflow_dir / "bm25" / "metrics.csv")
dense_metrics = pd.read_csv(workflow_dir / "dense" / "metrics.csv")
hybrid_metrics = pd.read_csv(workflow_dir / "hybrid" / "metrics.csv")

# Normalise split column name
bm25_metrics = bm25_metrics.rename(columns={"batch": "split"})
dense_metrics = dense_metrics.rename(columns={"batch": "split"})

bm25_metrics["method"] = "BM25"
dense_metrics["method"] = "Dense"
hybrid_metrics["method"] = "BM25 Dense Fusion"

output_dir = workflow_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loaded BM25:", bm25_metrics.shape)
print("Loaded Dense:", dense_metrics.shape)
print("Loaded Hybrid:", hybrid_metrics.shape)

# %% [markdown]
# ## 3. Stage 1 Recall Curves – Per Split (3×2 panels)

# %%
splits = [
    "training14b_10pct_sample",
    "13B1_golden",
    "13B2_golden",
    "13B3_golden",
    "13B4_golden",
]
split_labels = {
    "training14b_10pct_sample": "dev",
    "13B1_golden": "13B1",
    "13B2_golden": "13B2",
    "13B3_golden": "13B3",
    "13B4_golden": "13B4",
}

recall_cols_bm25 = sorted(
    [c for c in bm25_metrics.columns if c.startswith("MeanR@")],
    key=lambda c: int(c.split("@")[1]),
)
recall_cols_hybrid = sorted(
    [c for c in hybrid_metrics.columns if c.startswith("MeanR@")],
    key=lambda c: int(c.split("@")[1]),
)
recall_cols = sorted(
    set(recall_cols_bm25) & set(recall_cols_hybrid),
    key=lambda c: int(c.split("@")[1]),
)
k_values = [int(c.split("@")[1]) for c in recall_cols]

print("Shared K values:", k_values)

# Only show a subset of K ticks to avoid clutter.
tick_candidates = [50, 200, 500, 1000, 2000, 5000]
tick_values = [k for k in tick_candidates if k in k_values]

methods_cfg = {
    "BM25": {"df": bm25_metrics, "color": "#1f77b4", "marker": "o"},
    "Dense": {"df": dense_metrics, "color": "#ff7f0e", "marker": "s"},
    "BM25 Dense Fusion": {"df": hybrid_metrics, "color": "#2ca02c", "marker": "D"},
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
axes_flat = axes.flat

global_ymin, global_ymax = 1.0, 0.0
for split in splits:
    for cfg in methods_cfg.values():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols]
        global_ymin = min(global_ymin, min(vals))
        global_ymax = max(global_ymax, max(vals))

y_pad = (global_ymax - global_ymin) * 0.05
global_ymin = max(0, global_ymin - y_pad)
global_ymax = min(1, global_ymax + y_pad)

for idx, split in enumerate(splits):
    ax = axes_flat[idx]
    for method_name, cfg in methods_cfg.items():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols]
        ax.plot(
            k_values,
            vals,
            marker=cfg["marker"],
            color=cfg["color"],
            label=method_name,
            markersize=6,
            linewidth=1.8,
        )

    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(global_ymin, global_ymax)

    # Y-axis label only for left column (two left plots)
    if idx % 3 == 0:
        ax.set_ylabel("Mean Recall")
    else:
        ax.set_ylabel("")

    # X-axis label (and tick labels) only for bottom row (two bottom plots actually used)
    if idx >= 3:
        ax.set_xlabel("K")
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(k) for k in tick_values], rotation=90)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    # Grid: horizontal lines on all panels, vertical lines at selected K ticks
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

last_ax = axes_flat[len(splits)]
last_ax.axis("off")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.92, 0.12), fontsize=18)

fig.suptitle("Retrieval Mean_Recall@K Curves", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "01_stage1_recall_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 4. Hybrid vs Rerank vs Post-rerank Fusion – Recall Curves (K ≤ 300)

# %%
hybrid_stage1 = hybrid_metrics.copy()
rerank_metrics = pd.read_csv(workflow_dir / "rerank" / "metrics.csv")
rerank_fusion_metrics = pd.read_csv(workflow_dir / "rerank_hybrid_200" / "metrics.csv")

# Rerank uses 'label' for split name
rerank_metrics = rerank_metrics.rename(columns={"label": "split"})

methods_stage2 = {
    "Retrieval": {
        "df": hybrid_stage1,
        "color": "#2ca02c",
        "marker": "D",
    },
    "Rerank": {
        "df": rerank_metrics,
        "color": "#1f77b4",
        "marker": "o",
    },
}

# Shared MeanR@K columns with K in [10, 300]
recall_cols_all = []
for df in [hybrid_stage1, rerank_metrics]:
    cols = [c for c in df.columns if c.startswith("MeanR@")]
    recall_cols_all.append(set(cols))

recall_cols_common = sorted(
    set.intersection(*recall_cols_all),
    key=lambda c: int(c.split("@")[1]),
)
k_vals_recall = [int(c.split("@")[1]) for c in recall_cols_common if 10 <= int(c.split("@")[1]) <= 300]
recall_cols_common = [f"MeanR@{k}" for k in k_vals_recall]

print("Stage2+ K values (≤300):", k_vals_recall)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
axes_flat = axes.flat

global_ymin, global_ymax = 1.0, 0.0
for split in splits:
    for cfg in methods_stage2.values():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols_common]
        global_ymin = min(global_ymin, min(vals))
        global_ymax = max(global_ymax, max(vals))

y_pad = (global_ymax - global_ymin) * 0.05
global_ymin = max(0, global_ymin - y_pad)
global_ymax = min(1, global_ymax + y_pad)

tick_candidates_recall = [10, 50, 100, 200, 300,]
tick_values_recall = [k for k in tick_candidates_recall if k in k_vals_recall]

for idx, split in enumerate(splits):
    ax = axes_flat[idx]
    for method_name, cfg in methods_stage2.items():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols_common]
        ax.plot(
            k_vals_recall,
            vals,
            marker=cfg["marker"],
            color=cfg["color"],
            label=method_name,
            markersize=6,
            linewidth=1.8,
        )

    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(global_ymin, global_ymax)

    if idx % 3 == 0:
        ax.set_ylabel("Mean Recall")
    else:
        ax.set_ylabel("")

    if idx >= 3:
        ax.set_xlabel("K")
        ax.set_xticks(tick_values_recall)
        ax.set_xticklabels([str(k) for k in tick_values_recall], rotation=90)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

last_ax = axes_flat[len(splits)]
last_ax.axis("off")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.88, 0.12), fontsize=18)

fig.suptitle("Retrieval vs Rerank Mean_Recall@K (K ≤ 300)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "02_hybrid_rerank_fusion_recall_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 5. Hybrid vs Rerank vs Post-rerank Fusion – MAP@10 (Bar Plots)

# %%
fig, ax = plt.subplots(1, 5, figsize=(10, 4), sharey=True)
# fig.suptitle("Retrival vs Rerank vs Post-rerank Fusion – MAP@10", fontsize=16, fontweight="bold", y=1.08)
method_colors = {
    "Retrieval": "#2ca02c",
    "Rerank": "#1f77b4",
    "Post-rerank fusion": "#ff7f0e",
}

for idx, split in enumerate(splits):
    cur_ax = ax[idx]
    vals = []
    labels_methods = []

    # Hybrid retrieval
    row_h = hybrid_stage1[hybrid_stage1["split"] == split]
    if not row_h.empty:
        vals.append(float(row_h.iloc[0]["MAP@10"]))
        labels_methods.append("Retrieval")

    # Rerank
    row_r = rerank_metrics[rerank_metrics["split"] == split]
    if not row_r.empty:
        vals.append(float(row_r.iloc[0]["MAP@10"]))
        labels_methods.append("Rerank")

    # Post-rerank fusion
    row_f = rerank_fusion_metrics[rerank_fusion_metrics["split"] == split]
    if not row_f.empty:
        vals.append(float(row_f.iloc[0]["MAP@10"]))
        labels_methods.append("Post-rerank fusion")

    x = np.arange(len(labels_methods))
    colors = [method_colors[label] for label in labels_methods]
    cur_ax.bar(x, vals, color=colors)
    cur_ax.set_xticks(x)
    cur_ax.set_xticklabels("", rotation=30, ha="right")
    cur_ax.set_title(split_labels.get(split, split), fontsize=13, fontweight="bold")

# Legend for methods instead of y-axis label
handles = [
    plt.matplotlib.patches.Patch(color=method_colors["Retrieval"], label="Retrieval"),
    plt.matplotlib.patches.Patch(color=method_colors["Rerank"], label="Rerank"),
    plt.matplotlib.patches.Patch(color=method_colors["Post-rerank fusion"], label="Post-rerank fusion"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.9])
fig_path = output_dir / "03_hybrid_rerank_fusion_map10_bars_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 6. Hybrid vs Rerank vs Post-rerank Fusion – MAP@K Curves (K = 1, 3, 5, 10, 20, 30, 40, 50)

# %%
from collections import defaultdict

map_ks = [1, 3, 5, 10, 20, 30, 40, 50, 75, 100]


def _extract_pmid(doc_entry):
    if isinstance(doc_entry, dict):
        doc_entry = doc_entry.get("document", "")
    if not isinstance(doc_entry, str):
        return None
    if "/" in doc_entry:
        return doc_entry.rsplit("/", 1)[-1]
    return doc_entry


def _load_qrels(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qrels: dict[str, set[str]] = {}
    for q in data.get("questions", []):
        qid = str(q.get("id"))
        docs = q.get("documents", [])
        pmids = {
            _extract_pmid(d)
            for d in docs
            if _extract_pmid(d)
        }
        if qid and pmids:
            qrels[qid] = pmids
    return qrels


def _load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid")
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc")
    rank_col = cols.get("rank")
    if qid_col is None or doc_col is None:
        raise ValueError(f"Missing qid/doc columns in {path}")
    df[qid_col] = df[qid_col].astype(str)
    df[doc_col] = df[doc_col].astype(str)
    if rank_col:
        df = df.sort_values([qid_col, rank_col])
    return df[[qid_col, doc_col]]


def _ap_at_k(docs: list[str], rels: set[str], k: int) -> float:
    """BioASQ-style AP@k, matching retrieval_eval.common.ap_at_k."""
    if not rels:
        return 0.0
    denom = min(len(rels), k)
    if denom == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc in enumerate(docs[:k], start=1):
        if doc in rels:
            hits += 1
            score += hits / i
    return score / denom


def _map_at_ks_for_run(run_df: pd.DataFrame, qrels: dict[str, set[str]], ks: list[int]) -> dict[int, float]:
    qid_col, doc_col = run_df.columns.tolist()
    per_q: dict[int, list[float]] = {k: [] for k in ks}
    for qid, group in run_df.groupby(qid_col, sort=False):
        rels = qrels.get(str(qid))
        if not rels:
            continue
        docs = group[doc_col].tolist()
        for k in ks:
            per_q[k].append(_ap_at_k(docs, rels, k))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in per_q.items()}


qrels_paths = {
    "training14b_10pct_sample": base_dir / "example" / "training14b_10pct_sample.json",
    "13B1_golden": base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B1_golden.json",
    "13B2_golden": base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B2_golden.json",
    "13B3_golden": base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B3_golden.json",
    "13B4_golden": base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B4_golden.json",
}

qrels_by_split = {s: _load_qrels(p) for s, p in qrels_paths.items()}

run_dirs = {
    "Hybrid (retrieval)": workflow_dir / "hybrid" / "runs",
    "Rerank": workflow_dir / "rerank" / "runs",
    "Post-rerank fusion": workflow_dir / "rerank_hybrid_200" / "runs",
}

map_curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

for split in splits:
    qrels_split = qrels_by_split.get(split, {})
    for method_name, runs_dir in run_dirs.items():
        # Hybrid and rerank use simple pattern; fusion has extra suffix but same prefix.
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)
        map_vals = _map_at_ks_for_run(run_df, qrels_split, map_ks)
        map_curves[method_name][split] = map_vals

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
axes_flat = axes.flat

all_maps = []
for method_dict in map_curves.values():
    for split_vals in method_dict.values():
        all_maps.extend(split_vals.values())

if all_maps:
    y_min = max(0.0, min(all_maps) - 0.02)
    y_max = min(1.0, max(all_maps) + 0.02)
else:
    y_min, y_max = 0.0, 1.0

colors_map = {
    "Hybrid (retrieval)": "#2ca02c",
    "Rerank": "#1f77b4",
    "Post-rerank fusion": "#ff7f0e",
}

for idx, split in enumerate(splits):
    ax = axes_flat[idx]
    for method_name, method_dict in map_curves.items():
        if split not in method_dict:
            continue
        vals = [method_dict[split].get(k, 0.0) for k in map_ks]
        ax.plot(
            map_ks,
            vals,
            marker="o",
            color=colors_map.get(method_name),
            label=method_name,
            linewidth=1.8,
        )
    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    if idx % 3 == 0:
        ax.set_ylabel("MAP@K")
    else:
        ax.set_ylabel("")

    if idx >= 3:
        ax.set_xlabel("K")
        ax.set_xticks(map_ks)
        ax.set_xticklabels([str(k) for k in map_ks], rotation=90)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    ax.set_xlim(0, 100)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

last_ax = axes_flat[len(splits)]
last_ax.axis("off")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.88, 0.12), fontsize=13)

fig.suptitle("Retrieval vs Rerank vs Post Rerank Fusion – MAP@K)", fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "04_hybrid_rerank_fusion_mapk_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 7. Gold Count Histogram per Query (Dev + Test Combined)

# %%
dev_split = "training14b_10pct_sample"
test_splits = ["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"]

gold_counts_all = [len(v) for v in qrels_by_split[dev_split].values()]
for s in test_splits:
    gold_counts_all.extend(len(v) for v in qrels_by_split[s].values())

max_gold = max(gold_counts_all) if gold_counts_all else 0
bins = range(1, max_gold + 2) if max_gold > 0 else [0, 1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(gold_counts_all, bins=bins, color="#4c72b0", alpha=0.8, edgecolor="white")
ax.set_xlabel("|gold| (relevant docs)")
ax.set_ylabel("# queries")
ax.set_title(f"Dev + Test (n={len(gold_counts_all)})", fontweight="bold")

fig.suptitle("Gold Document Count per Query (All Splits)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = output_dir / "05_gold_count_hist_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 8. MAP@K Stratified by Gold Count (All Splits; |gold| = 1, 2, 3–5, >5)

# %%
bucket_order = ["1", "2", "3–5", ">5"]


def _gold_bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if 3 <= n <= 5:
        return "3–5"
    return ">5"


records = []
for method_name, runs_dir in run_dirs.items():
    for split in [dev_split] + test_splits:
        qrels_split = qrels_by_split[split]
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)

        qid_col, doc_col = run_df.columns.tolist()
        for qid, group in run_df.groupby(qid_col, sort=False):
            rels = qrels_split.get(str(qid))
            if not rels:
                continue
            gold_n = len(rels)
            bucket = _gold_bucket(gold_n)
            docs = group[doc_col].tolist()
            for k in map_ks:
                ap = _ap_at_k(docs, set(rels), k)
                records.append(
                    {
                        "split": split,
                        "method": method_name,
                        "qid": str(qid),
                        "gold_n": gold_n,
                        "bucket": bucket,
                        "k": k,
                        "AP@K": ap,
                    }
                )

map_pq_df = pd.DataFrame(records)
print("Per-query AP@K rows:", len(map_pq_df))

# Compute MAP@K aggregated over all splits
summary = (
    map_pq_df.groupby(["method", "bucket", "k"], as_index=False)["AP@K"]
    .mean()
    .rename(columns={"AP@K": "MAP@K"})
)

# Compute per-bucket sample sizes (unique queries)
bucket_counts = (
    map_pq_df[["qid", "bucket"]]
    .drop_duplicates()
    .groupby("bucket")["qid"]
    .nunique()
    .to_dict()
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes_flat = axes.flat

for idx, bucket in enumerate(bucket_order):
    ax = axes_flat[idx]
    bucket_df = summary[summary["bucket"] == bucket]
    if bucket_df.empty:
        ax.set_visible(False)
        continue
    for method_name in run_dirs.keys():
        m_df = bucket_df[bucket_df["method"] == method_name]
        if m_df.empty:
            continue
        ax.plot(
            m_df["k"],
            m_df["MAP@K"],
            marker="o",
            label=method_name,
        )
    n_bucket = bucket_counts.get(bucket, 0)
    ax.set_title(f"|gold| = {bucket}, n={n_bucket}", fontweight="bold")
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx % 2 == 0:
        ax.set_ylabel("MAP@K")
    if idx >= 2:
        ax.set_xlabel("K")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(run_dirs), bbox_to_anchor=(0.5, 1.05), fontsize=16)

#fig.suptitle("MAP@K by Gold Count – Dev + Test Combined", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "06_mapk_by_gold_bucket_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 9. MAP@K Curves – Rerankers Across Configs (Train + Test Splits)
# Note: for the train/dev split we drop any queries whose IDs also appear in the
# test batches (13B1–4) to avoid training/eval overlap.

# %%
map_ks = [1, 3, 5, 10, 20, 30, 40, 50, 75, 100]

splits_rerank = [
    "training14b_10pct_sample",
    "13B1_golden",
    "13B2_golden",
    "13B3_golden",
    "13B4_golden",
]

rerank_only_run_dirs = {
    "ms-marco-MiniLM-L-12-v2, 33.4M": base_dir / "output" / "eval_stage2_rerank_miniLM" / "runs",
    "bge-reranker-v2-m3, 568M, tok_len=200": base_dir / "output" / "eval_stage2_rerank_bge_reranker_v2_m3_len200" / "runs",
    "bge-reranker-v2-m3, 568M": base_dir / "output" / "eval_stage2_rerank_bge_reranker_v2_m3_len512" / "runs",
    "bge-reranker-v2-gemma, 2.5B": base_dir / "output" / "workflow_baseline_full_run_both_routes_gemma" / "rerank" / "runs",
}

# Precompute overlapping qids between train and test splits based on rerank runs.
# We use the MiniLM runs as the canonical source of which train queries overlap
# with the 13B1–4 test batches, then exclude those qids from all train analyses.
train_split_id = "training14b_10pct_sample"
test_split_ids = ["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"]
train_overlap_qids: set[str] = set()

minilm_runs_dir = rerank_only_run_dirs["ms-marco-MiniLM-L-12-v2, 33.4M"]

# Train run: best_rrf_train_subset_top*.tsv
train_candidates = list((minilm_runs_dir).glob("best_rrf_train_subset_top*.tsv"))
if train_candidates:
    train_run_df = _load_run(train_candidates[0])
    train_qid_col, _ = train_run_df.columns.tolist()
    train_qids_run = set(train_run_df[train_qid_col].astype(str).tolist())
else:
    train_qids_run = set()

test_qids_run: set[str] = set()
for s in test_split_ids:
    test_candidates = list((minilm_runs_dir).glob(f"best_rrf_{s}_top*.tsv"))
    if not test_candidates:
        continue
    test_run_df = _load_run(test_candidates[0])
    qid_col, _ = test_run_df.columns.tolist()
    test_qids_run.update(test_run_df[qid_col].astype(str).tolist())

train_overlap_qids = train_qids_run & test_qids_run

rerank_map_curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

for split in splits_rerank:
    qrels_split = qrels_by_split.get(split, {})
    for method_name, runs_dir in rerank_only_run_dirs.items():
        # Train/dev filenames use different stems ("train_subset" vs "training14b_10pct_sample")
        # across reranker outputs, so try both patterns for that split.
        stems = [split]
        if split == "training14b_10pct_sample":
            stems = ["train_subset", split]

        run_path = None
        for stem in stems:
            pattern = f"best_rrf_{stem}_top"
            candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
            if candidates:
                run_path = candidates[0]
                break

        if run_path is None:
            continue

        run_df = _load_run(run_path)

        # For train/dev, drop any qids that overlap with test batches.
        if split == train_split_id and train_overlap_qids:
            qid_col, _ = run_df.columns.tolist()
            run_df = run_df[~run_df[qid_col].astype(str).isin(train_overlap_qids)]

        map_vals = _map_at_ks_for_run(run_df, qrels_split, map_ks)
        rerank_map_curves[method_name][split] = map_vals

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
axes_flat = list(axes.flat)

all_vals = []
for method_dict in rerank_map_curves.values():
    for split_vals in method_dict.values():
        all_vals.extend(split_vals.values())

if all_vals:
    y_min = max(0.0, min(all_vals) - 0.02)
    y_max = min(1.0, max(all_vals) + 0.02)
else:
    y_min, y_max = 0.0, 1.0

colors_rerank = {
    "ms-marco-MiniLM-L-12-v2, 33.4M": "#1f77b4",
    "bge-reranker-v2-m3, 568M, tok_len=200": "#ff7f0e",
    "bge-reranker-v2-m3, 568M": "#2ca02c",
    "bge-reranker-v2-gemma, 2.5B": "#9467bd",
}

for idx, split in enumerate(splits_rerank):
    ax = axes_flat[idx]
    for method_name, method_dict in rerank_map_curves.items():
        if split not in method_dict:
            continue
        vals = [method_dict[split].get(k, 0.0) for k in map_ks]
        ax.plot(
            map_ks,
            vals,
            marker="o",
            linewidth=1.8,
            color=colors_rerank[method_name],
            label=method_name,
        )
    ax.set_title(split_labels.get(split, split), fontsize=14, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx % 3 == 0:
        ax.set_ylabel("MAP@K")
    if idx >= 3:
        ax.set_xlabel("K")
        ax.set_xticks(map_ks)
        ax.set_xticklabels([str(k) for k in map_ks], rotation=90)

# Hide unused last panel (6th subplot)
for j in range(len(splits_rerank), len(axes_flat)):
    axes_flat[j].set_visible(False)

from matplotlib.lines import Line2D

legend_handles = [
    Line2D([0], [0], color=colors_rerank[name], marker="o", linestyle="-", label=name)
    for name in colors_rerank.keys()
]
fig.legend(
    handles=legend_handles,
    labels=list(colors_rerank.keys()),
    loc="lower right",
    bbox_to_anchor=(1.02, 0.05),
    fontsize=16,
)
fig.suptitle("MAP@K Curves – Rerankers", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "07_rerankers_mapk_test_splits.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 10. Reranker Top-1 Overlap, Win/Loss, and Jaccard@10

# %%
# Methods and runs for pairwise comparison (ignore BGE len=200)
pair_methods = {
    "MiniLM": base_dir / "output" / "eval_stage2_rerank_miniLM" / "runs",
    "BGE-m3": base_dir / "output" / "eval_stage2_rerank_bge_reranker_v2_m3_len512" / "runs",
    "Gemma 2.5B": base_dir / "output" / "workflow_baseline_full_run_both_routes_gemma" / "rerank" / "runs",
}

splits_pairs = [
    "training14b_10pct_sample",
    "13B1_golden",
    "13B2_golden",
    "13B3_golden",
    "13B4_golden",
]

# Reuse the same train/test overlap set so that train-only tables exclude
# any queries whose IDs also appear in the test batches.
pair_train_overlap_qids = train_overlap_qids

# Collect top-1 and top-10 docs per (method, split, qid)
top1_docs: dict[str, dict[tuple[str, str], str | None]] = {m: {} for m in pair_methods}
top10_docs: dict[str, dict[tuple[str, str], list[str]]] = {m: {} for m in pair_methods}

for split in splits_pairs:
    qrels_split = qrels_by_split.get(split, {})
    for method_name, runs_dir in pair_methods.items():
        # Train/dev filenames use different stems ("train_subset" vs "training14b_10pct_sample")
        stems = [split]
        if split == "training14b_10pct_sample":
            stems = ["train_subset", split]

        run_path = None
        for stem in stems:
            pattern = f"best_rrf_{stem}_top"
            candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
            if candidates:
                run_path = candidates[0]
                break

        if run_path is None:
            continue

        run_df = _load_run(run_path)
        qid_col, doc_col = run_df.columns.tolist()
        for qid, group in run_df.groupby(qid_col, sort=False):
            qid_str = str(qid)
            if split == "training14b_10pct_sample" and pair_train_overlap_qids and qid_str in pair_train_overlap_qids:
                continue
            key = (split, qid_str)
            docs = group[doc_col].astype(str).tolist()
            top1_docs[method_name][key] = docs[0] if docs else None
            top10_docs[method_name][key] = docs[:10]

# All query keys where at least one method has output
all_keys: set[tuple[str, str]] = set()
for m in pair_methods:
    all_keys.update(top1_docs[m].keys())

# Use all keys (train overlap qids were already excluded when building top1_docs/top10_docs).
analysis_keys = all_keys

method_pairs = [
    ("MiniLM", "BGE-m3"),
    ("MiniLM", "Gemma 2.5B"),
    ("BGE-m3", "Gemma 2.5B"),
]

# 1. Top-1 overlap counts
rows_overlap = []
for a, b in method_pairs:
    same = 0
    different = 0
    total = 0
    for split, qid in analysis_keys:
        a_doc = top1_docs[a].get((split, qid))
        b_doc = top1_docs[b].get((split, qid))
        if not a_doc or not b_doc:
            continue
        total += 1
        if a_doc == b_doc:
            same += 1
        else:
            different += 1
    rows_overlap.append(
        {
            "pair": f"{a} vs {b}",
            "same_top1": same,
            "different_top1": different,
            "total_queries": total,
            "same_frac": same / total if total else 0.0,
        }
    )

df_overlap = pd.DataFrame(rows_overlap)
print("Top-1 overlap (MiniLM, BGE-m3 len512, Gemma):")
print(df_overlap.round(3).to_string(index=False))

overlap_path = output_dir / "08_reranker_top1_overlap.csv"
df_overlap.to_csv(overlap_path, index=False)
print("Saved:", overlap_path)

# 2. Top-1 win/loss vs relevance
rows_winloss = []
for a, b in method_pairs:
    a_rel_b_not = 0
    b_rel_a_not = 0
    both_rel = 0
    neither_rel = 0
    total = 0
    for split, qid in analysis_keys:
        a_doc = top1_docs[a].get((split, qid))
        b_doc = top1_docs[b].get((split, qid))
        if not a_doc or not b_doc:
            continue
        rels = qrels_by_split.get(split, {}).get(qid, set())
        if not rels:
            continue
        a_is_rel = a_doc in rels
        b_is_rel = b_doc in rels
        total += 1
        if a_is_rel and not b_is_rel:
            a_rel_b_not += 1
        elif b_is_rel and not a_is_rel:
            b_rel_a_not += 1
        elif a_is_rel and b_is_rel:
            both_rel += 1
        else:
            neither_rel += 1

    rows_winloss.append(
        {
            "pair": f"{a} vs {b}",
            "A_name": a,
            "B_name": b,
            "A_win": a_rel_b_not,
            "B_win": b_rel_a_not,
            "both_rel": both_rel,
            "neither_rel": neither_rel,
            "total_queries": total,
            "A_win_frac": a_rel_b_not / total if total else float("nan"),
            "B_win_frac": b_rel_a_not / total if total else float("nan"),
            "both_rel_frac": both_rel / total if total else float("nan"),
            "neither_rel_frac": neither_rel / total if total else float("nan"),
        }
    )

df_winloss = pd.DataFrame(rows_winloss)
print("\nTop-1 win/loss vs relevance (standardized table):")
print(df_winloss.round(3).to_string(index=False))

winloss_path = output_dir / "09_reranker_top1_winloss.csv"
df_winloss.to_csv(winloss_path, index=False)
print("Saved:", winloss_path)

# 3. Jaccard overlap of top-10 sets
rows_jaccard = []
for a, b in method_pairs:
    jaccs = []
    n_pairs = 0
    for split, qid in analysis_keys:
        a_docs = top10_docs[a].get((split, qid))
        b_docs = top10_docs[b].get((split, qid))
        if not a_docs or not b_docs:
            continue
        set_a = set(a_docs)
        set_b = set(b_docs)
        union = set_a | set_b
        if not union:
            continue
        inter = set_a & set_b
        j = len(inter) / len(union)
        jaccs.append(j)
        n_pairs += 1
    mean_j = float(np.mean(jaccs)) if jaccs else 0.0
    median_j = float(np.median(jaccs)) if jaccs else 0.0
    rows_jaccard.append(
        {
            "pair": f"{a} vs {b}",
            "mean_jaccard@10": mean_j,
            "median_jaccard@10": median_j,
            "n_queries": n_pairs,
        }
    )

df_jacc = pd.DataFrame(rows_jaccard)
print("\nJaccard overlap of top-10 (MiniLM, BGE-m3 len512, Gemma):")
print(df_jacc.round(3).to_string(index=False))

jacc_path = output_dir / "10_reranker_jaccard_top10.csv"
df_jacc.to_csv(jacc_path, index=False)
print("Saved:", jacc_path)

# %% [markdown]
# ## 11. RRF fusion sweep: MiniLM + BGE-m3 (keep Gemma on plot)

# %%
from collections import defaultdict

rrf_k = 60
fusion_weights = [
    (0.0, 1.0),
    (0.33, 0.67),
    (0.5, 0.5),
    (0.67, 0.33),
    (1.0, 0.0),
]

fusion_labels = {
    (0.0, 1.0): "BGE-m3 only",
    (0.33, 0.67): "Fusion w=(0.33,0.67)",
    (0.5, 0.5): "Fusion w=(0.5,0.5)",
    (0.67, 0.33): "Fusion w=(0.67,0.33)",
    (1.0, 0.0): "MiniLM only",
}

minilm_runs_dir = base_dir / "output" / "eval_stage2_rerank_miniLM" / "runs"
bge_runs_dir = base_dir / "output" / "eval_stage2_rerank_bge_reranker_v2_m3_len512" / "runs"
gemma_runs_dir = base_dir / "output" / "workflow_baseline_full_run_both_routes_gemma" / "rerank" / "runs"

def _find_run_path(runs_dir: Path, split: str) -> Path | None:
    # Train/dev split uses both "train_subset" and "training14b_10pct_sample"
    stems = [split]
    if split == "training14b_10pct_sample":
        stems = ["train_subset", split]
    for stem in stems:
        cands = list(runs_dir.glob(f"best_rrf_{stem}_top*.tsv"))
        if cands:
            return cands[0]
    return None

def _rrf_fuse_two_lists(
    docs_a: list[str],
    docs_b: list[str],
    pool_a: int,
    pool_b: int,
    k_rrf: int,
    w_a: float,
    w_b: float,
) -> list[str]:
    # Adapted from rerank_rrf_hybrid._rrf_fuse_docs
    a_top = docs_a[:pool_a]
    b_top = docs_b[:pool_b]
    rank_a = {d: i + 1 for i, d in enumerate(a_top)}
    rank_b = {d: i + 1 for i, d in enumerate(b_top)}
    union = list(dict.fromkeys(a_top + b_top))
    scored = []
    for d in union:
        s = 0.0
        ra = rank_a.get(d)
        rb = rank_b.get(d)
        if ra is not None:
            s += w_a / (k_rrf + ra)
        if rb is not None:
            s += w_b / (k_rrf + rb)
        scored.append((d, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored]

# Build MAP@K curves for: MiniLM, BGE-m3, Gemma, and each fusion weight
fusion_curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

for split in splits_rerank:
    qrels_split = qrels_by_split.get(split, {})
    if not qrels_split:
        continue

    # Load base runs
    minilm_path = _find_run_path(minilm_runs_dir, split)
    bge_path = _find_run_path(bge_runs_dir, split)
    gemma_path = _find_run_path(gemma_runs_dir, split)

    if not minilm_path or not bge_path:
        continue  # need both for fusion

    minilm_df = _load_run(minilm_path)
    bge_df = _load_run(bge_path)

    # Optional: apply same train-overlap filtering as section 9
    if split == "training14b_10pct_sample" and train_overlap_qids:
        qid_col, _ = minilm_df.columns.tolist()
        mask = ~minilm_df[qid_col].astype(str).isin(train_overlap_qids)
        minilm_df = minilm_df[mask]
        mask = ~bge_df[qid_col].astype(str).isin(train_overlap_qids)
        bge_df = bge_df[mask]

    # Build per-qid doc lists
    qid_col, doc_col = minilm_df.columns.tolist()
    minilm_docs = {
        q: g[doc_col].astype(str).tolist()
        for q, g in minilm_df.groupby(qid_col, sort=False)
    }
    bge_docs = {
        q: g[doc_col].astype(str).tolist()
        for q, g in bge_df.groupby(qid_col, sort=False)
    }

    # 1) Base curves for MiniLM, BGE-m3, and Gemma
    for name, df_path in [
        ("MiniLM", minilm_path),
        ("BGE-m3", bge_path),
        ("Gemma 2.5B", gemma_path),
    ]:
        if not df_path:
            continue
        base_df = _load_run(df_path)
        if split == "training14b_10pct_sample" and train_overlap_qids:
            qid_col, _ = base_df.columns.tolist()
            base_df = base_df[~base_df[qid_col].astype(str).isin(train_overlap_qids)]
        fusion_curves[name][split] = _map_at_ks_for_run(base_df, qrels_split, map_ks)

    # 2) Fusion curves for each weight pair
    union_qids = sorted(set(minilm_docs.keys()) | set(bge_docs.keys()), key=str)
    for w_a, w_b in fusion_weights:
        label = fusion_labels[(w_a, w_b)]
        rows = []
        for qid in union_qids:
            docs_a = minilm_docs.get(qid, [])
            docs_b = bge_docs.get(qid, [])
            fused = _rrf_fuse_two_lists(
                docs_a,
                docs_b,
                pool_a=50,
                pool_b=50,
                k_rrf=rrf_k,
                w_a=w_a,
                w_b=w_b,
            )
            for rank, doc in enumerate(fused, start=1):
                rows.append((str(qid), doc))
        if not rows:
            continue
        fused_df = pd.DataFrame(rows, columns=["qid", doc_col])
        if split == "training14b_10pct_sample" and train_overlap_qids:
            fused_df = fused_df[~fused_df["qid"].astype(str).isin(train_overlap_qids)]
        fusion_curves[label][split] = _map_at_ks_for_run(fused_df, qrels_split, map_ks)

# Plot: 2x3 grid, one panel per split (train + 4 tests), lines for base + fusions
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
axes_flat = list(axes.flat)

series_order = [
    "MiniLM",
    "BGE-m3",
    "Gemma 2.5B",
    "BGE-m3 only",
    "Fusion w=(0.33,0.67)",
    "Fusion w=(0.5,0.5)",
    "Fusion w=(0.67,0.33)",
    "MiniLM only",
]

colors_fusion = {
    "MiniLM": "#1f77b4",
    "BGE-m3": "#ff7f0e",
    "Gemma 2.5B": "#9467bd",
    "BGE-m3 only": "#ffbb78",
    "Fusion w=(0.33,0.67)": "#2ca02c",
    "Fusion w=(0.5,0.5)": "#17becf",
    "Fusion w=(0.67,0.33)": "#8c564b",
    "MiniLM only": "#d62728",
}

# y-range
all_vals = []
for name in series_order:
    for split_vals in fusion_curves.get(name, {}).values():
        all_vals.extend(split_vals.values())
y_min = max(0.0, min(all_vals) - 0.02) if all_vals else 0.0
y_max = min(1.0, max(all_vals) + 0.02) if all_vals else 1.0

for idx, split in enumerate(splits_rerank):
    ax = axes_flat[idx]
    for name in series_order:
        split_map = fusion_curves.get(name, {}).get(split)
        if not split_map:
            continue
        ys = [split_map.get(k, 0.0) for k in map_ks]
        ax.plot(
            map_ks,
            ys,
            marker="o",
            linewidth=1.6,
            color=colors_fusion.get(name, "#999999"),
            label=name,
        )
    ax.set_title(split_labels.get(split, split), fontsize=14, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx % 3 == 0:
        ax.set_ylabel("MAP@K")
    if idx >= 3:
        ax.set_xlabel("K")
        ax.set_xticks(map_ks)
        ax.set_xticklabels([str(k) for k in map_ks], rotation=90)

# Hide unused 6th panel if any
for j in range(len(splits_rerank), len(axes_flat)):
    axes_flat[j].set_visible(False)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=colors_fusion[name], marker="o", linestyle="-", label=name)
    for name in series_order
]
fig.legend(
    handles=legend_handles,
    labels=series_order,
    loc="lower right",
    bbox_to_anchor=(1.02, 0.05),
    fontsize=10,
)

fig.suptitle("RRF Fusion Sweep: MiniLM + BGE-m3 (MAP@K, k_rrf=60)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "11_rrf_fusion_minilm_bge_m3_mapk.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %%
