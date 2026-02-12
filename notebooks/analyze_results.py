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
# # BioASQ Results Analysis
# Static analysis and plots for BM25, Dense, Hybrid, and rerankers (MiniLM, BGE v2).

# %% [markdown]
# ## 1. Imports and Setup

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True

# %% [markdown]
# ## 2. Set Input Paths

# %%
base_dir = Path.cwd().resolve()
if not (base_dir / "output").exists() and (base_dir.parent / "output").exists():
    # Notebook run from notebooks/ or a subdir
    base_dir = base_dir.parent

print("Base dir:", base_dir)

# %%
results = {
    "BM25+RM3": base_dir / "output/eval_bm25_rm3",
    "Dense (MedEmbed)": base_dir / "output/eval_dense_medembed_small",
    "Hybrid (RRF)": base_dir / "output/eval_hybird_production_test",
    "Reranker MiniLM": base_dir / "output/eval_stage2_rerank",
    "BGE v2 (len=200)": base_dir / "output/eval_stage2_rerank_bge_reranker_v2_m3_len200",
    "BGE v2 (len=512)": base_dir / "output/eval_stage2_rerank_bge_reranker_v2_m3_len512",
}

output_dir = base_dir / "output" / "analysis_output"
figures_dir = output_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(exist_ok=True)

print("Outputs:", output_dir)
print("Methods:", list(results.keys()))


# %% [markdown]
# ## 3. Load Configuration and Run Metrics

# %%
def load_metrics_from_dir(result_dir: Path) -> pd.DataFrame | None:
    result_dir = Path(result_dir)
    metrics_file = result_dir / "rerank_metrics.csv"
    if not metrics_file.exists():
        metrics_file = result_dir / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if "split" not in df.columns and "batch" in df.columns:
            df["split"] = df["batch"]
        return df

    # Hybrid outputs recall-focused tables in results_all.csv
    results_all = result_dir / "results_all.csv"
    if results_all.exists():
        df = pd.read_csv(results_all)
        best_cfg = result_dir / "best_config.json"
        if best_cfg.exists():
            with best_cfg.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            k_rrf = cfg.get("k_rrf")
            w_bm25 = cfg.get("w_bm25")
            w_dense = cfg.get("w_dense")
            if k_rrf is not None and w_bm25 is not None and w_dense is not None:
                df = df[
                    np.isclose(df["k_rrf"], float(k_rrf))
                    & np.isclose(df["w_bm25"], float(w_bm25))
                    & np.isclose(df["w_dense"], float(w_dense))
                ]
        if "split" not in df.columns and "batch" in df.columns:
            df["split"] = df["batch"]
        return df

    print(f"Missing metrics in {result_dir}")
    return None

all_metrics = {}
for name, result_dir in results.items():
    df = load_metrics_from_dir(result_dir)
    if df is not None and not df.empty:
        all_metrics[name] = df
        print(f"Loaded {name}: {result_dir}")

print("Loaded sets:", len(all_metrics))

# %% [markdown]
# ## 4. Filter and Validate BGE Configs (len=200, len=512)

# %%
for bge_len in [200, 512]:
    config_path = base_dir / f"output/eval_stage2_rerank_bge_reranker_v2_m3_len{bge_len}/config.json"
    if not config_path.exists():
        print(f"Missing config for BGE len={bge_len}")
        continue
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    model = config.get("model")
    model_max_length = config.get("model_max_length")
    model_batch = config.get("model_batch")
    print(f"BGE len={bge_len}: model={model}, max_length={model_max_length}, batch={model_batch}")
    if model_max_length != bge_len:
        print(f"Warning: expected model_max_length={bge_len}, found {model_max_length}")

# %% [markdown]
# ## 5. Consolidate Metrics

# %%
rows = []

for method_name, df in all_metrics.items():
    df_copy = df.copy()
    df_copy["method"] = method_name
    rows.append(df_copy)

df_all = pd.concat(rows, ignore_index=True)

if "split" not in df_all.columns:
    raise ValueError("Missing 'split' column after loading metrics.")

def normalize_split(split_name: str) -> str:
    if pd.isna(split_name):
        return split_name
    split_str = str(split_name)
    if split_str.startswith("best_rrf_"):
        split_str = split_str.replace("best_rrf_", "")
        split_str = split_str.replace("_top2000", "")
    return split_str

df_all["split_normalized"] = df_all["split"].apply(normalize_split)

metric_cols = [c for c in df_all.columns if c.startswith("MeanR@")] + ["MAP@10", "GMAP@10", "MRR@10"]
print("Metrics columns:", metric_cols)
print("Splits:", df_all["split_normalized"].unique().tolist())

# %% [markdown]
# ## 6. Summary Table - Test Average (13B1-4)

# %%
test_splits = ["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"]
df_test = df_all[df_all["split_normalized"].isin(test_splits)].copy()

summary_rows = []
for method in df_all["method"].unique():
    method_data = df_test[df_test["method"] == method]
    if method_data.empty:
        continue
    row = {"method": method}
    for col in ["MAP@10", "GMAP@10", "MRR@10", "MeanR@50", "MeanR@100", "MeanR@500", "MeanR@2000"]:
        if col in method_data.columns:
            row[col] = method_data[col].mean()
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows).sort_values("MAP@10", ascending=False)
print(df_summary.round(3).to_string(index=False))

summary_path = output_dir / "summary_test_avg.csv"
df_summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)

# %% [markdown]
# ## 7. Stage 1 Recall Curves - Train and Test Avg
# Plot train and test average on the same axes. Use only K values that exist for all three stage-1 methods.

# %%
df_train = df_all[df_all["split_normalized"] == "train_subset"].copy()

test_splits = ["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"]
df_test = df_all[df_all["split_normalized"].isin(test_splits)].copy()

stage1_methods = ["BM25+RM3", "Dense (MedEmbed)", "Hybrid (RRF)"]

# Use a fixed K list shared across methods.
fixed_k = [100, 200, 500, 2000, 5000]

# Keep only K values that are present and non-NaN for every method in train and test.
valid_k = []
for k in fixed_k:
    col = f"MeanR@{k}"
    ok = True
    for method in stage1_methods:
        method_train = df_train[df_train["method"] == method]
        method_test = df_test[df_test["method"] == method]
        if method_train.empty or method_test.empty:
            ok = False
            break
        train_val = method_train.iloc[0].get(col)
        test_val = method_test[col].mean() if col in method_test.columns else np.nan
        if pd.isna(train_val) or pd.isna(test_val):
            ok = False
            break
    if ok:
        valid_k.append(k)

k_list = valid_k
if not k_list:
    raise ValueError("No overlapping MeanR@K columns found for stage-1 methods.")

print("Stage 1 K values (fixed + available):", k_list)

colors = {
    "BM25+RM3": "#1f77b4",
    "Dense (MedEmbed)": "#ff7f0e",
    "Hybrid (RRF)": "#2ca02c",
}

fig, ax = plt.subplots()
for method in stage1_methods:
    train_row = df_train[df_train["method"] == method]
    test_rows = df_test[df_test["method"] == method]
    if train_row.empty or test_rows.empty:
        continue
    train_row = train_row.iloc[0]
    train_vals = [train_row.get(f"MeanR@{k}", np.nan) for k in k_list]
    test_vals = [test_rows[f"MeanR@{k}"].mean() for k in k_list]

    color = colors.get(method)
    ax.plot(k_list, train_vals, marker="o", label=f"{method} (train)", color=color)
    ax.plot(k_list, test_vals, marker="o", linestyle="--", label=f"{method} (test avg)", color=color)

ax.set_xlabel("K (Recall Cutoff)")
ax.set_ylabel("Mean Recall")
ax.set_title("Stage 1 Recall")
ax.set_xscale("log")
ax.legend(fontsize=9, loc="lower right")

fig_path = figures_dir / "01_stage1_recall_train_test.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 8. Hybrid vs Reranker Recall
# Compare reranker recall gains at small K (50/100/200) and large K (2000) against the Hybrid baseline.

# %%
df_test = df_all[df_all["split_normalized"].isin(["13B1_golden", "13B2_golden", "13B3_golden", "13B4_golden"])].copy()
df_train = df_all[df_all["split_normalized"] == "train_subset"].copy()

test_avg = df_test.groupby("method").mean(numeric_only=True)
train_avg = df_train.groupby("method").mean(numeric_only=True)

baseline_method = "Hybrid (RRF)"
reranker_methods = [
    m for m in test_avg.index
    if m.startswith("Reranker") or m.startswith("BGE v2")
]

colors = {
    baseline_method: "#444444",
    "Reranker MiniLM": "#1f77b4",
    "BGE v2 (len=200)": "#ff7f0e",
    "BGE v2 (len=512)": "#2ca02c",
    "BGE v2 (len=1024)": "#d62728",
}

k_candidates = [50, 100, 200, 2000]


def _build_compare_methods(avg_df):
    compare = []
    if baseline_method in avg_df.index:
        compare.append(baseline_method)
    compare.extend([m for m in reranker_methods if m in avg_df.index])
    return compare


def _build_k_list(avg_df, compare):
    k_values = []
    for k in k_candidates:
        col = f"MeanR@{k}"
        if col not in avg_df.columns:
            continue
        if avg_df.loc[compare, col].isna().all():
            continue
        k_values.append(k)
    return k_values


def _plot_recall_map(avg_df, compare, k_list, title_prefix, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax_recall = axes[0]
    for method in compare:
        values = [avg_df.loc[method, f"MeanR@{k}"] for k in k_list]
        ax_recall.plot(
            k_list,
            values,
            marker="o",
            label=f"{method} (avg)",
            color=colors.get(method),
        )

    ax_recall.set_xlabel("K (Recall Cutoff)")
    ax_recall.set_ylabel("Mean Recall")
    ax_recall.set_title(f"{title_prefix} Recall")
    ax_recall.set_xscale("log")
    ax_recall.legend(fontsize=9, loc="lower right")

    ax_map = axes[1]
    map_values = [avg_df.loc[method, "MAP@10"] for method in compare]
    bar_colors = [colors.get(method) for method in compare]
    ax_map.bar(compare, map_values, color=bar_colors)
    ax_map.set_ylabel("MAP@10")
    ax_map.set_title(f"{title_prefix} MAP@10")
    ax_map.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print("Saved:", fig_path)
    plt.show()


compare_test = _build_compare_methods(test_avg)
k_list_test = _build_k_list(test_avg, compare_test)
if not k_list_test:
    raise ValueError("No overlapping MeanR@K columns found for hybrid/reranker recall plot (test).")

print("Hybrid/Reranker K values (test):", k_list_test)
_plot_recall_map(
    test_avg,
    compare_test,
    k_list_test,
    "Hybrid vs Reranker (Test Avg)",
    figures_dir / "02_hybrid_reranker_recall_map10_test.png",
)

compare_train = _build_compare_methods(train_avg)
k_list_train = _build_k_list(train_avg, compare_train)
if not k_list_train:
    raise ValueError("No overlapping MeanR@K columns found for hybrid/reranker recall plot (train).")

print("Hybrid/Reranker K values (train):", k_list_train)
_plot_recall_map(
    train_avg,
    compare_train,
    k_list_train,
    "Hybrid vs Reranker (Train)",
    figures_dir / "02_hybrid_reranker_recall_map10_train.png",
)

# Delta vs hybrid baseline to emphasize small-K gains (test only).
reranker_recall_rows = []
if baseline_method in test_avg.index:
    baseline = test_avg.loc[baseline_method]
    for method in reranker_methods:
        if method not in test_avg.index:
            continue
        row = {"method": method}
        for k in k_list_test:
            col = f"MeanR@{k}"
            base_val = baseline.get(col)
            method_val = test_avg.loc[method, col]
            if pd.isna(base_val) or pd.isna(method_val):
                row[f"delta@{k}"] = np.nan
            else:
                row[f"delta@{k}"] = method_val - base_val
        reranker_recall_rows.append(row)

reranker_recall_df = pd.DataFrame(reranker_recall_rows)
if not reranker_recall_df.empty:
    display(reranker_recall_df)

# %%
