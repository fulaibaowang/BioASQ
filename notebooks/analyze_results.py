# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
base_dir = Path("/Users/yun/develop/BioASQ")

results = {
    "BM25+RM3": base_dir / "output/eval_bm25_rm3",
    "Dense (MedEmbed)": base_dir / "output/eval_dense_medembed_small",
    "Hybrid (RRF)": base_dir / "output/eval_hybird",
    "Reranker MiniLM": base_dir / "output/eval_stage2_rerank",
    "BGE v2 (len=200)": base_dir / "output/eval_stage2_rerank_bge_reranker_v2_m3_len200",
    "BGE v2 (len=512)": base_dir / "output/eval_stage2_rerank_bge_reranker_v2_m3_len512",
}

output_dir = base_dir / "notebooks" / "analysis_output"
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
    if not metrics_file.exists():
        print(f"Missing metrics in {result_dir}")
        return None
    return pd.read_csv(metrics_file)

all_metrics = {}
for name, result_dir in results.items():
    df = load_metrics_from_dir(result_dir)
    if df is not None:
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
# ## 7. Recall Curves - Train Subset

# %%
df_train = df_all[df_all["split_normalized"] == "train_subset"].copy()

k_list = [50, 100, 200, 300, 400, 500, 1000, 2000]

fig, ax = plt.subplots()
for _, row in df_train.iterrows():
    values = []
    for k in k_list:
        col = f"MeanR@{k}"
        values.append(row[col] if col in row.index else np.nan)
    if np.all(np.isnan(values)):
        continue
    ax.plot(k_list, values, marker="o", label=row["method"])

ax.set_xlabel("K (Recall Cutoff)")
ax.set_ylabel("Mean Recall")
ax.set_title("Recall Curves - Train Subset")
ax.set_xscale("log")
ax.legend(fontsize=9, loc="lower right")

fig_path = figures_dir / "01_recall_curves_train.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 8. Recall Curves - Test Average (13B1-4)

# %%
fig, ax = plt.subplots()
for method in df_test["method"].unique():
    method_data = df_test[df_test["method"] == method]
    if method_data.empty:
        continue
    values = []
    for k in k_list:
        col = f"MeanR@{k}"
        values.append(method_data[col].mean() if col in method_data.columns else np.nan)
    if np.all(np.isnan(values)):
        continue
    ax.plot(k_list, values, marker="o", label=method)

ax.set_xlabel("K (Recall Cutoff)")
ax.set_ylabel("Mean Recall")
ax.set_title("Recall Curves - Test Avg (13B1-4)")
ax.set_xscale("log")
ax.legend(fontsize=9, loc="lower right")

fig_path = figures_dir / "02_recall_curves_test_avg.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 9. Precision Comparison (MAP@10, MRR@10)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

methods_sorted = df_summary["method"].tolist()

# MAP@10
ax = axes[0]
map_vals = df_summary["MAP@10"].values
ax.barh(methods_sorted, map_vals)
ax.set_xlabel("MAP@10")
ax.set_title("MAP@10 (Test Avg)")
for i, v in enumerate(map_vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

# MRR@10
ax = axes[1]
mrr_vals = df_summary["MRR@10"].values
ax.barh(methods_sorted, mrr_vals)
ax.set_xlabel("MRR@10")
ax.set_title("MRR@10 (Test Avg)")
for i, v in enumerate(mrr_vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

fig_path = figures_dir / "03_precision_comparison.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 10. BGE Length Sensitivity (200 vs 512)

# %%
bge_200 = df_all[df_all["method"] == "BGE v2 (len=200)"].copy()
bge_512 = df_all[df_all["method"] == "BGE v2 (len=512)"].copy()

bge_200_sel = bge_200[["split_normalized", "MAP@10", "MRR@10", "MeanR@500", "MeanR@2000"]].copy()
bge_200_sel.columns = ["split", "MAP@10_200", "MRR@10_200", "MeanR@500_200", "MeanR@2000_200"]

bge_512_sel = bge_512[["split_normalized", "MAP@10", "MRR@10", "MeanR@500", "MeanR@2000"]].copy()
bge_512_sel.columns = ["split", "MAP@10_512", "MRR@10_512", "MeanR@500_512", "MeanR@2000_512"]

bge_merged = pd.merge(bge_200_sel, bge_512_sel, on="split", how="inner")

for metric in ["MAP@10", "MRR@10", "MeanR@500", "MeanR@2000"]:
    bge_merged[f"{metric}_delta"] = bge_merged[f"{metric}_512"] - bge_merged[f"{metric}_200"]

print(bge_merged[["split", "MAP@10_delta", "MRR@10_delta", "MeanR@500_delta", "MeanR@2000_delta"]].round(3).to_string(index=False))

fig, ax = plt.subplots()
metrics = ["MAP@10_delta", "MRR@10_delta", "MeanR@500_delta", "MeanR@2000_delta"]
labels = ["MAP@10", "MRR@10", "MeanR@500", "MeanR@2000"]

x = np.arange(len(bge_merged))
width = 0.2
for i, (metric, label) in enumerate(zip(metrics, labels)):
    ax.bar(x + i * width, bge_merged[metric], width, label=label)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(bge_merged["split"], rotation=0)
ax.set_ylabel("Delta (len=512 - len=200)")
ax.set_title("BGE Length Sensitivity")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(fontsize=9)

fig_path = figures_dir / "04_bge_length_sensitivity.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 11. Stage-to-Stage Improvements

# %%
baseline_row = df_summary[df_summary["method"] == "BM25+RM3"]
if baseline_row.empty:
    raise ValueError("BM25+RM3 baseline missing from summary")

baseline_map = baseline_row["MAP@10"].values[0]
baseline_meanr = baseline_row["MeanR@500"].values[0]

stages = [
    "BM25+RM3",
    "Dense (MedEmbed)",
    "Hybrid (RRF)",
    "Reranker MiniLM",
    "BGE v2 (len=200)",
    "BGE v2 (len=512)",
]

improve_rows = []
for method in stages:
    if method not in df_summary["method"].values:
        continue
    row = df_summary[df_summary["method"] == method].iloc[0]
    improve_rows.append(
        {
            "method": method,
            "MAP@10": row["MAP@10"],
            "MAP@10_improve_pct": (row["MAP@10"] - baseline_map) / baseline_map * 100,
            "MeanR@500": row["MeanR@500"],
            "MeanR@500_improve_pct": (row["MeanR@500"] - baseline_meanr) / baseline_meanr * 100,
        }
    )

df_improvements = pd.DataFrame(improve_rows)
print(df_improvements.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(df_improvements["method"], df_improvements["MAP@10_improve_pct"])
ax.set_xlabel("Improvement (%)")
ax.set_title("MAP@10 Improvement vs BM25+RM3")
ax.axvline(0, color="black", linewidth=0.5)

ax = axes[1]
ax.barh(df_improvements["method"], df_improvements["MeanR@500_improve_pct"])
ax.set_xlabel("Improvement (%)")
ax.set_title("MeanR@500 Improvement vs BM25+RM3")
ax.axvline(0, color="black", linewidth=0.5)

fig_path = figures_dir / "05_stage_improvements.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 12. Save Outputs

# %%
consolidated_path = output_dir / "consolidated_all_metrics.csv"
df_all.to_csv(consolidated_path, index=False)
print("Saved:", consolidated_path)

improvements_path = output_dir / "stage_improvements.csv"
df_improvements.to_csv(improvements_path, index=False)
print("Saved:", improvements_path)

bge_path = output_dir / "bge_length_comparison.csv"
bge_merged.to_csv(bge_path, index=False)
print("Saved:", bge_path)

print("Done. Outputs in:", output_dir)
