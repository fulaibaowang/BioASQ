# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Results analysis: workflow_local_3pct_hpc_bge
#
# Focused analysis for `output/workflow_local_3pct_hpc_bge` using **bm25**, **dense**, **hybrid**, and **rerank** outputs.
#
# **Goals:**
# 1. **Ceiling (Hybrid @ K=2000):** Does Hybrid even contain the gold by K=2000? Table of queries with recall@2000 = 0.
# 2. **n_rel vs low recall:** Is number of relevant docs (`n_rel`) driving the low-recall tail? Scatter (n_rel vs R@200) + Spearman correlation.

# %% [markdown]
# ## Setup and paths

# %%
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

base_dir = Path(".").resolve()
if not (base_dir / "example").exists():
    base_dir = base_dir.parent  # run from notebooks/
    assert (base_dir / "example").exists(), "Run from repo root or notebooks/"

workflow_dir = base_dir / "output" / "workflow_local_3pct_hpc_bge"
assert workflow_dir.exists(), f"Missing {workflow_dir}"

bm25_dir = workflow_dir / "bm25"
dense_dir = workflow_dir / "dense"
hybrid_dir = workflow_dir / "hybrid"
rerank_dir = workflow_dir / "rerank"

# Splits and data paths (from workflow config)
splits = ["training14b_3pct_sample", "13b_golden_50q_sample"]
qrels_paths = {
    "training14b_3pct_sample": base_dir / "example" / "training14b_3pct_sample.json",
    "13b_golden_50q_sample": base_dir / "example" / "13b_golden_50q_sample.json",
}
for s, p in qrels_paths.items():
    assert p.exists(), f"Missing {p}"

print("Workflow dir:", workflow_dir)
print("Splits:", splits)


# %% [markdown]
# ## Load qrels, question text, and n_rel

# %%
def _extract_pmid(doc_entry):
    if isinstance(doc_entry, dict):
        doc_entry = doc_entry.get("document", "")
    if not isinstance(doc_entry, str):
        return None
    if "/" in doc_entry:
        return doc_entry.rsplit("/", 1)[-1]
    return doc_entry


def load_qrels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for q in data.get("questions", []):
        qid = q.get("id")
        if not qid:
            continue
        qid = str(qid)
        docs = q.get("documents", [])
        pmids = {str(_extract_pmid(d)) for d in docs if _extract_pmid(d)}
        out[qid] = pmids
    return out


def load_question_text_and_n_rel(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text_by_qid = {}
    n_rel_by_qid = {}
    for q in data.get("questions", []):
        qid = q.get("id")
        if not qid:
            continue
        qid = str(qid)
        text_by_qid[qid] = (q.get("body") or "").strip()
        docs = q.get("documents", [])
        n_rel_by_qid[qid] = len([d for d in docs if _extract_pmid(d)])
    return text_by_qid, n_rel_by_qid


qrels_by_split = {s: load_qrels(p) for s, p in qrels_paths.items()}
question_text_by_split = {}
n_rel_by_split = {}
for s, p in qrels_paths.items():
    txt, nrel = load_question_text_and_n_rel(p)
    question_text_by_split[s] = txt
    n_rel_by_split[s] = nrel

print("Qrels: ", {s: len(q) for s, q in qrels_by_split.items()})
print("Questions with text: ", {s: len(q) for s, q in question_text_by_split.items()})


# %% [markdown]
# ## Compute Hybrid per-query recall (from run TSVs)

# %%
def load_run(path: Path):
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid")
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc")
    if qid_col is None or doc_col is None:
        raise ValueError(f"Missing qid/doc columns in {path}")
    df[qid_col] = df[qid_col].astype(str)
    df[doc_col] = df[doc_col].astype(str)
    rank_col = cols.get("rank")
    if rank_col:
        df = df.sort_values([qid_col, rank_col])
    return df[[qid_col, doc_col]]


def recall_at_k(docs: list, rels: set, k: int) -> float:
    if not rels:
        return np.nan
    top = docs[:k]
    hit = len(set(top) & rels)
    return hit / len(rels)


def compute_hybrid_per_query_recall(hybrid_runs_dir: Path, qrels_by_split: dict, k_values=(200, 2000)):
    run_template = "best_rrf_{split}_top5000.tsv"
    rows = []
    for split, qrels in qrels_by_split.items():
        run_path = hybrid_runs_dir / run_template.format(split=split)
        if not run_path.exists():
            print(f"Missing: {run_path}")
            continue
        run_df = load_run(run_path)
        qid_col, doc_col = run_df.columns.tolist()
        for qid, group in run_df.groupby(qid_col, sort=False):
            rels = qrels.get(qid, set())
            if not rels:
                continue
            docs = group[doc_col].tolist()
            row = {"split": split, "qid": qid}
            for k in k_values:
                row[f"R@{k}"] = recall_at_k(docs, rels, k)
            rows.append(row)
    return pd.DataFrame(rows)


hybrid_runs_dir = hybrid_dir / "runs"
hybrid_per_query = compute_hybrid_per_query_recall(
    hybrid_runs_dir, qrels_by_split, k_values=(200, 2000)
)
print("Hybrid per-query shape:", hybrid_per_query.shape)
display(hybrid_per_query.head(10))

# %% [markdown]
# ---
# ## 1. Ceiling: Does Hybrid contain the gold by K=2000?
#
# Table: **split**, **qid**, **n_rel**, **query** for every query where **Recall@2000 == 0** (Hybrid retrieved no relevant doc in top 2000).

# %%
df = hybrid_per_query.copy()
df["n_rel"] = df.apply(
    lambda r: n_rel_by_split.get(r["split"], {}).get(r["qid"], np.nan), axis=1
)
df["query"] = df.apply(
    lambda r: question_text_by_split.get(r["split"], {}).get(r["qid"], ""), axis=1
)

ceiling_zero = df[df["R@2000"] == 0].copy()
ceiling_zero = ceiling_zero[["split", "qid", "n_rel", "query"]].reset_index(drop=True)

print("Queries with Hybrid Recall@2000 == 0:", len(ceiling_zero))
print()
display(ceiling_zero)

# %% [markdown]
# ---
# ## 2. Is n_rel driving the low-recall tail?
#
# - **Scatter:** per query, Hybrid; x = `n_rel`, y = R@200.
# - **Spearman correlation** between `n_rel` and R@200.

# %%
df = hybrid_per_query.copy()
df["n_rel"] = df.apply(
    lambda r: n_rel_by_split.get(r["split"], {}).get(r["qid"], np.nan), axis=1
)
df_clean = df.dropna(subset=["n_rel", "R@200"])

rho, pval = spearmanr(df_clean["n_rel"], df_clean["R@200"])
print(f"Spearman(n_rel, R@200): rho = {rho:.4f}, p = {pval:.4e}")
print()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df_clean["n_rel"], df_clean["R@200"], alpha=0.6, s=25)
ax.set_xlabel("n_rel (number of relevant documents)")
ax.set_ylabel("Recall@200 (Hybrid)")
ax.set_title("Hybrid: n_rel vs R@200 (per query)")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# Optional: same scatter faceted by split
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, split in zip(axes, splits):
    sub = df_clean[df_clean["split"] == split]
    if sub.empty:
        ax.set_visible(False)
        continue
    rho_s, p_s = spearmanr(sub["n_rel"], sub["R@200"])
    ax.scatter(sub["n_rel"], sub["R@200"], alpha=0.6, s=25)
    ax.set_xlabel("n_rel")
    ax.set_ylabel("R@200")
    ax.set_title(f"{split}\nSpearman = {rho_s:.3f} (p = {p_s:.2e})")
plt.suptitle("Hybrid: n_rel vs R@200 by split")
plt.tight_layout()
plt.show()
