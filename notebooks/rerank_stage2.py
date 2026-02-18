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
# # Stage 2 Reranking (Adaptive Cutoff)
#
# This notebook builds a Stage 2 cross-encoder reranker on top of the Stage 1 hybrid run.
#
# **Inputs**: BioASQ golden queries, hybrid top-2000 candidates, and PubMed subset texts.
# **Outputs**: reranked runs + adaptive cutoff stats + metrics (Recall, MAP@10, MRR@10, etc.).

# %% [markdown]
# ## 1) Setup

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
# Allow importing from scripts/public/shared_scripts
import sys
sys.path.insert(0, str(Path("..") / "scripts" / "public" / "shared_scripts"))

from retrieval_eval.common import (
    build_topics_and_gold,
    collect_qids_from_questions,
    evaluate_run,
    load_questions,
    normalize_pmid,
    recall_at_k,
    run_df_to_run_map,
)

# %% [markdown]
# ## 2) Paths and Config

# %%
# ---- data inputs ----
SUBSET_PATH = Path("../example/training14b_10pct_sample.json")
GOLD_DIR = Path("../bioasq_data/Task13BGoldenEnriched")
GOLD_FILES = [
    GOLD_DIR / "13B1_golden.json",
    GOLD_DIR / "13B2_golden.json",
    GOLD_DIR / "13B3_golden.json",
    GOLD_DIR / "13B4_golden.json",
]

RUNS_DIR = Path("../output/eval_hybird_production_test/runs")
RUN_FILES = {
    "train_subset": RUNS_DIR / "best_rrf_train_subset_top2000.tsv",
    "13B1_golden": RUNS_DIR / "best_rrf_13B1_golden_top2000.tsv",
    "13B2_golden": RUNS_DIR / "best_rrf_13B2_golden_top2000.tsv",
    "13B3_golden": RUNS_DIR / "best_rrf_13B3_golden_top2000.tsv",
    "13B4_golden": RUNS_DIR / "best_rrf_13B4_golden_top2000.tsv",
}

DOCS_JSONL = Path("../output/subset_pubmed.jsonl")

# ---- selection ----
SELECTED_RUNS = ["13B1_golden"]  # or ["train_subset"]
MAX_QUERIES_PER_SPLIT = None  # e.g., 50 for a quick test
CANDIDATE_LIMIT = 2000  # stage-1 top K

# ---- reranker ----
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEVICE = "cpu"  # "cuda", "mps", or "cpu"
BATCH_SIZE = 16

# ---- adaptive cutoff ----
P_TARGET = 0.95
K_CAP = 300

# ---- evaluation ----
KS_RECALL = (50, 100, 200, 300, 500, 1000, 2000)
OUTPUT_DIR = Path("../output/eval_stage2_rerank")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 3) Load queries + gold

# %%
def build_topics_map(questions: List[dict]) -> Dict[str, str]:
    out = {}
    for i, q in enumerate(questions):
        qid = str(q.get("id") or q.get("qid") or i)
        query = str(q.get("body") or q.get("query") or q.get("question") or "").strip()
        out[qid] = query
    return out

gold_maps: Dict[str, Dict[str, List[str]]] = {}
topics_map: Dict[str, str] = {}

for p in GOLD_FILES:
    questions = load_questions(p)
    topics_map.update(build_topics_map(questions))
    _, gold_map = build_topics_and_gold(questions)
    gold_maps[str(p.stem)] = gold_map

subset_questions = load_questions(SUBSET_PATH)
subset_topics = build_topics_map(subset_questions)
_, subset_gold = build_topics_and_gold(subset_questions)

if "train_subset" in SELECTED_RUNS:
    gold_maps["train_subset"] = subset_gold
    topics_map.update(subset_topics)

print("gold splits:", list(gold_maps.keys()))
print("selected runs:", SELECTED_RUNS)


# %% [markdown]
# ## 4) Load Stage 1 hybrid runs

# %%
def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")

    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(normalize_pmid),
    })

    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1

    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan

    return out.sort_values(["qid", "rank"]).reset_index(drop=True)

missing_runs = [r for r in SELECTED_RUNS if r not in RUN_FILES]
if missing_runs:
    raise ValueError(f"Unknown run names: {missing_runs}")

selected_files = {name: RUN_FILES[name] for name in SELECTED_RUNS}
runs_df = {}
for name, path in selected_files.items():
    df = load_run_tsv(path)
    if CANDIDATE_LIMIT:
        df = df[df["rank"] <= int(CANDIDATE_LIMIT)]
    if MAX_QUERIES_PER_SPLIT:
        qids = sorted(df["qid"].unique())[: int(MAX_QUERIES_PER_SPLIT)]
        df = df[df["qid"].isin(qids)]
    runs_df[name] = df

runs_map = {name: run_df_to_run_map(df) for name, df in runs_df.items()}
run_names = list(runs_map.keys())
for name, df in runs_df.items():
    print(name, df["qid"].nunique(), "queries", "|", len(df), "rows")


# %% [markdown]
# ## 5) Load subset PubMed texts (on-demand)

# %%
def extract_docno(rec: dict) -> str:
    for k in ("docno", "pmid", "id"):
        if k in rec:
            return normalize_pmid(rec[k])
    return ""

def extract_text(rec: dict) -> str:
    if "text" in rec and rec["text"]:
        return str(rec["text"]).strip()
    parts = []
    if rec.get("title"):
        parts.append(str(rec["title"]).strip())
    if rec.get("abstract"):
        parts.append(str(rec["abstract"]).strip())
    if rec.get("abstractText"):
        parts.append(str(rec["abstractText"]).strip())
    return " ".join([p for p in parts if p])

def load_doc_texts(docnos: Iterable[str], jsonl_path: Path) -> Dict[str, str]:
    wanted = set(map(str, docnos))
    out: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docno = extract_docno(rec)
            if docno in wanted and docno not in out:
                text = extract_text(rec)
                out[docno] = text
                if len(out) == len(wanted):
                    break
    return out

candidate_docnos = set()
for name in run_names:
    for docs in runs_map[name].values():
        candidate_docnos.update(docs)

print("candidate docnos:", len(candidate_docnos))
doc_texts = load_doc_texts(candidate_docnos, DOCS_JSONL)
print("loaded texts:", len(doc_texts))

# %% [markdown]
# ## 6) Load reranker

# %%
try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e

reranker = CrossEncoder(MODEL_NAME, device=DEVICE)


# %% [markdown]
# ## 7) Rerank candidates

# %%
def rerank_run(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model: CrossEncoder,
    batch_size: int = 32,
    log_every: int = 10,
    ) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    items = list(run_map.items())
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            out[qid] = docs
            continue
        pairs = [(query, doc_texts.get(d, "")) for d in docs]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[qid] = [d for d, _ in scored]
        if idx == 1 or idx % log_every == 0 or idx == total:
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[rerank] {idx}/{total} queries | {rate:.2f} q/s")
    return out

reranked_runs = {}
for name in run_names:
    if name not in gold_maps:
        print("skip rerank, no gold for", name)
        continue
    reranked_runs[name] = rerank_run(
        runs_map[name],
        topics=topics_map,
        doc_texts=doc_texts,
        model=reranker,
        batch_size=BATCH_SIZE,
    )
    print("reranked", name, "queries:", len(reranked_runs[name]))


# %% [markdown]
# ## 8) Adaptive cutoff (per-query K)
# Find the smallest $K \le 300$ that reaches $0.95 \times$ the per-query maximum recall achievable within the top-2000 candidates.

# %%
def adaptive_k_for_query(gold: List[str], ranked: List[str], p: float, cap: int) -> int:
    gold_set = set(map(str, gold))
    if not gold_set:
        return 0
    max_recall = recall_at_k(gold_set, ranked, k=len(ranked))
    target = p * max_recall
    if target <= 0:
        return 0
    k_max = min(cap, len(ranked))
    for k in range(1, k_max + 1):
        if recall_at_k(gold_set, ranked, k=k) >= target:
            return k
    return k_max

def evaluate_adaptive(
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
    p: float,
    cap: int,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    recalls = []
    keffs = []
    shortfalls = []

    for qid, gold in gold_map.items():
        ranked = run_map.get(qid, [])
        k_rec = adaptive_k_for_query(gold, ranked, p=p, cap=cap)
        k_eff = min(k_rec, len(ranked))
        shortfall = 1.0 if k_rec > len(ranked) else 0.0
        r = recall_at_k(set(map(str, gold)), ranked, k=k_eff) if k_eff > 0 else 0.0

        rows.append({"qid": qid, "K_rec": k_rec, "K_eff": k_eff, "R@Krec": r})
        recalls.append(r)
        keffs.append(k_eff)
        shortfalls.append(shortfall)

    metrics = {
        "MeanR@Krec": float(np.mean(recalls)) if recalls else 0.0,
        "MeanKeff@Krec": float(np.mean(keffs)) if keffs else 0.0,
        "ShortfallRate@Krec": float(np.mean(shortfalls)) if shortfalls else 0.0,
    }
    return metrics, pd.DataFrame(rows)

adaptive_stats = {}
adaptive_perq = {}
for name in reranked_runs.keys():
    stats, perq = evaluate_adaptive(gold_maps[name], reranked_runs[name], p=P_TARGET, cap=K_CAP)
    adaptive_stats[name] = stats
    adaptive_perq[name] = perq
    print("adaptive", name, stats)

# %% [markdown]
# ## 9) Standard metrics (Recall, MAP@10, MRR@10, etc.)

# %%
summary_rows = []
for name in reranked_runs.keys():
    metrics, _ = evaluate_run(gold_maps[name], reranked_runs[name], ks_recall=KS_RECALL)
    row = {"split": name}
    row.update(metrics)
    row.update(adaptive_stats.get(name, {}))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df

# %% [markdown]
# ## 10) Save outputs

# %%
summary_df.to_csv(OUTPUT_DIR / "rerank_metrics.csv", index=False)

for name in adaptive_perq.keys():
    adaptive_perq[name].to_csv(OUTPUT_DIR / f"adaptive_{name}.csv", index=False)

print("saved to", OUTPUT_DIR)

# %%
