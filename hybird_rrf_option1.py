#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hybrid retrieval tuning (RRF-only) with Option 1 (relative-to-best).

Option 1
--------
For each RRF config, compute:
  R_max = MeanRecall@K_MAX_EVAL
Then choose the smallest K <= CAP such that:
  MeanRecall@K >= P * R_max
If no such K exists within CAP, K_rec is set to CAP+1.

We also record:
  - ShortfallRate@CAP  : fraction of queries where fused list has < CAP docs
  - MeanKeff@CAP       : mean k_eff used at CAP (k_eff = min(CAP, len(fused_list_q)))

Important relationship between (KB, KD), CAP, K_MAX_EVAL
-------------------------------------------------------
- KB and KD control how many docs you *feed* into fusion.
- K_MAX_EVAL is the depth where we measure R_max.
- CAP is the maximum candidate budget you are willing to send to stage-2 reranking.

To keep this internally consistent, the script computes:
  K_MAX_EVAL_EFF = min(K_MAX_EVAL, max_union_upper)
Where max_union_upper is approximated as KB + KD (worst case, no overlap).
We always evaluate at depths <= K_MAX_EVAL_EFF.

Defaults:
  KB = KD = K_MAX_EVAL (good starting point).
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------ Paths (keep aligned with your previous script) ------------------
BM25_ROOT = Path("../output/eval_bm25_rm3")
BM25_RUNS_DIR = BM25_ROOT / "runs"
DENSE_ROOT = Path("../output/eval_dense_medembed_small")

SUBSET_PATH = Path("../example/training14b_10pct_sample.json")

TEST_DIR = Path("../Task13BGoldenEnriched")
TEST_BATCHES = [
    TEST_DIR / "13B1_golden.json",
    TEST_DIR / "13B2_golden.json",
    TEST_DIR / "13B3_golden.json",
    TEST_DIR / "13B4_golden.json",
]

BM25_METHOD_NAME = "BM25_RM3"


# ------------------ Main knobs ------------------
# Candidate pool cap for stage-2 reranker
CAP = 2000

# Depth to estimate best achievable recall (can be > CAP)
K_MAX_EVAL = 5000

# Relative target (e.g. 0.95 means "reach 95% of best achievable")
P = 0.95

# How many docs to take from BM25 and Dense before fusion
# If you want them tied to K_MAX_EVAL: keep defaults.
KB = K_MAX_EVAL
KD = K_MAX_EVAL

# RRF tuning grid
K_RRF_LIST = [10, 30, 60, 100, 150, 200]
WEIGHTS = [
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
    (3.0, 1.0),
    (1.0, 3.0),
]


# ------------------ Helpers: cutoffs generation ------------------

def make_ks(k_max: int, k0: int = 200, n: int = 7) -> tuple[int, ...]:
    """Geometric cutoffs between k0 and k_max (inclusive)."""
    if k_max <= 0:
        return tuple()
    if k_max <= k0:
        return (k_max,)
    ratio = (k_max / k0) ** (1.0 / max(1, n - 1))
    ks = [k0]
    for i in range(1, n - 1):
        ks.append(int(round(k0 * (ratio**i))))
    ks.append(k_max)
    ks = sorted(set(int(k) for k in ks if k > 0))
    if ks[0] != k0:
        ks = [k0] + ks
    if ks[-1] != k_max:
        ks.append(k_max)
    return tuple(sorted(set(ks)))


# ------------------ Helpers: gold + metrics ------------------

def normalize_pmid(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    m = re.search(r"/pubmed/(\d+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+", s):
        return s
    return s


def load_questions_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def build_gold_map_from_questions(questions: list[dict]) -> dict[str, list[str]]:
    gold = {}
    for i, q in enumerate(questions):
        qid = str(q.get("id") or q.get("qid") or i)
        docs = q.get("documents") or []
        pmids = [normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]
        gold[qid] = pmids
    return gold


def recall_at_k_eff(gold: set[str], ranked: list[str], k: int) -> tuple[float, int]:
    k_eff = min(k, len(ranked))
    if not gold or k_eff <= 0:
        return 0.0, k_eff
    return len(gold.intersection(ranked[:k_eff])) / len(gold), k_eff


def evaluate_recall_points(
    gold_map: dict[str, list[str]],
    run_map: dict[str, list[str]],
    ks: tuple[int, ...],
) -> dict:
    """Compute MeanRecall@k (with per-query k_eff) for each k in ks.

    Also returns ShortfallRate@k and MeanKeff@k for any k you ask for.
    """
    out = {}
    qids = list(gold_map.keys())

    for k in ks:
        recalls = []
        shortfalls = []
        keffs = []
        for qid in qids:
            gold = set(map(str, gold_map.get(qid, [])))
            ranked = list(map(str, run_map.get(qid, [])))
            r, k_eff = recall_at_k_eff(gold, ranked, k)
            recalls.append(r)
            keffs.append(k_eff)
            shortfalls.append(1.0 if len(ranked) < k else 0.0)
        out[f"MeanR@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        out[f"ShortfallRate@{k}"] = float(np.mean(shortfalls)) if shortfalls else 0.0
        out[f"MeanKeff@{k}"] = float(np.mean(keffs)) if keffs else 0.0

    return out


# ------------------ Load runs ------------------

def load_bm25_tsv_run(path: Path) -> pd.DataFrame:
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
        if score_col:
            tmp = df.copy()
            tmp["_qid"] = tmp[qid_col].astype(str)
            tmp["_score"] = tmp[score_col].astype(float)
            tmp = tmp.sort_values(["_qid", "_score"], ascending=[True, False])
            tmp["_rank"] = tmp.groupby("_qid").cumcount() + 1
            out = pd.DataFrame({
                "qid": tmp["_qid"],
                "docno": tmp[doc_col].astype(str).map(normalize_pmid),
                "rank": tmp["_rank"].astype(int),
                "score": tmp["_score"].astype(float),
            })
            return out.sort_values(["qid", "rank"]).reset_index(drop=True)
        out["rank"] = out.groupby("qid").cumcount() + 1

    out["score"] = df[score_col].astype(float) if score_col else np.nan
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def load_dense_parquet_run(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"qid", "docno", "rank"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {needed - set(df.columns)}")
    out = df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str).map(normalize_pmid)
    out["rank"] = out["rank"].astype(int)
    if "score" in out.columns:
        out["score"] = out["score"].astype(float)
    else:
        out["score"] = np.nan
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def bm25_run_path(split: str) -> Path:
    # Your runs are stored as top5000; we still cut to KB when fusing.
    return BM25_RUNS_DIR / f"{BM25_METHOD_NAME}__{split}__top5000.tsv"


def dense_run_path(split: str) -> Path:
    return DENSE_ROOT / f"dense_{split}.parquet"


# ------------------ Fusion: weighted RRF ------------------

def cut_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["rank"] <= k].copy()


def fuse_rrf(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    kb: int,
    kd: int,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
    k_out: int,
) -> dict[str, list[str]]:
    b = cut_topk(bm25_df, kb)
    d = cut_topk(dense_df, kd)

    run: dict[str, list[str]] = {}
    b_grp = {qid: grp for qid, grp in b.groupby("qid", sort=False)}
    d_grp = {qid: grp for qid, grp in d.groupby("qid", sort=False)}
    all_qids = list(dict.fromkeys(list(b_grp.keys()) + list(d_grp.keys())))

    for qid in all_qids:
        scores: dict[str, float] = {}

        if qid in b_grp:
            for _, row in b_grp[qid].iterrows():
                doc = str(row["docno"])
                r = int(row["rank"])
                scores[doc] = scores.get(doc, 0.0) + (w_bm25 / (k_rrf + r))

        if qid in d_grp:
            for _, row in d_grp[qid].iterrows():
                doc = str(row["docno"])
                r = int(row["rank"])
                scores[doc] = scores.get(doc, 0.0) + (w_dense / (k_rrf + r))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        run[str(qid)] = [doc for doc, _ in ranked[:k_out]]

    return run


# ------------------ Option 1: compute K_rec ------------------

def find_k_relative_to_best(metrics: dict, ks_cap: tuple[int, ...], k_best: int, p: float) -> int:
    """Return smallest K in ks_cap s.t. MeanR@K >= p * MeanR@k_best.

    Returns CAP+1 if not reached.
    """
    r_best = metrics.get(f"MeanR@{k_best}", None)
    if r_best is None:
        return max(ks_cap) + 1
    target = p * float(r_best)
    for k in ks_cap:
        if metrics.get(f"MeanR@{k}", -1.0) >= target:
            return int(k)
    return max(ks_cap) + 1


# ------------------ Main ------------------

def main() -> None:
    # Splits
    splits = ["train_subset"] + [fp.stem for fp in TEST_BATCHES]

    # Gold maps
    gold_maps: dict[str, dict[str, list[str]]] = {}

    # train_subset gold: exclude test qids (same as your earlier script)
    train_questions_all = load_questions_from_json(SUBSET_PATH)

    test_qids = set()
    for fp in TEST_BATCHES:
        qs = load_questions_from_json(fp)
        for i, q in enumerate(qs):
            test_qids.add(str(q.get("id") or q.get("qid") or i))

    train_questions = [
        q for i, q in enumerate(train_questions_all)
        if str(q.get("id") or q.get("qid") or i) not in test_qids
    ]
    gold_maps["train_subset"] = build_gold_map_from_questions(train_questions)

    for fp in TEST_BATCHES:
        gold_maps[fp.stem] = build_gold_map_from_questions(load_questions_from_json(fp))

    # Runs
    bm25_runs: dict[str, pd.DataFrame] = {}
    dense_runs: dict[str, pd.DataFrame] = {}
    for split in splits:
        print("Loading:", split)
        bm25_runs[split] = load_bm25_tsv_run(bm25_run_path(split))
        dense_runs[split] = load_dense_parquet_run(dense_run_path(split))

    # Effective eval depths (depend on KB/KD)
    # Upper bound for union size is KB + KD (worst-case no overlap)
    max_union_upper = KB + KD
    k_max_eval_eff = int(min(K_MAX_EVAL, max_union_upper))
    cap_eff = int(min(CAP, k_max_eval_eff))

    # Recall points: a curve up to cap_eff, plus k_max_eval_eff
    ks_cap = make_ks(cap_eff, k0=200, n=7)
    ks_eval = tuple(sorted(set(ks_cap + (k_max_eval_eff,))))

    print(f"KB={KB} KD={KD} CAP={CAP} K_MAX_EVAL={K_MAX_EVAL} => cap_eff={cap_eff} k_max_eval_eff={k_max_eval_eff}")
    print("ks_cap:", ks_cap)
    print("ks_eval:", ks_eval)

    rows: list[dict] = []

    test_splits = [fp.stem for fp in TEST_BATCHES]

    for (w_bm25, w_dense) in WEIGHTS:
        for k_rrf in K_RRF_LIST:
            for split in splits:
                gold_map = gold_maps[split]

                fused = fuse_rrf(
                    bm25_df=bm25_runs[split],
                    dense_df=dense_runs[split],
                    kb=KB,
                    kd=KD,
                    k_rrf=k_rrf,
                    w_bm25=w_bm25,
                    w_dense=w_dense,
                    k_out=k_max_eval_eff,
                )

                metrics = evaluate_recall_points(gold_map, fused, ks=ks_eval)

                k_rec = find_k_relative_to_best(metrics, ks_cap=ks_cap, k_best=k_max_eval_eff, p=P)

                row = {
                    "split": split,
                    "k_rrf": int(k_rrf),
                    "w_bm25": float(w_bm25),
                    "w_dense": float(w_dense),
                    "KB": int(KB),
                    "KD": int(KD),
                    "CAP": int(CAP),
                    "CAP_eff": int(cap_eff),
                    "K_MAX_EVAL": int(K_MAX_EVAL),
                    "K_MAX_EVAL_eff": int(k_max_eval_eff),
                    "P": float(P),
                    "K_rec": int(k_rec),
                    # Required diagnostics at CAP
                    f"ShortfallRate@{cap_eff}": metrics.get(f"ShortfallRate@{cap_eff}", np.nan),
                    f"MeanKeff@{cap_eff}": metrics.get(f"MeanKeff@{cap_eff}", np.nan),
                    # The two key recalls
                    f"MeanR@{cap_eff}": metrics.get(f"MeanR@{cap_eff}", np.nan),
                    f"MeanR@{k_max_eval_eff}": metrics.get(f"MeanR@{k_max_eval_eff}", np.nan),
                }

                # Also store curve points for convenience
                for k in ks_eval:
                    row[f"MeanR@{k}"] = metrics.get(f"MeanR@{k}", np.nan)

                rows.append(row)

    df = pd.DataFrame(rows)

    # Aggregate on test splits and rank configs.
    # Primary: smaller K_rec (<= cap_eff)
    # Secondary: higher MeanR@cap_eff
    # Tertiary: lower ShortfallRate@cap_eff
    agg_cols = {
        "K_rec": "mean",
        f"MeanR@{cap_eff}": "mean",
        f"MeanR@{k_max_eval_eff}": "mean",
        f"ShortfallRate@{cap_eff}": "mean",
        f"MeanKeff@{cap_eff}": "mean",
    }

    dtest = df[df["split"].isin(test_splits)].copy()
    grp = dtest.groupby(["k_rrf", "w_bm25", "w_dense"], as_index=False).agg(agg_cols)

    # K_rec is CAP+1 when not reached; keep smaller is better
    grp = grp.sort_values(
        by=["K_rec", f"MeanR@{cap_eff}", f"ShortfallRate@{cap_eff}"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    print("\n=== Top configs (aggregated on test splits) ===")
    print(grp.head(30).to_string(index=False))

    best = grp.iloc[0].to_dict() if len(grp) else {}
    print("\nBEST:")
    print(best)

    # Optional: plot recall curve for best config on each test split
    if best:
        best_krrf = int(best["k_rrf"])
        best_wb = float(best["w_bm25"])
        best_wd = float(best["w_dense"])

        for split in test_splits:
            fused = fuse_rrf(
                bm25_df=bm25_runs[split],
                dense_df=dense_runs[split],
                kb=KB,
                kd=KD,
                k_rrf=best_krrf,
                w_bm25=best_wb,
                w_dense=best_wd,
                k_out=k_max_eval_eff,
            )
            metrics = evaluate_recall_points(gold_maps[split], fused, ks=ks_eval)
            ys = [metrics.get(f"MeanR@{k}", np.nan) for k in ks_eval]
            plt.figure()
            plt.plot(list(ks_eval), ys, marker="o")
            plt.xlabel("K")
            plt.ylabel("Mean Recall@K (k_eff per query)")
            plt.title(f"Best RRF recall curve ({split})")
            plt.show()


if __name__ == "__main__":
    main()
