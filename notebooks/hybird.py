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
import json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# %% [markdown]
# # 1) paths

# %%
# -------- paths --------
BM25_ROOT = Path("../output/eval_bm25_rm3")
BM25_RUNS_DIR = BM25_ROOT / "runs"

DENSE_ROOT = Path("../output/eval_dense_MedEmbed")

SUBSET_PATH = Path("../example/training14b_10pct_sample.json")

TEST_DIR = Path("../Task13BGoldenEnriched")
TEST_BATCHES = [
    TEST_DIR / "13B1_golden.json",
    TEST_DIR / "13B2_golden.json",
    TEST_DIR / "13B3_golden.json",
    TEST_DIR / "13B4_golden.json",
]

# We'll use BM25_RM3 as the BM25 side
BM25_METHOD_NAME = "BM25_RM3"

# -------- evaluation cutoffs --------
K_EVAL = 5000
KS_RECALL = (50, 100, 200, 500, 2000, 5000)


# %% [markdown]
# # 2) Gold building + evaluation helpers

# %%
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

def build_gold_map_from_questions(questions: list[dict]) -> dict[str, list[str]]:
    gold = {}
    for i, q in enumerate(questions):
        qid = str(q.get("id") or q.get("qid") or i)
        docs = q.get("documents") or []
        pmids = [normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]
        gold[qid] = pmids
    return gold

def ap_bioasq(gold: set[str], ranked: list[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    hit = 0
    s = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in gold:
            hit += 1
            s += hit / i
    # BioASQ MAP@10 uses AP@10, normalized by min(|gold|, 10)
    return s / min(len(gold), k)

def rr_at_k(gold: set[str], ranked: list[str], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0

def success_at_k(gold: set[str], ranked: list[str], k: int = 10) -> float:
    return 1.0 if any(doc in gold for doc in ranked[:k]) else 0.0

def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)

def evaluate_run(gold_map: dict[str, list[str]], run_map: dict[str, list[str]],
                 ks_recall=KS_RECALL, eps: float = 1e-5) -> dict:
    qids = list(gold_map.keys())
    ap10s, rr10s, succ10s = [], [], []
    recalls = {k: [] for k in ks_recall}

    for qid in qids:
        gold = set(map(str, gold_map.get(qid, [])))
        ranked = list(map(str, run_map.get(qid, [])))

        ap10s.append(ap_bioasq(gold, ranked, k=10))
        rr10s.append(rr_at_k(gold, ranked, k=10))
        succ10s.append(success_at_k(gold, ranked, k=10))

        for k in ks_recall:
            recalls[k].append(recall_at_k(gold, ranked, k=k))

    gmap10 = math.exp(sum(math.log(max(eps, x)) for x in ap10s) / max(1, len(ap10s)))

    summary = {
        "MAP@10": float(np.mean(ap10s)) if ap10s else 0.0,
        "GMAP@10": float(gmap10) if ap10s else 0.0,
        "MRR@10": float(np.mean(rr10s)) if rr10s else 0.0,
        "Success@10": float(np.mean(succ10s)) if succ10s else 0.0,
    }
    for k in ks_recall:
        summary[f"MeanR@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    return summary



# %% [markdown]
# # 3) Load runs (BM25_RM3 TSV + dense parquet)
# ## 3.1 BM25_RM3 TSV loader

# %%
def load_bm25_tsv_run(path: Path) -> pd.DataFrame:
    """
    Expected columns usually include: qid, docno, rank, score
    But we'll handle common variants.
    """
    df = pd.read_csv(path, sep="\t")
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}

    # Identify qid / docno / rank / score columns
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
        # infer rank by sorting score descending per qid if score exists, else by input order
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
            return out
        else:
            out["rank"] = out.groupby("qid").cumcount() + 1

    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan

    # sort
    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    return out



# %% [markdown]
# # 3.2 Dense parquet loader

# %%
def load_dense_parquet_run(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"qid", "docno", "rank"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {needed - set(df.columns)}")
    out = df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str).map(normalize_pmid)
    out["rank"] = out["rank"].astype(int)
    # score exists in your dense parquet
    if "score" in out.columns:
        out["score"] = out["score"].astype(float)
    else:
        out["score"] = np.nan
    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    return out



# %% [markdown]
# # 4) Fusion methods: Union + RRF
# ## 4.1 Helpers to cut top-K and to build run_map

# %%
def cut_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["rank"] <= k].copy()

def df_to_run_map(df: pd.DataFrame) -> dict[str, list[str]]:
    run = {}
    for qid, grp in df.groupby("qid", sort=False):
        run[str(qid)] = grp.sort_values("rank")["docno"].astype(str).tolist()
    return run



# %% [markdown]
# # 4.2 Union (BM25-first ordering)

# %%
def fuse_union(bm25_df: pd.DataFrame, dense_df: pd.DataFrame, k_bm25: int, k_dense: int, k_out: int = K_EVAL) -> dict[str, list[str]]:
    b = cut_topk(bm25_df, k_bm25)
    d = cut_topk(dense_df, k_dense)

    run = {}
    # group once for speed
    b_grp = {qid: grp for qid, grp in b.groupby("qid", sort=False)}
    d_grp = {qid: grp for qid, grp in d.groupby("qid", sort=False)}
    all_qids = list(dict.fromkeys(list(b_grp.keys()) + list(d_grp.keys())))

    for qid in all_qids:
        b_docs = b_grp.get(qid)
        d_docs = d_grp.get(qid)
        out = []
        seen = set()

        if b_docs is not None:
            for doc in b_docs.sort_values("rank")["docno"].astype(str):
                if doc not in seen:
                    out.append(doc); seen.add(doc)
                if len(out) >= k_out:
                    break

        if len(out) < k_out and d_docs is not None:
            for doc in d_docs.sort_values("rank")["docno"].astype(str):
                if doc not in seen:
                    out.append(doc); seen.add(doc)
                if len(out) >= k_out:
                    break

        run[str(qid)] = out
    return run



# %% [markdown]
# ## 4.3 RRF (optionally weighted)

# %%
def fuse_rrf(bm25_df: pd.DataFrame, dense_df: pd.DataFrame,
             k_bm25: int, k_dense: int,
             k_rrf: int = 60,
             w_bm25: float = 1.0, w_dense: float = 1.0,
             k_out: int = K_EVAL) -> dict[str, list[str]]:
    b = cut_topk(bm25_df, k_bm25)
    d = cut_topk(dense_df, k_dense)

    run = {}
    b_grp = {qid: grp for qid, grp in b.groupby("qid", sort=False)}
    d_grp = {qid: grp for qid, grp in d.groupby("qid", sort=False)}
    all_qids = list(dict.fromkeys(list(b_grp.keys()) + list(d_grp.keys())))

    for qid in all_qids:
        scores = {}
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

        # sort by fused score desc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        run[str(qid)] = [doc for doc, _ in ranked[:k_out]]

    return run



# %% [markdown]
# # 5) Load gold + runs for each split

# %%
def load_questions_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]

def load_gold_for_split(split: str) -> dict[str, list[str]]:
    if split == "train_subset":
        questions = load_questions_from_json(SUBSET_PATH)
    else:
        fp = TEST_DIR / f"{split}.json"  # not used; we use TEST_BATCHES list
        raise ValueError("Use load_gold_for_test_file for test batches.")
    return build_gold_map_from_questions(questions)

def load_gold_for_test_file(fp: Path) -> dict[str, list[str]]:
    questions = load_questions_from_json(fp)
    return build_gold_map_from_questions(questions)

def bm25_run_path(split: str) -> Path:
    return BM25_RUNS_DIR / f"{BM25_METHOD_NAME}__{split}__top5000.tsv"

def dense_run_path(split: str) -> Path:
    return DENSE_ROOT / f"dense_{split}.parquet"



# %% [markdown]
# # 6) Evaluate baseline components + hybrids for one split

# %%
def evaluate_split(split: str, bm25_df: pd.DataFrame, dense_df: pd.DataFrame, gold_map: dict[str, list[str]],
                   k_bm25: int, k_dense: int, k_rrf: int = 60) -> list[dict]:
    out = []

    # Baselines (cut to K_EVAL)
    bm25_run = df_to_run_map(cut_topk(bm25_df, K_EVAL))
    dense_run = df_to_run_map(cut_topk(dense_df, K_EVAL))

    out.append({"method": BM25_METHOD_NAME, "split": split, **evaluate_run(gold_map, bm25_run)})
    out.append({"method": "Dense", "split": split, **evaluate_run(gold_map, dense_run)})

    # Union
    union_run = fuse_union(bm25_df, dense_df, k_bm25=k_bm25, k_dense=k_dense, k_out=K_EVAL)
    out.append({"method": f"Hybrid-Union(kb={k_bm25},kd={k_dense})", "split": split, **evaluate_run(gold_map, union_run)})

    # RRF
    rrf_run = fuse_rrf(bm25_df, dense_df, k_bm25=k_bm25, k_dense=k_dense, k_rrf=k_rrf, k_out=K_EVAL)
    out.append({"method": f"Hybrid-RRF(kb={k_bm25},kd={k_dense},krrf={k_rrf})", "split": split, **evaluate_run(gold_map, rrf_run)})

    return out



# %% [markdown]
# # 7) Load all splits and run a tuning grid

# %%
# Load BM25_RM3 and dense for all splits
splits = ["train_subset"] + [fp.stem for fp in TEST_BATCHES]

bm25_runs = {}
dense_runs = {}
gold_maps = {}

# train_subset gold
EXCLUDE_TEST_QIDS_FROM_TRAIN = True

train_questions_all = load_questions_from_json(SUBSET_PATH)

test_qids = set()
for fp in TEST_BATCHES:
    qs = load_questions_from_json(fp)
    for i, q in enumerate(qs):
        test_qids.add(str(q.get("id") or q.get("qid") or i))

if EXCLUDE_TEST_QIDS_FROM_TRAIN:
    train_questions = [
        q for i, q in enumerate(train_questions_all)
        if str(q.get("id") or q.get("qid") or i) not in test_qids
    ]
else:
    train_questions = train_questions_all

print("train_questions_all:", len(train_questions_all), "train_questions_used:", len(train_questions))
gold_maps["train_subset"] = build_gold_map_from_questions(train_questions)

# test golds
for fp in TEST_BATCHES:
    gold_maps[fp.stem] = load_gold_for_test_file(fp)

# %%
# Quick sanity-check: BM25_RM3 only (no dense load)
split = "13B1_golden"
bm25_df = load_bm25_tsv_run(bm25_run_path(split))
bm25_run = df_to_run_map(cut_topk(bm25_df, K_EVAL))
print(split, evaluate_run(gold_maps[split], bm25_run))

# %%
# BM25_RM3 recall sanity-check across all splits
rows = []
for split in splits:
    bm25_df = load_bm25_tsv_run(bm25_run_path(split))
    bm25_run = df_to_run_map(cut_topk(bm25_df, K_EVAL))
    m = evaluate_run(gold_maps[split], bm25_run)
    rows.append({"split": split, **m})

pd.DataFrame(rows)[["split","MAP@10","MeanR@50","MeanR@100","MeanR@200","MeanR@500","MeanR@2000","MeanR@5000"]]

# %%

# runs
for split in splits:
    bm25_path = bm25_run_path(split)
    dense_path = dense_run_path(split)

    print("Loading:", split)
    bm25_runs[split] = load_bm25_tsv_run(bm25_path)
    dense_runs[split] = load_dense_parquet_run(dense_path)

print("Loaded splits:", splits)

# %%
# Tuning grid
K_BM25_LIST = [500, 1000, 2000, 5000]
K_DENSE_LIST = [100, 200, 500, 1000]
K_RRF_LIST = [60]  # expand later if you want: [10, 30, 60, 100]

all_rows = []

for krrf in K_RRF_LIST:
    for kb in K_BM25_LIST:
        for kd in K_DENSE_LIST:
            # Evaluate all splits for this config
            for split in splits:
                rows = evaluate_split(
                    split=split,
                    bm25_df=bm25_runs[split],
                    dense_df=dense_runs[split],
                    gold_map=gold_maps[split],
                    k_bm25=kb,
                    k_dense=kd,
                    k_rrf=krrf,
                )
                for r in rows:
                    # attach config fields
                    r = dict(r)
                    r["k_bm25"] = kb
                    r["k_dense"] = kd
                    r["k_rrf"] = krrf
                    all_rows.append(r)

results_df = pd.DataFrame(all_rows)
results_df.head()


# %%
results_df

# %% [markdown]
# # 8) Pick best config by MeanR@200/500 on test batches only

# %%
test_splits = [fp.stem for fp in TEST_BATCHES]

def agg_score(df: pd.DataFrame) -> pd.DataFrame:
    # Only hybrid rows, only test splits
    d = df[df["split"].isin(test_splits)].copy()
    d = d[d["method"].str.startswith("Hybrid-")].copy()

    grp = d.groupby(["method", "k_bm25", "k_dense", "k_rrf"], as_index=False).agg({
        "MeanR@200": "mean",
        "MeanR@500": "mean",
        "MeanR@50": "mean",
        "MeanR@100": "mean",
        "MeanR@5000": "mean",
        "MAP@10": "mean",
        "MRR@10": "mean",
    })
    grp["score_R200_R500"] = 0.5 * grp["MeanR@200"] + 0.5 * grp["MeanR@500"]
    return grp.sort_values("score_R200_R500", ascending=False)

hybrid_ranked = agg_score(results_df)
hybrid_ranked.head(20)



# %%
best = hybrid_ranked.iloc[0].to_dict()
best


# %%
