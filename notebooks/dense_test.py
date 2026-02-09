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

# %%
import os
import re
import numpy as np
import pandas as pd
import json, math
from tqdm import tqdm
from pathlib import Path

from sentence_transformers import SentenceTransformer
import hnswlib

# %%
from dotenv import load_dotenv
load_dotenv()


# %% [markdown]
# # one example

# %%
def read_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)

toy_path = "/Users/yun/develop/BioASQ/example/dense_test/pubmed26n0064.jsonl"
docs = list(read_jsonl(toy_path))
len(docs), docs[0].keys()


# %%
def parse_mesh_terms(mesh_terms: str) -> list[str]:
    """
    mesh_terms string like:
    'D000445:Aldehyde Oxidoreductases; D000818:Animals; ...'
    Return just names: ['Aldehyde Oxidoreductases', 'Animals', ...]
    """
    if not mesh_terms:
        return []
    parts = [p.strip() for p in mesh_terms.split(";") if p.strip()]
    names = []
    for p in parts:
        # split 'D000445:Name'
        if ":" in p:
            names.append(p.split(":", 1)[1].strip())
        else:
            names.append(p)
    return names

def build_doc_text(d: dict, include_mesh: bool = False) -> str:
    title = (d.get("title") or "").strip()
    abstract = (d.get("abstract") or "").strip()
    text = title
    if abstract:
        text = f"{title}\n\n{abstract}" if title else abstract

    if include_mesh:
        mesh_names = parse_mesh_terms(d.get("mesh_terms") or "")
        if mesh_names:
            # keep it compact
            text = f"{text}\n\nMeSH: " + "; ".join(mesh_names)

    return text.strip()

include_mesh = False  # try True later

rows = []
for d in docs:
    pmid = str((d.get("pmid") or d.get("docno") or "")).strip()
    if not pmid:
        continue
    if d.get("is_deleted") is True:
        continue
    rows.append({
        "pmid": pmid,
        "text": build_doc_text(d, include_mesh=include_mesh),
        "title": (d.get("title") or "").strip(),
    })

df = pd.DataFrame(rows)
df.head()


# %%
with pd.option_context("display.max_rows", None,
                       "display.max_columns", None,
                       "display.max_colwidth", None,
                       "display.width", None):
    display(df.head(2))

# %%
model_name = "abhinand/MedEmbed-small-v0.1"  # HF model
model = SentenceTransformer(model_name)
# Optional: enforce max sequence length (model default is usually fine)
model.max_seq_length = 512

# Embed doc texts
doc_texts = df["text"].tolist()

# %%
emb = model.encode(
    doc_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # cosine similarity
)

emb = emb.astype(np.float32)
emb.shape

# %%
dim = emb.shape[1]
num_elements = emb.shape[0]

# HNSW params (reasonable starting points)
M = 32
ef_construction = 200

index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

# Add vectors (use integer ids 0..N-1)
index.add_items(emb, ids=np.arange(num_elements))

# Query-time ef (higher = better recall, slower)
index.set_ef(100)

print("HNSW built:", num_elements, "vectors,", dim, "dim")


# %%
def dense_search(query: str, topk: int = 10):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q_emb, k=topk)
    # cosine space distance in hnswlib is (1 - cosine_sim) depending on space; smaller is better
    hits = []
    for idx, dist in zip(labels[0], distances[0]):
        hits.append((int(idx), float(dist)))
    return hits

def pretty_print_hits(query: str, topk: int = 5):
    hits = dense_search(query, topk=topk)
    print("QUERY:", query)
    for rank, (row_id, dist) in enumerate(hits, 1):
        pmid = df.iloc[row_id]["pmid"]
        title = df.iloc[row_id]["title"]
        print(f"{rank:2d}. pmid={pmid}  dist={dist:.4f}  title={title[:120]}")

pretty_print_hits("What is the prognostic role of alterred thyroid profile after cardiosurgery?", topk=5)


# %% [markdown]
# top 1 result is in our ground truth and bm25 failed to get it

# %% [markdown]
# ## check 1 how many are truncated (>512 words)

# %%

# SentenceTransformer -> underlying HF tokenizer
# Works for most ST models
hf_tokenizer = model.tokenizer
max_len = getattr(model, "max_seq_length", 512) or 512  # ST sometimes stores it here

def count_tokens(texts, tokenizer, max_len: int, batch_size: int = 512):
    lengths = []
    truncated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=False,
            truncation=False,   # count full length
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # lengths include special tokens
        lens = [len(ids) for ids in enc["input_ids"]]
        lengths.extend(lens)
        truncated.extend([l > max_len for l in lens])
    return np.array(lengths, dtype=np.int32), np.array(truncated, dtype=bool)

texts = df["text"].astype(str).tolist()
tok_len, is_trunc = count_tokens(texts, hf_tokenizer, max_len=max_len)

df_stats = df.copy()
df_stats["tok_len"] = tok_len
df_stats["would_truncate"] = is_trunc

print("Model max_seq_length:", max_len)
print("Docs:", len(df_stats))
print("Would truncate:", int(df_stats["would_truncate"].sum()),
      f"({df_stats['would_truncate'].mean()*100:.2f}%)")

# quick distribution
print("\nToken length quantiles:")
print(df_stats["tok_len"].quantile([0, .5, .75, .9, .95, .99, 1.0]).to_string())


# %% [markdown]
# ## NaN/shape check

# %%
assert emb.dtype == np.float32
assert emb.ndim == 2
assert not np.isnan(emb).any()
print("emb ok:", emb.shape)



# %% [markdown]
# # run scripts/public/data/build_dense_hnsw_index_from_jsonl_shards.py to build vector database (10% data first)
#
# we first use abhinand/MedEmbed-small-v0.1

# %% [markdown]
# # test on subset

# %%
import torch

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# %%
# ---------- Paths ----------
SUBSET_PATH = Path("../example/training14b_10pct_sample.json")

TEST_DIR = Path("../bioasq_data/Task13BGoldenEnriched")
TEST_BATCHES = [
    TEST_DIR / "13B1_golden.json",
    TEST_DIR / "13B2_golden.json",
    TEST_DIR / "13B3_golden.json",
    TEST_DIR / "13B4_golden.json",
]

# Dense index output dir from your build script
DENSE_INDEX_DIR = Path("/Users/yun/develop/pubmed_medembed_2026_subset_index")  # <-- change to yours
HNSW_INDEX_PATH = DENSE_INDEX_DIR / "hnsw_index.bin"
ROWID_MAP_PATH  = DENSE_INDEX_DIR / "rowid_to_pmid.tsv"
META_PATH       = DENSE_INDEX_DIR / "meta.json"

# ---------- Eval settings ----------
K_CHECK_SUBSET = 5000   # evaluate recall up to 5000
K_QUERY = 5000          # how many hits to retrieve per query (match eval)


# %%
# Load metadata if available (recommended)
meta = None
if META_PATH.exists():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    print("Loaded meta:", {k: meta[k] for k in ["model_name","dim","max_seq_length","hnsw_space","hnsw_ef_search","hnsw_M","hnsw_ef_construction"] if k in meta})

# Model name: prefer meta, otherwise set manually
model_name = meta["model_name"] if meta and "model_name" in meta else "abhinand/MedEmbed-small-v0.1"

model = SentenceTransformer(model_name, device=device)
# Ensure truncation is consistent with how you built doc embeddings
if meta and "max_seq_length" in meta:
    model.max_seq_length = int(meta["max_seq_length"])
else:
    model.max_seq_length = 512

print("Model:", model_name, "device:", device, "max_seq_length:", model.max_seq_length)

# Load rowid -> pmid mapping
rowid_to_pmid = []
with open(ROWID_MAP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        rid, pmid = line.split("\t", 1)
        rowid_to_pmid.append(pmid)

n_docs = len(rowid_to_pmid)
print("Docs in dense index:", n_docs)

# Load HNSW index
# Need dim + space. Use meta if possible. Otherwise probe dim from model and use cosine.
dim = int(meta["dim"]) if meta and "dim" in meta else int(model.get_sentence_embedding_dimension())
space = meta["hnsw_space"] if meta and "hnsw_space" in meta else "cosine"

index = hnswlib.Index(space=space, dim=dim)
index.load_index(str(HNSW_INDEX_PATH), max_elements=n_docs)

# Set efSearch (query-time). Can override meta.
ef_search = int(meta["hnsw_ef_search"]) if meta and "hnsw_ef_search" in meta else 100
index.set_ef(ef_search)

print("HNSW loaded:", HNSW_INDEX_PATH.name, "space:", space, "dim:", dim, "efSearch:", ef_search)


# %%
def normalize_pmid(x) -> str:
    # BioASQ sometimes stores document URLs; sometimes raw pmid strings
    if x is None:
        return ""
    s = str(x).strip()
    # common BioASQ format: "http://www.ncbi.nlm.nih.gov/pubmed/12345"
    m = re.search(r"/pubmed/(\d+)", s)
    if m:
        return m.group(1)
    # or just digits
    m = re.fullmatch(r"\d+", s)
    if m:
        return s
    # fallback: return as-is
    return s

def build_topics_and_gold(questions: list[dict]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    gold_map = {}
    for i, q in enumerate(questions):
        qid = q.get("id") or q.get("qid") or str(i)
        qid = str(qid)
        query = q.get("body") or q.get("query") or q.get("question") or ""
        query = str(query).strip()

        # BioASQ "documents" are typically a list of URLs; normalize to PMIDs
        docs = q.get("documents") or []
        pmids = [normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]  # drop empties

        rows.append({"qid": qid, "query": query})
        gold_map[qid] = pmids

    topics_df = pd.DataFrame(rows)
    return topics_df, gold_map



# %%
def ap_bioasq(gold: set[str], ranked: list[str], k: int = 10) -> float:
    """Average precision @k."""
    if not gold:
        return 0.0
    hit = 0
    s = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in gold:
            hit += 1
            s += hit / i
    return s / len(gold)

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

def evaluate_run(
    gold_map: dict[str, list[str]],
    run_map: dict[str, list[str]],
    ks_recall=(50, 100, 200, 500, 2000, 5000),
    eps: float = 1e-8,
) -> tuple[dict, pd.DataFrame]:
    qids = list(gold_map.keys())
    perq = []

    ap10s, rr10s, succ10s = [], [], []
    recalls = {k: [] for k in ks_recall}

    for qid in qids:
        gold = set(map(str, gold_map.get(qid, [])))
        ranked = list(map(str, run_map.get(qid, [])))

        ap10 = ap_bioasq(gold, ranked, k=10)
        rr10 = rr_at_k(gold, ranked, k=10)
        succ10 = success_at_k(gold, ranked, k=10)

        ap10s.append(ap10)
        rr10s.append(rr10)
        succ10s.append(succ10)

        row = {"qid": qid, "AP@10": ap10, "RR@10": rr10, "Success@10": succ10}
        for k in ks_recall:
            r = recall_at_k(gold, ranked, k=k)
            recalls[k].append(r)
            row[f"R@{k}"] = r
        perq.append(row)

    # GMAP@10 uses geometric mean of AP@10 with epsilon smoothing (BioASQ style)
    gmap10 = math.exp(sum(math.log(max(eps, x)) for x in ap10s) / max(1, len(ap10s)))

    summary = {
        "MAP@10": float(np.mean(ap10s)) if ap10s else 0.0,
        "GMAP@10": float(gmap10) if ap10s else 0.0,
        "MRR@10": float(np.mean(rr10s)) if rr10s else 0.0,
        "Success@10": float(np.mean(succ10s)) if succ10s else 0.0,
    }
    for k in ks_recall:
        summary[f"MeanR@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0

    return summary, pd.DataFrame(perq)



# %%
def dense_retrieve_topics(
    topics_df: pd.DataFrame,
    topk: int,
    batch_size: int = 256,
    ef: int = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: qid, docno, score, rank
    docno is PMID.
    score is similarity-ish (we convert distance -> similarity for cosine)
    
    Args:
        ef: efSearch parameter for HNSW (higher = better recall but slower).
            Note: hnswlib effectively requires efSearch >= k (topk). If ef < topk,
            it will behave like ef=topk, so you won't see differences.
    """
    # Set ef on the index if provided
    if ef is not None:
        ef = int(ef)
        topk_i = int(topk)
        eff_ef = max(ef, topk_i)
        if eff_ef != ef:
            print(f"[warn] efSearch={ef} < k={topk_i}; clamping efSearch -> {eff_ef}")
        index.set_ef(eff_ef)
    
    qids = topics_df["qid"].astype(str).tolist()
    queries = topics_df["query"].astype(str).tolist()

    out_rows = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Dense retrieve", unit="batch"):
        batch_q = queries[i:i+batch_size]
        batch_qids = qids[i:i+batch_size]

        # Encode batch queries
        q_emb = model.encode(
            batch_q,
            batch_size=min(64, batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,  # must match how docs were embedded if using cosine
            show_progress_bar=False,
        ).astype(np.float32)

        # Query with the index's current ef setting
        labels, dists = index.knn_query(q_emb, k=topk)

        # Convert to rows
        for qi, qid in enumerate(batch_qids):
            # hnswlib cosine returns distance (smaller is better).
            # We convert to a "similarity" score = 1 - distance for convenience.
            for rank, (rid, dist) in enumerate(zip(labels[qi], dists[qi]), start=1):
                pmid = rowid_to_pmid[int(rid)]
                score = 1.0 - float(dist)
                out_rows.append((qid, pmid, score, rank))

    res = pd.DataFrame(out_rows, columns=["qid", "docno", "score", "rank"])
    return res


# %%
def results_df_to_run_map(res_df: pd.DataFrame) -> dict[str, list[str]]:
    # Ensure sorted by rank (it already is), then build run map
    run_map = {}
    for qid, grp in res_df.groupby("qid", sort=False):
        run_map[str(qid)] = grp.sort_values("rank")["docno"].astype(str).tolist()
    return run_map

def evaluate_dense_on_questions(questions: list[dict], label: str, topk: int = K_QUERY, ef: int = None) -> dict:
    topics_df, gold_map = build_topics_and_gold(questions)

    res = dense_retrieve_topics(topics_df, topk=topk, batch_size=256, ef=ef)

    # cut to K_CHECK_SUBSET for evaluation (should already be == topk)
    res = res[res["rank"] <= K_CHECK_SUBSET].copy()

    run_map = results_df_to_run_map(res)
    summary, _ = evaluate_run(
        gold_map,
        run_map,
        ks_recall=(50, 100, 200, 500, 2000, 5000),
        eps=1e-5,
    )
    return {"method": "Dense", "batch": label, **summary}


# %%
# ---- Run evaluations ----
all_dense_summaries = []

# Train subset
train_data = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
train_questions = train_data["questions"]
print("Train questions:", len(train_questions))

all_dense_summaries.append(evaluate_dense_on_questions(train_questions, "train_subset", topk=K_QUERY))

# Test batches
for fp in TEST_BATCHES:
    data = json.loads(fp.read_text(encoding="utf-8"))
    label = fp.stem
    print("Test batch:", label, "questions:", len(data["questions"]))
    all_dense_summaries.append(evaluate_dense_on_questions(data["questions"], label, topk=K_QUERY))

metrics_df_dense = pd.DataFrame(all_dense_summaries)
metrics_df_dense


# %%
# Try higher efSearch for better recall (slower queries)
print(f"K_QUERY={K_QUERY}. For hnswlib, efSearch should be >= k to have any effect.")

# These values are meaningful when k=5000
for ef in [K_QUERY, 2 * K_QUERY, 4 * K_QUERY]:
    print("\n== efSearch =", ef, "==")
    s = evaluate_dense_on_questions(train_questions, f"train_subset_ef{ef}", topk=K_QUERY, ef=ef)
    print({k: s[k] for k in ["MAP@10", "MRR@10", "MeanR@200", "MeanR@500", "MeanR@5000"]})

# %% [markdown]
# # save long table for hybird

# %%
# where to save per-query dense results for later hybrid tuning
DENSE_RUNS_DIR = Path("../output/eval_dense_MedEmbed")   # change if you want
DENSE_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# %%
def ensure_dense_schema(res_df: pd.DataFrame) -> pd.DataFrame:
    req = {"qid", "docno", "score", "rank"}
    missing = req - set(res_df.columns)
    if missing:
        raise ValueError(f"Missing columns in dense results: {missing}")

    out = res_df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out["rank"] = out["rank"].astype(int)
    out["score"] = out["score"].astype(float)
    out = out.sort_values(["qid", "rank"], ascending=[True, True]).reset_index(drop=True)
    return out

def save_dense_split(res_df: pd.DataFrame, split: str, meta: dict | None = None, save_run_map: bool = True):
    res_df = ensure_dense_schema(res_df)

    pq_path = DENSE_RUNS_DIR / f"dense_{split}.parquet"
    res_df.to_parquet(pq_path, index=False, compression="zstd")

    if save_run_map:
        run_map = results_df_to_run_map(res_df)  # you already defined this
        jm_path = DENSE_RUNS_DIR / f"dense_{split}_run_map.json"
        with open(jm_path, "w", encoding="utf-8") as f:
            json.dump(run_map, f)

    if meta is not None:
        meta_path = DENSE_RUNS_DIR / f"dense_{split}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("Saved:", pq_path)
    if save_run_map:
        print("Saved:", jm_path)
    if meta is not None:
        print("Saved:", meta_path)


# %%
dense_run_meta = {
    "model_name": model_name,
    "device": device,
    "max_seq_length": int(getattr(model, "max_seq_length", 0) or 0),
    "space": space,
    "dim": int(dim),
    "ef_search_default": int(ef_search),
    "topk_retrieved": int(K_QUERY),
    "k_eval": int(K_CHECK_SUBSET),
    "index_dir": str(DENSE_INDEX_DIR),
    "index_file": str(HNSW_INDEX_PATH),
    "rowid_map_file": str(ROWID_MAP_PATH),
}
dense_run_meta


# %%
def evaluate_and_save_dense_on_questions(
    questions: list[dict],
    label: str,
    topk: int = K_QUERY,
    ef: int | None = None,
    save: bool = True,
) -> dict:
    """
    Runs dense retrieval on questions, saves per-query results, evaluates, returns summary.
    """
    topics_df, gold_map = build_topics_and_gold(questions)

    # dense retrieval (this returns the long-form results DF)
    res_df = dense_retrieve_topics(topics_df, topk=topk, batch_size=256, ef=ef)

    # Keep consistent with evaluation cutoff
    res_df = res_df[res_df["rank"] <= K_CHECK_SUBSET].copy()

    if save:
        meta = {**dense_run_meta,
                "split": label,
                "n_queries": int(topics_df.shape[0]),
                "ef_search_used": int(ef) if ef is not None else int(index.get_ef()),
               }
        save_dense_split(res_df, split=label, meta=meta, save_run_map=True)

    run_map = results_df_to_run_map(res_df)
    summary, _ = evaluate_run(
        gold_map,
        run_map,
        ks_recall=(50, 100, 200, 500, 2000, 5000),
        eps=1e-5,
    )
    return {"method": "Dense", "batch": label, **summary}



# %%
def evaluate_and_save_dense_on_questions(
    questions: list[dict],
    label: str,
    topk: int = K_QUERY,
    ef: int | None = None,
    save: bool = True,
 ) -> dict:
    """
    Runs dense retrieval on questions, saves per-query results, evaluates, returns summary.
    """
    topics_df, gold_map = build_topics_and_gold(questions)

    # dense retrieval (this returns the long-form results DF)
    res_df = dense_retrieve_topics(topics_df, topk=topk, batch_size=256, ef=ef)

    # Keep consistent with evaluation cutoff
    res_df = res_df[res_df["rank"] <= K_CHECK_SUBSET].copy()

    if save:
        meta = {**dense_run_meta,
                "split": label,
                "n_queries": int(topics_df.shape[0]),
                "ef_search_used": int(ef) if ef is not None else int(ef_search),
               }
        save_dense_split(res_df, split=label, meta=meta, save_run_map=True)

    run_map = results_df_to_run_map(res_df)
    summary, _ = evaluate_run(
        gold_map,
        run_map,
        ks_recall=(50, 100, 200, 500, 2000, 5000),
        eps=1e-5,
    )
    return {"method": "Dense", "batch": label, **summary}



# %%
all_dense_summaries = []

# Train subset
train_data = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
train_questions = train_data["questions"]
print("Train questions:", len(train_questions))

all_dense_summaries.append(
    evaluate_and_save_dense_on_questions(train_questions, "train_subset", topk=K_QUERY, save=True)
)

# Test batches
for fp in TEST_BATCHES:
    data = json.loads(fp.read_text(encoding="utf-8"))
    label = fp.stem
    print("Test batch:", label, "questions:", len(data["questions"]))

    all_dense_summaries.append(
        evaluate_and_save_dense_on_questions(data["questions"], label, topk=K_QUERY, save=True)
    )

metrics_df_dense = pd.DataFrame(all_dense_summaries)
metrics_df_dense


# %%
