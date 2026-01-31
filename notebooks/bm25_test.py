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
import re, json, math
import pandas as pd
import pyterrier as pt
import random
import matplotlib.pyplot as plt
import os
# -------------------------
# Load Terrier index
# -------------------------
if not pt.java.started():
    pt.java.init()

# %% [markdown]
# # testing with example

# %%
# index
INDEX_PROPERTIES = "../example/toy/index/pubmed_bm25_example/data.properties"
index = pt.IndexFactory.of(INDEX_PROPERTIES)

bm25 = pt.BatchRetrieve(index, wmodel="BM25")  # you can pass num_results later via slicing

# %%
# -------------------------
# Load BioASQ golden file
# -------------------------
GOLD_PATH = "../example/toy/example_query.json"  # adjust if needed

with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold = json.load(f)

questions = gold["questions"]

def url_to_pmid(url: str) -> str | None:
    m = re.search(r"pubmed/(\d+)", url)
    return m.group(1) if m else None

topics = []
gold_map = {}  # qid -> set(docno=pmid)
for q in questions:
    qid = str(q["id"])
    query = q["body"]
    topics.append({"qid": qid, "query": query})

    pmids = set()
    for u in q.get("documents", []):
        pmid = url_to_pmid(u)
        if pmid:
            pmids.add(pmid)
    gold_map[qid] = pmids

topics_df = pd.DataFrame(topics)

print("n_questions:", len(topics_df))
print("example q:", topics_df.iloc[0].to_dict())
print("example gold docnos:", list(gold_map[topics_df.iloc[0]["qid"]])[:5])

# %%
topics_df


# %%
# -------------------------
# BioASQ-style eval helpers (from your notebook)
# -------------------------
def ap_bioasq(ranked: list[str], relset: set[str], k: int = 10) -> float:
    ranked = ranked[:k]
    if not relset:
        return 0.0
    denom = min(len(relset), k)
    if denom == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, docno in enumerate(ranked, start=1):
        if docno in relset:
            hits += 1
            s += hits / i
    return s / denom

def rr_at_k(ranked: list[str], relset: set[str], k: int = 10) -> float:
    ranked = ranked[:k]
    for i, docno in enumerate(ranked, start=1):
        if docno in relset:
            return 1.0 / i
    return 0.0

def success_at_k(ranked: list[str], relset: set[str], k: int = 10) -> int:
    ranked = ranked[:k]
    return int(any(docno in relset for docno in ranked))

def recall_at_k(ranked: list[str], relset: set[str], k: int) -> float:
    ranked = ranked[:k]
    if not relset:
        return 0.0
    return len(set(ranked) & relset) / len(relset)

def evaluate_run(gold_map: dict[str, set[str]],
                 run_map: dict[str, list[str]],
                 ks_recall=(50, 100, 200, 500),
                 eps=1e-5):
    perq = []
    APs, RRs, S10s = [], [], []
    recalls = {K: [] for K in ks_recall}

    for qid, relset in gold_map.items():
        ranked = run_map.get(qid, [])

        ap = ap_bioasq(ranked, relset, k=10)
        rr = rr_at_k(ranked, relset, k=10)
        s10 = success_at_k(ranked, relset, k=10)

        APs.append(ap); RRs.append(rr); S10s.append(s10)

        rec_k_vals = {}
        for K in ks_recall:
            rK = recall_at_k(ranked, relset, k=K)
            recalls[K].append(rK)
            rec_k_vals[f"R@{K}"] = rK

        perq.append({
            "qid": qid,
            "n_gold": len(relset),
            "AP@10": ap,
            "RR@10": rr,
            "Success@10": s10,
            **rec_k_vals
        })

    MAP10 = sum(APs) / len(APs) if APs else 0.0
    GMAP10 = math.exp(sum(math.log(a + eps) for a in APs) / len(APs)) if APs else 0.0
    MRR10 = sum(RRs) / len(RRs) if RRs else 0.0
    Success10 = sum(S10s) / len(S10s) if S10s else 0.0
    RecallK = {K: (sum(vals) / len(vals) if vals else 0.0) for K, vals in recalls.items()}

    summary = {
        "MAP@10": MAP10,
        "GMAP@10": GMAP10,
        "MRR@10": MRR10,
        "Success@10": Success10,
        **{f"MeanR@{K}": v for K, v in RecallK.items()}
    }
    perq_df = pd.DataFrame(perq).sort_values("qid")
    return summary, perq_df


# %%
# -------------------------
# Run BM25 and build run_map
# -------------------------
K_MAX = 500  # overfetch for recall@K; eval AP@10 still uses top 10
res = bm25.transform(topics_df)

# sort by score desc within qid, add rank, cut at K_MAX
res = res.sort_values(["qid", "score"], ascending=[True, False])
res["rank"] = res.groupby("qid").cumcount() + 1
res = res[res["rank"] <= K_MAX].copy()

run_map = {qid: grp["docno"].astype(str).tolist()
           for qid, grp in res.groupby("qid", sort=False)}

summary, perq_df = evaluate_run(gold_map, run_map, ks_recall=(50,100,200,500), eps=1e-5)
print(summary)
perq_df.head()

# %% [markdown]
# # after indexing 2026 baseline
#
# for index we shall include articles that does not have abstracts

# %%
index = pt.IndexFactory.of("/Users/yun/develop/pubmed_bm25_index/data.properties")
coll = index.getCollectionStatistics()

print("Number of documents:", coll.getNumberOfDocuments())
print("Number of tokens:", coll.getNumberOfTokens())
print("Number of unique terms:", coll.getNumberOfUniqueTerms())
print("Avg doc length:", coll.getAverageDocumentLength())

# %% [markdown]
# # Evaluate with BioASQ Training13b data

# %%
# Load BioASQ training13b data
TRAIN_PATH = "/Users/yun/develop/BioASQ/BioASQ-training13b/training13b.json"

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_questions = train_data["questions"]

# Build topics and gold_map for training data
train_topics = []
train_gold_map = {}

for q in train_questions:
    qid = str(q["id"])
    query = q["body"]
    train_topics.append({"qid": qid, "query": query})

    pmids = set()
    for u in q.get("documents", []):
        pmid = url_to_pmid(u)
        if pmid:
            pmids.add(pmid)
    train_gold_map[qid] = pmids

train_topics_df = pd.DataFrame(train_topics)

print("n_questions:", len(train_topics_df))
print("example q:", train_topics_df.iloc[0].to_dict())
print("example gold docnos:", list(train_gold_map[train_topics_df.iloc[0]["qid"]])[:5])

# %%
# Run BM25 with full 2025 baseline index
bm25_full = pt.BatchRetrieve(index, wmodel="BM25")

K_MAX = 2000
train_res_raw = bm25_full.transform(train_topics_df)

# Sort and rank
train_res_raw = train_res_raw.sort_values(["qid", "score"], ascending=[True, False])
train_res_raw["rank"] = train_res_raw.groupby("qid").cumcount() + 1

# %%
train_res = train_res_raw[train_res_raw["rank"] <= K_MAX].copy()

train_run_map = {qid: grp["docno"].astype(str).tolist()
                 for qid, grp in train_res.groupby("qid", sort=False)}

# Evaluate
train_summary, train_perq_df = evaluate_run(train_gold_map, train_run_map, ks_recall=(50,100,200,500,2000), eps=1e-5)

print("Training13b Evaluation Results:")
print(train_summary)
train_perq_df.head()

# %%
out = "../tmp/train_bm25_k500_initial.parquet"
train_res.to_parquet(out, index=False)

# %%
pdf = train_perq_df

# 1) RR@10 distribution
plt.figure()
plt.hist(pdf["RR@10"], bins=50)
plt.title("Per-query RR@10 distribution")
plt.xlabel("RR@10")
plt.ylabel("Number of queries")
plt.show()

# 2) AP@10 distribution
plt.figure()
plt.hist(pdf["AP@10"], bins=50)
plt.title("Per-query AP@10 distribution")
plt.xlabel("AP@10")
plt.ylabel("Number of queries")
plt.show()

# 3) Success@10 distribution (0/1)
plt.figure()
plt.hist(pdf["Success@10"], bins=[-0.5, 0.5, 1.5])
plt.title("Per-query Success@10 distribution")
plt.xlabel("Success@10")
plt.ylabel("Number of queries")
plt.xticks([0, 1])
plt.show()

# %%
train_summary

# %%
# 4) Mean Recall@K bar chart
Ks = [50, 100, 200, 500,2000]
rec_vals = [train_summary.get(f"MeanR@{k}", 0.0) for k in Ks]

plt.figure()
plt.bar([str(k) for k in Ks], rec_vals)
plt.title("Mean Recall@K (BM25 candidate set coverage)")
plt.xlabel("K")
plt.ylabel("Mean Recall@K")
plt.ylim(0, 1)
plt.show()

# %%
# 5) Optional: per-query Recall@K distributions

col = f"R@2000"
if col in pdf.columns:
    plt.figure()
    plt.hist(pdf[col], bins=30)
    plt.title(f"Per-query {col} distribution")
    plt.xlabel(col)
    plt.ylabel("Number of queries")
    plt.ylim(bottom=0)
    plt.show()

# %% [markdown]
# # Build 10% Subset with Gold + Retrieved PMIDs

# %%
# Step 1: Collect all gold PMIDs from training data
all_gold_pmids = set()
for pmids in train_gold_map.values():
    all_gold_pmids.update(pmids)

print(f"Total gold PMIDs: {len(all_gold_pmids)}")

# Step 2: Randomly pick 10% of questions (with fixed seed)
random.seed(42)
all_qids = list(train_gold_map.keys())
sample_size = max(1, int(len(all_qids) * 0.1))
sampled_qids = set(random.sample(all_qids, sample_size))

print(f"Total questions: {len(all_qids)}")
print(f"Sampled questions (10%): {len(sampled_qids)}")

# Build sampled questions data for JSON export
sampled_questions = [q for q in train_questions if str(q["id"]) in sampled_qids]
sampled_data = {"questions": sampled_questions}

# Save to example folder
sample_json_path = "../example/training13b_10pct_sample.json"
with open(sample_json_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"Saved sampled questions to: {sample_json_path}")

# %%
# Step 3: Collect top N=2000 retrieved PMIDs for sampled questions
N_TOP = 2000

retrieved_pool_pmids = set()

# Filter train_res to only sampled questions and get top 2000 per question
sampled_res = train_res[train_res["qid"].isin(sampled_qids)].copy()

for qid in sampled_qids:
    qid_results = sampled_res[sampled_res["qid"] == qid].head(N_TOP)
    retrieved_pmids = qid_results["docno"].astype(str).tolist()
    retrieved_pool_pmids.update(retrieved_pmids)

print(f"Retrieved PMIDs from top {N_TOP} per question: {len(retrieved_pool_pmids)}")

# %%
# Step 4: Build subset PMIDs = gold ∪ retrieved
subset_pmids = all_gold_pmids | retrieved_pool_pmids

print(f"\nSubset Statistics:")
print(f"  Gold PMIDs: {len(all_gold_pmids)}")
print(f"  Retrieved PMIDs: {len(retrieved_pool_pmids)}")
print(f"  Union (subset): {len(subset_pmids)}")
print(f"  Overlap: {len(all_gold_pmids & retrieved_pool_pmids)}")

# Save subset PMIDs to file
subset_pmids_path = "../example/subset_pmids.txt"
with open(subset_pmids_path, "w") as f:
    for pmid in sorted(subset_pmids):
        f.write(f"{pmid}\n")

print(f"\nSaved subset PMIDs to: {subset_pmids_path}")

# %% [markdown]
# # check recall 0 ids

# %%
# Check if all gold PMIDs are in the index
meta_index = index.getMetaIndex()

gold_in_index = []
gold_not_in_index = []

for pmid in all_gold_pmids:
    try:
        docid = meta_index.getDocument("docno", pmid)
        if docid >= 0:
            gold_in_index.append(pmid)
        else:
            gold_not_in_index.append(pmid)
    except:
        gold_not_in_index.append(pmid)

print(f"Gold PMIDs Coverage:")
print(f"  Total gold PMIDs: {len(all_gold_pmids)}")
print(f"  Found in index: {len(gold_in_index)}")
print(f"  NOT in index: {len(gold_not_in_index)}")
print(f"  Coverage: {100 * len(gold_in_index) / len(all_gold_pmids):.2f}%")

if gold_not_in_index:
    print(f"\nFirst 10 missing gold PMIDs: {gold_not_in_index[:10]}")

# %%
# Check beginSection for missing gold PMIDs
from collections import Counter

# Build a map of PMID -> list of beginSections from snippets
pmid_to_sections = {}

for q in train_questions:
    for snippet in q.get("snippets", []):
        # Extract PMID from document URL
        doc_url = snippet.get("document", "")
        pmid = url_to_pmid(doc_url)
        if pmid:
            section = snippet.get("beginSection", "unknown")
            if pmid not in pmid_to_sections:
                pmid_to_sections[pmid] = []
            pmid_to_sections[pmid].append(section)

# Check beginSection for missing PMIDs
missing_sections = []
for pmid in gold_not_in_index[:100]:  # Check first 100 missing
    sections = pmid_to_sections.get(pmid, [])
    if sections:
        missing_sections.extend(sections)

section_counts = Counter(missing_sections)

print(f"\nbeginSection distribution for missing gold PMIDs:")
print(f"  Total missing PMIDs checked: {min(100, len(gold_not_in_index))}")
print(f"  Total snippets found: {len(missing_sections)}")
print(f"\nSection breakdown:")
for section, count in section_counts.most_common():
    print(f"  {section}: {count} ({100*count/len(missing_sections):.1f}%)")

# Check if any missing PMIDs have only title sections
missing_with_only_title = []
missing_with_abstract = []
missing_with_both = []

for pmid in gold_not_in_index[:100]:
    sections = set(pmid_to_sections.get(pmid, []))
    if sections:
        if sections == {"title"}:
            missing_with_only_title.append(pmid)
        elif "abstract" in sections:
            missing_with_abstract.append(pmid)
        elif sections and "abstract" not in sections:
            missing_with_both.append(pmid)

print(f"\nMissing PMIDs by section type:")
print(f"  Only title: {len(missing_with_only_title)}")
print(f"  Has abstract: {len(missing_with_abstract)}")
print(f"  Other (no abstract): {len(missing_with_both)}")

# %%
