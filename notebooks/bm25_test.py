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


# %%
def url_to_pmid(url: str) -> str | None:
    m = re.search(r"pubmed/(\d+)", url)
    return m.group(1) if m else None


# %%
os.chdir('/Users/yun/develop/BioASQ/notebooks')

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
# for index we shall include also articles that does not have abstracts, because BIOASQ uses also titles

# %%
index = pt.IndexFactory.of("/Users/yun/develop/pubmed_bm25_2026_index/data.properties")
coll = index.getCollectionStatistics()

print("Number of documents:", coll.getNumberOfDocuments())
print("Number of tokens:", coll.getNumberOfTokens())
print("Number of unique terms:", coll.getNumberOfUniqueTerms())
print("Avg doc length:", coll.getAverageDocumentLength())

# %% [markdown]
# #### Evaluate with BioASQ Training14b data

# %%
# Load BioASQ training14b data
TRAIN_PATH = "../BioASQ-training14b/trainining14b.json"

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
K_MAX = 10_000
bm25_full = pt.BatchRetrieve(index, wmodel="BM25", num_results=K_MAX)

# %%
train_res_raw = bm25_full.transform(train_topics_df)
# Sort and rank
train_res_raw = train_res_raw.sort_values(["qid", "score"], ascending=[True, False])
train_res_raw["rank"] = train_res_raw.groupby("qid").cumcount() + 1

# %%
# Check how many results you're actually getting with K_MAX=2000
print(f"train_res shape: {train_res_raw.shape}")
print(f"Unique queries: {train_res_raw['qid'].nunique()}")
print(f"Avg results per query: {len(train_res_raw) / train_res_raw['qid'].nunique()}")

# %%
K_CHECK = 10_000
train_res = train_res_raw[train_res_raw["rank"] <= K_CHECK].copy()

train_run_map = {qid: grp["docno"].astype(str).tolist()
                 for qid, grp in train_res.groupby("qid", sort=False)}

# Evaluate
train_summary, train_perq_df = evaluate_run(train_gold_map, train_run_map, ks_recall=(50,100,200,500,2000,5000,10_000), eps=1e-5)

print("Training14b Evaluation Results:")
print(train_summary)
train_perq_df.head()

# %%
# Check how many results you're actually getting with K_MAX=2000
print(f"train_res shape: {train_res.shape}")
print(f"Unique queries: {train_res['qid'].nunique()}")
print(f"Avg results per query: {len(train_res) / train_res['qid'].nunique()}")

# %%
out = "../tmp/train_bm25_k10000_initial.parquet"
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
Ks = [50, 100, 200, 500,2000,5000,10_000]
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
#  'MeanR@2000': 0.8720740787086406,
#  'MeanR@5000': 0.9160551444706584,
#  'MeanR@10000': 0.9398105257177296}
#
# let us 
# - check those zeros for recall@5000
# - build a subset with
#  - for  10% Subset with Gold retreieve top 5000
#  - gold pmids
# - use this subset we test our system furthur

# %% [markdown]
# # check zero-recall

# %% [markdown]
# ## quick check if gold pmids are in the pubmed index at all

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

# %% [markdown]
# those ids are fine, it seems there are either gene books (not in FTP) or deleted paper

# %% [markdown]
# ## extract the zero-recall questions

# %%
zero_recall_df = train_perq_df[train_perq_df[f"R@{K_CHECK}"] == 0].copy()
print("Zero-recall @10000 count:", len(zero_recall_df))
zero_recall_df.head(1)

# %%
# Attach the question body + gold PMIDs for those qids

qid_to_body = {str(q["id"]): q["body"] for q in train_questions}

# Add body + a compact gold pmid list column for quick viewing
zero_recall_df["body"] = zero_recall_df["qid"].map(qid_to_body)
zero_recall_df["gold_pmids"] = zero_recall_df["qid"].apply(lambda qid: sorted(train_gold_map.get(qid, [])))

# View a few (truncate gold list in print to avoid huge output)
for _, row in zero_recall_df.head(20).iterrows():
    qid = row["qid"]
    print("\n---")
    print("qid:", qid)
    print("n_gold:", row["n_gold"])
    print("body:", row["body"])
    print("gold_pmids (first 20):", row["gold_pmids"][:30])

# %%
dbg_topics = pd.DataFrame([
    {"qid":"t1", "query":"CRT0066101"},
    {"qid":"t3", "query":"What are the 3 types of immunoglobulin heavy chain containing antibodies found in human breast milk"},
])

dbg = bm25_full.transform(dbg_topics).sort_values(["qid","score"], ascending=[True, False])
for qid in ["t1","t3"]:
    print(qid, dbg[dbg.qid==qid].head(5)[["docno","score"]].to_string(index=False))


# %% [markdown]
# **I saw several issues here**
#
# - RM3 / PRF: eravacycline, miR
# - hyphen issue LB-100, CPX351(query rewrite)
# - CRT0066101 been discarded (tokenize issue)
# - words like "how, what are noisy" (query rewrite)
# - broad topic e,g Alzheimer

# %% [markdown]
# # Build 10% Subset with Gold + zero recall ids + Retrieved PMIDs top 5000

# %%
# Step 1: Collect all gold PMIDs from training data
all_gold_pmids = set()
for pmids in train_gold_map.values():
    all_gold_pmids.update(pmids)

print(f"Total gold PMIDs: {len(all_gold_pmids)}")

# %%
# Step 2: Randomly pick 10% of questions (with fixed seed)
random.seed(42)
all_qids = list(train_gold_map.keys())
sample_size = max(1, int(len(all_qids) * 0.1))
sampled_qids = set(random.sample(all_qids, sample_size))

print(f"Total questions: {len(all_qids)}")
print(f"Sampled questions (10%): {len(sampled_qids)}")

# %%
# Step 3: add zero recall qids and test batch qids

# Load test batch qids
test_batch_qids_path = "../Task13BGoldenEnriched/test_batch_qids.txt"
test_batch_qids = set()
with open(test_batch_qids_path, "r", encoding="utf-8") as f:
    for line in f:
        qid = line.strip()
        if qid:
            test_batch_qids.add(qid)
            
sampled_qids_expanded = set(zero_recall_df['qid']) | sampled_qids | test_batch_qids
print(f"Sampled questions plus zero recall qids: {len(sampled_qids_expanded)}")

# %%
# Build sampled questions data for JSON export
sampled_questions = [q for q in train_questions if str(q["id"]) in sampled_qids_expanded]
sampled_data = {"questions": sampled_questions}

# Save to example folder
sample_json_path = "../example/training14b_10pct_sample.json"
with open(sample_json_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"Saved sampled questions to: {sample_json_path}")

# %%
# Step 4: Collect top N=5000 retrieved PMIDs for sampled questions
N_TOP = 5000

retrieved_pool_pmids = set()

# Filter train_res to only sampled questions and get top 2000 per question
sampled_res = train_res[train_res["qid"].isin(sampled_qids_expanded)].copy()

for qid in sampled_qids_expanded:
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
# # rebuild bm25 index

# %% [markdown]
# python3 data/extract_jsonl_subset_by_pmids.py  --jsonl_glob "../biolab/pubmed/jsonl_2026/*.jsonl" --pmid_list "example/subset_pmids.txt" --output_jsonl "output/subset_pubmed.jsonl" --dedup --stop_when_complete 
#
# python data/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/subset_pubmed.jsonl"   --index_path "/work/output/pubmed_bm25_2026_subset_index"   --threads 4   --overwrite
#
# in this step, build_bm25_index_from_jsonl_shards.py was changed slightly for handle digitals longer than 4 (cases like crt0066101)

# %% [markdown]
# # our new baseline on subset

# %%
index = pt.IndexFactory.of("../output/pubmed_bm25_2026_subset_index/data.properties")
coll = index.getCollectionStatistics()

print("Number of documents:", coll.getNumberOfDocuments())
print("Number of tokens:", coll.getNumberOfTokens())
print("Number of unique terms:", coll.getNumberOfUniqueTerms())
print("Avg doc length:", coll.getAverageDocumentLength())

# %%
# Define the regex and helper for augmentation (used in both query and index)
CODE_RE = re.compile(r"\b([A-Za-z]{2,12})\s*[-–-]?\s*(\d{2,})\b")

def chunk_digits(d: str, k: int = 4) -> list[str]:
    # chunk into <=4 digits; keeps leading zeros
    return [d[i:i+k] for i in range(0, len(d), k)]


# %%
def augment_text_for_codes(text: str) -> str:
    extras = []
    for pfx, digits in CODE_RE.findall(text):
        p = pfx.lower()
        # always keep prefix as a token (often survives even when digits are dropped)
        extras.append(p)

        if len(digits) >= 5:
            # critical: create <=4-digit chunks so Terrier won't discard them
            chunks = chunk_digits(digits, 4)
            extras.extend(chunks)                 # e.g. 0066101 -> 0066 101
            extras.append(p + " " + " ".join(chunks))
        else:
            # for short digits (2-4), variants usually survive
            extras.append(digits)
            extras.append(f"{p}{digits}")
            extras.append(f"{p}-{digits}")
            extras.append(f"{p} {digits}")

    if extras:
        # append extras as additional terms (index-only hints)
        return text + "\n\n" + " ".join(sorted(set(extras)))
    return text


# %%
# Load subset questions from saved file (independent of earlier cells)
SUBSET_PATH = "../example/training14b_10pct_sample.json"

with open(SUBSET_PATH, "r", encoding="utf-8") as f:
    subset_data = json.load(f)

subset_questions = subset_data["questions"]

# Build topics and gold_map from loaded data
subset_topics = []
subset_gold_map = {}

for q in subset_questions:
    qid = str(q["id"])
    query = q["body"]
    subset_topics.append({"qid": qid, "query": query})
    
    pmids = set()
    for u in q.get("documents", []):
        pmid = url_to_pmid(u)
        if pmid:
            pmids.add(pmid)
    subset_gold_map[qid] = pmids

subset_topics_df = pd.DataFrame(subset_topics)

print(f"Subset evaluation questions: {len(subset_topics_df)}")
print(f"Example query: {subset_topics_df.iloc[0]['query'][:100]}")

# %% [markdown]
# **small fix**
#
# we need to seprate train_qids and test_batches_qids here for stats

# %%
subset_topics_df_trainset = subset_topics_df[
    ~subset_topics_df["qid"].isin(test_batch_qids)
].reset_index(drop=True)

# %%
subset_topics_df_trainset.shape

# %%
# Run BM25 with query augmentation on subset index
K_MAX = 5_000
bm25_subset = pt.BatchRetrieve(index, wmodel="BM25", num_results=K_MAX)

# Apply query augmentation to subset topics
qe_subset = pt.apply.query(lambda r: augment_text_for_codes(r["query"]))
pipe_subset = qe_subset >> bm25_subset

subset_res_raw = pipe_subset.transform(subset_topics_df_trainset)

# Sort and rank
subset_res_raw = subset_res_raw.sort_values(["qid", "score"], ascending=[True, False])
subset_res_raw["rank"] = subset_res_raw.groupby("qid").cumcount() + 1

print(f"subset_res shape: {subset_res_raw.shape}")
print(f"Unique queries: {subset_res_raw['qid'].nunique()}")
print(f"Avg results per query: {len(subset_res_raw) / subset_res_raw['qid'].nunique():.1f}")

# %%
# Evaluate on subset with K up to 5000
K_CHECK_SUBSET = 5_000
subset_res = subset_res_raw[subset_res_raw["rank"] <= K_CHECK_SUBSET].copy()

subset_run_map = {qid: grp["docno"].astype(str).tolist()
                  for qid, grp in subset_res.groupby("qid", sort=False)}

# Evaluate with recall up to 5000
subset_summary, subset_perq_df = evaluate_run(
    subset_gold_map, 
    subset_run_map, 
    ks_recall=(50, 100, 200, 500, 2000, 5000), 
    eps=1e-5
)

print("Subset Evaluation Results (with query augmentation):")
print(subset_summary)
subset_perq_df.head(2)

# %%
# Extract zero-recall questions from subset
subset_zero_recall_df = subset_perq_df[subset_perq_df[f"R@{K_CHECK_SUBSET}"] == 0].copy()
print(f"Zero-recall @{K_CHECK_SUBSET} count: {len(subset_zero_recall_df)}")
subset_zero_recall_df.head(1)

# %%
# Attach question body + gold PMIDs for zero-recall questions
subset_qid_to_body = {str(q["id"]): q["body"] for q in subset_questions}

subset_zero_recall_df["body"] = subset_zero_recall_df["qid"].map(subset_qid_to_body)
subset_zero_recall_df["gold_pmids"] = subset_zero_recall_df["qid"].apply(
    lambda qid: sorted(subset_gold_map.get(qid, []))
)

# Print examples of zero-recall questions
print(f"\n{'='*80}")
print(f"ZERO-RECALL QUESTIONS (n={len(subset_zero_recall_df)})")
print(f"{'='*80}\n")

for _, row in subset_zero_recall_df.head(10).iterrows():
    qid = row["qid"]
    print(f"\n{'-'*80}")
    print(f"QID: {qid}")
    print(f"Gold docs: {row['n_gold']}")
    print(f"Query: {row['body']}")
    print(f"Gold PMIDs (first 30): {row['gold_pmids'][:30]}")
    
if len(subset_zero_recall_df) == 0:
    print("✓ No zero-recall questions! All queries found at least one relevant doc in top 5000.")

# %%
# Plot distributions for subset evaluation
pdf_subset = subset_perq_df

# 1) RR@10 distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.hist(pdf_subset["RR@10"], bins=30)
plt.title("Per-query RR@10 (subset)")
plt.xlabel("RR@10")
plt.ylabel("Count")

# 2) AP@10 distribution
plt.subplot(1, 3, 2)
plt.hist(pdf_subset["AP@10"], bins=30)
plt.title("Per-query AP@10 (subset)")
plt.xlabel("AP@10")
plt.ylabel("Count")

# 3) Success@10 distribution
plt.subplot(1, 3, 3)
plt.hist(pdf_subset["Success@10"], bins=[-0.5, 0.5, 1.5])
plt.title("Success@10 (subset)")
plt.xlabel("Success@10")
plt.ylabel("Count")
plt.xticks([0, 1])

plt.tight_layout()
plt.show()

# %%
# Mean Recall@K bar chart for subset
Ks_subset = [50, 100, 200, 500, 2000, 5000]
rec_vals_subset = [subset_summary.get(f"MeanR@{k}", 0.0) for k in Ks_subset]

plt.figure()
plt.bar([str(k) for k in Ks_subset], rec_vals_subset)
plt.title("Mean Recall@K (BM25 subset with query augmentation)")
plt.xlabel("K")
plt.ylabel("Mean Recall@K")
plt.ylim(0, 1)
for i, v in enumerate(rec_vals_subset):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
plt.show()

# %%
# Quick check for specific qid CRT0066101
target_qid = "5c83ff91617e120c34000005"

# Get gold PMIDs
gold_pmids = sorted(subset_gold_map.get(target_qid, set()))
print(f"QID: {target_qid}")
print(f"Gold PMIDs ({len(gold_pmids)}): {gold_pmids}\n")

# Get query
query_text = subset_topics_df_trainset[subset_topics_df_trainset["qid"] == target_qid]["query"].values
if len(query_text) > 0:
    print(f"Query: {query_text[0]}\n")

# Get retrieved results
retrieved = subset_res[subset_res["qid"] == target_qid].sort_values("rank")

# Get metrics
metrics = subset_perq_df[subset_perq_df["qid"] == target_qid]
metrics

# %%
# Evaluate BM25 (with query augmentation) on test batches + merge metrics
from pathlib import Path

TEST_DIR = Path("../Task13BGoldenEnriched")
TEST_BATCHES = [
    TEST_DIR / "13B1_golden.json",
    TEST_DIR / "13B2_golden.json",
    TEST_DIR / "13B3_golden.json",
    TEST_DIR / "13B4_golden.json",
]


def build_topics_and_gold(questions: list[dict]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    topics = []
    gold_map = {}
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
    return pd.DataFrame(topics), gold_map


def evaluate_bm25_on_questions(questions: list[dict], label: str) -> dict:
    topics_df, gold_map = build_topics_and_gold(questions)

    # apply augmentation
    qe = pt.apply.query(lambda r: augment_text_for_codes(r["query"]))
    pipe = qe >> bm25_subset
    res_raw = pipe.transform(topics_df)

    # sort + rank + cut
    res_raw = res_raw.sort_values(["qid", "score"], ascending=[True, False])
    res_raw["rank"] = res_raw.groupby("qid").cumcount() + 1
    res = res_raw[res_raw["rank"] <= K_CHECK_SUBSET].copy()

    run_map = {
        qid: grp["docno"].astype(str).tolist()
        for qid, grp in res.groupby("qid", sort=False)
    }

    summary, _ = evaluate_run(
        gold_map,
        run_map,
        ks_recall=(50, 100, 200, 500, 2000, 5000),
        eps=1e-5,
    )
    summary = {"batch": label, **summary}
    return summary


# Build merged metrics table (includes train subset_summary)
all_summaries = []

# Train subset summary already computed above
all_summaries.append({"batch": "train_subset", **subset_summary})

# Test batches
for fp in TEST_BATCHES:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    batch_label = fp.stem  # e.g., 13B1_golden
    all_summaries.append(evaluate_bm25_on_questions(data["questions"], batch_label))

metrics_df = pd.DataFrame(all_summaries)
metrics_df

# %%

# %%

# %%
