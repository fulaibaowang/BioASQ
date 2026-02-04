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
import os
import re
import numpy as np
import pandas as pd
import orjson
from tqdm import tqdm

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



# %%

# %%
