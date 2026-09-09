# BioASQ 14b (2026): retrieval, reranking, and generation

This repository supports **BioASQ Task 14b** (2026) **Phase A/A+**: a **RAG-style** stack that retrieves biomedical literature, reranks with a cross-encoder and optionally snippet windows, and generates answers from evidence.

**Where to read next**

- [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md) as backbone:
  - there you can read **Pipeline overview and flowchart** as well as detailed commands and parameters setting
- **indexes, and BioASQ-oriented commands, e.g input and output conversion for BioASQ format** [docs/USAGE.md](docs/USAGE.md)


At a high level: **hybrid retrieval** (BM25 + RM3 and dense HNSW with fusion), **document reranking** and post-rerank fusion, an **optional snippet ranking and snippet-document-fusion** branch for snippet-style evidence, and **LLM generation** with **different prompts per `query_type`** (factoid, list, yesno, summary). 

## Paper

This repository accompanies our BioASQ Task 14b (2026) working note:

> Yun Wang. *A Multistage Evidence Retrieval System for BioASQ Task 14b: Hybrid Retrieval, Reranking, and Snippet Selection.* CLEF 2026 Working Notes, BioASQ Lab. \
> **[[PDF]](https://clef-staging.pages.dev/paper49.pdf)** &nbsp;·&nbsp; archived code: [release v0.1.0](https://github.com/fulaibaowang/BioASQ/releases/tag/v0.1.0)

### Prompts and schemas

All LLM prompts used by the pipeline are in the repository:

- **Query parsing, normalization, and the per-question HyDE decision/generation** — [`scripts/public/query_parsing/prompt.md`](scripts/public/query_parsing/prompt.md) (background: [`MULTI_QUERY_HYDE.md`](scripts/public/query_parsing/MULTI_QUERY_HYDE.md)). This step sets a per-question `hyde_enabled` flag and disables HyDE for numeric/measurement, exact-identifier, and other highly specific-target questions.
- **Answer generation** — [`scripts/public/shared_scripts/prompts/system.txt`](scripts/public/shared_scripts/prompts/system.txt) and [`scripts/public/shared_scripts/prompts/user_base.txt`](scripts/public/shared_scripts/prompts/user_base.txt).
- **Per-`query_type` answer schemas** — [`scripts/public/prompts/schemas/`](scripts/public/prompts/schemas/) (`factoid`, `list`, `yesno`, `summary`, `default`).
- **Query-rewriting ablation (Variants A/B)** — [`notebooks/archive/oneoff_query_rewrite_llm.py`](notebooks/archive/oneoff_query_rewrite_llm.py).

## Usage (Docker)

Build the image, build indexes, and run the orchestrator with a config file: [docs/USAGE.md](docs/USAGE.md).

## Results

Some results: [docs/RESULTS.md](docs/RESULTS.md).

## Environment

Container image and Python pins live only under the vendored [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main) tree: [Dockerfile](scripts/public/shared_scripts/Dockerfile), [requirements-docker-pytorch.txt](scripts/public/shared_scripts/requirements-docker-pytorch.txt), [requirements-docker.txt](scripts/public/shared_scripts/requirements-docker.txt). Build from the repo root: `docker build -t bioasq-pipeline -f scripts/public/shared_scripts/Dockerfile scripts/public/shared_scripts` (see [docs/USAGE.md](docs/USAGE.md)).

## Estimated resource requirements

Measured on a SLURM cluster with one GPU per job, indexing the **PubMed 2026 baseline
(39,958,371 abstracts)**. Treat them as a rough guide — everything scales with corpus size,
`TOP_K`, `RERANK_CANDIDATE_LIMIT` and your generation backend.

### Corpus and indexes (one-time)

Parsing the PubMed baseline XML gives ~**54 GB** of JSONL. From that, the BM25 (Terrier) index is
~**14 GB** and takes ~2 h on 16 CPU cores with 96 GB RAM, no GPU. The dense HNSW index
(`MedEmbed-small-v0.1`, 384-dim) is ~**69 GB**, built as 10 shards of ~4 M documents at roughly
4 h per shard on one A100-80GB — run them as a job array. Add ~10 GB for the container image and
~3 GB for model weights.

**Budget ~150 GB of disk**, built once and reused by every run.

### Per run

| | Typical request | Observed peak |
|---|---|---|
| GPU | 1 | ~5 GB VRAM (see below) |
| CPU | 16 cores | — |
| RAM | 256 GB | **155–177 GB** |
| Wall clock | — | 1–4 h per batch |

**RAM is the binding constraint, not the GPU** — dense retrieval holds all HNSW shards in memory at
once. Running BM25-only (`STAGE1_SOURCE=bm25`) drops this to a few GB and costs ~0.014 MAP@10 after
reranking, which is the cheapest saving available here.

**VRAM depends on which reranker you choose.** The default `bge-reranker-v2-m3` peaks at ~5 GB at
batch 64 / max_length 512, so an 8 GB card is enough for the whole pipeline; the snippet models are
smaller. LLM-based rerankers are a different class — `bge-reranker-v2-gemma` alone is ~9.4 GB of
fp32 weights before activations — so measure before switching, or lower `RERANK_MODEL_BATCH`.
A faster GPU buys wall time, not headroom: an A100 reranks ~2.3× faster than an L4.

### Worked example: one batch of 80 questions, both routes

`TOP_K=5000`, `RERANK_CANDIDATE_LIMIT=2000`, one A100-40GB, generation on a self-hosted
`llama3.3:70b`:

| Stage | Time |
|---|---|
| BM25 + dense retrieval + fusion | ~14 min |
| Cross-encoder rerank | ~62 min |
| Snippet windows + rerank + fusion | ~12 min |
| Evidence build + answer generation | ~84 min |
| **Total** | **~2 h 50 m** |

Reranking scales as `n_queries × RERANK_CANDIDATE_LIMIT`, so halving the candidate limit roughly
halves that step. Generation is bound by the LLM backend, not the pipeline (~20 s per question per
route here); a faster served backend cuts it to minutes. Output is ~150 MB per batch.
