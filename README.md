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

## Estimated resource requirements

Measured on the **Frida** HPC (SLURM + enroot, container `bioasq_08.03.26.sqfs`) against the
**PubMed 2026 baseline — 39,958,371 abstracts**. Numbers scale with corpus size, `TOP_K`,
`RERANK_CANDIDATE_LIMIT` and the generation backend.

### One-time: corpus and indexes

Built once, reused by every run.

| Artifact | Disk | Resources | Wall time |
|---|---|---|---|
| Parsed corpus `jsonl_2026/` (1,334 shards) | **54 GB** (17 GB gzipped) | CPU only | a few hours, parallel per XML file |
| BM25 / Terrier index (39.9 M docs, 6.5 M terms) | **14 GB** | 16 CPU, **96 GB RAM**, no GPU | **1 h 45 m** |
| Dense HNSW, 10 shards à ~4.0 M docs (`MedEmbed-small-v0.1`, 384-dim, M=32, efC=200) | **69 GB** (6.9 GB/shard) | 1 GPU + 16 CPU + 256 GB RAM **per shard** | **3 h 52 m/shard** (A100-80GB) or **6 h 44 m/shard** (L4), run as a 10-way array |
| Model weights (HF cache, default models) | ~3 GB | — | download once |
| Container image (`.sqfs` / `.sif`) | 10 GB | — | — |

**≈ 140 GB steady state** for corpus + both indexes; budget ~200 GB with the tarball and scratch.
Terrier indexing peaked at the full 96 GB it was given — do not go below that.

### Per run

| | Requested | Observed peak |
|---|---|---|
| GPU | 1 × A100 or L4 | fits in **24 GB**; the reranker is the largest resident model (568 M params, fp32) |
| CPU | 16 cores | — |
| **Host RAM** | 256 GB | **155–177 GB** |
| Wall clock | 12 h limit | 1–4 h, see below |
| Output per batch | — | 110–150 MB |

**Host RAM, not GPU, is the binding constraint.** `retrieve_dense.py` loads *all* HNSW shards into
RAM at once, so the 69 GB index plus Terrier, the candidate frames and the corpus scan land around
170 GB. Skip the dense route and this collapses to a few GB.

### Worked example: one 14b batch

Same config, same 80-question test set, only the GPU differs (`[timing]` lines in the job log):

| Step | 80 q, **L4** | 80 q, **A100-40GB** | 60 q, A100 (no generation) |
|---|---|---|---|
| 1 BM25 + RM3 | 280 s | 308 s | 284 s |
| 2 Dense HNSW (HyDE, 2 sub-runs) | 367 s | 454 s | 210 s |
| 3 Retrieval fusion | 30 s | 62 s | 26 s |
| **4 Cross-encoder rerank** | **8,521 s** | **3,727 s** | **2,907 s** |
| 5 Post-rerank RRF | 3 s | 3 s | 2 s |
| 6 Snippet window + CE rerank | 1,086 s | 708 s | 630 s |
| 7 Snippet/doc fusion | 1 s | 0 s | 1 s |
| Evidence build + generation | 4,930 s | 5,011 s | 635 s |
| **Total** | **4 h 14 m** | **2 h 52 m** | **1 h 19 m** |

- **Reranking dominates** and scales as `n_queries × RERANK_CANDIDATE_LIMIT` (60 q × 2,000 = 120 k
  query–document pairs). An A100 is **2.3× faster** than an L4 here; halving the candidate limit
  roughly halves the step.
- **Evidence build costs ~4.5 min per route** almost regardless of question count — it is a linear
  scan of the 54 GB corpus to pull a few hundred abstracts. Both routes ≈ 10 min.
- **Generation is LLM-bound, not pipeline-bound**: `llama3.3:70b` on a self-hosted Ollama at
  `concurrency=1` takes **20–23 s per question per route** (~45 min for 60 questions, both routes).
  A vLLM backend on newer hardware brought the same step down to ~5 min. The GPU in the table is
  idle during this step if the LLM is served elsewhere.

### Running smaller

| Goal | Change | Cost (MAP@10, mean over 5 splits) |
|---|---|---|
| Drop 69 GB disk and ~140 GB RAM | BM25 only: `STAGE1_SOURCE=bm25`, no dense index | **−0.014** (0.399 → 0.385) — the cheapest saving by far |
| Halve wall clock | `RERANK_CANDIDATE_LIMIT=1000`, `TOP_K=1000` | some recall at deep cutoffs |
| No GPU | `--no-rerank` | **−0.096** (0.399 → 0.303) |
| Just try it | `example/` — toy PubMed XMLs + 50-question sample | runs on a laptop |

## Results

Some results: [docs/RESULTS.md](docs/RESULTS.md).

## Environment

Container image and Python pins live only under the vendored [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main) tree: [Dockerfile](scripts/public/shared_scripts/Dockerfile), [requirements-docker-pytorch.txt](scripts/public/shared_scripts/requirements-docker-pytorch.txt), [requirements-docker.txt](scripts/public/shared_scripts/requirements-docker.txt). Build from the repo root: `docker build -t bioasq-pipeline -f scripts/public/shared_scripts/Dockerfile scripts/public/shared_scripts` (see [docs/USAGE.md](docs/USAGE.md)).

