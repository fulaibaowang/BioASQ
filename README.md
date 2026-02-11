# BioASQ: retrieval + reranking experiments

This repo collects data prep, retrieval, and evaluation code for BioASQ Phase-A style document retrieval.

## Plan and Goals

- Stage 1 retrieval: combine retrievers (BM25+RM3, dense, hybrid) , fetch ~500-2000 docs per query, .
- Stage 2 retrieval: cross-encoder model, fetch ~50-200 to keep recall while narrowing candidates.
- Stage 3 reranking: focus on precision at top ranks (MAP@10, MRR@10).
- Metrics: use MeanR@K for stages 1-2 and MAP@10/MRR@10 for stage 3 (BioASQ official metrics).

## Methods

### First Stage Retrieval

- **BM25 + RM3** ([script](scripts/public/retrieval/eval_bm25_rm3_bioasq.py), [notebook](notebooks/bm25_test.ipynb))
  - Keyword retrieval with RM3 query expansion
  - Tunable: `--rm3_fb_docs`, `--rm3_fb_terms`, `--rm3_lambda`

- **Dense Retrieval** ([script](scripts/public/retrieval/eval_dense.py), [notebook](notebooks/dense_test.ipynb))
  - SentenceTransformer embeddings + HNSW index
  - Models: MedEmbed-small-v0.1 (default), PubMedBERT
  - Tunable: `--ef_search`, `--ef_cap`, embedding model

### Hybrid / Reranking
- **Hybrid RRF** ([notebook](notebooks/hybird.ipynb))
  - Reciprocal Rank Fusion combining BM25 and dense
  - Tuning knobs: `K_RRF`, BM25/dense weight ratio

Tunable parameter ranges: [docs/PARAMETERS.md](docs/PARAMETERS.md)

## Results

Recall metrics for stages 1-2. Full tables are in [docs/RESULTS.md](docs/RESULTS.md).

| Method | MeanR@2000 (13B test avg) |
|--------|---------------------------|
| BM25_RM3 | 0.856 |
| Dense (MedEmbed)    | 0.787 |
| Hybrid | 0.907 |

Hybrid details: [notebooks/hybird.ipynb](notebooks/hybird.ipynb)

## Detailed commands

See [docs/USAGE.md](docs/USAGE.md) for detailed setup and evaluation commands.

## Environment

 `pip install pyterrier sentence-transformers hnswlib pyarrow pandas numpy scipy`

## TODO

- in production we don't know the recall so adaptive eval method needs to be changed in hybird and reranker
