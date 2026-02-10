# BioASQ: retrieval + reranking experiments

This repo collects data prep, retrieval, and evaluation code for BioASQ Phase-A style document retrieval.

## Methods

### First Stage Retrieval

- **BM25 + RM3** ([script](scripts/public/retrieval/eval_bm25_rm3_bioasq.py), [notebook](notebooks/bm25_test.ipynb))
  - Keyword retrieval with RM3 query expansion
  - Tunable: `--rm3_fb_docs`, `--rm3_fb_terms`, `--rm3_lambda`
  - See [docs/PARAMETERS.md](docs/PARAMETERS.md) for ranges

- **Dense Retrieval** ([script](scripts/public/retrieval/eval_dense.py), [notebook](notebooks/dense_test.ipynb))
  - SentenceTransformer embeddings + HNSW index
  - Models: MedEmbed-small-v0.1 (default), PubMedBERT
  - Tunable: `--ef_search`, `--ef_cap`, embedding model
  - See [docs/PARAMETERS.md](docs/PARAMETERS.md)

### Hybrid / Reranking
- **Hybrid RRF** ([notebook](notebooks/hybird.ipynb))
  - Reciprocal Rank Fusion combining BM25 and dense
  - Tuning knobs: `K_RRF`, BM25/dense weight ratio
  - See [docs/PARAMETERS.md](docs/PARAMETERS.md)

## Results (sample, BioASQ 13B)

| Method | MAP@10 | MRR@10 | Success@10 | MeanR@5000 |
|--------|--------|--------|------------|-----------|
| BM25_RM3 | 0.285 | 0.524 | 0.824 | 0.891 |
| Dense    | 0.218 | 0.410 | 0.710 | 0.848 |
| Hybrid* | TBD | TBD | TBD | TBD |

*See [notebooks/hybird.ipynb](notebooks/hybird.ipynb) for hybrid grid search results and analysis.  
Full results with per-batch breakdown: [docs/RESULTS.md](docs/RESULTS.md)

## Quick Start

See [docs/USAGE.md](docs/USAGE.md) for detailed setup and evaluation commands.

## Environment

- Python 3.10+
- PyTerrier (BM25 indexing & retrieval)
- SentenceTransformers + HNSW (dense retrieval)
- PyArrow (outputs)

Install: `pip install -r requirements.txt`  
Or: `pip install pyterrier sentence-transformers hnswlib pyarrow pandas numpy scipy`
