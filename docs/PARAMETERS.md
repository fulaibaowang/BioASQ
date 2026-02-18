# Tunable Parameters

Pipeline config (env vars and script mapping): [scripts/public/README.md](../scripts/public/README.md).

## BM25 + RM3

| Parameter | Range | Notes |
|-----------|-------|-------|
| `--k_feedback` | 30 – 100 | Feedback pool size for RM3 |
| `--rm3_fb_docs` | 10 – 50 | Feedback documents for term extraction |
| `--rm3_fb_terms` | 20 – 50 | Expanded terms to add |
| `--rm3_lambda` | 0.4 – 0.8 | Interpolation weight (0 = pure RM3, 1 = pure original) |

See [notebooks/bm25_test.ipynb](../notebooks/bm25_test.ipynb) for detailed notes.

## Dense Retrieval (HNSW)

| Parameter | Range | Notes |
|-----------|-------|-------|
| `--M` | 16 – 64 | HNSW graph degree |
| `--ef_construction` | 100 – 400 | HNSW build-time quality |
| `--ef_search` | 50 – 300 | HNSW query-time expansion |
| `--batch_size` | 64 – 256 | Embedding batch size |
| `--max_seq_length` | 256 – 1024 | Encoder truncation length |

## Dense Models Tested

- [abhinand/MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1)
- [sentence-transformers/pubmedbert](https://huggingface.co/sentence-transformers/pubmedbert)

See [notebooks/dense_test.ipynb](../notebooks/dense_test.ipynb) for detailed notes.

## Hybrid Reranking (RRF)

| Parameter | Range | Notes |
|-----------|-------|-------|
| `K_RRF` | 30 – 200 | RRF constant |
| BM25 weight | 0.5 – 3.0 | Weight multiplier for BM25 scores |
| Dense weight | 0.5 – 3.0 | Weight multiplier for dense scores |

See [notebooks/hybird.ipynb](../notebooks/hybird.ipynb) for the RRF grid search.

## Stage 2 Rerank (Cross-Encoder)

| Parameter | Range | Notes |
|-----------|-------|-------|
| `--candidate-limit` | 200 – 2000 | Stage-1 candidates per query to rerank |
| `--model` | Cross-encoder HF model | Default: `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `--model-device` | cpu / cuda / mps / auto | Device selection (auto picks best available) |
| `--model-batch` | 8 – 64 | Cross-encoder batch size |
| `--model-max-length` | 128 – 1024 | Cross-encoder token truncation length |
| `--adaptive-p` | 0.90 – 0.99 | Target recall ratio for adaptive K |
| `--adaptive-cap` | 100 – 500 | Max adaptive cutoff K |
| `--ks-recall` | 50 – 2000 | Recall K values (comma-separated) |

See [scripts/public/shared_scripts/rerank/rerank_stage2.py](../scripts/public/shared_scripts/rerank/rerank_stage2.py) for full arguments.
