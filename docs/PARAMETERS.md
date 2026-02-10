# Tunable Parameters

## BM25 + RM3

| Parameter | Range | Notes |
|-----------|-------|-------|
| `--k_feedback` | 30 – 100 | Feedback pool size for RM3 |
| `--rm3_fb_docs` | 10 – 50 | Feedback documents for term extraction |
| `--rm3_fb_terms` | 20 – 50 | Expanded terms to add |
| `--rm3_lambda` | 0.4 – 0.8 | Interpolation weight (0 = pure RM3, 1 = pure original) |

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

See [notebooks/dense_test.ipynb](../notebooks/dense_test.ipynb) for model comparison notes.

## Hybrid Reranking (RRF)

| Parameter | Range | Notes |
|-----------|-------|-------|
| `K_RRF` | 30 – 200 | RRF constant |
| BM25 weight | 0.5 – 3.0 | Weight multiplier for BM25 scores |
| Dense weight | 0.5 – 3.0 | Weight multiplier for dense scores |

See [notebooks/hybird.ipynb](../notebooks/hybird.ipynb) for the RRF grid search.
