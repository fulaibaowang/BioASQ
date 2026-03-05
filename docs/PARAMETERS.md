# Tunable Parameters

Pipeline config (env vars and script mapping): [scripts/public/README.md](../scripts/public/README.md).

## BM25 + RM3

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `fb_docs` | 5 – 20 | **20** | Feedback documents for term extraction |
| `fb_terms` | 10 – 30 | **30** | Expanded terms to add |
| `fb_lambda` | 0.6 – 0.8 | **0.6** | Interpolation weight (0 = pure RM3, 1 = pure original) |

**Decision:** Aggressive config (`fb_docs=20, fb_terms=30, fb_lambda=0.6`) chosen by optimising `0.5 × MeanR@200 + 0.5 × MeanR@500` across test batches. It achieved the highest combined recall (score 0.6709) while balanced configs had marginally higher MAP@10 but lower recall.

See [notebooks/bm25_test.ipynb](../notebooks/bm25_test.ipynb) for the RM3 parameter sweep.

## Dense Retrieval (HNSW)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `M` | — | **32** | HNSW graph degree (common value, not swept) |
| `ef_construction` | — | **200** | HNSW build-time quality (common value, not swept) |
| `ef_search` | 5000 – 20000 | **100** | HNSW query-time expansion (diminishing returns above 5000) |
| `batch_size` | — | **128** (index) / **256** (retrieval) | Embedding batch size (not swept) |
| `max_seq_length` | — | **512** | Encoder truncation length (~2.8% of docs truncate at 512) |
| `hnsw_space` | — | **cosine** | Distance metric |

**Decision:** Only `ef_search` was swept (5000/10000/20000); differences in MeanR@5000 were marginal (0.845 → 0.850). Other HNSW parameters use standard values from the literature.

See [notebooks/dense_test.ipynb](../notebooks/dense_test.ipynb) for details.

### Dense Models Tested

| Model | Result |
|-------|--------|
| [abhinand/MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1) | **Default** — used in production pipeline |
| [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | Worse hybrid recall than MedEmbed |

**Decision:** MedEmbed is the default dense model. PubMedBERT was tested in a separate hybrid pipeline and produced lower recall.

See [notebooks/hybrid_pubmedbert.ipynb](../notebooks/hybrid_pubmedbert.ipynb) for the model comparison.

## Hybrid Retrieval (RRF)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `K_RRF` | 30, 60, 100, 150, 200 | **150** | RRF constant in `1 / (k_rrf + rank)` |
| BM25 weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for BM25 RRF scores |
| Dense weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for Dense RRF scores |

**Decision:** Equal weights (`1.0 / 1.0`) with `K_RRF=150` selected via grid search (25 configs). Equal weights beat both BM25-heavy (2:1, 3:1) and Dense-heavy (1:2, 1:3) configurations. Best MeanR@2000 = 0.9012. `K_RRF` has a small effect — 60 and 200 are nearly identical to 150.

See [notebooks/hybrid.ipynb](../notebooks/hybrid.ipynb) for the RRF grid search.

## Stage 2 Rerank (Cross-Encoder)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `--candidate-limit` | — | **2000** | Stage-1 candidates per query to rerank |
| `--model` | MiniLM, BGE v2 | **BAAI/bge-reranker-v2-m3** | Cross-encoder model |
| `--model-device` | cpu / cuda / mps / auto | **auto** | Device selection |
| `--model-batch` | — | **16** | Cross-encoder batch size |
| `--model-max-length` | 200, 512 | **512** | Cross-encoder token truncation length |
| `--ks-recall` | — | 50,100,200,300,400,500,1000,2000,5000 | Recall K values for evaluation |

### Reranker Models Tested

| Model | max_length | MAP@10 (test avg) |
|-------|-----------|-------------------|
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 512 | 0.385 |
| BAAI/bge-reranker-v2-m3 | 200 | 0.351 |
| **BAAI/bge-reranker-v2-m3** | **512** | **0.418** |

**Decision:** `candidate-limit=2000` chosen from recall curves — the smallest K where hybrid recall reaches 95% of maximum (P=0.95). BGE v2 at `max_length=512` is the default reranker, outperforming MiniLM (MAP@10 0.418 vs 0.385) and BGE at `max_length=200` (0.351).

See [notebooks/analyze_results.ipynb](../notebooks/analyze_results.ipynb) for recall curves, reranker comparison, and BM25/Dense/Hybrid baselines.

### Post-Rerank Fusion

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `k_rrf` (fusion) | 30, 60 | **60** | RRF constant for BGE + Hybrid fusion |
| `w_bge` | 0.5 – 1.0 | **0.8** | BGE reranker weight |
| `w_hybrid` | 0.0 – 0.5 | **0.2** | Hybrid stage-1 weight |
| `pool_top_rerank` | 50, 100, 200 | **50** | Top-K from reranker for fusion pool |
| `pool_top_hybrid` | 20, 50, 100, 200 | **50** | Top-K from hybrid for fusion pool |

**Decision:** Reranker-dominant fusion (`w_bge=0.8, w_hybrid=0.2`) outperforms pure reranker output. Optimal for MAP@10: `k_rrf=30, pool_rerank=50, pool_hybrid=50`. Optimal for Recall@50: `k_rrf=60, pool_rerank=100, pool_hybrid=200`.

See [notebooks/analyze_workflow_results.ipynb](../notebooks/analyze_workflow_results.ipynb) for the fusion sweep.

## Answer Generation (LLM)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `--temperature` | 0.0, 0.3, 0.8 | **0.0** | LLM sampling temperature |
| `--top-p` | — | **1.0** | Nucleus sampling (no truncation) |
| `--max-contexts` | — | **10** | Cap on evidence passages per question |
| `--max-chars-per-context` | — | **1300** | Truncation length per context |
| Model | — | **llama3.3:latest** | Ollama model |

**Decision:** Temperature differences are marginal (F_MRR: 0.4994 at 0.3 vs 0.4988 at 0.8; R_SU4_Rec: 0.3478 at 0.8 vs 0.3474 at 0.3). Script default is `temperature=0.0` for deterministic output. Prompt tests are to be updated.

See [notebooks/generation_test.ipynb](../notebooks/generation_test.ipynb) for temperature and prompt tests.

## Retrieval Pipeline Summary

| Stage | Method | Key Defaults |
|-------|--------|-------------|
| 1a | BM25 + RM3 | `fb_docs=20, fb_terms=30, fb_lambda=0.6` |
| 1b | Dense (MedEmbed) | `M=32, ef_construction=200, ef_search=100` |
| 2 | Hybrid RRF | `K_RRF=150, weights=1.0/1.0` |
| 3 | Rerank (BGE v2) | `candidate_limit=2000, max_length=512` |
| 4 | Fusion (BGE + Hybrid) | `k_rrf=60, w_bge=0.8, w_hybrid=0.2` |
| 5 | Generation (Llama 3.3) | `temperature=0.0, max_contexts=10` |
