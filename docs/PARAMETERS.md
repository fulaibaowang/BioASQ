# Parameter Tuning Guide

## BM25 + RM3

### RM3 Query Expansion Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--k_feedback` | 50 | 30 – 100 | Documents used for RM3 feedback pool; more coverage but may add noise |
| `--rm3_fb_docs` | 20 | 10 – 50 | How many top-ranked documents to use for term extraction |
| `--rm3_fb_terms` | 30 | 20 – 50 | How many expanded terms to add to query |
| `--rm3_lambda` | 0.6 | 0.4 – 0.8 | Query interpolation weight (0 = pure RM3, 1 = pure original query) |

### Rationale

**`--rm3_lambda` (interpolation weight)**
- Higher λ (e.g., 0.8) keeps original query dominant → **higher precision**, lower recall
- Lower λ (e.g., 0.4) emphasizes expanded terms → **broader recall**, may add false positives
- **Current choice (0.6):** Good balance for BioASQ; leans toward precision

**`--rm3_fb_docs` and `--rm3_fb_terms`**
- Default (20, 30) balances expansion breadth vs. noise
- For **recall-heavy** tasks: increase both (e.g., 50, 50)
- For **precision-heavy** tasks: decrease (e.g., 10, 20)

**`--k_feedback`**
- Must be ≥ `--rm3_fb_docs` to have enough candidates
- Default (50) is reasonable; going much higher adds marginal gain

### Example Configurations

**Precision-focused:**
```bash
--rm3_lambda 0.75 --rm3_fb_docs 10 --rm3_fb_terms 20 --k_feedback 40
```

**Recall-focused:**
```bash
--rm3_lambda 0.45 --rm3_fb_docs 40 --rm3_fb_terms 40 --k_feedback 80
```

**Balanced (current default):**
```bash
--rm3_lambda 0.6 --rm3_fb_docs 20 --rm3_fb_terms 30 --k_feedback 50
```

---

## Dense Retrieval (HNSW)

### Index Building Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--M` | 32 | 16 – 64 | Graph degree; controls connectivity (higher = better recall, more memory) |
| `--ef_construction` | 200 | 100 – 400 | Quality during index build (higher = slower, better recall for fixed M) |
| `--ef_search` | 100 | 50 – 300 | Query-time expansion (higher = slower, better recall) |
| `--batch_size` | 128 | 64 – 256 | Embedding batch size (GPU-dependent) |
| `--max_seq_length` | 512 | 256 – 1024 | Token limit per document before truncation |

### Rationale

**`--M` (graph degree)**
- Smaller M (16 – 24): Faster search, lower memory, slightly lower recall
- Larger M (48 – 64): Better recall, more memory, slower search
- **Current choice (32):** Standard compromise for million-scale indexes

**`--ef_construction` vs. `--ef_search`**
- `ef_construction`: Build-time cost (one-time); directly affects index quality
- `ef_search`: Query-time (runtime); adjustable after build
- **Current choices:** Standard for balanced quality/speed

**`--max_seq_length`**
- Longer texts are truncated; balance between coverage and model capacity
- **512 tokens:** High enough for titles + most abstracts; standard for MedEmbed

### Index Building vs. Querying

```bash
# Build (slow, one-time)
python build_dense_hnsw_index_from_jsonl_shards.py \
  --M 32 --ef_construction 200  # → index quality is fixed

# Query (can adjust at eval time)
python eval_dense.py \
  --ef_search 100              # → use 100 at query time (can override others)
  --ef_cap 200                 # → but cap it at 200 for speed
```

---

## Model Selection (Dense)

### Models Tested

| Model | Dim | Speed | Quality | Remarks |
|-------|-----|-------|---------|---------|
| `abhinand/MedEmbed-small-v0.1` | 384 | ★★★★★ | ★★★★☆ | **Default**: small, fast, domain-aware |
| `allenai/specter` | 768 | ★★★☆☆ | ★★★★★ | Citation embeddings; slower |
| `sentence-transformers/pubmedbert` | 768 | ★★★☆☆ | ★★★★☆ | Biomedical BERT-based |
| `sentence-transformers/all-mpnet-base-v2` | 768 | ★★☆☆☆ | ★★★☆☆ | General-purpose; slower |

### Recommendation

**For BioASQ:** `abhinand/MedEmbed-small-v0.1` (default) offers the best speed-quality tradeoff for biomedical literature. Switching models requires rebuilding the index.

---

## Hybrid Reranking (RRF)

### Reciprocal Rank Fusion Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `K_RRF` | 30 – 200 | RRF constant (higher = smoother ranking, lower recall gap emphasis) |
| BM25 weight | 0.5 – 3.0 | Relative weight in final score (>1 boosts BM25) |
| Dense weight | 0.5 – 3.0 | Relative weight in final score (>1 boosts dense) |

### RRF Score Formula

For doc $d$ at rank $r$ in BM25 and $d'$ in dense:

$$\text{RRF\_score}(d) = \sum_i \frac{w_i}{K + \text{rank}_i(d)}$$

where:
- $w_i \in \{w_{\text{BM25}}, w_{\text{dense}}\}$ = weight for method $i$
- $K$ = RRF constant
- Higher $K$ → ranks become less important (all docs valued more equally)
- Lower $K$ → top-ranked docs dominate

### Rationale

**`K_RRF` (constant)**
- `K ≈ 60`: BM25 top ranks matter significantly
- `K ≈ 150`: Smooths ranking; more emphasis on method agreement
- **Current choice (100–150):** Balanced; allows both methods to contribute meaningfully

**Weight Ratio (BM25 / Dense)**
- `(2.0, 1.0)`: BM25-heavy; good for high-precision recall@5000
- `(1.0, 2.0)`: Dense-heavy; better coverage for diverse queries
- `(1.0, 1.0)`: Equal; reasonable default
- **Suggested approach:** Grid search over test batches; final choice depends on metric priority (e.g., MAP@10 vs. Recall@5000)

### Example Hybrid Configurations

**BM25-focused (precision):**
```python
K_RRF = 60
WEIGHTS = [(2.0, 1.0), (3.0, 1.0)]
```

**Balanced:**
```python
K_RRF = 100
WEIGHTS = [(1.0, 1.0), (1.5, 1.5)]
```

**Dense-focused (recall):**
```python
K_RRF = 150
WEIGHTS = [(1.0, 2.0), (1.0, 3.0)]
```

See [notebooks/hybird.ipynb](../notebooks/hybird.ipynb) for full grid search over these parameters.

---

## Summary: Current Chosen Defaults

| Component | Setting | Rationale |
|-----------|---------|-----------|
| **BM25+RM3** | λ=0.6, fb_docs=20, fb_terms=30 | Balanced precision/recall on BioASQ test sets |
| **Dense** | MedEmbed-small, M=32, ef_construction=200, ef_search=100 | Speed-quality tradeoff; domain-specific embeddings |
| **HNSW** | L2/cosine after normalization | Standard for normalized embeddings |
| **Hybrid** | RRF K=100, equal weights (1:1) | Grid search shows good balance across metrics |

To reproduce current reported results, use [docs/USAGE.md](USAGE.md) command examples with these parameter defaults.
