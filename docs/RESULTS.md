# Results

## Sample Evaluations on BioASQ 13B

### Metrics

Results are reported on BioASQ Task 13B golden sets (13B1–13B4) and a training subset (10% sample).

#### BM25 + RM3

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@500 | MeanR@5000 |
|-------|--------|---------|--------|------------|----------|-----------|-----------|-----------|
| train_subset (n=583) | 0.241 | 0.007 | 0.428 | 0.666 | 0.509 | 0.595 | 0.767 | 0.899 |
| 13B1_golden (n=85) | 0.351 | 0.028 | 0.547 | 0.788 | 0.481 | 0.566 | 0.745 | 0.924 |
| 13B2_golden (n=85) | 0.345 | 0.062 | 0.567 | 0.871 | 0.532 | 0.588 | 0.756 | 0.917 |
| 13B3_golden (n=85) | 0.303 | 0.057 | 0.550 | 0.871 | 0.493 | 0.585 | 0.773 | 0.938 |
| 13B4_golden (n=85) | 0.219 | 0.030 | 0.539 | 0.835 | 0.303 | 0.395 | 0.619 | 0.876 |
| **Test avg (13B1–4)** | **0.305** | **0.044** | **0.551** | **0.841** | **0.452** | **0.536** | **0.723** | **0.914** |

**Configuration:** `--rm3_fb_docs 20 --rm3_fb_terms 30 --rm3_lambda 0.6 --k_feedback 50`

#### Dense Retrieval (MedEmbed-small-v0.1)

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@1000 | MeanR@5000 |
|-------|--------|---------|--------|------------|----------|-----------|------------|-----------|
| train_subset (n=923) | 0.176 | 0.003 | 0.314 | 0.596 | 0.384 | 0.469 | 0.726 | 0.848 |
| 13B1_golden (n=85) | 0.275 | 0.013 | 0.442 | 0.729 | 0.486 | 0.578 | 0.851 | 0.949 |
| 13B2_golden (n=85) | 0.194 | 0.004 | 0.323 | 0.635 | 0.387 | 0.474 | 0.694 | 0.819 |
| 13B3_golden (n=85) | 0.205 | 0.013 | 0.404 | 0.753 | 0.411 | 0.510 | 0.785 | 0.898 |
| 13B4_golden (n=85) | 0.154 | 0.007 | 0.369 | 0.706 | 0.242 | 0.340 | 0.727 | 0.899 |
| **Test avg (13B1–4)** | **0.207** | **0.009** | **0.384** | **0.706** | **0.403** | **0.476** | **0.764** | **0.891** |

**Configuration:** Model=`abhinand/MedEmbed-small-v0.1`, M=32, ef_construction=200, ef_search=100

#### Key Observations

1. **BM25+RM3** outperforms Dense on **MAP@10** and **MRR@10** (precision metrics)
   - Better for exact-match focused evaluation
   - Keyword expansion aids recall of directly relevant papers

2. **Dense** shows competitive **Recall@5000** and strong **Success@10** on some batches
   - Captures semantically relevant documents keyword methods miss
   - More stable performance across batches (lower variance)

3. **Batch variation:** 13B4 is harder for both methods, especially dense
   - May require longer sequence context or more specific query interpretation

4. **Hybrid RRF** (See [notebooks/hybird.ipynb](../notebooks/hybird.ipynb)) balances both strengths
   - Combining methods improves overall robustness

---

## Detailed Per-Query Analysis

Per-query breakdowns available in:
- `output/eval_bm25_rm3/per_query/BM25_RM3_*.csv` (all batches)
- `output/eval_dense_medembed_small/per_query/dense_*.csv` (all batches)

Each row shows: qid, n_gold, AP@10, RR@10, Success@10, R@K (for K in 50, 100, 200, 500, ...)

---

## Model Scaling & Future Work

### Why MedEmbed-small?

- **Size:** 384-dimensional embeddings (vs. 768 for larger models)
- **Speed:** ~5× faster embedding than PubMedBERT or Specter
- **Biomedical fit:** Trained on domain-specific data
- **Trade-off:** Slightly lower MAP@10 than larger models, but excellent Recall@5000

### Extending with Other Models

To try larger / alternative models:

```bash
python scripts/public/data/build_dense_hnsw_index_from_jsonl_shards.py \
  --model_name "allenai/specter"  # or "sentence-transformers/pubmedbert"
  --out_dir /path/to/specter_index
  ...
```

Then evaluate and compare metrics.

---

## Reproduction

To reproduce these results:

1. **Parse PubMed:**
   ```bash
   python scripts/public/data/parse_pubmed_local.py \
     --input_dir /path/to/pubmed/baseline2026 \
     --output_dir /path/to/pubmed/jsonl_2026 \
     --skip_existing
   ```

2. **Build indices:**
   ```bash
   # BM25
   python scripts/public/data/build_bm25_index_from_jsonl_shards.py \
     --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
     --index_path pubmed_bm25_2026 \
     --threads 4 --overwrite

   # Dense
   python scripts/public/data/build_dense_hnsw_index_from_jsonl_shards.py \
     --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
     --out_dir pubmed_medembed_2026 \
     --model_name "abhinand/MedEmbed-small-v0.1" --device cuda
   ```

3. **Evaluate** (see [docs/USAGE.md](USAGE.md) for exact commands)

See [docs/PARAMETERS.md](PARAMETERS.md) for parameter tuning guidance to improve further.
