# Results

## BM25 + RM3

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@500 | MeanR@5000 |
|-------|--------|---------|--------|------------|----------|-----------|-----------|-----------|
| train_subset (n=583) | 0.241 | 0.007 | 0.428 | 0.666 | 0.509 | 0.595 | 0.767 | 0.899 |
| 13B1_golden (n=85) | 0.351 | 0.028 | 0.547 | 0.788 | 0.481 | 0.566 | 0.745 | 0.924 |
| 13B2_golden (n=85) | 0.345 | 0.062 | 0.567 | 0.871 | 0.532 | 0.588 | 0.756 | 0.917 |
| 13B3_golden (n=85) | 0.303 | 0.057 | 0.550 | 0.871 | 0.493 | 0.585 | 0.773 | 0.938 |
| 13B4_golden (n=85) | 0.219 | 0.030 | 0.539 | 0.835 | 0.303 | 0.395 | 0.619 | 0.876 |
| **Test avg (13B1–4)** | **0.305** | **0.044** | **0.551** | **0.841** | **0.452** | **0.536** | **0.723** | **0.914** |

## Dense Retrieval (MedEmbed-small-v0.1)

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@1000 | MeanR@5000 |
|-------|--------|---------|--------|------------|----------|-----------|------------|-----------|
| train_subset (n=923) | 0.176 | 0.003 | 0.314 | 0.596 | 0.384 | 0.469 | 0.726 | 0.848 |
| 13B1_golden (n=85) | 0.275 | 0.013 | 0.442 | 0.729 | 0.486 | 0.578 | 0.851 | 0.949 |
| 13B2_golden (n=85) | 0.194 | 0.004 | 0.323 | 0.635 | 0.387 | 0.474 | 0.694 | 0.819 |
| 13B3_golden (n=85) | 0.205 | 0.013 | 0.404 | 0.753 | 0.411 | 0.510 | 0.785 | 0.898 |
| 13B4_golden (n=85) | 0.154 | 0.007 | 0.369 | 0.706 | 0.242 | 0.340 | 0.727 | 0.899 |
| **Test avg (13B1–4)** | **0.207** | **0.009** | **0.384** | **0.706** | **0.403** | **0.476** | **0.764** | **0.891** |

## Hybrid (BM25 + Dense RRF, k_rrf=150, w=1.0:1.0)

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@200 | MeanR@300 | MeanR@400 | MeanR@2000 | MeanR@5000 | K_rec |
|-------|--------|---------|--------|------------|----------|-----------|-----------|-----------|-----------|------------|------------|-------|
| train_subset (n=923) | 0.232 | 0.008 | 0.432 | 0.686 | 0.527 | 0.618 | 0.694 | 0.740 | 0.769 | 0.894 | 0.933 | 2000 |
| 13B1_golden (n=85) | 0.393 | 0.088 | 0.640 | 0.894 | 0.536 | 0.654 | 0.719 | 0.767 | 0.794 | 0.929 | 0.968 | 2000 |
| 13B2_golden (n=85) | 0.371 | 0.091 | 0.675 | 0.894 | 0.540 | 0.637 | 0.704 | 0.742 | 0.773 | 0.904 | 0.958 | 2001 |
| 13B3_golden (n=85) | 0.332 | 0.114 | 0.611 | 0.929 | 0.546 | 0.648 | 0.746 | 0.791 | 0.817 | 0.927 | 0.967 | 2000 |
| 13B4_golden (n=85) | 0.273 | 0.058 | 0.617 | 0.882 | 0.343 | 0.448 | 0.558 | 0.615 | 0.660 | 0.872 | 0.944 | 2001 |
| **Test avg (13B1–4)** | **0.342** | **0.088** | **0.636** | **0.900** | **0.491** | **0.597** | **0.682** | **0.729** | **0.761** | **0.908** | **0.959** | **2001** |

---

**About BioASQ Snippets:**

BioASQ Phase A evaluates **document retrieval** (returning ranked PMIDs). Your current pipeline (BM25 + Dense + Reranker) produces document rankings, which is what the metrics above measure.

**Snippets** are for Phase B (question answering):
- Participants extract relevant text passages from retrieved documents
- Typically done after Phase A using the top-ranked documents
- Can use extractive QA models, sliding windows, or sentence ranking on document text
- Your reranker outputs (top 200-300 docs) would feed into a snippet extraction step

To generate snippets, you'd need an additional pipeline stage that:
1. Takes your reranked top-K documents
2. Retrieves full text for each PMID
3. Extracts/ranks sentences or passages relevant to the query
4. Returns formatted snippets with offsets

Your current scripts evaluate Phase A only (document retrieval metrics).

## Stage 2 Rerank (cross-encoder/ms-marco-MiniLM-L-12-v2)

| Split | MAP@10 | GMAP@10 | MRR@10 | Success@10 | MeanR@50 | MeanR@100 | MeanR@200 | MeanR@300 | MeanR@500 | MeanR@1000 | MeanR@2000 | MeanR@Krec | MeanKeff@Krec |
|-------|--------|---------|--------|------------|----------|-----------|-----------|-----------|-----------|------------|------------|-----------|---------------|
| train_subset (n=923) | 0.296 | 0.025 | 0.552 | 0.780 | 0.511 | 0.606 | 0.692 | 0.738 | 0.790 | 0.842 | 0.870 | 0.738 | 186.5 |
| 13B1_golden (n=85) | 0.431 | 0.145 | 0.735 | 0.929 | 0.585 | 0.675 | 0.757 | 0.802 | 0.867 | 0.907 | 0.927 | 0.801 | 199.6 |
| 13B2_golden (n=85) | 0.459 | 0.180 | 0.733 | 0.941 | 0.591 | 0.658 | 0.743 | 0.781 | 0.826 | 0.875 | 0.901 | 0.780 | 185.1 |
| 13B3_golden (n=85) | 0.357 | 0.149 | 0.689 | 0.953 | 0.542 | 0.634 | 0.735 | 0.787 | 0.843 | 0.893 | 0.926 | 0.785 | 211.1 |
| 13B4_golden (n=85) | 0.293 | 0.101 | 0.660 | 0.929 | 0.383 | 0.522 | 0.620 | 0.682 | 0.754 | 0.835 | 0.873 | 0.681 | 263.9 |
| **Test avg (13B1–4)** | **0.385** | **0.144** | **0.704** | **0.938** | **0.525** | **0.622** | **0.714** | **0.763** | **0.823** | **0.878** | **0.907** | **0.762** | **215.0** |
