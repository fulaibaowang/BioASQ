# Results summary: workflow_baseline_full_run_both_routes

Metrics are taken from `output/workflow_baseline_full_run_both_routes/<method>/metrics.csv` for each method.  
Reported: **MAP@10**, **MeanR@10**, **MeanR@50** per split.

---

## Split: 13B1_golden

| Method            | MAP@10  | MeanR@10 | MeanR@50 |
|------------------|---------|----------|----------|
| Bm25             | 0.3371  | 0.3471   | 0.4610   |
| Dense            | 0.2492  | 0.3081   | 0.4457   |
| Hybrid           | 0.3690  | 0.3711   | 0.5231   |
| Rerank           | 0.4392  | 0.4121   | 0.5918   |
| Rerank_hybrid    | 0.4499  | 0.4164   | 0.5918   |
| Rerank_hybrid_200| 0.4568  | 0.4302   | 0.6068   |
| snippet_rerank   | 0.4079  | 0.4006   | 0.5830   |
| Snippet_rrf      | 0.4632  | 0.4252   | 0.6070   |

---

## Split: 13B2_golden

| Method            | MAP@10  | MeanR@10 | MeanR@50 |
|------------------|---------|----------|----------|
| Bm25             | 0.3260  | 0.3618   | 0.5310   |
| Dense            | 0.1846  | 0.2416   | 0.3341   |
| Hybrid           | 0.3393  | 0.3464   | 0.5277   |
| Rerank           | 0.4959  | 0.4397   | 0.6345   |
| Rerank_hybrid    | 0.4862  | 0.4376   | 0.6345   |
| Rerank_hybrid_200| 0.4917  | 0.4393   | 0.6428   |
| snippet_rerank   | 0.4912  | 0.4369   | 0.6287   |
| Snippet_rrf      | 0.4980  | 0.4430   | 0.6447   |

---

## Split: 13B3_golden

| Method            | MAP@10  | MeanR@10 | MeanR@50 |
|------------------|---------|----------|----------|
| Bm25             | 0.2916  | 0.2992   | 0.5035   |
| Dense            | 0.1705  | 0.2170   | 0.3674   |
| Hybrid           | 0.3249  | 0.3205   | 0.5346   |
| Rerank           | 0.3852  | 0.3478   | 0.6078   |
| Rerank_hybrid    | 0.4296  | 0.3653   | 0.6078   |
| Rerank_hybrid_200| 0.4276  | 0.3653   | 0.6227   |
| snippet_rerank   | 0.3756  | 0.3503   | 0.6080   |
| Snippet_rrf      | 0.4355  | 0.3676   | 0.6218   |

---

## Split: 13B4_golden

| Method            | MAP@10  | MeanR@10 | MeanR@50 |
|------------------|---------|----------|----------|
| Bm25             | 0.2194  | 0.1374   | 0.2942   |
| Dense            | 0.1304  | 0.0967   | 0.2112   |
| Hybrid           | 0.2586  | 0.1492   | 0.3209   |
| Rerank           | 0.3264  | 0.1773   | 0.4155   |
| Rerank_hybrid    | 0.3451  | 0.1763   | 0.4155   |
| Rerank_hybrid_200| 0.3465  | 0.1787   | 0.4276   |
| snippet_rerank   | 0.3141  | 0.1753   | 0.4147   |
| Snippet_rrf      | 0.3454  | 0.1763   | 0.4302   |

---

## Split: training14b_10pct_sample

| Method            | MAP@10  | MeanR@10 | MeanR@50 |
|------------------|---------|----------|----------|
| Bm25             | 0.2307  | 0.3019   | 0.4904   |
| Dense            | 0.1450  | 0.2174   | 0.3637   |
| Hybrid           | 0.2249  | 0.2907   | 0.5140   |
| Rerank           | 0.2770  | 0.3528   | 0.5832   |
| Rerank_hybrid    | 0.2838  | 0.3674   | 0.5832   |
| Rerank_hybrid_200| 0.2833  | 0.3657   | 0.5975   |
| snippet_rerank   | 0.2756  | 0.3454   | 0.5795   |
| Snippet_rrf      | 0.2829  | 0.3677   | 0.6009   |

---

## Source paths

| Method            | metrics path |
|------------------|--------------|
| Bm25             | `output/workflow_baseline_full_run_both_routes/bm25/metrics.csv` |
| Dense            | `output/workflow_baseline_full_run_both_routes/dense/metrics.csv` |
| Hybrid           | `output/workflow_baseline_full_run_both_routes/hybrid/metrics.csv` |
| Rerank           | `output/workflow_baseline_full_run_both_routes/rerank/metrics.csv` |
| Rerank_hybrid    | `output/workflow_baseline_full_run_both_routes/rerank_hybrid/metrics.csv` |
| Rerank_hybrid_200| `output/workflow_baseline_full_run_both_routes/rerank_hybrid_200/metrics.csv` |
| snippet_rerank   | `output/workflow_baseline_full_run_both_routes/snippet_rerank/metrics.csv` |
| Snippet_rrf      | `output/workflow_baseline_full_run_both_routes/snippet_rrf/metrics.csv` |
