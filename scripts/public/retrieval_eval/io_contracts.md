# Run IO contract

All retrieval stages (BM25, Dense, Hybrid, Reranker) use the same run format for interchange.

## Run format (TSV only)

- **Columns:** `qid`, `docno`, `rank`, `score`
- **Encoding:** UTF-8, tab-separated, no header required but recommended
- **Semantics:** One row per (query, document) pair; `docno` is the document ID (e.g. PMID); `rank` is 1-based; `score` may be NaN where not used

No parquet or JSON for run output. Per-query evaluation CSVs (when ground truth is present) use the same run plus columns such as `AP@10`, `RR@10`, `R@50`, `R@100`, etc., using the shared `RECALL_KS` grid from `common.py`.

## File layout (workflow)

- **BM25:** `{out_dir}/runs/{method}__{split}__top{k}.tsv`
- **Dense:** `{out_dir}/runs/dense_{split}.tsv`
- **Hybrid:** `{out_dir}/runs/best_rrf_{split}_top{k}.tsv`
- **Reranker:** `{out_dir}/runs/{name}.tsv`

Hybrid reads Dense runs from `{dense_root}/runs/dense_{split}.tsv` (or falls back to `{dense_root}/dense_{split}.parquet` for backward compatibility).
