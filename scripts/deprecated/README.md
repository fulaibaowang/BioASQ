# Deprecated / experimental scripts

These scripts are **not used in production**. They are kept for reference or one-off comparisons.

- **`rerank_guard_rail_topk.py`** — Guard-rail fusion: BGE top-m + Hybrid anchors for a fixed top-k prefix. Superseded by RRF fusion (Hybrid + Rerank) in the main pipeline.
- **`rerank_stage3_sentence.py`** — Stage-3 reranking with CE on (query, title + sentence) and max/top2mean pooling.
- **`rerank_stage3_merged.py`** — Stage-3 variant: one CE call per doc on (query, title + top-3 sentences merged).

(Result-dir comparison and plotting is in shared scripts: `scripts/public/shared_scripts/compare_result_dirs.py`.)

Internal experiments showed that these sentence-level reranking variants did not improve over stage-2 (full-document CE); they are preserved here in case the idea is revisited (e.g. with differently trained cross-encoders). See [docs/RESULTS.md](../../docs/RESULTS.md) for a short note.

Run from repo root. They resolve `scripts/public/shared_scripts` and `scripts/public/shared_scripts/rerank` for imports.
