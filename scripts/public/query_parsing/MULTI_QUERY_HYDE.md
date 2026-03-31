# Multi-Query Fields & HyDE

## Overview

The pipeline supports running retrieval and reranking with **multiple query variants** per stage, then fusing their results with weighted Reciprocal Rank Fusion (RRF). The primary use case is **HyDE (Hypothetical Document Embeddings)**: an LLM rewrites each question into an abstract-like passage that better matches the vocabulary of relevant papers. Running both the original `body` and the HyDE-rewritten `body_hyde` as query variants and fusing their results improves dense retrieval recall.

In testing, `DENSE_QUERY_FIELD=body,body_hyde` showed meaningful improvement in dense retrieval metrics over `body` alone.

## HyDE Query Preparation

### Script

The canonical HyDE preparation script is at:

```
scripts/public/query_parsing/prepare_hyde_query.py
```

It reads a `*.hyde_ready.json` file (output from an LLM HyDE generation step) and flattens the HyDE text into a top-level `body_hyde` field on each question. The LLM prompt template is alongside at `scripts/public/query_parsing/hyde_prompt.md`.

### Usage

```bash
python scripts/public/query_parsing/prepare_hyde_query.py \
  input.hyde_ready.json \
  -o output.json \
  --no-fallback
```

**`--no-fallback`** is recommended for multi-query fusion: when HyDE did not produce a different text (gating disabled, or LLM returned empty), the field is set to `null` instead of falling back to `body`. This avoids running identical queries twice through retrieval/reranking (see *Smart Deduplication* below).

### Input format

The script supports two JSON shapes inside each question object:

- **New shape** (`query_parse`): `{"query_parse": {"hyde_enabled": true, "hyde_text": "..."}}`
- **Legacy shape** (`hyde`): `{"hyde": {"enabled": true, "hyde_text": "..."}}`

### Experimental version

The full version with facet and listwise support remains at `scripts/private_scripts/hpc/retrieval_test/hyde/prepare_hyde_query.py` as an experimental reference.

## Multi-Query Configuration

Set comma-separated fields in the `*_QUERY_FIELD` environment variables. Each field names a key in the question JSON (e.g., `body`, `body_hyde`).

| Variable | Applies to | Example |
|----------|-----------|---------|
| `BM25_QUERY_FIELD` | BM25 retrieval | `body` |
| `DENSE_QUERY_FIELD` | Dense retrieval | `body,body_hyde` |
| `RERANK_QUERY_FIELD` | Cross-encoder reranking + snippet reranking | `body` |
| `LISTWISE_QUERY_FIELD` | Listwise reranker (single field only) | `body` |

When a `*_QUERY_FIELD` contains a single value (or is unset), the pipeline behaves exactly as before. When comma-separated, it:

1. Runs one sub-run per field under `_sub_<field>/` directories.
2. Fuses the sub-runs with weighted RRF into the canonical output directory (e.g., `dense/runs/`).
3. Generates `metrics.csv` and comparison eval plots for the fused result.
4. Downstream stages consume the fused output transparently.

### Weighting modes

Three weighting modes, in priority order:

| Mode | Config | Behavior |
|------|--------|----------|
| **Fixed weights** | `*_QUERY_FUSION_WEIGHTS=0.4,0.6` | Explicit per-field weights (order matches field order) |
| **Adaptive body weight** | `*_QUERY_BODY_WEIGHT=0.5` | Body gets fixed share; remainder split equally among active non-body fields per query |
| **Equal (default)** | Neither set | Each field gets `1/N` |

Fixed weights take precedence over adaptive body weight. Adaptive body weight is useful when different questions have different numbers of active non-body fields (e.g., variable facets).

### RRF k parameter

Override with `*_QUERY_FUSION_K_RRF` (default: 60).

### Example config

```bash
DENSE_QUERY_FIELD=body,body_hyde
# Uses default equal weights (0.5, 0.5) and k_rrf=60

# Or with explicit weights:
# DENSE_QUERY_FUSION_WEIGHTS=0.4,0.6

# Or with adaptive body weight (useful for variable facets):
# RERANK_QUERY_FIELD=body,body_facet1,body_facet2,body_facet3
# RERANK_QUERY_BODY_WEIGHT=0.5
```

## Smart Deduplication

When using `--no-fallback` in `prepare_hyde_query.py`, questions where HyDE did not produce a different text get `body_hyde = null`. The pipeline handles this efficiently:

1. Sub-runs for `body_hyde` pass `--skip-empty-query-field`, which skips questions with null/empty `body_hyde` in retrieval and reranking.
2. The `body` sub-run processes all questions as usual.
3. RRF fusion handles the asymmetry: for questions where only `body` has results, the fused output equals the `body` result; for questions where both have results, they are fused normally.

This avoids redundant computation (running the same query text twice through expensive reranking) while producing the same final result.

### Warnings

- **Multi-field sub-runs**: When `--skip-empty-query-field` skips questions, a prominent `[WARNING]` is printed with the count and percentage of skipped questions.
- **Single-field usage**: If only one field is configured (e.g., `DENSE_QUERY_FIELD=body_hyde`) and some questions have empty values, the pipeline raises an error rather than silently skipping. This prevents accidental data loss.

## Evaluation and Plots

When multi-query fusion runs, `multi_query_fuse.py` generates:

- **`metrics.csv`**: Standard retrieval metrics (MAP@K, MRR, Recall@K) for the fused result.
- **Comparison plot** (on the "different queries" subset): Shows per-field curves alongside the fused curve, evaluated only on queries where at least one non-body field was active. This enables fair comparison of what multi-query fusion actually changes.
- **All-queries plot**: The fused result evaluated on all queries, including those where only `body` contributed.

## Resume Behavior

The pipeline checks for existing sub-run outputs before running. If one sub-run is already complete and another is not, only the incomplete sub-run is executed on re-run. The fusion step re-runs whenever any sub-run is newer or the fused output does not exist.

## Files

| File | Purpose |
|------|---------|
| `scripts/public/query_parsing/prepare_hyde_query.py` | HyDE query preparation (canonical) |
| `scripts/public/query_parsing/hyde_prompt.md` | LLM prompt template for HyDE generation |
| `scripts/public/shared_scripts/retrieval/multi_query_fuse.py` | N-way weighted RRF fusion + eval plots |
| `scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh` | Pipeline orchestrator (multi-query logic) |
| `scripts/public/shared_scripts/workflow_config_full.env` | All configuration parameters |
