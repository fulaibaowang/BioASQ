# Notebooks index

Personal experiment notebooks. Filename prefixes match the pipeline stages in
`workingnotes/figures/01_pipeline.png`:

- `retrieval_*` — BM25, dense, retrieval fusion
- `rerank_*` — cross-encoder, post-rerank fusion, listwise, score-cutoffs
- `snippet_*` — snippet extraction / snippet-doc fusion
- `generation_*` — LLM answer generation
- `analyze_*` / `workflow_*` — cross-stage results analysis
- `oneoff_*` — one-shot side experiments
- `archive/` — superseded notebooks, kept for reference

Notebooks are paired with `.ipynb` via [jupytext](https://jupytext.readthedocs.io/)
(`formats = "ipynb,py:percent"` in `pyproject.toml`). The `.py` file is the
source of truth.

## Retrieval

| Notebook | Purpose | Data |
|---|---|---|
| `retrieval_bm25_baseline_and_subset.py` | BM25+RM3 build & sanity; **builds the 10% subset** used by USAGE.md (gold + zero-recall + top-5000 PMIDs); RM3 parameter sweep. | full PubMed 2026 + BioASQ 14b |
| `retrieval_dense_baseline_index.py` | Dense (MedEmbed) sanity, sequence-length truncation check, builds the 10% HNSW vector DB. | 10% subset |
| `retrieval_fusion_sweep_medembed.py` | Hybrid (BM25_RM3 + dense MedEmbed) **RRF weight × `k_rrf` grid search**. | 10% subset, train/test splits |
| `retrieval_fusion_sweep_pubmedbert.py` | Same RRF grid with **PubMedBERT** dense encoder. Loses to MedEmbed. | 10% subset |
| `retrieval_fusion_hyde_sweep.py` | Weighted RRF over two dense runs (HyDE multi-query), **Recall@5000**. | 10% subset |

## Reranking

| Notebook | Purpose | Data |
|---|---|---|
| `rerank_stage2.py` | Stage-2 CE rerank (MiniLM) with adaptive cutoff. Has a **local CPU** preset and an HPC GPU preset in the config block (merged from the old `rerank_stage2_hpc.py`). | best-RRF top-2000 |
| `rerank_post_fusion_analysis.py` | Reranker top-1 overlap, win/loss, Jaccard@10, and RRF fusion sweep over **MiniLM + BGE-m3 + Gemma**. Plots for `workflow_baseline_full_run_both_routes/`. | workflow output |
| `rerank_listwise.py` | RankZephyr-7B-V1-Full listwise rerank on snippets (vLLM). | snippet windows |
| `rerank_cutoff_01_adaptive_gap_oracle.py` | Adaptive score-gap cutoff vs an oracle cutoff (remaining AP mass). Multiple reranker families side-by-side. | gold + reranked runs |
| `rerank_cutoff_02_hard_score_global.py` | Hard global score threshold sweep per family (`bgem3`, `bge_gemma`). | as above |
| `rerank_cutoff_03_two_stage_iqr_gap.py` | Two-stage: global `t*` + IQR-gated gap on survivors. | as above |

## Snippets

| Notebook | Purpose | Data |
|---|---|---|
| `snippet_extraction_ce_bgem3.py` | Snippet route: window CE rerank (bge-reranker-v2-m3) → snippet/doc RRF fusion. | training + 13B golden |
| `snippet_extraction_medcpt.py` | Same route with **MedCPT** reranker. | as above |

## Generation

| Notebook | Purpose | Data |
|---|---|---|
| `generation_temperature_sweep.py` | Compare BioASQ generation metrics across ground-truth temperatures (0.0 / 0.3 / 0.8). | `using_ground_truth_generation/` |
| `generation_ollama_quickrun.py` | Single-file generation against one `contexts.json` via the FRI Ollama endpoint (qwen3:32b). Prompt-tuning sandbox. | one workflow split |
| `generation_openrouter_demo.py` | Four OpenRouter experiments: chat/completions, planner template, planner→answer chain. | one workflow split |

## Cross-stage analysis

| Notebook | Purpose | Data |
|---|---|---|
| `analyze_workflow_results.py` | Workflow-runner results: MAP@10 (ours) vs `d_MAP` (official), per-stage stability. Supports `3pct` and `10pct`. | workflow output |
| `workflow_baseline_full_run_report.py` | Plots for `output/workflow_baseline_full_run_both_routes/{bm25,dense,hybrid}` — GitHub issue #4. | workflow output |
| `oneoff_llm_evidence_vs_map10.py` | One-off: BioASQ AP@10 for original `documents` order vs LLM `evidence_ids` reordering (cited PMIDs first). | one workflow split |

## Utilities

| File | Purpose |
|---|---|
| `_rerank_plot_sample.py` | Shared helper: canonical stratified `(split, qid)` sample for cross-notebook comparability. No paired `.ipynb`. |

## Archive

`archive/` holds superseded or non-improving experiments, kept for reference only.

| Notebook | Why archived |
|---|---|
| `archive/analyze_results.py` | First-pass results analysis (BM25/Dense/Hybrid + MiniLM vs BGE configs). Superseded by `analyze_workflow_results.py`. |
| `archive/oneoff_query_rewrite_llm.py` | LLM query rewrite (A: normalize_only, B: normalize_and_enrich). Did not improve MAP (see CLAUDE.md). |
