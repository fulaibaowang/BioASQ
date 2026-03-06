## Pipeline Highlights and Submission Title Ideas

### High-level approach

- **Multi-stage, multi-hybrid retrieval stack**: BM25+RM3, dense HNSW retrieval, reciprocal-rank-fusion (RRF) hybrid, cross-encoder reranker, and two RRF fusion stages (Hybrid+Rerank, Docs+Snippets).
- **Single, reproducible pipeline**: one entrypoint (`run_retrieval_rerank_pipeline.sh`) that orchestrates all stages, with idempotent steps (skip-when-output-exists), shared `TOP_K` semantics, and consistent run formats.
- **Engineering and HPC readiness**: config-driven via `.env` files, Slurm/HPC scripts and container image usage, support for sharded dense indexes, and re-runnable experiments that reuse previous outputs.
- **Snippet-RRF innovation**:
  - Stage A: BM25 + dense RRF over abstract windows to keep top windows per doc.
  - Stage B: cross-encoder reranking over windows, aggregating to doc scores.
  - Final fusion: RRF fusion of `rerank_hybrid` (0.8) with `snippet_rerank` (0.2) to form `snippet_rrf/`, then building snippet-based evidence contexts from top windows.
- **End-to-end answer generation**: evidence builders for both baseline and snippet routes plus an LLM generation step, all wired into the same pipeline so that Phase A retrieval and Phase A+/B generation are run consistently.

### Key components (one-line bullets)

- **Retrieval**: BM25+RM3 and dense HNSW retrieval with tuned parameters and unified depth control via `TOP_K`.
- **Hybrid fusion**: RRF-based fusion of BM25 and dense with tuned `K_RRF` and weights, feeding the reranker.
- **Cross-encoder reranking**: BGE v2 cross-encoder with calibrated candidate limit and sequence length, producing strong MAP@10.
- **Baseline RRF fusion**: fusion of Cross-Encoder scores with Hybrid scores (`rerank_hybrid/`) to combine neural reranker and first-stage diversity.
- **Snippet window rerank**: two-stage hybrid + CE ranking of overlapping abstract windows with per-doc aggregation.
- **Snippet-RRF fusion**: final RRF that combines document-only and snippet-augmented signals into a single run (`snippet_rrf/`).
- **Evidence & generation**: document-based and snippet-based evidence builders, followed by LLM answer generation and rescue.

### Candidate system / pipeline titles

Below are several candidate titles with short taglines that could be used for the CLEF/BioASQ working notes submission. They all assume a BioASQ 13b / 2025 context; you can add the exact task/year as a suffix.

1. **\"Snippet-RRF: A Multi-Stage Hybrid Retrieval and Snippet Reranking Pipeline for BioASQ\"**  
   Emphasizes the new snippet-RRF component and the multi-stage hybrid design from BM25+RM3 and dense to Cross-Encoder and snippet windows.

2. **\"An End-to-End Hybrid Retrieval and Snippet-RRF Pipeline for BioASQ Question Answering\"**  
   Highlights that the system spans retrieval, reranking, snippet extraction, evidence building, and LLM answer generation in one reproducible pipeline.

3. **\"Engineering a Reproducible Multi-Stage Hybrid and Snippet-RRF Pipeline for BioASQ\"**  
   Focuses on the engineering aspects: single script, env-based configs, HPC scripts, idempotent stages, and re-runnable experiments.

4. **\"From Hybrid Retrieval to Snippet-RRF: A Unified Pipeline for BioASQ Document and Snippet QA\"**  
   Stresses the unified treatment of document-level and snippet-level evidence within the same pipeline, using final RRF fusion and dedicated evidence builders.

5. **\"Hybrid Retrieval and Snippet-RRF with End-to-End Evidence and Generation for BioASQ\"**  
   Makes clear that evidence construction and LLM generation are first-class stages, not add-ons, for both baseline and snippet routes.

You can mix and match phrasing; for example, a slightly shorter version might be:

- **\"Snippet-RRF: A Unified Hybrid Retrieval and Snippet Pipeline for BioASQ\"**

and the abstract/introduction can expand on the three main pillars: (1) multi-stage hybrid retrieval, (2) engineered, reproducible pipeline, and (3) snippet-RRF with snippet-based evidence.

