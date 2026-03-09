# Pipeline Highlights

- **Multi-stage retrieval, reranking, and evidence fusion pipeline**

## Stage 1: Retrieval (recall-oriented)

- A reciprocal rank fusion (RRF) hybrid combines **BM25+RM3** and **dense HNSW retrieval**, leveraging both lexical exact matching and semantic similarity.

## Stage 2: Reranking (early-precision oriented)

- A cross-encoder reranker rescoring the stage-1 candidates improves top-rank precision.

- The reranked list is further fused with the stage-1 hybrid ranking, combining the robustness of lexical-semantic retrieval with the precision of cross-encoder scoring.

## Stage 3: Snippet-aware evidence route (optional)

- Top-ranked abstracts are split into overlapping sentence windows and reranked.

- A final fusion combines document-level and snippet-level signals, with the main goal of improving evidence selection and context compression for downstream generation rather than substantially improving document-level MAP@10.

# Key observations

- In BioASQ, **BM25 often achieves stronger recall than dense retrieval**, likely because many questions depend on highly specific biomedical terms, abbreviations, or entity mentions.

- Cross-encoder reranking consistently improves early precision over first-stage hybrid retrieval.

- Larger rerankers often perform better, although this should be reported as an empirical trend in the tested setup rather than a universal rule.

- Document–snippet fusion has limited impact on early document-ranking precision, but it may still be useful for selecting compact evidence passages for LLM input.

# Lessons learned

- Query rewriting (e.g. A. typo correction, or, B. question paraphrasing and generalization) did not improve reranking performance in the current setup, suggesting that the reranker is already robust to modest query variation.

- First-stage retrieval misses relatively few gold documents overall, although broader “Wikipedia-style” questions remain difficult (TODO).

- Reranking full abstracts generally outperforms reranking preselected windows at the document-ranking level; snippet selection is more useful for evidence compression than for improving MAP.

- Reranking gains may vary by question type and question length (TODO).

- Shorter questions tend to have lower MAP@10. (TODO)

- After reranking, recall should not be the primary headline metric, because queries differ in the number of relevant documents and reranking mainly targets early precision. Metrics such as **MAP@10** are more appropriate for comparing rerankers.
