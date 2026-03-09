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


# Scientific Discussion

Your first point, about BM25 often beating dense on recall, is believable and worth highlighting, but I would not write “almost always” unless your plots really show that across nearly all splits. A safer formulation is “BM25 tended to outperform dense retrieval in recall in our experiments.” The scientific interpretation is good: BioASQ questions are often entity-centric and terminology-heavy, so lexical exact matching stays very strong. Dense retrieval still matters because it captures semantic similarity and complements BM25, which is exactly why your hybrid stage is a core design choice in both the README and the pipeline.

Your second point, that rerankers help early precision, is one of the strongest parts of the story. That is also how the repo is framed: stage 1 is broad retrieval, stage 2 is reranking focused on top-rank metrics.

I would say “consistently improved MAP@10 / MRR@10 over the stage-1 hybrid” rather than “always help,” because “always” is easy for a reviewer to challenge.

The third point about snippet fusion is scientifically interesting, and the script actually supports your intuition. The snippet route is not a fresh retrieval branch from the full corpus; it is a late branch from the document route, using rerank_hybrid_200 as input, then extracting windows, reranking them, and finally fusing document and snippet signals with default weights of 0.8 for docs and 0.2 for snippets.

Because of that design, large MAP@10 gains are naturally hard to get: the snippet branch mostly rearranges evidence inside an already strong shortlisted document set. That makes your interpretation strong: snippet routing is better presented as evidence refinement / compression than as a major document-retrieval improvement.

Your “query rewriting does not help reranking” point is useful, but it should be scoped carefully. The script allows separate query fields for BM25, dense, and rerank stages, so the system is already set up for stage-specific query representation choices.

I would therefore write: “In the current setup, lightweight query rewriting did not improve reranking performance.” That leaves room for the possibility that rewriting could still help stage-1 retrieval, or help under other models.

Your “hard miss by first retrieval stage is already low” point is good, but it needs a precise definition in the report. Use something measurable such as:

- fraction of questions with zero relevant documents in top-K,
- recall@1000,
- or count of unrecoverable failures before reranking.

That will make the claim much more defensible. The note about “Wikipedia-style questions” is also good; I would describe them as broader, more descriptive, or multi-aspect questions whose evidence is more diffuse and therefore harder for both lexical and semantic retrieval.

Your point that full-abstract reranking is better than selected-window reranking is also quite plausible. A scientific explanation is that BioASQ relevance labels are document-level, and full abstracts preserve context, disambiguation, and supporting cues that small windows may drop. So again, snippet scoring may be more useful for downstream evidence packaging than for document ranking itself.

The “is reranker delta MAP@10 biased by question type or question length?” point should not sit under “lessons learned.” That is not yet a finding; it is a hypothesis or planned analysis. I would move it to a section like Open questions / further analysis. Same for “shorter question seems always have lower MAP@10”: unless you have already run the stratified analysis, soften it to “shorter questions appear to have lower MAP@10.” Also, question length is confounded with question type, ambiguity, and entity specificity, so I would recommend analyzing length within each BioASQ question type rather than across all questions pooled together.

Your final point on recall is important, but I would make it more nuanced. I would not say recall is “not a good metric anymore.” I would say:

“After reranking, recall@k is less suitable as the primary comparison metric, because reranking mainly targets early precision and because recall is strongly affected by the number of relevant documents per query. For reranker comparison, MAP@10, MRR@10, and related early-precision metrics are more informative, while recall@k remains a secondary coverage measure.”

That sounds much more balanced.

## Additional Scientific Points

Two extra scientific points jumped out from the script that are worth being explicit about in the report.

First, the snippet route and baseline route are not compared under identical upstream settings by default. The script uses a pool-50 post-rerank fusion for the baseline rerank_hybrid, but a pool-200 branch for snippet processing via rerank_hybrid_200.

run_retrieval_rerank_pipeline

So if you compare baseline vs snippet downstream, make clear whether the comparison is:

- baseline route vs snippet route as actually used in the system, or
- a controlled ablation with matched upstream pool size.

Second, generation is also not identical by default between routes. The script gives different default context budgets to baseline and snippet generation: baseline defaults to 8 contexts and 1300 chars per context, while snippet defaults to 10 contexts and 960 chars per context.

run_retrieval_rerank_pipeline

That means any generation comparison should state this clearly, because differences may come from evidence packaging budget, not only retrieval quality.

## Naming Scheme Recommendation

For naming, I would strongly suggest this scheme in the report:

- Stage 1: Hybrid retrieval
- Stage 2: Document reranking
- Stage 2b: Post-rerank fusion
- Stage 3: Snippet-aware evidence reranking
- Stage 3b: Evidence fusion
- Stage 4: Answer generation

Then, when you describe the three combination points, call them:

- retrieval fusion
- post-rerank fusion
- evidence fusion

That will keep the reader oriented.

## Title Ideas

Some better title ideas:

- A Multi-Stage Hybrid Retrieval and Evidence Fusion Pipeline for BioASQ
- Hybrid Retrieval, Cross-Encoder Reranking, and Snippet-Aware Evidence Construction for BioASQ
- From Hybrid Retrieval to Snippet-Based Evidence: A BioASQ RAG Pipeline
- A Multi-Route BioASQ Pipeline for Retrieval, Reranking, and Evidence-Aware Generation

My honest view: the strongest scientific story here is not “we have many hybrids.” The stronger story is:

A recall-oriented retrieval stage is followed by precision-oriented reranking, and an optional snippet-aware evidence route improves evidence packaging for generation even when document-level MAP gains are limited.

That is cleaner, more defensible, and much easier to expand into a full report.
