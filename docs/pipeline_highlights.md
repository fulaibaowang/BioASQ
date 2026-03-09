## Pipeline Highlights and Title Ideas

### Multi-stage, multi-hybrid retrieval stack:
- retrival (recall foucused)
  -  reciprocal-rank-fusion (RRF) hybrid in retrival steps on BM25, dense HNSW retrieval, combining advantages keyword and semantic understanding
- rerank (top titer precision focused)
  -  doc level rrf on rerank lists with hybrid lists (from retrival), combining advantages keyword and cross encoder understanding
  -  rrf on doc level (abstracts) and snippets list combining doc level and small window level

Key
- bioasq test often focus on specific terms, bm25 almost always get higher recall to dense
- reranker always help on early precision, and often the bigger model have better results
- three hybrid
  - bm25 and dense fusion usually helps
  - hybrid and rerank fusion usually helps
  - doc level and snippets fusion does not impactful on early pricision, because snippet are from top docs, but might help on selction window and feed llm key evidence passage (smaller context)
 
lesson learned:
- query rewriting (either basic tpyo correction, or short question generailze / refrase) does not help reranking, prosumably reranker was robust on this already
- hard miss by first retrieval stage is already low in my system. those wikipedia style questions is always diffcult for retrival though
- reranking on whole abstract is almost always better than on select windows overall, so snippet selection can help on compress evidence but not increase map
- is raranker (delta map@10 rerank - hybrid) bias by question types or question length (this is worthy one more closer look)
- shorter question seems always have lower map@10
- once move to reranking, recall is not anymore a good eval metrices, epecially number of  gold docs are not the same (the higher the # gold docs, the likely the lower the recall)
- how to name terms better?
