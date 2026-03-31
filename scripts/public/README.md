# Public scripts

- **Shared retrieval pipeline and scripts:** see [shared_scripts/README.md](shared_scripts/README.md) for the end-to-end workflow (BM25 → Dense → retrieval fusion → Reranker → Snippet → Generation), config env vars, and script catalogue.
- **Parameter tuning and notebook links:** see [shared_scripts/docs/PARAMETERS.md](shared_scripts/docs/PARAMETERS.md) for recommended ranges and pointers to analysis notebooks.
- **Step-by-step CLI examples:** see [shared_scripts/docs/USAGE.md](shared_scripts/docs/USAGE.md) for indexing, standalone eval scripts, and full pipeline commands.
- **Multi-query fields & HyDE:** see [query_parsing/MULTI_QUERY_HYDE.md](query_parsing/MULTI_QUERY_HYDE.md) for multi-query fusion, HyDE query preparation, and smart deduplication.
- **Query parsing utilities:** [query_parsing/](query_parsing/) contains the HyDE query preparation script, prompt template, and documentation.
