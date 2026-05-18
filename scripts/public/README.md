# Public scripts

## Overview

- **[shared_scripts/](shared_scripts/)** — Retrieval pipeline shared across tasks: orchestration, indexes, retrieval, rerank, optional snippet-RRF, evidence, generation. Narrative and diagram: [shared_scripts/README.md](shared_scripts/README.md).
- **`format/`**, **`data/`**, **`evidence/`**, **`prompts/`**, **`query_parsing/`** — BioASQ-specific helpers documented below.

## BioASQ format (adapt-in and adapt-out)

- **Adapt-in (official JSON → pipeline `.jsonl`):** [format/bioasq_json_to_queries_jsonl.py](format/bioasq_json_to_queries_jsonl.py) converts wrapped `{"questions":[...]}` task JSON into one JSON object per line for the orchestrator.
- **Adapt-out (pipeline answers → BioASQ-style JSON):** [format/queries_jsonl_to_bioasq_json.py](format/queries_jsonl_to_bioasq_json.py) maps query/answer JSONL back into a `questions` array for submission or inspection.

End-to-end Docker, indexing, and BioASQ-oriented evaluation examples: [docs/USAGE.md](../../docs/USAGE.md).

## Corpus and dataset utilities (`data/`)

| Script | Purpose |
|---|---|
| [data/parse_pubmed_local.py](data/parse_pubmed_local.py) | Parse PubMed XML.gz baselines into JSONL shards (canonical `{docno, pmid, type, title, text, mesh_terms, keywords, is_deleted}`). |
| [data/migrate_jsonl_schema.py](data/migrate_jsonl_schema.py) | In-place migration of older PubMed JSONL shards to the unified schema (`abstract` → `text`, adds `type: "abstract"`). Idempotent and atomic per shard. |
| [data/extract_jsonl_subset_by_pmids.py](data/extract_jsonl_subset_by_pmids.py) | Filter PubMed JSONL shards to a PMID allow-list — used to materialise the 3 % / 10 % training subsets. |
| [data/simplify_bioasq_questions_json.py](data/simplify_bioasq_questions_json.py) | Strip official BioASQ question JSON down to `{id, body, type, snippet.text}` (drops `documents` and snippet offset metadata). |

## Evidence to BioASQ submission (`evidence/`)

- [evidence/contexts_json_to_bioasq_snippets.py](evidence/contexts_json_to_bioasq_snippets.py) — Convert pipeline `*_contexts.jsonl` (top-10 docs per question with selected windows) into a BioASQ-style question JSON with `documents` URLs and abstract/title snippet entries (with offsets). Requires the PubMed JSONL corpus for span alignment.

## Generation prompts (`prompts/`)

Per-question-type answer schemas loaded by `shared_scripts/generation/generate_answers.py`. One file per BioASQ type:

```
prompts/schemas/factoid.txt
prompts/schemas/list.txt
prompts/schemas/yesno.txt
prompts/schemas/summary.txt
prompts/schemas/default.txt  # fallback
```

## Shared pipeline configuration

- **All env variables (commented):** [shared_scripts/workflow_config_full.env](shared_scripts/workflow_config_full.env) (also links to tuning notes in PARAMETERS).
- **Tuning ranges, constraints, cap tables:** [shared_scripts/docs/PARAMETERS.md](shared_scripts/docs/PARAMETERS.md).

## Query parsing and HyDE

- **Multi-query fusion, HyDE, deduplication:** [query_parsing/MULTI_QUERY_HYDE.md](query_parsing/MULTI_QUERY_HYDE.md)
- **Scripts and templates:** [query_parsing/](query_parsing/)

## Operational docs

| Doc | Purpose |
|-----|---------|
| [docs/USAGE.md](../../docs/USAGE.md) | BioASQ + Docker + task paths |
| [shared_scripts/docs/USAGE.md](shared_scripts/docs/USAGE.md) | Generic CLI recipes |
| [shared_scripts/docs/output.md](shared_scripts/docs/output.md) | Output directories, TSV runs, logs |
| [shared_scripts/docs/PARAMETERS.md](shared_scripts/docs/PARAMETERS.md) | Parameter tuning |
