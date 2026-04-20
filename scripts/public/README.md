# Public scripts

**Which usage doc?** 

Use **[docs/USAGE.md](../../docs/USAGE.md)** for BioASQ-oriented examples and codes including indexing, bioASQ format conversion. 

Use **[shared_scripts/docs/USAGE.md](shared_scripts/docs/USAGE.md)** for generic indexing and per-stage CLI commands with placeholder paths.

## Overview

- **[shared_scripts/](shared_scripts/)** — Retrieval pipeline shared across tasks: orchestration, indexes, retrieval, rerank, optional snippet-RRF, evidence, generation. Narrative and diagram: [shared_scripts/README.md](shared_scripts/README.md).
- **`format/`**, **`query_parsing/`**, **`data/`** — Task-facing helpers (BioASQ JSON conversion, query variants, PubMed parsing) documented below.

## BioASQ format (adapt-in and adapt-out)

- **Adapt-in (official JSON → pipeline `.jsonl`):** [format/bioasq_json_to_queries_jsonl.py](format/bioasq_json_to_queries_jsonl.py) converts wrapped `{"questions":[...]}` task JSON into one JSON object per line for the orchestrator.
- **Adapt-out (pipeline answers → BioASQ-style JSON):** [format/queries_jsonl_to_bioasq_json.py](format/queries_jsonl_to_bioasq_json.py) maps query/answer JSONL back into a `questions` array for submission or inspection.

End-to-end Docker, indexing, and BioASQ-oriented evaluation examples: [docs/USAGE.md](../../docs/USAGE.md).

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
