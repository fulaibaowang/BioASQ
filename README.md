# BioASQ 14b (2026): retrieval, reranking, and generation

This repository supports **BioASQ Task 14b** (2026) **Phase A–style** document retrieval: a **RAG-style** stack that retrieves biomedical literature, optionally reranks with a cross-encoder and snippet windows, and generates answers from evidence.

**Where to read next**

- **Pipeline overview and flowchart:** [RAG-scripts README](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md) (vendored as `scripts/public/shared_scripts/`)
- **Docker, indexes, and BioASQ-oriented commands:** [docs/USAGE.md](docs/USAGE.md) (new to the repo or using official-style paths → start here)
- **Generic per-stage CLI examples (placeholder paths):** [RAG-scripts docs/USAGE.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/USAGE.md)
- **Adapt-in / adapt-out and public script layout:** [scripts/public/README.md](scripts/public/README.md)
- **Tuning, cap chains, and env parity:** [RAG-scripts docs/PARAMETERS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/PARAMETERS.md) and [workflow_config_full.env](https://github.com/fulaibaowang/RAG-scripts/blob/main/workflow_config_full.env)

At a high level: **hybrid retrieval** (BM25 + RM3 and dense HNSW with fusion), **document reranking** and post-rerank fusion, an **optional snippet-RRF** branch for snippet-style evidence, and **LLM generation** with **different prompts per `query_type`** (factoid, list, yesno, summary). Generation behaviour and backends are implemented in [`generate_answers.py`](https://github.com/fulaibaowang/RAG-scripts/blob/main/generation/generate_answers.py) and commented in `workflow_config_full.env`.

## Usage (Docker)

Build the image, build indexes, and run the orchestrator with a config file: [docs/USAGE.md](docs/USAGE.md).

## Results

Full results: [docs/RESULTS.md](docs/RESULTS.md).

## Environment and CI

Reproducible runtime: [Dockerfile](Dockerfile). Python stacks: [requirements-docker-pytorch.txt](requirements-docker-pytorch.txt) (CUDA 12.8 PyTorch wheels) and [requirements-docker.txt](requirements-docker.txt).

CI runs a small smoke workflow (`compileall` + pipeline `--help`) on changes under the vendored copy of [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main) at `scripts/public/shared_scripts/`; see [.github/workflows/pipeline-smoke.yml](.github/workflows/pipeline-smoke.yml).
