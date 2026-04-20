# BioASQ 14b (2026): retrieval, reranking, and generation

This repository supports **BioASQ Task 14b** (2026) **Phase A/A+**: a **RAG-style** stack that retrieves biomedical literature, reranks with a cross-encoder and optionally snippet windows, and generates answers from evidence.

**Where to read next**

- Use [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md) as backbone:
  - there you can read **Pipeline overview and flowchart** as well as detailed commands and parameters setting
- **indexes, and BioASQ-oriented commands, e.g input and output conversion for BioASQ format** [docs/USAGE.md](docs/USAGE.md)


At a high level: **hybrid retrieval** (BM25 + RM3 and dense HNSW with fusion), **document reranking** and post-rerank fusion, an **optional snippet ranking and snippet-document-fusion** branch for snippet-style evidence, and **LLM generation** with **different prompts per `query_type`** (factoid, list, yesno, summary). 

## Usage (Docker)

Build the image, build indexes, and run the orchestrator with a config file: [docs/USAGE.md](docs/USAGE.md).

## Results

Some results: [docs/RESULTS.md](docs/RESULTS.md).

## Environment and CI

Reproducible runtime: [Dockerfile](Dockerfile) at the repository root (it copies Python requirements from the [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main) subtree). Canonical copies of the image recipe and pins: [`Dockerfile`](https://github.com/fulaibaowang/RAG-scripts/blob/main/Dockerfile), [`requirements-docker-pytorch.txt`](https://github.com/fulaibaowang/RAG-scripts/blob/main/requirements-docker-pytorch.txt), [`requirements-docker.txt`](https://github.com/fulaibaowang/RAG-scripts/blob/main/requirements-docker.txt) under `scripts/public/shared_scripts/` in this repo.

