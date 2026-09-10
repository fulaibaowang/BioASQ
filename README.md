# BioASQ 14b (2026): retrieval, reranking, and generation

This repository supports **BioASQ Task 14b** (2026) **Phase A/A+**: a **RAG-style** stack that retrieves biomedical literature, reranks with a cross-encoder and optionally snippet windows, and generates answers from evidence.

**Where to read next**

- [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md) as backbone:
  - there you can read **Pipeline overview and flowchart** as well as detailed commands and parameters setting
- **indexes, and BioASQ-oriented commands, e.g input and output conversion for BioASQ format** [docs/USAGE.md](docs/USAGE.md)
- **changing anything in this repo (with or without a coding agent)** — repo map, the adapt-in/adapt-out contract, and what belongs upstream instead: [AGENTS.md](AGENTS.md)


At a high level: **hybrid retrieval** (BM25 + RM3 and dense HNSW with fusion), **document reranking** and post-rerank fusion, an **optional snippet ranking and snippet-document-fusion** branch for snippet-style evidence, and **LLM generation** with **different prompts per `query_type`** (factoid, list, yesno, summary). 

## Paper

This repository accompanies our BioASQ Task 14b (2026) working note:

> Yun Wang. *A Multistage Evidence Retrieval System for BioASQ Task 14b: Hybrid Retrieval, Reranking, and Snippet Selection.* CLEF 2026 Working Notes, BioASQ Lab. \
> **[[PDF]](https://clef-staging.pages.dev/paper49.pdf)** &nbsp;·&nbsp; archived code: [release v0.1.0](https://github.com/fulaibaowang/BioASQ/releases/tag/v0.1.0)

The submissions were produced at **`v0.1.0`**, whose vendored pipeline is identical to
[RAG-scripts `v0.1.0`](https://github.com/fulaibaowang/RAG-scripts/releases/tag/v0.1.0) — that pair
of tags is what reproduces the paper. The vendored copy under `scripts/public/shared_scripts/` is
**frozen** at its 2026-07-06 state and no longer tracks upstream; RAG-scripts `main` has moved on.

### Prompts and schemas

All LLM prompts used by the pipeline are in the repository:

- **Query parsing, normalization, and the per-question HyDE decision/generation** — [`scripts/public/query_parsing/prompt.md`](scripts/public/query_parsing/prompt.md) (background: [`MULTI_QUERY_HYDE.md`](scripts/public/query_parsing/MULTI_QUERY_HYDE.md)). This step sets a per-question `hyde_enabled` flag and disables HyDE for numeric/measurement, exact-identifier, and other highly specific-target questions.
- **Answer generation** — [`scripts/public/shared_scripts/prompts/system.txt`](scripts/public/shared_scripts/prompts/system.txt) and [`scripts/public/shared_scripts/prompts/user_base.txt`](scripts/public/shared_scripts/prompts/user_base.txt).
- **Per-`query_type` answer schemas** — [`scripts/public/prompts/schemas/`](scripts/public/prompts/schemas/) (`factoid`, `list`, `yesno`, `summary`, `default`).
- **Query-rewriting ablation (Variants A/B)** — [`notebooks/archive/oneoff_query_rewrite_llm.py`](notebooks/archive/oneoff_query_rewrite_llm.py).

## Usage (Docker)

Build the image, build indexes, and run the orchestrator with a config file: [docs/USAGE.md](docs/USAGE.md).

## Results

Some results: [docs/RESULTS.md](docs/RESULTS.md).

## Reproducing a 14b batch

The questions, the configuration and the container are all here; **the corpus is the cost**. Once
the PubMed indexes exist, a batch is one command.

1. **Build the corpus and indexes** once — parse the PubMed baseline to JSONL, then build the BM25
   and dense indexes ([docs/USAGE.md](docs/USAGE.md)). Budget ~150 GB and ~6 h, and note that
   running a batch afterwards needs ~170 GB RAM (see [requirements](#estimated-resource-requirements)).
2. **Copy the config for the system you want** from
   [`bioasq_data/14b/conf/`](bioasq_data/14b/conf/) and fill in your paths. The files are named
   after our submitted systems: `dictycite-baseline` (document route), `dictycite-snippet`
   (snippet route), `dictycite-max` (snippet route with the 2.5 B LLM reranker), `dictycite-max-rew`
   (the same, driven by normalized queries and HyDE).
3. **Run it**, inside the container from [Environment](#environment):

   ```bash
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config /path/to/your_copy.env
   ```

4. **Convert the outputs** into BioASQ JSON. Documents and snippets (Phase A) come from the snippet
   route's contexts; answers (Phase B) come from the generation output:

   ```bash
   python3 scripts/public/evidence/contexts_json_to_bioasq_snippets.py \
     --contexts-jsonl $WORKFLOW_OUTPUT_DIR/evidence/evidence_snippet/*_contexts.jsonl \
     --corpus-path "/path/to/pubmed_corpus/jsonl_shards/*.jsonl" \
     --output-json phaseA_submission.json

   python3 scripts/public/format/queries_jsonl_to_bioasq_json.py \
     --input $WORKFLOW_OUTPUT_DIR/generation/generation_snippet/*_answers.jsonl \
     --output phaseB_submission.json
   ```

Some of our submitted runs added a RankZephyr listwise reranking stage on top of the snippet route.
It is not part of the working note's results and we do not recommend it, so none of the configs
above enable it; `listwise_script/` holds the tooling if you want to try it anyway.

## Environment

Container image and Python pins live only under the vendored [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main) tree: [Dockerfile](scripts/public/shared_scripts/Dockerfile), [requirements-docker-pytorch.txt](scripts/public/shared_scripts/requirements-docker-pytorch.txt), [requirements-docker.txt](scripts/public/shared_scripts/requirements-docker.txt). Build from the repo root: `docker build -t bioasq-pipeline -f scripts/public/shared_scripts/Dockerfile scripts/public/shared_scripts` (see [docs/USAGE.md](docs/USAGE.md)).

## Estimated resource requirements

Measured on a single GPU node, indexing the **PubMed 2026 baseline (~40 million abstracts)**.
A smaller corpus needs proportionally less.

### Building the indexes (one-time)

| | Disk | Build cost |
|---|---|---|
| Parsed PubMed corpus (JSONL) | 54 GB | CPU only |
| BM25 index | 14 GB | ~2 h on 16 cores, 96 GB RAM |
| Dense HNSW index (10 shards) | 69 GB | ~4 h per shard on one A100-80GB |
| Container image + model weights | 13 GB | download once |
| **Total** | **~150 GB** | **~6 h** with the shards built in parallel |

### Running one batch

One BioASQ 14b batch — 80 questions, both routes, on a single A100:

| | |
|---|---|
| CPU | 16 cores |
| RAM | ~80 GB |
| GPU | ~5 GB VRAM |
| Runtime | ~1.5 h retrieval and reranking, plus ~1.5 h generation |
| Output | ~150 MB |

**RAM is the constraint, not the GPU** — dense retrieval holds all ten HNSW shards (~70 GB) in
memory at once.
BM25-only retrieval needs a few GB instead and loses little accuracy once the reranker runs.

**VRAM depends on the reranker and its batch size.** The default `bge-reranker-v2-m3` needs ~5 GB at
the shipped batch of 64, so an 8 GB card is enough; lower `RERANK_MODEL_BATCH` to fit a smaller one.
LLM rerankers such as `bge-reranker-v2-gemma` (2.5 B params, fp16) need a larger card.

**Generation time depends on the generator.** The ~1.5 h above is a self-hosted `llama3.3:70b`
answering one question at a time; a batched OpenAI-compatible API takes minutes.
