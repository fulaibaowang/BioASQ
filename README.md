# BioASQ 14b (2026): retrieval, reranking, and generation

Our system for **BioASQ Task 14b** (2026), Phases A and A+: **hybrid retrieval** (BM25 + RM3 and dense
HNSW, fused), **cross-encoder reranking**, an optional **snippet ranking** branch, and **LLM
generation** with one prompt schema per `query_type` (factoid, list, yesno, summary).

The pipeline itself is [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts), vendored under
`scripts/public/shared_scripts/`. This repo adds the BioASQ side: the PubMed corpus, the adapters to
and from the BioASQ JSON format, the per-type answer schemas, and the task data.

## Paper

> Yun Wang. *A Multistage Evidence Retrieval System for BioASQ Task 14b: Hybrid Retrieval, Reranking, and Snippet Selection.* CLEF 2026 Working Notes, BioASQ Lab, CEUR-WS Vol-4283, pp. 778–793. \
> **[[PDF]](https://ceur-ws.org/Vol-4283/paper49.pdf)** &nbsp;·&nbsp; **[[Poster]](docs/poster.pdf)** &nbsp;·&nbsp; archived code: [release v0.1.0](https://github.com/fulaibaowang/BioASQ/releases/tag/v0.1.0)

The submissions were produced at tag **`v0.1.0`** (pipeline identical to
[RAG-scripts `v0.1.0`](https://github.com/fulaibaowang/RAG-scripts/releases/tag/v0.1.0)). On `main`,
the vendored pipeline is frozen at its 2026-07-08 state, used for the post-paper analysis; upstream
has moved on. BibTeX: [Citation](#citation).

## Documentation

| | |
|---|---|
| Docker image, building indexes, BioASQ format conversion | [docs/USAGE.md](docs/USAGE.md) |
| Running against existing PubMed indexes | [docs/REUSE.md](docs/REUSE.md) |
| Results | [docs/RESULTS.md](docs/RESULTS.md) |
| Pipeline commands and parameters (vendored, matches the frozen code) | [USAGE.md](scripts/public/shared_scripts/docs/USAGE.md), [PARAMETERS.md](scripts/public/shared_scripts/docs/PARAMETERS.md) |
| Pipeline overview and flowchart | [RAG-scripts README](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md) |
| Changing this repo (repo map, adapter contract, what belongs upstream) | [AGENTS.md](AGENTS.md) |

### Prompts

- **Query parsing, normalization, per-question HyDE** — [`scripts/public/query_parsing/prompt.md`](scripts/public/query_parsing/prompt.md) (background: [`MULTI_QUERY_HYDE.md`](scripts/public/query_parsing/MULTI_QUERY_HYDE.md)). HyDE is disabled for numeric/measurement, exact-identifier, and other narrow-target questions.
- **Answer generation** — [`system.txt`](scripts/public/shared_scripts/prompts/system.txt) and [`user_base.txt`](scripts/public/shared_scripts/prompts/user_base.txt), plus the per-`query_type` schemas in [`scripts/public/prompts/schemas/`](scripts/public/prompts/schemas/).
- **Query-rewriting ablation (Variants A/B)** — [`notebooks/archive/oneoff_query_rewrite_llm.py`](notebooks/archive/oneoff_query_rewrite_llm.py).

## Reproducing a 14b batch

The questions, the configuration and the container are all here; **the corpus is the cost**. Once
the PubMed indexes exist, a batch is one command.

1. **Build the corpus and indexes** once — parse the PubMed baseline to JSONL, then build the BM25
   and dense indexes ([docs/USAGE.md](docs/USAGE.md)). Budget ~150 GB disk and ~6 h; a batch then
   needs ~80 GB RAM ([requirements](#estimated-resource-requirements)).
2. **Build the container**:

   ```bash
   docker build -t bioasq-pipeline -f scripts/public/shared_scripts/Dockerfile scripts/public/shared_scripts
   ```

3. **Copy the config for the system you want** from
   [`bioasq_data/14b/conf/`](bioasq_data/14b/conf/) and fill in your paths. Files are named after
   our submitted systems: `*-baseline` (document route), `*-snippet` (snippet route), `*-max`
   (snippet route with the 2.5 B LLM reranker), `*-max-rew` (the same, with normalized queries and HyDE).
4. **Run it** inside the container:

   ```bash
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config /path/to/your_copy.env
   ```

5. **Convert the outputs** into BioASQ JSON. Documents and snippets (Phase A) come from the snippet
   route's contexts; answers (Phase B) from the generation output:

   ```bash
   python3 scripts/public/evidence/contexts_json_to_bioasq_snippets.py \
     --contexts-jsonl $WORKFLOW_OUTPUT_DIR/evidence/evidence_snippet/*_contexts.jsonl \
     --corpus-path "/path/to/pubmed_corpus/jsonl_shards/*.jsonl" \
     --output-json phaseA_submission.json

   python3 scripts/public/format/queries_jsonl_to_bioasq_json.py \
     --input $WORKFLOW_OUTPUT_DIR/generation/generation_snippet/*_answers.jsonl \
     --output phaseB_submission.json
   ```

Some submitted runs added a RankZephyr listwise reranking stage. It is not part of the working
note's results and we do not recommend it, so no config enables it; the tooling is in
`scripts/public/shared_scripts/listwise_script/`.

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

## Citation

```bibtex
@inproceedings{wang2026bioasq14b,
  author    = {Yun Wang},
  title     = {A Multistage Evidence Retrieval System for {BioASQ} Task 14b: Hybrid Retrieval, Reranking, and Snippet Selection},
  booktitle = {Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2026)},
  editor    = {Eva S{\'a}nchez Salido and Alberto Barr{\'o}n-Cede{\~n}o and Alba Garc{\'{\i}}a Seco de Herrera and Sean MacAvaney and Julia Maria Stru{\ss}},
  series    = {CEUR Workshop Proceedings},
  volume    = {4283},
  pages     = {778--793},
  publisher = {CEUR-WS.org},
  address   = {Jena, Germany},
  year      = {2026},
  url       = {https://ceur-ws.org/Vol-4283/paper49.pdf}
}
```

