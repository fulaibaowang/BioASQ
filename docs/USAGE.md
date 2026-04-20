# Usage (BioASQ repo, Docker, and task data)

The retrieval pipeline is vendored from upstream **[fulaibaowang/RAG-scripts](https://github.com/fulaibaowang/RAG-scripts/tree/main)** at `scripts/public/shared_scripts/` in this repo. Commands below assume you run them from the **BioASQ repo root** using that path.

**This file** is the BioASQ-oriented runbook: Docker, indexes, official-style paths, and adapt-in/adapt-out examples. For **pipeline-generic** commands (indexing and per-stage CLIs with placeholder paths), see [RAG-scripts docs/USAGE.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/USAGE.md).

---

## Docker: build image and run the pipeline

From the **repository root** (clone with `.git` present, or use absolute paths in your config):

```bash
docker build -t bioasq-pipeline -f Dockerfile .
```

Python dependencies for a **local venv** (optional) are listed at the repo root in [requirements-docker-pytorch.txt](../requirements-docker-pytorch.txt) and [requirements-docker.txt](../requirements-docker.txt). Install a matching `torch` for your platform from [pytorch.org](https://pytorch.org), then `pip install -r requirements-docker.txt`. You still need Java and the system libraries the [Dockerfile](../Dockerfile) installs.

---

## Build indexes (inside the container or on the host)

Assuming you [downloaded pubmed baselines](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/), parse PubMed XML to JSONL once:

```bash
python scripts/public/data/parse_pubmed_local.py \
  --input_dir /path/to/pubmed/baseline2026 \
  --output_dir /path/to/pubmed/jsonl_2026 \
  --skip_existing
```
After you have JSONL document, build both indexes once:

**BM25 (Terrier):**

```bash
python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/path/to/indexes/pubmed_bm25_2026" \
  --threads 4 \
  --overwrite
```

**Dense (HNSW):**

```bash
python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
  --out_dir /path/to/indexes/pubmed_medembed_2026 \
  --model_name "abhinand/MedEmbed-small-v0.1" \
  --device "cuda" \
  --batch_size 128 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100
```

Point `BM25_INDEX_PATH` and `DENSE_INDEX_DIR` in your workflow config at these outputs. More argument detail: [RAG-scripts docs/USAGE.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/USAGE.md).

---

## Adapt-in: BioASQ JSON to pipeline queries

The orchestrator reads **`.jsonl` query streams** only. Convert wrapped BioASQ `{"questions":[...]}` JSON to query JSONL:

```bash
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input path/to/task.json \
  --output path/to/queries.jsonl
```

See [scripts/public/README.md](../scripts/public/README.md) for other format scripts and adapt-out.

---

## BioASQ data preparation and subsets

### Build 10% training subset (BioASQ QAs)

This is only related to BioASQ working note. For pipeline development and evaluaion we made a small subset from the whole training dataset. We ship an example subset at [example/training14b_10pct_sample.json](../example/training14b_10pct_sample.json). It is built from gold QAs, zero-recall IDs, and top-5000 retrieved PMIDs.

See the notebook section **Build 10% Subset with Gold + zero recall ids + Retrieved PMIDs top 5000** in [notebooks/bm25_test.ipynb](../notebooks/bm25_test.ipynb) for the exact steps.

---

### Retrieval, rerank, and full pipeline

For running the pipeline, use the generic examples in [RAG-scripts docs/USAGE.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/USAGE.md) and swap in your paths. Full pipeline:

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh \
  --config /path/to/config.env
```

Config templates: [config.env](bioasq_data/14b/workflow_config_14b_example.env), [workflow_config_full.env](https://github.com/fulaibaowang/RAG-scripts/blob/main/workflow_config_full.env). Parameter tuning: [PARAMETERS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/PARAMETERS.md).


Copy [config.env](bioasq_data/14b/workflow_config_14b_example.env) to a private file and set at least: `WORKFLOW_OUTPUT_DIR`, `INPUT_JSONL` and/or `INPUT_BATCH_JSONLS` (`.jsonl` query streams), `BM25_INDEX_PATH`, `DENSE_INDEX_DIR`, and for rerank and downstream steps `DOCS_JSONL`. Use `HAVE_GROUND_TRUTH=0` when you have no documents/snippets.

---

## Outputs and layout

Under `$WORKFLOW_OUTPUT_DIR`: `retrieval/{bm25,dense,fusion}/`, `rerank/...`, optional `snippet/...`, then `evidence/...` and `generation/...`. Full layout: [RAG-scripts docs/output.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/output.md).

Pipeline narrative and diagram: [RAG-scripts README](https://github.com/fulaibaowang/RAG-scripts/blob/main/README.md).
