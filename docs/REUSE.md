# Reusing prebuilt PubMed indexes

How to run the pipeline against an existing copy of the parsed PubMed corpus and its indexes,
instead of spending ~150 GB and ~6 h rebuilding them ([README](../README.md#estimated-resource-requirements)).
To build your own, see [USAGE.md](USAGE.md#build-indexes-inside-the-container-or-on-the-host).

## What a prebuilt set must contain

All four pieces must come from the same PubMed baseline. The pipeline reads the corpus shards for
reranker text and snippets, so if the shards and the indexes disagree you get missing documents
rather than an error.

| Config variable | Points at | Notes |
|---|---|---|
| `DOCS_JSONL` | parsed shards, e.g. `jsonl_2026/*.jsonl` | 1,334 shards for the 2026 baseline, schema `{docno, pmid, type, title, text, mesh_terms, keywords, is_deleted}` |
| `BM25_INDEX_PATH` | the BM25 index | Use the `_with_keywords` index if there are two; it is the one the repo's run configs use. ~40M documents. |
| `DENSE_INDEX_GLOB` | the 10 dense shards, e.g. `pubmed_medembed_shard0*` | `abhinand/MedEmbed-small-v0.1`, 384-d, cosine HNSW. Each shard's `meta.json` records the settings. Comment out `DENSE_INDEX_DIR`: the glob wins anyway. |
| (container) | the pipeline image | Built from `scripts/public/shared_scripts/Dockerfile`. Its PyTerrier and hnswlib versions are not pinned, so prefer the image that built the indexes |

The dense index only works with queries embedded by the same model. The pipeline reads the model
name from `meta.json`, so don't override it.

## Running against them

1. Clone the repo from GitHub.
2. Copy [`bioasq_data/14b/workflow_config_14b_example.env`](../bioasq_data/14b/workflow_config_14b_example.env)
   to a private path. Fill in the four variables above, plus `WORKFLOW_OUTPUT_DIR` (somewhere you
   can write) and `INPUT_JSONL`. The rest of [AGENTS.md § Running a batch](../AGENTS.md#running-a-batch)
   applies, in particular `GENERATION_SCHEMAS_DIR` and `HAVE_GROUND_TRUTH`.
3. On a SLURM cluster with pyxis/enroot, start from
   [`scripts/private_scripts/hpc/bioasq_runs/sbatch_pipeline.sh`](../scripts/private_scripts/hpc/bioasq_runs/sbatch_pipeline.sh).
   It hardcodes the authors' setup. Change `cd ~/BioASQ`, `CONTAINER_IMG`, the extra output mount
   (`yun=...`, mounted as `/yun`), `PIPELINE_CONFIG`, and the `#SBATCH` partition and GPU lines.
   Inside the container, the repo is `/work` (`REPO_ROOT=/work`) and the corpus folder is `/pubmed`.

### The model cache

The sbatch script points `HF_HOME` at a shared Hugging Face cache. If that cache is read-only for
you, add `export HF_HUB_OFFLINE=1` next to `HF_HOME` in the script. Models already in the cache then
load without the library trying to write lock files or contact the Hub. A model that isn't cached
will fail; use your own `HF_HOME` for that.

### Retrieval only, or end to end

Retrieval, reranking and evidence need nothing beyond the above. Generation also needs an LLM
endpoint and its key in `.env` (`GEN_API_KEY` or `LLAMA_API_KEY`, depending on the
backend). Set `RUN_GENERATION_DOCUMENT=0` and
`RUN_GENERATION_SNIPPET=0` to stop before it.
