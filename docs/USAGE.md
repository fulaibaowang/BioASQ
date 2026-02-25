# Usage Guide

## Data Preparation

### Parse PubMed XML to JSONL

Convert PubMed baseline XML files to JSONL format with title, abstract, and MeSH metadata:

```bash
python scripts/public/data/parse_pubmed_local.py \
  --input_dir /path/to/pubmed/baseline2026 \
  --output_dir /path/to/pubmed/jsonl_2026 \
  --skip_existing
```

**Arguments:**
- `--input_dir`: Directory containing PubMed XML baseline files
- `--output_dir`: Output directory for JSONL shards
- `--skip_existing`: Skip files that already exist in output

### Build 10% Training Subset (BioASQ QAs)

We generate a smaller training set at [example/training14b_10pct_sample.json](../example/training14b_10pct_sample.json).
This subset is built from gold QAs plus zero-recall IDs and top-5000 retrieved PMIDs.

See the notebook section **Build 10% Subset with Gold + zero recall ids + Retrieved PMIDs top 5000** in
[notebooks/bm25_test.ipynb](../notebooks/bm25_test.ipynb) for the exact steps.

## Indexing & Retrieval

### BM25 Index

Build a Terrier-based BM25 index from JSONL shards:

```bash
python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/path/to/indexes/pubmed_bm25_2026" \
  --threads 4 \
  --overwrite
```

**Arguments:**
- `--jsonl_glob`: Glob pattern for input JSONL files
- `--index_path`: Output Terrier index directory
- `--threads`: Number of indexing threads
- `--overwrite`: Recreate index if it exists

### Dense (HNSW) Index

Build an HNSW dense vector index using SentenceTransformer embeddings:

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

**Arguments:**
- `--model_name`: SentenceTransformer model (default: MedEmbed-small-v0.1)
- `--device`: `cuda`, `cpu`, or `mps` (default: cuda)
- `--batch_size`: Embedding batch size (default: 128)
- `--M`: HNSW graph degree (default: 32)
- `--ef_construction`: HNSW construction parameter (default: 200)
- `--ef_search`: HNSW query-time parameter (default: 100)
- `--max_docs`: Limit index to N docs (for testing)
- `--dedup_pmids`: De-duplicate documents by PMID

## Evaluation

### Full pipeline (BM25 → Dense → Hybrid → Reranker)

Run all stages with one script and a config file. The script skips any stage whose output already exists (e.g. if hybrid is done, only reranker (and optional RRF fusion) run). Use `--no-rerank` to run only retrieval (BM25, Dense, Hybrid). Use `--no-rrf-fusion` to disable the Hybrid+Rerank RRF fusion step.

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/public/workflow_config_small.env
```

**Config files:** [workflow_config_small.env](../scripts/public/workflow_config_small.env), [workflow_config_full.env](../scripts/public/workflow_config_full.env). For all options and env→script mapping see [scripts/public/README.md](../scripts/public/README.md).

### BM25 + RM3

Evaluate BM25 and BM25+RM3 on training and test sets:

```bash
python scripts/public/shared_scripts/retrieval/eval_bm25_rm3.py \
  --index_path "/path/to/indexes/pubmed_bm25_2026/data.properties" \
  --train_json "example/training14b_10pct_sample.json" \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_bm25_rm3" \
  --threads 4 \
  --k_eval 5000 \
  --k_feedback 50 \
  --rm3_fb_docs 20 \
  --rm3_fb_terms 30 \
  --rm3_lambda 0.6 \
  --save_runs --save_per_query --save_zero_recall
```

**Key Arguments:**
- `--k_eval`: Maximum documents to retrieve (default: 5000)
- `--k_feedback`: Documents used for RM3 feedback (default: 50)
- `--rm3_fb_docs`: Feedback documents for RM3 (default: 20)
- `--rm3_fb_terms`: Feedback terms for RM3 (default: 30)
- `--rm3_lambda`: RM3 interpolation weight (default: 0.6)
  - Higher λ = more influence from original query
  - Range: [0, 1]
- `--include_bm25`: Also output BM25-only baseline
- `--java_mem`: JVM heap size, e.g., "8g"

### Dense Retrieval

Evaluate dense retrieval using pre-built HNSW index:

```bash
python scripts/public/shared_scripts/retrieval/eval_dense.py \
  --index_dir "/path/to/indexes/pubmed_medembed_2026" \
  --train-json "example/training14b_10pct_sample.json" \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_dense" \
  --topk 5000 \
  --ks "50,100,200,500,2000,5000" \
  --ef_search 100 \
  --batch_size 256
```

**Key Arguments:**
- `--topk`: Maximum retrieve depth (default: 5000)
- `--ks`: Recall evaluation points, comma-separated (default: 50,100,200,500,1000,2000,5000)
- `--ef_search`: HNSW query-time expansion (higher ≈ slower but better recall)
- `--ef_cap`: Optional cap on effective efSearch
- `--device`: `cpu`, `cuda`, or `mps`
- `--model_name`: Override SentenceTransformer model

### Hybrid Reranking

Fuse BM25 and dense runs with reciprocal rank fusion (RRF):

```bash
python scripts/public/shared_scripts/retrieval/eval_hybird.py \
  --bm25_runs_dir "output/eval_bm25_rm3/runs" \
  --dense_root "output/eval_dense" \
  --train-json "example/training14b_10pct_sample.json" \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_hybird" \
  --mode "default" \
  --k_rrf 150 \
  --w_bm25 1.0 \
  --w_dense 1.0
```

**Key Arguments:**
- `--bm25_runs_dir`: BM25 run TSVs (from BM25+RM3 eval)
- `--dense_root`: Dense output folder with `dense_*.parquet`
- `--mode`: `default` for a single config or `sweep` for grid search
- `--k_rrf`, `--w_bm25`, `--w_dense`: RRF tuning knobs


### Stage 2 Rerank (Cross-Encoder)

Re-rank stage-1 runs with a cross-encoder using query + doc text pairs:

```bash
python scripts/public/shared_scripts/rerank/rerank_stage2.py \
  --runs-dir "output/eval_hybird/runs" \
  --docs-jsonl "output/subset_pubmed.jsonl" \
  --train-json "example/training14b_10pct_sample.json" \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --output-dir "output/eval_stage2_rerank" \
  --candidate-limit 2000 \
  --model "cross-encoder/ms-marco-MiniLM-L-12-v2" \
  --model-device "cpu" \
  --model-batch 16 \
  --model-max-length 512
```

**Key Arguments:**
- `--runs-dir`: Stage-1 run TSVs to rerank
- `--docs-jsonl`: JSONL corpus with title/abstract text
- `--candidate-limit`: Candidates per query to rerank
- `--model`: Cross-encoder model name
- `--model-device`: `auto`, `cuda`, `mps`, or `cpu`
- `--adaptive-p`, `--adaptive-cap`: Adaptive cutoff parameters
- `--model-max-length`: Token truncation length for the cross-encoder

## Output

Both BM25 and dense evaluations produce:
- `metrics.csv`: Aggregate metrics (MAP@10, GMAP@10, MRR@10, Success@10, MeanR@K)
- `runs/`: Per-method run TSVs (qid, rank, docno, score)
- `per_query/`: Per-query CSV breakdown (if `--save_per_query`)
- `*_meta.json`: Metadata and parameters used

## Tuning

See [docs/PARAMETERS.md](PARAMETERS.md) for parameter ranges and short notes.


