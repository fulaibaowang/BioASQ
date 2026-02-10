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

## Indexing & Retrieval

### BM25 Index

Build a Terrier-based BM25 index from JSONL shards:

```bash
python scripts/public/data/build_bm25_index_from_jsonl_shards.py \
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
python scripts/public/data/build_dense_hnsw_index_from_jsonl_shards.py \
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

### BM25 + RM3

Evaluate BM25 and BM25+RM3 on training and test sets:

```bash
python scripts/public/retrieval/eval_bm25_rm3_bioasq.py \
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
python scripts/public/retrieval/eval_dense.py \
  --index_dir "/path/to/indexes/pubmed_medembed_2026" \
  --train_subset_json "example/training14b_10pct_sample.json" \
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

## Output

Both BM25 and dense evaluations produce:
- `metrics.csv`: Aggregate metrics (MAP@10, GMAP@10, MRR@10, Success@10, MeanR@K)
- `runs/`: Per-method run TSVs (qid, rank, docno, score)
- `per_query/`: Per-query CSV breakdown (if `--save_per_query`)
- `*_meta.json`: Metadata and parameters used

## Tuning

See [docs/PARAMETERS.md](PARAMETERS.md) for recommended ranges and rationale for RM3, HNSW, and dense model choices.

## Hybrid Reranking

See [notebooks/hybird.ipynb](../notebooks/hybird.ipynb) for RRF-based hybrid fusion and parameter grid search.
