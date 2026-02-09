# BioASQ: retrieval + reranking experiments

This repo collects data prep, retrieval, and evaluation code for BioASQ Phase-A style document retrieval.
The focus is on BM25/RM3 baselines, dense retrieval, and evaluation utilities.

## Repository layout

```
BioASQ/
  scripts/
    public/              # reusable CLI scripts
      data/              # data parsing + index builders
      retrieval/         # BM25/RM3 + dense eval
      retrieval_eval/    # shared eval helpers
    private_scripts/     # host-specific or hardcoded notes
      hpc/
      local/
  notebooks/             # exploration (paired .ipynb/.py)
  bioasq_data/            # official datasets (not tracked)
  output/                # generated outputs (gitignored)
  indexes/               # BM25/HNSW indexes (gitignored)
  runs_dense/            # dense runs (gitignored)
  example/               # small samples
```

## Data placement

Put official datasets under `bioasq_data/`:

```
bioasq_data/
  BioASQ-training13b/
  BioASQ-training14b/
  Task13BGoldenEnriched/
```

Large PubMed baselines and indexes are not tracked; store them in `output/` or external disks and point scripts via paths.

## Methods (current)

- BM25 + RM3 expansion for baseline retrieval
- Dense retrieval via sentence-transformers + HNSW
- Evaluation using official BioASQ measures

## Quickstart (minimal)

1) Parse PubMed XML to JSONL
```
python scripts/public/data/parse_pubmed_local.py \
  --input_dir /path/to/pubmed/baseline2026 \
  --output_dir /path/to/pubmed/jsonl_2026 \
  --skip_existing
```

2) Build a BM25 index
```
python scripts/public/data/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/path/to/indexes/pubmed_bm25_2026" \
  --threads 4 \
  --overwrite
```

3) Evaluate BM25/RM3 on a sample + test batches
```
python scripts/public/retrieval/eval_bm25_rm3_bioasq.py \
  --index_path "/path/to/indexes/pubmed_bm25_2026" \
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
  --rm3_fb_docs 20 --rm3_fb_terms 30 --rm3_lambda 0.6 \
  --save_runs --save_per_query --save_zero_recall
```

## Notes

- `scripts/private_scripts/` contains host-specific commands; keep it for your records but do not publish.
- Notebooks are exploratory; scripts under `scripts/public/` are the reproducible entry points.