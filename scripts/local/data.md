docker run -it \
  -v $(pwd):/BioASQ/ \
  --platform=linux/amd64 fulaibaowang/bioasq:28.01.26 \
  bash

python /BioASQ/data/parse_pubmed_local.py \
    --input_dir /BioASQ/example/ \
    --output_dir /BioASQ/example/jsonl/ \
    --skip_existing

python /BioASQ/data/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/BioASQ/example/jsonl/*.jsonl" \
  --index_path "/BioASQ/example/index/pubmed_bm25_example" \
  --threads 4 \
  --overwrite

python data/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "example/dense_test/*.jsonl" \
  --out_dir "tmp/toy_hnsw_medembed" \
  --device "mps" \
  --batch_size 64 \
  --dedup_pmids \
  --max_elements 150000

/usr/bin/time -l \
python retrieval/eval_bm25_rm3_bioasq.py \
  --index_path "output/pubmed_bm25_2026_subset_index" \
  --train_json "example/training14b_10pct_sample.json" \
  --test_batch_jsons Task13BGoldenEnriched/13B1_golden.json Task13BGoldenEnriched/13B2_golden.json Task13BGoldenEnriched/13B3_golden.json Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_bm25_rm3" \
  --threads 4 \
  --k_eval 5000 \
  --k_feedback 50 \
  --rm3_fb_docs 20 --rm3_fb_terms 30 --rm3_lambda 0.6 \
  --save_runs --save_per_query --save_zero_recall

# Add this if you also want BM25 baseline numbers:
#   --include_bm25
/usr/bin/time -l \
python retrieval/eval_dense.py \
  --train_subset_json example/training14b_10pct_sample.json \
  --index_dir /Users/yun/develop/pubmed_medembed_2026_subset_index \
  --test_batch_jsons Task13BGoldenEnriched/13B1_golden.json Task13BGoldenEnriched/13B2_golden.json Task13BGoldenEnriched/13B3_golden.json Task13BGoldenEnriched/13B4_golden.json \
  --out_dir output/eval_dense_medembed_small \
  --topk 5000 \
  --ef_cap 2000
