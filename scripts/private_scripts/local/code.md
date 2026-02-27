# prepare 
docker run -it \
  -v $(pwd):/BioASQ/ \
  --platform=linux/amd64 fulaibaowang/bioasq:28.01.26 \
  bash

python /BioASQ/scripts/public/data/parse_pubmed_local.py \
    --input_dir /BioASQ/example/ \
    --output_dir /BioASQ/example/jsonl/ \
    --skip_existing

# build index
python /BioASQ/scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/BioASQ/example/jsonl/*.jsonl" \
  --index_path "/BioASQ/example/index/pubmed_bm25_example" \
  --threads 4 \
  --overwrite

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "example/dense_test/*.jsonl" \
  --out_dir "tmp/toy_hnsw_medembed" \
  --device "mps" \
  --batch_size 64 \
  --dedup_pmids \
  --max_elements 150000

# eval
/usr/bin/time -l \
python scripts/public/shared_scripts/retrieval/eval_bm25_rm3.py \
  --index_path "output/pubmed_bm25_2026_subset_index" \
  --train_json "example/training14b_10pct_sample.json" \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_bm25_rm3" \
  --threads 4 \
  --k_eval 5000 \
  --k_feedback 50 \
  --rm3_fb_docs 20 --rm3_fb_terms 30 --rm3_lambda 0.6 \
  --save_runs --save_per_query --save_zero_recall

#<--- add this  --include_bm25 if you also want BM25 baseline numbers  

/usr/bin/time -l \
python scripts/public/shared_scripts/retrieval/eval_dense.py \
  --train-json example/training14b_10pct_sample.json \
  --index_dir /Users/yun/develop/pubmed_medembed_2026_subset_index \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir output/eval_dense_medembed_small \
  --topk 5000 \
  --ef_cap 2000

python scripts/public/shared_scripts/retrieval/eval_hybird.py \
  --bm25_runs_dir output/eval_bm25_rm3/runs \
  --dense_root output/eval_dense_medembed_small \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir output/eval_hybird_production_test \
  --mode default \
  --jobs 4

# rerank
python scripts/public/shared_scripts/rerank/rerank_stage2.py \
  --output-dir output/eval_stage2_rerank_minitest \
  --runs-dir output/eval_hybird_production_test/runs \
  --docs-jsonl output/subset_pubmed.jsonl \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --model-device cpu \
  --model-batch 16 \
  --max-queries 5 

python scripts/public/shared_scripts/compare_result_dirs.py \
  --dirs output/workflow_local_3pct_hpc_bge/rerank_merged output/workflow_local_3pct_hpc_bge/rerank_sentence \
  --labels "Stage 3 merged" "Stage 3 sentence" \
  --plot both \
  --map-ks 10,20,50,100,200 \
  --train-json example/training14b_3pct_sample.json \
  --test_batch_jsons example/13b_golden_50q_sample.json \
  --output-dir output/workflow_local_3pct_hpc_bge/compare_plots_mergedtest

python scripts/public/shared_scripts/rerank/rerank_rrf_hybrid.py \
  --hybrid-runs-dir output/workflow_local_10pct_hpc_bge/hybrid/runs \
  --rerank-runs-dir output/workflow_local_10pct_hpc_bge/rerank/runs \
  --output-dir output/workflow_local_10pct_hpc_bge/rerank_hybrid \
  --train-json example/training14b_10pct_sample.json \
  --test-batch-jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --pool-top 50 --k-rrf 60 --w-bge 0.8 --w-hybrid 0.2

python scripts/public/shared_scripts/compare_result_dirs.py \
  --dirs output/workflow_local_10pct_hpc_bge/rerank output/workflow_local_10pct_hpc_bge/rerank_hybrid \
  --labels "rerank" "rerank_rrf_hybrid" \
  --output-dir output/workflow_local_10pct_hpc_bge/rerank_hybrid \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json 

#evidence
python3 "scripts/public/shared_scripts/evidence/post_rerank_json.py" \
  --run-path "output/workflow_local_10pct_hpc_bge/rerank_hybrid/runs/best_rrf_training14b_10pct_sample_top5000_rrf_pool50_k60.tsv" \
  --query-json example/training14b_10pct_sample.json \
  --output-path "output/workflow_local_10pct_hpc_bge/post_rerank_training14b_10pct_sample.json" \
  --top-k 10

python3 "scripts/public/shared_scripts/evidence/build_contexts_from_documents.py" \
  --post-rerank-json "output/workflow_local_10pct_hpc_bge/post_rerank_training14b_10pct_sample.json" \
  --corpus-path output/subset_pubmed.jsonl \
  --output-path "output/workflow_local_10pct_hpc_bge/evidence/training14b_10pct_sample_contexts.jsonl"


#after query rewrting
python scripts/public/shared_scripts/compare_result_dirs.py \
  --dirs output/workflow_local_10pct_hpc_bge/rerank output/workflow_local_10pct_hpc_bge/rerank_body_rewrite_A output/workflow_local_10pct_hpc_bge/rerank_body_rewrite_B \
  --labels "rerank" "rewrite A" "rewrite B"\
  --plot both \
  --map-ks 10,20,50,100,200 \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --output-dir output/workflow_local_10pct_hpc_bge/compare_plots_query_rewrite \
  --plots-by-split 