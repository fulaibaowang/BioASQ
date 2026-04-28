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
python scripts/public/shared_scripts/retrieval/retrieve_bm25.py \
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
python scripts/public/shared_scripts/retrieval/retrieve_dense.py \
  --input-jsonl example/training14b_10pct_sample.jsonl \
  --index_dir /Users/yun/develop/pubmed_medembed_2026_subset_index \
  --input-batch-jsonls bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl \
  --out_dir output/eval_dense_medembed_small \
  --topk 5000 \
  --ef_cap 2000

python scripts/public/shared_scripts/retrieval/fuse_retrieval.py \
  --bm25-runs-dir output/retrieve_bm25/runs \
  --dense-root output/retrieve_dense \
  --input-jsonl example/training14b_10pct_sample.jsonl \
  --input-batch-jsonls \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl \
  --out_dir output/eval_hybrid_production_test \
  --mode default \
  --jobs 4

# rerank
python scripts/public/shared_scripts/rerank/rerank_crossencoder.py \
  --output-dir output/eval_stage2_rerank_minitest \
  --runs-dir output/fuse_retrieval/runs \
  --docs-jsonl output/subset_pubmed.jsonl \
  --input-jsonl example/training14b_10pct_sample.jsonl \
  --input-batch-jsonls \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl \
  --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --model-device cpu \
  --model-batch 16 \
  --max-queries 5 

python scripts/public/shared_scripts/analysis/compare_result_dirs.py \
  --dirs output/workflow_local_3pct_hpc_bge/rerank_merged output/workflow_local_3pct_hpc_bge/rerank_sentence \
  --labels "Stage 3 merged" "Stage 3 sentence" \
  --plot both \
  --map-ks 10,20,50,100,200 \
  --train-json example/training14b_3pct_sample.json \
  --test_batch_jsons example/13b_golden_50q_sample.json \
  --output-dir output/workflow_local_3pct_hpc_bge/compare_plots_mergedtest

python scripts/public/shared_scripts/rerank/fuse_rerank.py \
  --hybrid-runs-dir output/workflow_local_10pct_hpc_bge/retrieval/fusion/runs \
  --rerank-runs-dir output/workflow_local_10pct_hpc_bge/rerank/cross_encoder/runs \
  --output-dir output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion \
  --input-jsonl example/training14b_10pct_sample.jsonl \
  --input-batch-jsonls bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl \
  --pool-top 50 --k-rrf 60 --w-bge 0.8 --w-hybrid 0.2

python scripts/public/shared_scripts/analysis/compare_result_dirs.py \
  --dirs output/workflow_local_10pct_hpc_bge/rerank/cross_encoder output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion \
  --labels "rerank" "rerank_rrf_hybrid" \
  --output-dir output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json 

#evidence
python3 "scripts/public/shared_scripts/evidence/build_retrieval_jsonl.py" \
  --run-path "output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion/runs/best_rrf_training14b_10pct_sample_top5000_rrf_pool50_k60.tsv" \
  --query-jsonl example/training14b_10pct_sample.jsonl \
  --output-path "output/workflow_local_10pct_hpc_bge/post_rerank_training14b_10pct_sample.jsonl" \
  --top-k 10

python3 "scripts/public/shared_scripts/evidence/build_doc_contexts.py" \
  --post-rerank-jsonl "output/workflow_local_10pct_hpc_bge/post_rerank_training14b_10pct_sample.jsonl" \
  --corpus-path output/subset_pubmed.jsonl \
  --output-path "output/workflow_local_10pct_hpc_bge/evidence/training14b_10pct_sample_contexts.jsonl"

#rewriting is depracted
#after query rewriting
python scripts/public/shared_scripts/analysis/compare_result_dirs.py \
  --dirs output/workflow_local_10pct_hpc_bge/rerank/cross_encoder output/workflow_local_10pct_hpc_bge/rerank_body_rewrite_A output/workflow_local_10pct_hpc_bge/rerank_body_rewrite_B \
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

#generation
python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/evidence/13B1_golden_contexts.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation \
  --concurrency 2 \
  --max-contexts 8 \
  --max-chars-per-context 2000 \
  --sleep 0.5

./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh 

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --output-dir output/using_ground_truth_generation/snippet_generation_13B4 \
  --evidence-source snippets
  --max-contexts 12 \
  --max-chars-per-context 1200 \
  --sleep 0.5

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset1 \
  --output-dir bioasq14_output/batch_1_B/14b1.json \
  --evidence-source snippets \
  --max-contexts 12 \
  --max-chars-per-context 1200 \
  --sleep 0.5

python scripts/public/shared_scripts/analysis/low_recall_report.py --output-dir bioasq14_output/batch_1
python scripts/public/shared_scripts/analysis/question_recall_report.py --output-dir bioasq14_output/batch_1

python scripts/public/shared_scripts/analysis/low_recall_report.py \
  --output-dir output/retrieval_test/hyde \
  --stage dense \
  --ground-truth bioasq_data/14b/BioASQ-task14bPhaseB-testset1

python scripts/public/shared_scripts/analysis/low_recall_report.py \
  --output-dir output/retrieval_test/bm25_new \
  --stage dense 
python scripts/public/shared_scripts/analysis/low_recall_report.py \
  --output-dir output/retrieval_test/bgem3 \
  --stage dense 