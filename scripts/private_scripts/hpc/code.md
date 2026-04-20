# subset
## bm25
python3 scripts/public/data/extract_jsonl_subset_by_pmids.py  --jsonl_glob "../biolab/pubmed/jsonl_2026" --pmid_list "example/subset_pmids.txt" --output_jsonl "/yun/output/subset_pubmed.jsonl" --dedup --stop_when_complete 

srun -p dev --time=12:00:00 -c 4 --gres=gpu:1  --mem=64G \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed,/shared/workspace/biolab/yun:/yun" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/yun/output/subset_pubmed.jsonl"   --index_path "/yun/output/pubmed_bm25_2026_subset_index"   --threads 4   --overwrite

python3 scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/yun/output/subset_pubmed.jsonl" \
  --index_path "/yun/indexes/pubmed_bm25_2026_subset_index_with_keywords" \
  --threads 4 \
  --include_keywords

## dense
### small model medembed
cd /shared/home/yun.wang/biolab/yun
srun -p dev --container-image=fulaibaowang/bioasq:20.02.26 --container-save=./bioasq_20.02.26.sqfs  true

srun -p dev --time=12:00:00 --gres=gpu:L4:1 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/yun/output/subset_pubmed.jsonl" \
  --out_dir "/pubmed/pubmed_medembed_2026_subset_index" \
  --device "cuda" \
  --batch_size 256 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --dedup_pmids

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/yun/output/subset_pubmed.jsonl" \
  --model_name "BAAI/bge-m3" \
  --out_dir "/yun/indexes/pubmed_bgem3_2026_subset_index" \
  --device "cuda" \
  --batch_size 256 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --dedup_pmids

### bigger model medembed
sbatch sbatch_dense_pubmedbert.sh  #<--  change max_elements accordingly

srun -p dev --time=12:00:00 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/retrieval/eval_dense.py \
  --train-json example/training14b_10pct_sample.json \
  --index_dir /pubmed/pubmed_pubmedbert_2026_subset_index/pubmedbert_hnsw_69667 \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir /yun/output/eval_dense_pubmedbert_small \
  --topk 5000 \
  --ef_cap 5000

# reranker
stunnel -c16 --time=12:00:00 --mem=64G --gres=gpu:A100_80GB:1 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --job-name reranker \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work 


srun -p dev --time=1:00:00 -c 4 --mem=32G --gres=gpu:1 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash
  
python scripts/public/shared_scripts/rerank/rerank_stage2.py \
  --output-dir /yun/output/eval_stage2_rerank_hpc_minitest \
  --runs-dir /yun/output/eval_hybrid_production_test/runs \
  --docs-jsonl /yun/output/subset_pubmed.jsonl \
  --train-json example/training14b_10pct_sample.json \
  --test_batch_jsons \
    bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
    bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --model-device cuda \
  --model-batch 32 

  --use-multi-gpu \
  --num-gpus 2

sbatch scripts/private_scripts/hpc/sbatch_rerank_stage2.sh



# pipeline
srun -p dev --time=8:00:00 -c 4 --mem=32G --gres=gpu:A100:1 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/hpc/config_3pct.env

#test stage3 (deprecated; scripts moved to scripts/deprecated/)
python scripts/deprecated/rerank_stage3_sentence.py \
  --runs-dir /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs \
  --run-files \
    /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs/best_rrf_13b_golden_50q_sample_top5000.tsv \
    /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs/best_rrf_training14b_3pct_sample_top5000.tsv \
  --docs-jsonl /yun/output/subset_pubmed.jsonl \
  --train-json example/training14b_3pct_sample.json \
  --test-batch-jsons example/13b_golden_50q_sample.json \
  --candidate-limit 1000 \
  --dense-model abhinand/MedEmbed-small-v0.1 \
  --model BAAI/bge-reranker-v2-m3 \
  --model-batch 56 \
  --model-max-length 512 \
  --model-device cuda \
  --output-dir /yun/output/workflow_local_3pct_hpc_bge/rerank_sentence


python scripts/deprecated/rerank_stage3_merged.py \
  --runs-dir /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs \
  --run-files \
    /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs/best_rrf_13b_golden_50q_sample_top5000.tsv \
    /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs/best_rrf_training14b_3pct_sample_top5000.tsv \
  --docs-jsonl /yun/output/subset_pubmed.jsonl \
  --train-json example/training14b_3pct_sample.json \
  --test-batch-jsons example/13b_golden_50q_sample.json \
  --candidate-limit 1000 \
  --sentence-picks-dir /yun/output/workflow_local_3pct_hpc_bge/rerank_sentence/sentence_picks \
  --model BAAI/bge-reranker-v2-m3 \
  --model-batch 56 \
  --model-max-length 512 \
  --model-device cuda \
  --output-dir /yun/output/workflow_local_3pct_hpc_bge/rerank_merged


# Run RRF fusion (Hybrid + Rerank) offline on finished workflows:
python scripts/public/shared_scripts/rerank/rerank_rrf_hybrid.py \
  --hybrid-runs-dir /yun/output/workflow_local_10pct_hpc_bge/retrieval/fusion/runs \
  --rerank-runs-dir /yun/output/workflow_local_10pct_hpc_bge/rerank/cross_encoder/runs \
  --output-dir /yun/output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion \
  --train-json example/training14b_10pct_sample.json \
  --test-batch-jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --pool-top 50 --k-rrf 60 --w-bge 0.8 --w-hybrid 0.2

python scripts/public/shared_scripts/rerank/rerank_rrf_hybrid.py \
  --hybrid-runs-dir /yun/output/workflow_local_3pct_hpc_bge/retrieval/fusion/runs \
  --rerank-runs-dir /yun/output/workflow_local_3pct_hpc_bge/rerank/cross_encoder/runs \
  --output-dir /yun/output/workflow_local_3pct_hpc_bge/rerank/post_rerank_fusion \
  --train-json example/training14b_3pct_sample.json \
  --test-batch-jsons example/13b_golden_50q_sample.json \
  --pool-top 50 --k-rrf 60 --w-bge 0.8 --w-hybrid 0.2

python scripts/public/shared_scripts/analysis/compare_result_dirs.py \
  --dirs /yun/output/workflow_local_10pct_hpc_bge/rerank/cross_encoder /yun/output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion \
  --labels "rerank" "rerank_rrf_hybrid" \
  --plot both \
  --map-ks 10 \
  --train-json example/training14b_10pct_sample.json \
  --test-batch-jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --log-x --plots-by-split \
  --output-dir /yun/output/workflow_local_10pct_hpc_bge/rerank/post_rerank_fusion/compare_plots

scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/hpc/config_10pct_sbatch_test.env
