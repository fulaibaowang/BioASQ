# subset
## bm25
python3 scripts/public/data/extract_jsonl_subset_by_pmids.py  --jsonl_glob "../biolab/pubmed/jsonl_2026" --pmid_list "example/subset_pmids.txt" --output_jsonl "output/subset_pubmed.jsonl" --dedup --stop_when_complete 

srun -p dev --time=12:00:00 -c 4 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/subset_pubmed.jsonl"   --index_path "/work/output/pubmed_bm25_2026_subset_index"   --threads 4   --overwrite

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
  --jsonl_glob "/work/output/subset_pubmed.jsonl" \
  --out_dir "/pubmed/pubmed_medembed_2026_subset_index" \
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
  --train_subset_json example/training14b_10pct_sample.json \
  --index_dir /pubmed/pubmed_pubmedbert_2026_subset_index/pubmedbert_hnsw_69667 \
  --test_batch_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --out_dir output/eval_dense_pubmedbert_small \
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
  --output-dir output/eval_stage2_rerank_hpc_minitest \
  --runs-dir output/eval_hybird_production_test/runs \
  --docs-jsonl output/subset_pubmed.jsonl \
  --train_subset_json example/training14b_10pct_sample.json \
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
  --runs-dir output/workflow_local_3pct_hpc_bge/hybrid/runs \
  --run-files \
    output/workflow_local_3pct_hpc_bge/hybrid/runs/best_rrf_13b_golden_50q_sample_top5000.tsv \
    output/workflow_local_3pct_hpc_bge/hybrid/runs/best_rrf_training14b_3pct_sample_top5000.tsv \
  --docs-jsonl output/subset_pubmed.jsonl \
  --train-subset-json example/training14b_3pct_sample.json \
  --test-batch-jsons example/13b_golden_50q_sample.json \
  --candidate-limit 1000 \
  --dense-model abhinand/MedEmbed-small-v0.1 \
  --model BAAI/bge-reranker-v2-m3 \
  --model-batch 56 \
  --model-max-length 512 \
  --model-device cuda \
  --output-dir output/workflow_local_3pct_hpc_bge/rerank_sentence


python scripts/deprecated/rerank_stage3_merged.py \
  --runs-dir output/workflow_local_3pct_hpc_bge/hybrid/runs \
  --run-files \
    output/workflow_local_3pct_hpc_bge/hybrid/runs/best_rrf_13b_golden_50q_sample_top5000.tsv \
    output/workflow_local_3pct_hpc_bge/hybrid/runs/best_rrf_training14b_3pct_sample_top5000.tsv \
  --docs-jsonl output/subset_pubmed.jsonl \
  --train-subset-json example/training14b_3pct_sample.json \
  --test-batch-jsons example/13b_golden_50q_sample.json \
  --candidate-limit 1000 \
  --sentence-picks-dir output/workflow_local_3pct_hpc_bge/rerank_sentence/sentence_picks \
  --model BAAI/bge-reranker-v2-m3 \
  --model-batch 56 \
  --model-max-length 512 \
  --model-device cuda \
  --output-dir output/workflow_local_3pct_hpc_bge/rerank_merged


#Re-run guard‑rail offline on  finished workflow:
python scripts/public/shared_scripts/rerank/rerank_guard_rail_topk.py \
  --hybrid-runs-dir output/workflow_local_10pct_hpc_bge/hybrid/runs \
  --rerank-runs-dir output/workflow_local_10pct_hpc_bge/rerank/runs \
  --output-dir output/workflow_local_10pct_hpc_bge/rerank/guard_rail_topk \
  --train-subset-json example/training14b_10pct_sample.json \
  --test-batch-jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --k-top 10 --m-bge 8


##############################!##############################################################
# whole dataset
## parse xlm and build bm25
cd ~/BioASQ
srun -p dev --time=12:00:00 -c 4 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python scripts/public/data/parse_pubmed_local.py \
    --input_dir /pubmed/baseline2026 \
    --output_dir /pubmed/jsonl_2026/ \
    --skip_existing

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/pubmed/pubmed_bm25_2026_index" \
  --threads 4 \
  --overwrite

srun -p dev --time=12:00:00 --gres=gpu:A100_80GB:1 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/pubmed/jsonl_2026/*.jsonl" \
  --out_dir "/pubmed/pubmed_medembed_2026_index" \
  --device "cuda" \
  --batch_size 512 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --dedup_pmids
  
max_elements 39994988
max_elements 42000000
