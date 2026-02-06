cd /shared/home/yun.wang/biolab/yun
srun -p dev --container-image=fulaibaowang/bioasq:28.01.26 --container-save=./bioasq_28.01.26.sqfs true

# parse xlm and build bm25
cd ~/BioASQ
srun -p dev --time=12:00:00 -c 4 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_28.01.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash


python data/parse_pubmed_local.py \
    --input_dir /pubmed/baseline2026 \
    --output_dir /pubmed/jsonl_2026/ \
    --skip_existing


python data/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/pubmed/pubmed_bm25_2026_index" \
  --threads 4 \
  --overwrite


# subset
python3 data/extract_jsonl_subset_by_pmids.py  --jsonl_glob "../biolab/pubmed/jsonl_2026" --pmid_list "example/subset_pmids.txt" --output_jsonl "output/subset_pubmed.jsonl" --dedup --stop_when_complete 

srun -p dev --time=12:00:00 -c 4 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_28.01.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python data/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/subset_pubmed.jsonl"   --index_path "/work/output/pubmed_bm25_2026_subset_index"   --threads 4   --overwrite

# dense
cd /shared/home/yun.wang/biolab/yun
srun -p dev --container-image=fulaibaowang/bioasq:04.02.26 --container-save=./bioasq_04.02.26.sqfs  true

srun -p dev --time=12:00:00 --gres=gpu:L4:1 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python data/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/work/output/subset_pubmed.jsonl" \
  --out_dir "/pubmed/pubmed_medembed_2026_subset_index" \
  --device "cuda" \
  --batch_size 256 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --dedup_pmids

sbatch sbatch_dense_pubmedbert.sh
#change max_elements accordingly

srun -p dev --time=12:00:00 --gres=gpu:A100_80GB:1 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash

python data/build_dense_hnsw_index_from_jsonl_shards.py \
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


# eval (inside container, repo mounted at /work)

python retrieval/eval_bm25_rm3_bioasq.py \
  --index_path "output/pubmed_bm25_2026_subset_index" \
  --train_json "example/training14b_10pct_sample.json" \
  --test_batch_jsons Task13BGoldenEnriched/13B1_golden.json Task13BGoldenEnriched/13B2_golden.json Task13BGoldenEnriched/13B3_golden.json Task13BGoldenEnriched/13B4_golden.json \
  --out_dir "output/eval_bm25_rm3" \
  --threads 4 \
  --k_eval 5000 \
  --k_feedback 50 \
  --rm3_fb_docs 20 --rm3_fb_terms 30 --rm3_lambda 0.6 \
  --save_runs --save_per_query

# Add this if you also want BM25 baseline numbers:
#   --include_bm25

python retrieval/eval_dense.py \
  --train_subset_json example/training14b_10pct_sample.json \
  --index_dir /pubmed/pubmed_medembed_2026_subset_index \
  --test_batch_jsons Task13BGoldenEnriched/13B1_golden.json Task13BGoldenEnriched/13B2_golden.json Task13BGoldenEnriched/13B3_golden.json Task13BGoldenEnriched/13B4_golden.json \
  --out_dir output/eval_dense_medembed_small \
  --topk 5000 \
  --ef_cap 2000