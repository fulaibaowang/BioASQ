cd ~/BioASQ
srun -p dev --time=12:00:00 -c 16 --mem=96G \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed,/shared/workspace/biolab/yun:/yun" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/pubmed/jsonl_2026/*.jsonl" \
  --index_path "/pubmed/pubmed_bm25_2026_index" \
  --threads 16 \
  --overwrite

srun -p dev --time=12:00:00 -c 4 --gres=gpu:A100_80GB:1 --mem=256G \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed,/shared/workspace/biolab/yun:/yun" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/pubmed/jsonl_2026/*.jsonl" \
  --out_dir "/pubmed/pubmed_medembed_2026_index" \
  --device "cuda" \
  --batch_size 256 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --max_elements 42000000