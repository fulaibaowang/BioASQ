cd /shared/home/yun.wang/biolab/yun
srun -p dev --container-image=fulaibaowang/bioasq:28.01.26 --container-save=./bioasq_28.01.26.sqfs

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
  --index_path "/pubmed/pubmed_bm26_index" \
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
srun -p dev --time=12:00:00 --gres=gpu:L4:1 -c 4 --mem=64G\
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_28.01.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash