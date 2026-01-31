cd /shared/home/yun.wang/biolab/yun
srun -p dev --container-image=fulaibaowang/bioasq:28.01.26 --container-save=./bioasq_28.01.26.sqfs


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






srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/bioasq:28.01.26 \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed_test:/pubmed_test" \
  --container-workdir /work \
  --pty bash

python /app/parse_pubmed_local.py \
  --input_dir /pubmed_test \
  --output_dir /pubmed_test/jsonl/ \
  --skip_existing