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
  --dedup_pmids