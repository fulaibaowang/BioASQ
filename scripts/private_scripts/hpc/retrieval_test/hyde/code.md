python scripts/private_scripts/hpc/retrieval_test/hyde/prepare_hyde_query.py \
  example/hyde/raw/13b_golden_50q_sample.hyde_ready.json \
  -o example/hyde/ready/13b_golden_50q_sample.json

python scripts/private_scripts/hpc/retrieval_test/hyde/prepare_hyde_query.py \
  example/hyde/raw/BioASQ-task14bPhaseB-testset1.hyde_ready.json \
  -o example/hyde/ready/BioASQ-task14bPhaseB-testset1.json

python scripts/private_scripts/hpc/retrieval_test/hyde/prepare_hyde_query.py \
  example/hyde/raw/training14b_3pct_sample.hyde_ready.json \
  -o example/hyde/ready/training14b_3pct_sample.json