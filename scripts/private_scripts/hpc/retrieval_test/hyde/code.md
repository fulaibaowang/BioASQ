# Canonical HyDE-only script (no facet/listwise):
#   scripts/public/query_parsing/prepare_hyde_query.py
#   scripts/public/query_parsing/hyde_prompt.md
#
# The version in this directory (private_scripts) includes experimental
# facet + listwise support and is kept as reference.

python scripts/public/query_parsing/prepare_hyde_query.py \
  example/hyde/raw/13b_golden_50q_sample.hyde_ready.json \
  -o example/hyde/ready/13b_golden_50q_sample.json --no-fallback

python scripts/public/query_parsing/prepare_hyde_query.py \
  example/hyde/raw/BioASQ-task14bPhaseB-testset1.hyde_ready.json \
  -o example/hyde/ready/BioASQ-task14bPhaseB-testset1.json --no-fallback

python scripts/public/query_parsing/prepare_hyde_query.py \
  example/hyde/raw/training14b_3pct_sample.hyde_ready.json \
  -o example/hyde/ready/training14b_3pct_sample.json --no-fallback