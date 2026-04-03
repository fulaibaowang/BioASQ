# Canonical HyDE-only script (no facet/listwise):
#   scripts/public/query_parsing/prepare_hyde_query.py
#   scripts/public/query_parsing/hyde_prompt.md
#
# The version in this directory (private_scripts) includes experimental
# facet + listwise support and is kept as reference.

python scripts/public/query_parsing/prepare_query.py \
  bioasq_data/14b/BioASQ-task14bPhaseA-testset2-rewrite \
  -o bioasq_data/14b/BioASQ-task14bPhaseA-testset2-rewrite-ready.json --no-fallback

python3 scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset2 \
  --output-dir bioasq14_output/batch_2_B \
  --evidence-source snippets

python3 scripts/public/shared_scripts/generation/rescue_failed_generation.py \
  --input bioasq14_output/batch_2_B/BioASQ-task14bPhaseB-testset2_answers.json