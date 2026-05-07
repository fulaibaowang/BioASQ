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

python scripts/public/query_parsing/prepare_query.py \
  bioasq_data/14b/BioASQ-task14bPhaseA-testset3-rewrite \
  -o bioasq_data/14b/BioASQ-task14bPhaseA-testset3-rewrite-ready.json --no-fallback

python3 scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset3 \
  --output-dir bioasq14_output/batch_3_B_ \
  --evidence-source snippets --max-contexts 13

python3 scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset3 \
  --output-dir bioasq14_output/batch_3_B_18 \
  --evidence-source snippets --max-contexts 18

python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input bioasq_data/14b/BioASQ-task14bPhaseA-testset3 \
  --output bioasq_data/14b/BioASQ-task14bPhaseA-testset3.jsonl

python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input bioasq_data/14b/BioASQ-task14bPhaseA-testset4 \
  --output bioasq_data/14b/BioASQ-task14bPhaseA-testset4.jsonl

python scripts/public/query_parsing/prepare_query.py \
  bioasq_data/14b/BioASQ-task14bPhaseA-testset4-rewrite \
  -o bioasq_data/14b/BioASQ-task14bPhaseA-testset4-rewrite-ready.json --no-fallback

python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input bioasq_data/14b/BioASQ-task14bPhaseA-testset4-rewrite \
  --output bioasq_data/14b/BioASQ-task14bPhaseA-testset4-rewrite-ready.jsonl

# All *_answers.jsonl under bioasq14_output/batch_4*/... → sibling *_bioasq.json
for top in bioasq14_output/batch_4*/; do
  [ -d "$top" ] || continue
  find "$top" -name '*_answers.jsonl' -type f | sort | while IFS= read -r f; do
    out="${f%.jsonl}_bioasq.json"
    python3 scripts/public/format/queries_jsonl_to_bioasq_json.py --input "$f" --output "$out" --pretty --max-documents 10
  done
done

export GENERATION_BACKEND=openrouter
export GEN_API_BASE="https://openrouter.ai/api/v1"

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path "/Users/yun/develop/BioASQ/bioasq14_output/batch_4_gemma_b200/evidence/evidence_listwise/BioASQ-task14bPhaseA-testset4-rewrite-ready_contexts.jsonl" \
  --output-dir "/Users/yun/develop/BioASQ/bioasq14_output/batch_4_gemma_b200/generation/generation_openrouter_gpt41" \
  --model "openai/gpt-4.1:floor" \
  --max-contexts 22 \
  --timeout 300 \
  --schemas-dir "/Users/yun/develop/BioASQ/scripts/public/prompts/schemas" \
  --concurrency 1


# Conversion copies BioASQ ``snippets`` onto each JSONL line so ``--evidence-source snippets`` sees them.
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input bioasq_data/14b/BioASQ-task14bPhaseB-testset4 \
  --output bioasq_data/14b/BioASQ-task14bPhaseB-testset4.jsonl

python3 scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset4.jsonl \
  --output-dir bioasq14_output/batch_4_B_13 \
  --evidence-source snippets --max-contexts 13 --schemas-dir "/Users/yun/develop/BioASQ/scripts/public/prompts/schemas"

python3 scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/14b/BioASQ-task14bPhaseB-testset4.jsonl \
  --output-dir bioasq14_output/batch_4_B_18 \
  --evidence-source snippets --max-contexts 18 --schemas-dir "/Users/yun/develop/BioASQ/scripts/public/prompts/schemas"

 python3 scripts/public/format/queries_jsonl_to_bioasq_json.py --input "bioasq14_output/batch_4_B_13/BioASQ-task14bPhaseB-testset4_answers.jsonl" --output "bioasq14_output/batch_4_B_13/BioASQ-task14bPhaseB-testset4_answers.json" --pretty

  python3 scripts/public/format/queries_jsonl_to_bioasq_json.py --input "bioasq14_output/batch_4_B_18/BioASQ-task14bPhaseB-testset4_answers.jsonl" --output "bioasq14_output/batch_4_B_18/BioASQ-task14bPhaseB-testset4_answers.json" --pretty