python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/Task13BGoldenEnriched/13B1_golden.json \
  --output-dir output/using_ground_truth_generation/ \
  --evidence-source snippets  \
  --max-contexts 12 \
  --max-chars-per-context 1000 \
  --sleep 0.5

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/Task13BGoldenEnriched/13B2_golden.json \
  --output-dir output/using_ground_truth_generation/ \
  --evidence-source snippets  \
  --max-contexts 12 \
  --max-chars-per-context 1000 \
  --sleep 0.5

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/Task13BGoldenEnriched/13B3_golden.json \
  --output-dir output/using_ground_truth_generation/ \
  --evidence-source snippets  \
  --max-contexts 12 \
  --max-chars-per-context 1000 \
  --sleep 0.5

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path bioasq_data/Task13BGoldenEnriched/13B4_golden.json \
  --output-dir output/using_ground_truth_generation/ \
  --evidence-source snippets  \
  --max-contexts 12 \
  --max-chars-per-context 1000 \
  --sleep 0.5

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path example/training14b_10pct_sample.json \
  --output-dir output/using_ground_truth_generation/ \
  --evidence-source snippets  \
  --max-contexts 12 \
  --max-chars-per-context 1000 \
  --sleep 0.5