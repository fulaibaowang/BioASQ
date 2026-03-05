python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/generation/13B1_golden_answers.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation_after_tuning/ \
  --sleep 0.5 

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/generation/13B2_golden_answers.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation_after_tuning/ \
  --sleep 0.5 

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/generation/13B3_golden_answers.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation_after_tuning/ \
  --sleep 0.5 

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/generation/13B4_golden_answers.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation_after_tuning/ \    
  --sleep 0.5 

python scripts/public/shared_scripts/generation/generate_answers.py \
  --input-path output/workflow_local_10pct_hpc_bge/generation/training14b_10pct_sample_answers.json \
  --output-dir output/workflow_local_10pct_hpc_bge/generation_after_tuning/ \
  --sleep 0.5 
