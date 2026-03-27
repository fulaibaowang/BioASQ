#!/bin/bash
#SBATCH -J stage2_rerank
#SBATCH -p dev
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100_80GB:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=312G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/BioASQ

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs"
WORKDIR="${PWD}"

# Host path that will be mounted as /pubmed inside container
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"

# Input: hybrid Stage 1 runs (top 2000 candidates)
RUNS_DIR="/yun/output/eval_hybrid_production_test/runs"

# Input: PubMed subset texts
DOCS_JSONL="/yun/output/subset_pubmed.jsonl"

# Output location (inside container)
OUT_DIR="/yun/output/eval_stage2_rerank_bge_reranker_v2_m3"

# NOTE: Reranker parameters (MODEL_NAME, USE_MULTI_GPU, NUM_GPUS, BATCH_SIZE, SELECTED_RUNS)
# are hardcoded in rerank_stage2_bge.py, not passed via CLI arguments.
# To modify them, edit the Python script directly.

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running: rerank_stage2_bge.py with multi-GPU reranking"

# Generate Python script from notebook (if not using interactive)
# Uncomment if you need to sync notebook to .py first:
# jupytext --to py notebooks/rerank_stage2-hpc.ipynb --output notebooks/rerank_stage2_hpc.py

# -----------------------------
# Run inside container
# -----------------------------
srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail

    # --- Persistent HF cache on shared workspace ---
    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    # --- Set up environment for multi-GPU ---
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1

    # --- Run reranker script with multi-GPU support ---
    echo '[run] Starting reranker (see rerank_stage2_bge.py for config)'
    python -u scripts/private_scripts/hpc/rerank_stage2_bge.py

    echo '[done] Reranking complete'
    ls -lh '${OUT_DIR}' || true
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
