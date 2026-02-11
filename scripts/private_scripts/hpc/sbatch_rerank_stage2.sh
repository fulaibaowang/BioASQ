#!/bin/bash
#SBATCH -J stage2_rerank
#SBATCH -p dev
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100_80GB:2
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

# Input: hybrid Stage 1 runs (top 2000 candidates)
RUNS_DIR="output/eval_hybird_production_test/runs"

# Input: PubMed subset texts
DOCS_JSONL="output/subset_pubmed.jsonl"

# Output location (inside container)
OUT_DIR="output/eval_stage2_rerank_bge_reranker_v2_m3"

MODEL_NAME="BAAI/bge-reranker-v2-m3"

# Reranker parameters
USE_MULTI_GPU=true
NUM_GPUS=2
BATCH_SIZE=48

# Selected runs to rerank (all splits)
SELECTED_RUNS="13B1_golden 13B2_golden 13B3_golden 13B4_golden train_subset"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "USE_MULTI_GPU=${USE_MULTI_GPU}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SELECTED_RUNS=${SELECTED_RUNS}"

# Generate Python script from notebook (if not using interactive)
# Uncomment if you need to sync notebook to .py first:
# jupytext --to py notebooks/rerank_stage2-hpc.ipynb --output notebooks/rerank_stage2_hpc.py

# -----------------------------
# Run inside container
# -----------------------------
srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed" \
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
    echo '[run] Starting reranker with multi-GPU...'
    echo '[note] Make sure notebook is configured with:'
    echo '  - USE_MULTI_GPU = True'
    echo '  - NUM_GPUS = 2'
    echo '  - BATCH_SIZE = 48'
    echo '  - MODEL_NAME = "BAAI/bge-reranker-v2-m3"'
    echo '  - SELECTED_RUNS includes all splits'
    python -u scripts/private_scripts/hpc/rerank_stage2_bge.py

    echo '[done] Reranking complete'
    ls -lh '${OUT_DIR}' || true
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
