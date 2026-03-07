#!/bin/bash
#SBATCH -J pipeline_full_pubmed_sharded
#SBATCH -p frida
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:A100:3


set -euo pipefail

cd ~/BioASQ
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs"
WORKDIR="${PWD}"

# Host path that will be mounted as /pubmed inside container
PUBMED_HOST="/shared/workspace/biolab/pubmed"

# Pipeline config
PIPELINE_CONFIG="scripts/private_scripts/hpc/full_test/config_full_pubmed_sharded_gpus.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running pipeline script with config: ${PIPELINE_CONFIG}"

# Derive number of GPUs from Slurm allocation (for multi-GPU-aware code)
NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-1}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

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

    # --- Shared HF cache on /pubmed ---
    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    echo \"[run] Starting retrieval + rerank + evidence + generation pipeline\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Pipeline completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

