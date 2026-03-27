#!/bin/bash
#SBATCH -J bm25_index
#SBATCH -p frida
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/BioASQ
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "index_path=/pubmed/pubmed_bm25_2026_index"

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
    python -u scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
      --jsonl_glob '/pubmed/jsonl_2026/*.jsonl' \
      --index_path '/pubmed/pubmed_bm25_2026_index' \
      --threads 16 \
      --overwrite
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
