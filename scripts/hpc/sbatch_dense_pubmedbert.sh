#!/bin/bash
#SBATCH -J dense_pubmedbert_hnsw
#SBATCH -p frida
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

cd ~/BioASQ
# Adjust these paths as needed
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"

JSONL_GLOB="/work/output/subset_pubmed.jsonl" # <-- EDIT
OUT_DIR="/pubmed/pubmed_pubmedbert_2026_subset_index"    # <-- EDIT

MODEL_NAME="NeuML/pubmedbert-base-embeddings"

# Recommended starting batch size for PubMedBERT on L4
BATCH_SIZE=128

echo "Starting job on $(hostname) at $(date)"
echo "JSONL_GLOB=${JSONL_GLOB}"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}"

srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail
    python data/build_dense_hnsw_index_from_jsonl_shards.py \
      --jsonl_glob '${JSONL_GLOB}' \
      --model_name '${MODEL_NAME}' \
      --out_dir '${OUT_DIR}' \
      --device 'cuda' \
      --batch_size ${BATCH_SIZE} \
      --M 32 \
      --ef_construction 200 \
      --ef_search 100
  "

echo "Finished at $(date)"
