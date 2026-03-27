#!/bin/bash
#SBATCH -J dense_medembed_10shards
#SBATCH -p dev
#SBATCH --array=0-9
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

cd ~/BioASQ
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"

JSONL_SRC="/pubmed/jsonl_2026/*.jsonl"
SHARD_WORK_ROOT="/pubmed/shard_work"
OUT_FINAL_ROOT="/pubmed/pubmed_medembed_shard"
SEED=42
N_SHARDS=10
MAX_ELEMENTS=4200000

SHARD_ID="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
SHARD_DIR="${SHARD_WORK_ROOT}/shard${SHARD_ID}"
OUT_FINAL="${OUT_FINAL_ROOT}${SHARD_ID}"

# -----------------------------
# Params
# -----------------------------
BATCH_SIZE=256
M=32
EF_CONSTRUCTION=200
EF_SEARCH=100
SAVE_EVERY=250000

echo "Starting array task ${SLURM_ARRAY_TASK_ID} (shard ${SHARD_ID}) job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "SHARD_DIR=${SHARD_DIR}  OUT_FINAL=${OUT_FINAL}"
echo "MAX_ELEMENTS=${MAX_ELEMENTS}  SAVE_EVERY=${SAVE_EVERY}"

# -----------------------------
# Skip if final index already exists
# -----------------------------
if [ -f "${OUT_FINAL}/hnsw_index.bin" ] && \
   [ -f "${OUT_FINAL}/rowid_to_pmid.tsv" ] && \
   [ -f "${OUT_FINAL}/meta.json" ]; then
  echo "[skip] Final index already exists in ${OUT_FINAL}; skipping shard ${SHARD_ID}."
  exit 0
fi

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
    SHARD_ID='${SHARD_ID}'
    SHARD_DIR='${SHARD_DIR}'
    OUT_FINAL='${OUT_FINAL}'
    OUT_LOCAL=\"\${TMPDIR:-/tmp}/pubmed_medembed_shard\${SHARD_ID}_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}\"
    mkdir -p \"\$OUT_LOCAL\"

    # --- HF cache on shared workspace ---
    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"

    # --- 1) Prepare shard: shuffle, split, symlinks + manifest ---
    python -u scripts/private_scripts/hpc/full_test/prepare_jsonl_shards.py \
      --jsonl_glob '${JSONL_SRC}' \
      --seed ${SEED} \
      --n_shards ${N_SHARDS} \
      --shard_index ${SLURM_ARRAY_TASK_ID} \
      --out_shard_dir \"\$SHARD_DIR\"

    # --- 2) Build dense index (local out_dir then copy) ---
    python -u scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
      --jsonl_glob \"\$SHARD_DIR/*.jsonl\" \
      --out_dir \"\$OUT_LOCAL\" \
      --device 'cuda' \
      --batch_size ${BATCH_SIZE} \
      --M ${M} \
      --ef_construction ${EF_CONSTRUCTION} \
      --ef_search ${EF_SEARCH} \
      --max_elements ${MAX_ELEMENTS} \
      --save_every ${SAVE_EVERY}

    # --- 3) Copy to final location on shared storage ---
    mkdir -p \"\$OUT_FINAL\"
    cp -a \"\$OUT_LOCAL\"/. \"\$OUT_FINAL/\"
    echo \"[done] Copied to \$OUT_FINAL\"
    ls -lh \"\$OUT_FINAL\" || true
  "

echo "Finished array task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} at $(date)"
