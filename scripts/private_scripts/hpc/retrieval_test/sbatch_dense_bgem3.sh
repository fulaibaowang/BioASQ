#!/bin/bash
#SBATCH -J dense_bgem3_hnsw
#SBATCH -p frida
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

# Input JSONLs as seen inside container (/work is ${WORKDIR} on host)
JSONL_GLOB="/yun/output/subset_pubmed.jsonl"

# Final output location on shared storage (inside container under /pubmed)
OUT_FINAL="/yun/indexes/pubmed_bgem3_2026_subset_index"

MODEL_NAME="BAAI/bge-m3"

# -----------------------------
# Params (safer defaults)
# -----------------------------
BATCH_SIZE=512
MAX_ELEMENTS=3600000
M=32
EF_CONSTRUCTION=200
EF_SEARCH=100

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "JSONL_GLOB=${JSONL_GLOB}"
echo "OUT_FINAL=${OUT_FINAL}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}  MAX_ELEMENTS=${MAX_ELEMENTS}"
echo "HNSW: M=${M}  ef_construction=${EF_CONSTRUCTION}  ef_search=${EF_SEARCH}"

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

    # --- Persistent HF cache on shared workspace (avoid /shared/home which is full) ---
    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    # --- Build locally on node (/tmp) then copy to shared storage ---
    OUT_LOCAL=\"\${TMPDIR:-/tmp}/pubmedbert_hnsw_\${SLURM_JOB_ID}\"
    mkdir -p \"\$OUT_LOCAL\"
    echo \"[paths] OUT_LOCAL=\$OUT_LOCAL\"
    echo \"[paths] OUT_FINAL=${OUT_FINAL}\"

    python -u scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
      --jsonl_glob '${JSONL_GLOB}' \
      --model_name '${MODEL_NAME}' \
      --out_dir \"\$OUT_LOCAL\" \
      --device 'cuda' \
      --batch_size ${BATCH_SIZE} \
      --M ${M} \
      --ef_construction ${EF_CONSTRUCTION} \
      --ef_search ${EF_SEARCH} \
      --max_elements ${MAX_ELEMENTS}

    # Copy contents into OUT_FINAL (not OUT_FINAL/<jobdir>/)
    mkdir -p '${OUT_FINAL}'
    cp -a \"\$OUT_LOCAL\"/. '${OUT_FINAL}/'
    chmod -R a+rX '${OUT_FINAL}' || true

    echo \"[done] Copied outputs to ${OUT_FINAL}\"
    du -sh '${OUT_FINAL}' || true
    ls -lah '${OUT_FINAL}' || true
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
