#!/bin/bash
# Rerank sweep over query fields: body_rewrite_A, body_rewrite_B.
# Uses hybrid runs from 10pct workflow and train/test from example/query_rewrite/*_rewrite.json.
# One job per query field; outputs to /yun/output/rerank_sweep_query_rewrite/rerank_<query_field>/.
#SBATCH -J rerank_sweep_rewrite
#SBATCH -p dev
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/BioASQ

CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"

CONFIG="scripts/private_scripts/hpc/config_10pct_rerank_sweep_rewrite.env"
if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "$CONFIG"
set +a

# Query fields to sweep (must exist as keys in the rewrite JSONs)
QUERY_FIELDS=(body_rewrite_A body_rewrite_B)

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-1}}"
export NUM_GPUS

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Config: $CONFIG"
echo "Sweep: ${QUERY_FIELDS[*]}"
echo "Output base: $RERANK_SWEEP_OUTPUT_DIR"

for RERANK_QUERY_FIELD in "${QUERY_FIELDS[@]}"; do
  OUT_DIR="$RERANK_SWEEP_OUTPUT_DIR/rerank_${RERANK_QUERY_FIELD}"
  echo ""
  echo "[sweep] Running rerank with --query-field $RERANK_QUERY_FIELD -> $OUT_DIR"
  srun \
    --container-image="${CONTAINER_IMG}" \
    --container-mount-home \
    --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun" \
    --container-workdir /work \
    bash -lc "
      set -euo pipefail
      # Ensure Python can import retrieval_eval from scripts/public/shared_scripts
      export PYTHONPATH=\"/work/scripts/public/shared_scripts:\${PYTHONPATH:-}\"
      export HF_HOME='/pubmed/_hf_cache'
      export HF_HUB_CACHE=\"\$HF_HOME/hub\"
      export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
      export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
      mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
      export OMP_NUM_THREADS=8
      export PYTHONUNBUFFERED=1

      python -u scripts/public/shared_scripts/rerank/rerank_crossencoder.py \
        --runs-dir '${RUNS_DIR}' \
        --run-glob '${RUN_GLOB}' \
        --docs-jsonl '${DOCS_JSONL}' \
        --input-jsonl '${INPUT_JSONL}' \
        --input-batch-jsonls $INPUT_BATCH_JSONLS \
        --query-field '${RERANK_QUERY_FIELD}' \
        --candidate-limit ${RERANK_CANDIDATE_LIMIT} \
        --model '${RERANK_MODEL}' \
        --model-device '${RERANK_MODEL_DEVICE}' \
        --model-batch ${RERANK_MODEL_BATCH} \
        --model-max-length ${RERANK_MODEL_MAX_LENGTH} \
        --ks-recall '${RERANK_KS_RECALL}' \
        --output-dir '${OUT_DIR}' \
        $([ \"${RERANK_USE_MULTI_GPU}\" = \"1\" ] && echo --use-multi-gpu) \
        $([ -n \"${RERANK_NUM_GPUS}\" ] && echo --num-gpus \"${RERANK_NUM_GPUS}\")

      echo '[done] ${RERANK_QUERY_FIELD} -> ${OUT_DIR}'
    "
done

echo ""
echo "Finished job ${SLURM_JOB_ID} at $(date)"
echo "Outputs: $RERANK_SWEEP_OUTPUT_DIR/rerank_body_rewrite_A $RERANK_SWEEP_OUTPUT_DIR/rerank_body_rewrite_B"
