#!/bin/bash
#SBATCH -J pipeline_listwise
#SBATCH -p dev,frida
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/BioASQ
mkdir -p logs

# Stage control: override via env to skip stages already completed.
#   START_STAGE=3 END_STAGE=3  ->  run only generation
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-3}"

# -----------------------------
# Paths / inputs
# -----------------------------
MAIN_CONTAINER="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
LISTWISE_CONTAINER="/shared/home/yun.wang/biolab/yun/bioasqlistwisereranker_16.03.26v3.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"

# Pipeline config (must have RUN_SNIPPET_RRF=1 or use --run-both-routes)
PIPELINE_CONFIG="scripts/private_scripts/hpc/bioasq_runs/config_gemma.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Config: ${PIPELINE_CONFIG}"
echo "Main container:     ${MAIN_CONTAINER}"
echo "Listwise container: ${LISTWISE_CONTAINER}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

# Common srun flags
SRUN_COMMON=(
  --container-mount-home
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun"
  --container-workdir /work
)

# Common env setup (runs inside each container)
HF_CACHE_SETUP='
  export HF_HOME="/pubmed/_hf_cache"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"
  unset TRANSFORMERS_CACHE 2>/dev/null || true
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_HOME/transformers" "$SENTENCE_TRANSFORMERS_HOME"
  export OMP_NUM_THREADS=8
  export PYTHONUNBUFFERED=1
  export TQDM_DISABLE=1
'

# =====================================================================
# Stage 1: Main pipeline (BM25 -> Dense -> Hybrid -> Rerank -> RRF -> Snippet)
# =====================================================================
if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
echo ""
echo "========== Stage 1: Main Pipeline =========="
srun \
  --container-image="${MAIN_CONTAINER}" \
  "${SRUN_COMMON[@]}" \
  bash -lc "
    set -euo pipefail
    ${HF_CACHE_SETUP}

    echo '[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}'
    nvidia-smi -L || true

    source '${PIPELINE_CONFIG}'
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"
    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    echo '[stage-1] Starting main pipeline (with snippet-rrf)'
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh \
      --config '${PIPELINE_CONFIG}' --run-both-routes
    echo '[stage-1] Done'
  "
echo "Stage 1 completed at $(date)"
else
  echo "Skipping Stage 1 (START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE})"
fi

# =====================================================================
# Stage 2: Listwise reranking (RankZephyr in vLLM container)
# =====================================================================
if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
echo ""
echo "========== Stage 2: Listwise Reranking =========="
srun \
  --container-image="${LISTWISE_CONTAINER}" \
  "${SRUN_COMMON[@]}" \
  bash -lc "
    set -euo pipefail
    ${HF_CACHE_SETUP}

    echo '[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}'
    nvidia-smi -L || true

    echo '[stage-2] Starting listwise reranking'
    ./scripts/public/shared_scripts/listwise_script/run_listwise_rerank.sh \
      --config '${PIPELINE_CONFIG}'
    echo '[stage-2] Done'
  "
echo "Stage 2 completed at $(date)"
else
  echo "Skipping Stage 2 (START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE})"
fi

# =====================================================================
# Stage 3: Evidence + Generation on listwise output (main container)
# =====================================================================
if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
echo ""
echo "========== Stage 3: Listwise Evidence + Generation =========="
srun \
  --container-image="${MAIN_CONTAINER}" \
  "${SRUN_COMMON[@]}" \
  bash -lc "
    set -euo pipefail
    ${HF_CACHE_SETUP}

    echo '[stage-3] Starting evidence + generation for listwise output'
    ./scripts/public/shared_scripts/listwise_script/run_listwise_evidence_gen.sh \
      --config '${PIPELINE_CONFIG}'
    echo '[stage-3] Done'
  "
echo "Stage 3 completed at $(date)"
else
  echo "Skipping Stage 3 (START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE})"
fi

echo ""
echo "Finished job ${SLURM_JOB_ID} at $(date)"
