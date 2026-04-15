#!/bin/bash
#SBATCH -J pipeline_listwise
#SBATCH -p gpu
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
# Paths / inputs (VEGA: Apptainer .sif, same bind layout as sbatch_pipeline.sh)
# -----------------------------
# Main pipeline image (non-b200 build; mirror Frida bioasq_08.03.26.sqfs)
MAIN_CONTAINER="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
LISTWISE_CONTAINER="/ceph/hpc/data/s25t12-03-users/apptainer/bioasqlistwisereranker_16.03.26v3b200.sif"
WORKDIR="${PWD}"
PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

# Pipeline config (must have RUN_SNIPPET_RRF=1 or use --run-both-routes in stage 1)
PIPELINE_CONFIG="scripts/private_scripts/hpc/vega/config_gemma.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Config: ${PIPELINE_CONFIG}"
echo "Main container:     ${MAIN_CONTAINER}"
echo "Listwise container: ${LISTWISE_CONTAINER}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

module purge
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

APPTAINER_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  APPTAINER_GPU_ARGS+=(--nv)
else
  echo "No GPUs allocated; running container without --nv"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

SING_BINDS=(
  -B "${WORKDIR}:/work"
  -B "${PUBMED_HOST}:/pubmed"
  -B "${YUN_HOST}:/yun"
  -B "${HOME_HOST}:/home/wangy"
)

# Common env setup (runs inside each container; matches vega/sbatch_pipeline.sh HF layout)
HF_CACHE_SETUP='
  export HF_HOME="/yun/_hf_cache"
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
srun singularity exec \
  --cleanenv \
  "${APPTAINER_GPU_ARGS[@]}" \
  "${SING_BINDS[@]}" \
  --pwd /work \
  "${MAIN_CONTAINER}" \
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
srun singularity exec \
  --cleanenv \
  "${APPTAINER_GPU_ARGS[@]}" \
  "${SING_BINDS[@]}" \
  --pwd /work \
  "${LISTWISE_CONTAINER}" \
  bash -lc "
    set -euo pipefail
    ${HF_CACHE_SETUP}

    echo '[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}'
    nvidia-smi -L || true

    echo '[stage-2] Starting listwise reranking'
    ./scripts/public/shared_scripts/run_listwise_rerank.sh \
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
srun singularity exec \
  --cleanenv \
  "${APPTAINER_GPU_ARGS[@]}" \
  "${SING_BINDS[@]}" \
  --pwd /work \
  "${MAIN_CONTAINER}" \
  bash -lc "
    set -euo pipefail
    ${HF_CACHE_SETUP}

    echo '[stage-3] Starting evidence + generation for listwise output'
    ./scripts/public/shared_scripts/run_listwise_evidence_gen.sh \
      --config '${PIPELINE_CONFIG}'
    echo '[stage-3] Done'
  "
echo "Stage 3 completed at $(date)"
else
  echo "Skipping Stage 3 (START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE})"
fi

echo ""
echo "Finished job ${SLURM_JOB_ID} at $(date)"
