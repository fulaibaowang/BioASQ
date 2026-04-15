#!/bin/bash
#SBATCH -J pipeline_bioasq_cpu
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# CPU-only BioASQ pipeline. Request more cores to speed up cross-encoder reranking
# (OpenMP/MKL inside a single Python process). Multi-GPU rerank in the codebase is
# CUDA-only; extra Slurm CPUs do not spawn extra rerank replicas automatically.
#
# To use more CPUs: raise --cpus-per-task and optionally increase
# RERANK_MODEL_BATCH / SNIPPET_CE_BATCH in config_cpu.env if RAM allows.
#
# Oversubscription note: HYBRID_JOBS in config_cpu.env uses multiple processes.
# If hybrid is slow or CPU usage looks wrong, set CPU_OMP_THREADS before sbatch:
#   export CPU_OMP_THREADS=$(( SLURM_CPUS_PER_TASK / 8 ))
#   sbatch sbatch_pipeline_cpu.sh
# (Or lower HYBRID_JOBS in config_cpu.env.)

set -euo pipefail

cd ~/BioASQ
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

PIPELINE_CONFIG="scripts/private_scripts/hpc/vega/config_cpu.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running pipeline script with config: ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS} (CPU partition; expect 0)"

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

_SLURM_CPUS="${SLURM_CPUS_PER_TASK:-64}"
_OMP="${CPU_OMP_THREADS:-${_SLURM_CPUS}}"

srun singularity exec \
  --cleanenv \
  "${APPTAINER_GPU_ARGS[@]}" \
  -B "${WORKDIR}:/work" \
  -B "${PUBMED_HOST}:/pubmed" \
  -B "${YUN_HOST}:/yun" \
  -B "${HOME_HOST}:/home/wangy" \
  --pwd /work \
  "${CONTAINER_IMG}" \
  bash -lc "
    set -euo pipefail

    export HF_HOME='/yun/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$HF_HOME/transformers\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS='${_OMP}'
    export MKL_NUM_THREADS='${_OMP}'
    export OPENBLAS_NUM_THREADS='${_OMP}'
    export PYTHONUNBUFFERED=1
    export TQDM_DISABLE=1

    echo \"[cpu] SLURM_CPUS_PER_TASK=${_SLURM_CPUS} OMP_NUM_THREADS=\${OMP_NUM_THREADS}\"

    source '${PIPELINE_CONFIG}'
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"
    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    echo \"[run] Starting retrieval + rerank + evidence + generation pipeline (CPU)\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Pipeline completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
