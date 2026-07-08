#!/bin/bash
#SBATCH -J bioasq_rescue
#SBATCH -p dev,frida
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Rescue failed questions in one generator-config's arms (decoding fix).
#   GENERATOR, OLLAMA_SERVE_URL, OUT, GENERATION_THINK (for gemma), REPEAT_PENALTY, NUM_PREDICT
set -euo pipefail
cd ~/BioASQ; mkdir -p logs
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
PUBMED_HOST="/shared/workspace/biolab/pubmed"; yun="/shared/workspace/biolab/yun"; WORKDIR="${PWD}"
GENERATOR="${GENERATOR:-llama3.3:latest}"
GENERATION_THINK="${GENERATION_THINK:-}"
OLLAMA_SERVE_URL="${OLLAMA_SERVE_URL:-http://ana:11434/api/generate}"
OUT="${OUT:?set OUT}"
# Default: NO decoding override (match the arms' original decoding). The batch
# failures were transient serve glitches, cleared by a plain re-run.
REPEAT_PENALTY="${REPEAT_PENALTY:-}"; NUM_PREDICT="${NUM_PREDICT:-}"

srun --container-image="${CONTAINER_IMG}" --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun" \
  --container-workdir /work \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
    export GENERATION_BACKEND=ollama LLAMA_API_KEY=local-selfhost
    export OLLAMA_URL='"$OLLAMA_SERVE_URL"'
    export GENERATION_THINK="'"$GENERATION_THINK"'"
    export GENERATION_REPEAT_PENALTY='"$REPEAT_PENALTY"' GENERATION_NUM_PREDICT='"$NUM_PREDICT"'
    REPO_ROOT=/work PY=python OUT='"$OUT"' GENERATOR='"$GENERATOR"' \
      bash scripts/private_scripts/hpc/gen_pipeline_test/oracle/run_rescue.sh
  '
echo "Finished ${SLURM_JOB_ID} at $(date)"
