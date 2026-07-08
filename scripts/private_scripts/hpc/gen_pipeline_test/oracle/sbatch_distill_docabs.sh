#!/bin/bash
#SBATCH -J bioasq_distill_docabs
#SBATCH -p dev,frida
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Plan B long pole: shared distillation over the long-context (full gold abstract)
# oracle evidence + token-budget measurement. Generation is NOT run here (arm
# counts get locked from the measurement first). All LLM work is remote ollama.
#
#   SPLIT             default 13B4
#   EXTRACT_MODEL     default llama3.3:latest
#   CLAIM_SLOTS       default "16 50"   FACET_N default "8 16"
#   OLLAMA_SERVE_URL  default http://ana:11434/api/generate
#   DISTILL_DIR       default /yun/bioasq14_output/oracle_docabs_<SPLIT>_distill
set -euo pipefail
cd ~/BioASQ; mkdir -p logs
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
PUBMED_HOST="/shared/workspace/biolab/pubmed"; yun="/shared/workspace/biolab/yun"; WORKDIR="${PWD}"
SPLIT="${SPLIT:-13B4}"
EXTRACT_MODEL="${EXTRACT_MODEL:-llama3.3:latest}"
CLAIM_SLOTS="${CLAIM_SLOTS:-16 50}"
FACET_N="${FACET_N:-8 16}"
OLLAMA_SERVE_URL="${OLLAMA_SERVE_URL:-http://ana:11434/api/generate}"
DISTILL_DIR="${DISTILL_DIR:-/yun/bioasq14_output/oracle_docabs_${SPLIT}_distill}"

srun --container-image="${CONTAINER_IMG}" --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun" \
  --container-workdir /work \
  bash -lc '
    set -euo pipefail
    export HF_HOME=/pubmed/_hf_cache HF_HUB_CACHE=/pubmed/_hf_cache/hub
    export SENTENCE_TRANSFORMERS_HOME=/pubmed/_hf_cache/sentence_transformers
    export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
    export GENERATION_BACKEND=ollama LLAMA_API_KEY=local-selfhost
    export OLLAMA_URL='"$OLLAMA_SERVE_URL"' GENERATION_NUM_CTX=16384
    python -c "import rank_bm25" 2>/dev/null || pip install --quiet --user rank_bm25
    _base="${OLLAMA_URL%/api/generate}"
    curl -sS --max-time 15 "$_base/api/tags" | grep -q "'"$EXTRACT_MODEL"'" \
      || { echo "[preflight] ERROR: '"$EXTRACT_MODEL"' not served at $_base" >&2; exit 1; }
    echo "[preflight] ollama OK ('"$EXTRACT_MODEL"' @ $_base)"
    REPO_ROOT=/work SPLIT='"$SPLIT"' PY=python \
    EXTRACT_MODEL='"$EXTRACT_MODEL"' \
    DISTILL_DIR='"$DISTILL_DIR"' \
    CLAIM_SLOTS="'"$CLAIM_SLOTS"'" FACET_N="'"$FACET_N"'" \
      bash scripts/private_scripts/hpc/gen_pipeline_test/oracle/distill_docabs.sh
  '
echo "Finished ${SLURM_JOB_ID} at $(date)"
