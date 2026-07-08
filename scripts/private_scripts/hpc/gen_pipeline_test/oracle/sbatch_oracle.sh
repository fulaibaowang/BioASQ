#!/bin/bash
#SBATCH -J bioasq_oracle
#SBATCH -p dev,frida
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Oracle generation test (13B4, GT snippets): 4 arms x 4 question types.
# Generation only (no retrieval); hits the self-hosted ollama serve.
# Distillation (extract+summarize) is generator-independent -> built once into
# DISTILL_DIR and reused across generators. Score afterwards with the official
# BioASQ evaluator (../Evaluation-Measures), run locally on pulled-back answers.
#
# Env (override per submission):
#   GENERATOR         default llama3.3:latest   (final answer model)
#   OUT               default /yun/bioasq14_output/oracle_<SPLIT>_<gtag>
#   DISTILL_DIR       default /yun/bioasq14_output/oracle_<SPLIT>_distill (shared)
#   OLLAMA_SERVE_URL  default http://ana:11434/api/generate  (point at the generator's serve)
#   EXTRACT_MODEL     default llama3.3:latest
set -euo pipefail
cd ~/BioASQ; mkdir -p logs
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
PUBMED_HOST="/shared/workspace/biolab/pubmed"; yun="/shared/workspace/biolab/yun"; WORKDIR="${PWD}"
SPLIT="${SPLIT:-13B4}"
GENERATOR="${GENERATOR:-llama3.3:latest}"
EXTRACT_MODEL="${EXTRACT_MODEL:-llama3.3:latest}"
OLLAMA_SERVE_URL="${OLLAMA_SERVE_URL:-http://ana:11434/api/generate}"
_gtag="$(echo "$GENERATOR" | tr ':./' '___')"
OUT="${OUT:-/yun/bioasq14_output/oracle_${SPLIT}_${_gtag}}"
DISTILL_DIR="${DISTILL_DIR:-/yun/bioasq14_output/oracle_${SPLIT}_distill}"
# Evidence override (Plan B long-context oracle): set CTX_ALL/CTX_16 to the
# oracle_docabs_* files (repo-relative). Empty -> run_oracle_arms uses the
# GT-snippet defaults. GENERATION_THINK=false disables gemma's think-chain
# (finding 3 truncation fix); empty -> model default (llama needs nothing).
CTX_ALL="${CTX_ALL:-}"; CTX_16="${CTX_16:-}"
GENERATION_THINK="${GENERATION_THINK:-}"

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
    export GENERATION_THINK='"$GENERATION_THINK"'
    python -c "import rank_bm25" 2>/dev/null || pip install --quiet --user rank_bm25
    _base="${OLLAMA_URL%/api/generate}"
    curl -sS --max-time 15 "$_base/api/tags" | grep -q "'"$GENERATOR"'" \
      || { echo "[preflight] ERROR: '"$GENERATOR"' not served at $_base" >&2; exit 1; }
    echo "[preflight] ollama OK ('"$GENERATOR"' @ $_base)"
    REPO_ROOT=/work SPLIT='"$SPLIT"' PY=python \
    OUT='"$OUT"' DISTILL_DIR='"$DISTILL_DIR"' \
    CTX_ALL='"$CTX_ALL"' CTX_16='"$CTX_16"' \
    GENERATOR='"$GENERATOR"' EXTRACT_MODEL='"$EXTRACT_MODEL"' \
      bash scripts/private_scripts/hpc/gen_pipeline_test/oracle/run_oracle_arms.sh
  '
echo "Finished ${SLURM_JOB_ID} at $(date)"
