#!/bin/bash
#SBATCH -J snippet_rerank
#SBATCH -p frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/BioASQ

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs"
WORKDIR="${PWD}"

PUBMED_HOST="/shared/workspace/biolab/pubmed"

# Input: hybrid rerank runs
RUNS_DIR="output/workflow_local_10pct_hpc_bge/rerank_hybrid/runs"
RUN_GLOB="best_rrf_*_top*.tsv"

# Corpus
DOCS_JSONL="output/subset_pubmed.jsonl"

# Gold: training + test batches
TRAIN_JSON="example/training14b_10pct_sample.json"
TEST_BATCH_JSONS=(
  "bioasq_data/Task13BGoldenEnriched/13B1_golden.json"
  "bioasq_data/Task13BGoldenEnriched/13B2_golden.json"
  "bioasq_data/Task13BGoldenEnriched/13B3_golden.json"
  "bioasq_data/Task13BGoldenEnriched/13B4_golden.json"
)

# Output
OUT_DIR="output/workflow_local_10pct_hpc_bge/snippet_rerank"

# Stage A (hybrid filter)
DENSE_MODEL="abhinand/MedEmbed-small-v0.1"
N_DOCS=100
WINDOW_SIZE=2
WINDOW_STRIDE=1
TOP_W=12

# Stage B (CE rerank)
CE_MODEL="BAAI/bge-reranker-v2-m3"
CE_BATCH=64
CE_MAX_LENGTH=512

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running: scripts/private_scripts/hpc/snippet_rerank.py"

# -----------------------------
# Run inside container
# -----------------------------
srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail

    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1

    printf '[debug] CUDA_VISIBLE_DEVICES=%s\n' \"\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    TEST_BATCH_JSONS=(
      'bioasq_data/Task13BGoldenEnriched/13B1_golden.json'
      'bioasq_data/Task13BGoldenEnriched/13B2_golden.json'
      'bioasq_data/Task13BGoldenEnriched/13B3_golden.json'
      'bioasq_data/Task13BGoldenEnriched/13B4_golden.json'
    )

    echo '[run] Starting snippet rerank'
    python -u scripts/private_scripts/hpc/snippet_rerank.py \
      --runs-dir '${RUNS_DIR}' \
      --run-glob '${RUN_GLOB}' \
      --docs-jsonl '${DOCS_JSONL}' \
      --train-json '${TRAIN_JSON}' \
      --test-batch-jsons \${TEST_BATCH_JSONS[@]} \
      --n-docs ${N_DOCS} \
      --window-size ${WINDOW_SIZE} \
      --window-stride ${WINDOW_STRIDE} \
      --top-w ${TOP_W} \
      --dense-model '${DENSE_MODEL}' \
      --dense-device cpu \
      --ce-model '${CE_MODEL}' \
      --ce-device cuda \
      --ce-batch ${CE_BATCH} \
      --ce-max-length ${CE_MAX_LENGTH} \
      --output-dir '${OUT_DIR}'

    echo '[done] Snippet rerank complete'
    ls -lh '${OUT_DIR}' || true
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
