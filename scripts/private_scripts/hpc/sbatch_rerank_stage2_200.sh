#!/bin/bash
#SBATCH -J stage2_rerank_public
#SBATCH -p frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
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

# Input: hybrid Stage 1 runs (top 2000 candidates)
RUNS_DIR="/yun/output/eval_hybrid_production_test/runs"
RUN_GLOB="best_rrf_*_top2000.tsv"

# Input: PubMed subset texts
DOCS_JSONL="/yun/output/subset_pubmed.jsonl"

# Gold: training subset + test batches (.jsonl; adapt-in from BioASQ JSON if needed)
INPUT_JSONL="example/training14b_10pct_sample.jsonl"
INPUT_BATCH_JSONLS=(
  "bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl"
  "bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl"
  "bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl"
  "bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl"
)

# Output location (inside container)
OUT_DIR="/yun/output/eval_stage2_rerank_bge_reranker_v2_m3_len200"

# Reranker parameters
MODEL_NAME="BAAI/bge-reranker-v2-m3"
USE_MULTI_GPU=true
NUM_GPUS=2
BATCH_SIZE=512
MODEL_MAX_LENGTH=200
CANDIDATE_LIMIT=2000
ADAPTIVE_P=0.95
ADAPTIVE_CAP=300
KS_RECALL="50,100,200,300,400,500,1000,2000"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running: scripts/public/shared_scripts/rerank/rerank_crossencoder.py"

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

    # --- Persistent HF cache on shared workspace ---
    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    # --- Set up environment for multi-GPU ---
    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1

    INPUT_BATCH_JSONLS=(
      'bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl'
      'bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl'
      'bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl'
      'bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl'
    )

    printf '[debug] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi -L || true

    # --- Run reranker script with CLI flags ---
    echo '[run] Starting reranker (CLI config)'
    echo '[debug] use_multi_gpu flag: --use-multi-gpu --num-gpus ${NUM_GPUS}'
    python -u scripts/public/shared_scripts/rerank/rerank_crossencoder.py \
      --runs-dir '${RUNS_DIR}' \
      --run-glob '${RUN_GLOB}' \
      --docs-jsonl '${DOCS_JSONL}' \
      --input-jsonl '${INPUT_JSONL}' \
      --input-batch-jsonls \${INPUT_BATCH_JSONLS[@]} \
      --candidate-limit ${CANDIDATE_LIMIT} \
      --model '${MODEL_NAME}' \
      --model-batch ${BATCH_SIZE} \
      --model-max-length ${MODEL_MAX_LENGTH} \
      --use-multi-gpu \
      --num-gpus ${NUM_GPUS} \
      --adaptive-p ${ADAPTIVE_P} \
      --adaptive-cap ${ADAPTIVE_CAP} \
      --ks-recall '${KS_RECALL}' \
      --output-dir '${OUT_DIR}'

    echo '[done] Reranking complete'
    ls -lh '${OUT_DIR}' || true
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"
