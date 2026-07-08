#!/bin/bash
#SBATCH -J bioasq_gen_test
#SBATCH -p dev,frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Reproducibility + new-generation-pipeline test on the 13B golden splits.
#   Phase 1 (repro):   run each split with GENERATION_MODE=direct and confirm
#                      retrieval + Phase B metrics match docs/RESULTS.md.
#   Phase 2 (gen A/B/C): reuse Phase 1 retrieval/rerank/evidence; re-run ONLY
#                      the generation stage in direct|claims|facets modes.
#   Phase 3 (eval):    official BioASQ evaluator -> per-question TSV -> subset
#                      to the 60 oracle-rich list questions with
#                      compare_subset_metrics.py.
#
# Mirrors scripts/private_scripts/hpc/bioasq_runs/sbatch_pipeline.sh (container,
# mounts, HF cache). Adjust CONTAINER_IMG / mount hosts to your cluster.
set -euo pipefail
cd ~/BioASQ
mkdir -p logs

CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
yun="/shared/workspace/biolab/yun"
WORKDIR="${PWD}"
TESTDIR="scripts/private_scripts/hpc/gen_pipeline_test"

# Splits to run. 13B4 = richest split (18 oracle-rich list questions), the fast
# first pass. Set SPLITS="13B1 13B2 13B3 13B4" for full RESULTS.md reproduction.
SPLITS="${SPLITS:-13B4}"
MODES="${MODES:-direct claims facets}"
# Generators for the answer stage (space-separated ollama tags). Repro/anchor
# uses the first entry; the generation A/B/C matrix runs every generator.
# Only llama3.3 is self-host-ready for now; add gemma4:31b once its serve is set
# up:  GEN_MODELS="llama3.3:latest gemma4:31b"
GEN_MODELS="${GEN_MODELS:-llama3.3:latest}"
# Self-hosted ollama serve (job "ollama_serve" on node ana).
OLLAMA_SERVE_URL="${OLLAMA_SERVE_URL:-http://ana:11434/api/generate}"
# Model that must be reachable for the preflight to pass (first generator).
OLLAMA_PREFLIGHT_MODEL="${OLLAMA_PREFLIGHT_MODEL:-llama3.3:latest}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-1}}"; export NUM_GPUS

srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${yun}:/yun" \
  --container-workdir /work \
  bash -lc '
    set -euo pipefail
    export HF_HOME=/pubmed/_hf_cache
    export HF_HUB_CACHE="$HF_HOME/hub"
    export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME"
    export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TQDM_DISABLE=1

    # --- Self-hosted ollama (llama3.3 + gemma4:31b on node ana) ---
    export GENERATION_BACKEND=ollama
    export OLLAMA_URL='"$OLLAMA_SERVE_URL"'
    export LLAMA_API_KEY="${LLAMA_API_KEY:-local-selfhost}"
    _base="${OLLAMA_URL%/api/generate}"
    _pfmodel='"$OLLAMA_PREFLIGHT_MODEL"'
    echo "[preflight] ollama @ $_base (need $_pfmodel)"
    if ! curl -sS --max-time 15 "$_base/api/tags" | grep -q "$_pfmodel"; then
      echo "[preflight] ERROR: ollama serve not reachable / missing $_pfmodel at $_base" >&2
      echo "[preflight] is the ollama_serve job still RUNNING on ana? (squeue -u \$USER)" >&2
      exit 1
    fi
    echo "[preflight] ollama OK"

    PIPE=./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh
    TESTDIR='"$TESTDIR"'
    GOLDEN=bioasq_data/Task13BGoldenEnriched

    for SPLIT in '"$SPLITS"'; do
      echo "==================== SPLIT $SPLIT ===================="
      # 0. golden JSON -> queries jsonl (once per split)
      if [ ! -f "$GOLDEN/${SPLIT}_golden.jsonl" ]; then
        python scripts/public/format/bioasq_json_to_queries_jsonl.py \
          --input "$GOLDEN/${SPLIT}_golden.json" \
          --output "$GOLDEN/${SPLIT}_golden.jsonl"
      fi

      GENS=( '"$GEN_MODELS"' )
      ANCHOR_GEN="${GENS[0]}"

      # 1. Phase 1 (repro): direct mode + the anchor generator = the RESULTS.md
      #    comparison. Full pipeline, real metrics.
      REPRO_OUT=/yun/bioasq14_output/repro_${SPLIT}_direct
      WORKFLOW_OUTPUT_DIR=$REPRO_OUT \
      INPUT_JSONL=/work/$GOLDEN/${SPLIT}_golden.jsonl \
      GENERATION_MODEL="$ANCHOR_GEN" \
        $PIPE --config "$TESTDIR/config_repro_13b4.env"

      # 2. Phase 2: generation matrix (mode x generator). Every cell reuses the
      #    Phase-1 retrieval/rerank/evidence so only distill+generation run.
      for GEN in "${GENS[@]}"; do
        gtag=$(echo "$GEN" | tr ":./" "___")
        for MODE in '"$MODES"'; do
          # direct + anchor generator already produced by Phase 1
          [ "$MODE" = direct ] && [ "$GEN" = "$ANCHOR_GEN" ] && continue
          GEN_OUT=/yun/bioasq14_output/gen_${SPLIT}_${MODE}_${gtag}
          # Seed retrieval/rerank/evidence from Phase 1, drop generation so only
          # distill+generation re-run (cp+rm: no rsync dependency in-container).
          rm -rf "$GEN_OUT"; mkdir -p "$GEN_OUT"
          cp -a "$REPRO_OUT/." "$GEN_OUT/"
          rm -rf "$GEN_OUT"/generation*
          WORKFLOW_OUTPUT_DIR=$GEN_OUT \
          INPUT_JSONL=/work/$GOLDEN/${SPLIT}_golden.jsonl \
          GENERATION_MODEL="$GEN" \
            $PIPE --config "$TESTDIR/config_gen_${MODE}.env"
        done
      done
    done
    echo "[done] retrieval + generation modes complete"
  '

echo "Finished ${SLURM_JOB_ID} at $(date)"
echo
echo "NEXT (Phase 3, run after the job):"
echo "  1. Convert each mode answers -> BioASQ submission JSON (queries_jsonl_to_bioasq_json.py)."
echo "  2. Run the official BioASQ evaluator to produce per-question phaseB TSVs."
echo "  3. Compare on the nugget-rich subset (document route is the real test):"
echo "     python $TESTDIR/compare_subset_metrics.py \\"
echo "       --qids $TESTDIR/nugget_rich_list_qids.txt \\"
echo "       --mode direct=<direct_perq.tsv> --mode claims=<claims_perq.tsv> --mode facets=<facets_perq.tsv>"
