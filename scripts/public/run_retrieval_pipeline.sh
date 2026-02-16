#!/usr/bin/env bash
#
# Run retrieval pipeline: BM25 -> Dense -> Hybrid (and optionally Reranker).
#
# Usage:
#   ./run_retrieval_pipeline.sh --config <path/to/config.env>
#   ./run_retrieval_pipeline.sh -c <path/to/config.env>
#
# Or: source my.env && ./run_retrieval_pipeline.sh  (env already set)
#
# Config file sets: WORKFLOW_OUTPUT_DIR, TRAIN_JSON, TEST_BATCH_JSONS, TOP_K,
# RECALL_KS, BM25_INDEX_PATH, DENSE_INDEX_DIR, DOCS_JSONL (optional), and
# stage overrides (BM25_*, DENSE_*, HYBRID_*, RERANK_*). See workflow_config_full.env.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse -c / --config (and -h / --help)
CONFIG_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--config)
      [ -z "${2:-}" ] && { echo "Error: --config requires a path." >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--config|-c <config.env>]"
      echo "  -c, --config PATH   Source PATH as config (env vars) before running."
      echo "  -h, --help          Show this help."
      echo ""
      echo "Example: $0 --config scripts/private_scripts/config.env"
      echo "Example: source workflow_config_small.env && $0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (use -h for help)" >&2
      exit 1
      ;;
  esac
done

if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
  echo "Loaded config: $CONFIG_FILE"
fi

cd "$REPO_ROOT"

# Required env (set by config file or by sourcing before run)
: "${WORKFLOW_OUTPUT_DIR:?Set WORKFLOW_OUTPUT_DIR (e.g. output/workflow_run)}"
: "${TRAIN_JSON:?Set TRAIN_JSON (path to training questions JSON)}"
: "${TEST_BATCH_JSONS:?Set TEST_BATCH_JSONS (space-separated paths to test batch JSONs)}"
: "${BM25_INDEX_PATH:?Set BM25_INDEX_PATH (Terrier index directory)}"
: "${DENSE_INDEX_DIR:?Set DENSE_INDEX_DIR (Dense HNSW index directory)}"

TOP_K="${TOP_K:-5000}"
RECALL_KS="${RECALL_KS:-50,100,200,300,400,500,1000,2000,5000}"

# Stage-specific retrieval depth (default to TOP_K)
BM25_TOP_K="${BM25_TOP_K:-$TOP_K}"
DENSE_TOP_K="${DENSE_TOP_K:-$TOP_K}"
HYBRID_CAP="${HYBRID_CAP:-$TOP_K}"
HYBRID_K_MAX_EVAL="${HYBRID_K_MAX_EVAL:-$TOP_K}"
RERANK_CANDIDATE_LIMIT="${RERANK_CANDIDATE_LIMIT:-$TOP_K}"

BM25_OUT="$WORKFLOW_OUTPUT_DIR/bm25"
DENSE_OUT="$WORKFLOW_OUTPUT_DIR/dense"
HYBRID_OUT="$WORKFLOW_OUTPUT_DIR/hybrid"
RERANK_OUT="$WORKFLOW_OUTPUT_DIR/rerank"

mkdir -p "$BM25_OUT" "$DENSE_OUT" "$HYBRID_OUT"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ----- BM25 -----
BM25_ARGS=(
  --index_path "$BM25_INDEX_PATH"
  --train_json "$TRAIN_JSON"
  --test_batch_jsons $TEST_BATCH_JSONS
  --out_dir "$BM25_OUT"
  --k_eval "$BM25_TOP_K"
)
[ -n "${BM25_JAVA_MEM:-}" ] && BM25_ARGS+=(--java_mem "$BM25_JAVA_MEM")
[ -n "${BM25_THREADS:-}" ] && BM25_ARGS+=(--threads "$BM25_THREADS")
[ -n "${BM25_RM3_FEEDBACK_POOL:-}" ] && BM25_ARGS+=(--k_feedback "$BM25_RM3_FEEDBACK_POOL")
[ -n "${BM25_RM3_FB_DOCS:-}" ] && BM25_ARGS+=(--rm3_fb_docs "$BM25_RM3_FB_DOCS")
[ -n "${BM25_RM3_FB_TERMS:-}" ] && BM25_ARGS+=(--rm3_fb_terms "$BM25_RM3_FB_TERMS")
[ -n "${BM25_RM3_LAMBDA:-}" ] && BM25_ARGS+=(--rm3_lambda "$BM25_RM3_LAMBDA")
[ "${BM25_INCLUDE_BASELINE:-0}" = "1" ] && BM25_ARGS+=(--include_bm25)
[ "${BM25_NO_EVAL:-0}" = "1" ] && BM25_ARGS+=(--no_eval)
[ "${BM25_SAVE_RUNS:-1}" = "1" ] && BM25_ARGS+=(--save_runs)
[ "${BM25_SAVE_PER_QUERY:-0}" = "1" ] && BM25_ARGS+=(--save_per_query)
[ "${BM25_SAVE_ZERO_RECALL:-0}" = "1" ] && BM25_ARGS+=(--save_zero_recall)
[ "${BM25_NO_EXCLUDE_TEST_QIDS:-0}" = "1" ] && BM25_ARGS+=(--no_exclude_test_qids)

echo "[1/3] BM25..."
python "$SCRIPT_DIR/retrieval/eval_bm25_rm3.py" "${BM25_ARGS[@]}"

# ----- Dense -----
DENSE_ARGS=(
  --index_dir "$DENSE_INDEX_DIR"
  --out_dir "$DENSE_OUT"
  --train_subset_json "$TRAIN_JSON"
  --test_batch_jsons $TEST_BATCH_JSONS
  --topk "$DENSE_TOP_K"
  --ks "$RECALL_KS"
)
[ -n "${DENSE_EF_SEARCH:-}" ] && DENSE_ARGS+=(--ef_search "$DENSE_EF_SEARCH")
[ -n "${DENSE_EF_CAP:-}" ] && DENSE_ARGS+=(--ef_cap "$DENSE_EF_CAP")
[ -n "${DENSE_BATCH_SIZE:-}" ] && DENSE_ARGS+=(--batch_size "$DENSE_BATCH_SIZE")
[ -n "${DENSE_DEVICE:-}" ] && DENSE_ARGS+=(--device "$DENSE_DEVICE")
[ -n "${DENSE_MODEL_NAME:-}" ] && DENSE_ARGS+=(--model_name "$DENSE_MODEL_NAME")
[ "${DENSE_NO_EVAL:-0}" = "1" ] && DENSE_ARGS+=(--no_eval)
[ "${DENSE_SAVE_PER_QUERY:-0}" = "1" ] && DENSE_ARGS+=(--save_per_query)

echo "[2/3] Dense..."
python "$SCRIPT_DIR/retrieval/eval_dense.py" "${DENSE_ARGS[@]}"

# ----- Hybrid -----
HYBRID_ARGS=(
  --bm25_runs_dir "$BM25_OUT/runs"
  --bm25_topk "$BM25_TOP_K"
  --dense_root "$DENSE_OUT"
  --train_subset_json "$TRAIN_JSON"
  --test_batch_jsons $TEST_BATCH_JSONS
  --out_dir "$HYBRID_OUT"
  --k_max_eval "$HYBRID_K_MAX_EVAL"
  --cap "$HYBRID_CAP"
)
[ -n "${HYBRID_MODE:-}" ] && HYBRID_ARGS+=(--mode "$HYBRID_MODE")
[ -n "${HYBRID_K_RRF:-}" ] && HYBRID_ARGS+=(--k_rrf "$HYBRID_K_RRF")
[ -n "${HYBRID_W_BM25:-}" ] && HYBRID_ARGS+=(--w_bm25 "$HYBRID_W_BM25")
[ -n "${HYBRID_W_DENSE:-}" ] && HYBRID_ARGS+=(--w_dense "$HYBRID_W_DENSE")
[ -n "${HYBRID_WEIGHTS:-}" ] && HYBRID_ARGS+=(--weights "$HYBRID_WEIGHTS")
[ -n "${HYBRID_K_RRF_LIST:-}" ] && HYBRID_ARGS+=(--k_rrf_list "$HYBRID_K_RRF_LIST")
[ -n "${HYBRID_JOBS:-}" ] && HYBRID_ARGS+=(--jobs "$HYBRID_JOBS")
[ "${HYBRID_NO_EVAL:-0}" = "1" ] && HYBRID_ARGS+=(--no_eval)
[ "${HYBRID_NO_PLOTS:-0}" = "1" ] && HYBRID_ARGS+=(--no_plots)
[ "${HYBRID_SAVE_PLOTS:-0}" = "1" ] && HYBRID_ARGS+=(--save_plots)

echo "[3/3] Hybrid..."
python "$SCRIPT_DIR/retrieval/eval_hybird.py" "${HYBRID_ARGS[@]}"

# ----- Reranker (optional) -----
if [ -n "${DOCS_JSONL:-}" ]; then
  echo "[4/4] Reranker..."
  mkdir -p "$RERANK_OUT"
  RERANK_ARGS=(
    --runs-dir "$HYBRID_OUT/runs"
    --output-dir "$RERANK_OUT"
    --docs-jsonl "$DOCS_JSONL"
    --candidate_limit "$RERANK_CANDIDATE_LIMIT"
    --ks-recall "${RERANK_KS_RECALL:-$RECALL_KS}"
  )
  [ -n "${TRAIN_JSON:-}" ] && RERANK_ARGS+=(--train_subset_json "$TRAIN_JSON")
  [ -n "${TEST_BATCH_JSONS:-}" ] && RERANK_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
  [ -n "${RERANK_MODEL:-}" ] && RERANK_ARGS+=(--model "$RERANK_MODEL")
  [ -n "${RERANK_MODEL_DEVICE:-}" ] && RERANK_ARGS+=(--model-device "$RERANK_MODEL_DEVICE")
  [ -n "${RERANK_MODEL_BATCH:-}" ] && RERANK_ARGS+=(--model-batch "$RERANK_MODEL_BATCH")
  [ -n "${RERANK_MODEL_MAX_LENGTH:-}" ] && RERANK_ARGS+=(--model-max-length "$RERANK_MODEL_MAX_LENGTH")
  [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && RERANK_ARGS+=(--disable-metrics)
  [ "${RERANK_USE_MULTI_GPU:-0}" = "1" ] && RERANK_ARGS+=(--use-multi-gpu)
  [ -n "${RERANK_NUM_GPUS:-}" ] && RERANK_ARGS+=(--num-gpus "$RERANK_NUM_GPUS")
  python "$SCRIPT_DIR/rerank/rerank_stage2.py" "${RERANK_ARGS[@]}"
  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (bm25/, dense/, hybrid/, rerank/)"
else
  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (bm25/, dense/, hybrid/)"
  echo "Optional: set DOCS_JSONL and re-run to add reranker step."
fi
