#!/bin/bash
# Oracle generation test: 4 arms on GT snippets. Location-agnostic. Env:
#   REPO_ROOT (default .), SPLIT (default 13B4)
#   OUT          answers dir for THIS generator
#   DISTILL_DIR  shared distillation artifacts (default = OUT). Extraction +
#                facet summarization are GENERATOR-INDEPENDENT (validated on
#                llama3.3), so build them once here and reuse across generators.
#   EXTRACT_MODEL (default llama3.3:latest)  extractor + summarizer
#   GENERATOR     (default llama3.3:latest)  final answer model
#   OLLAMA_URL, LLAMA_API_KEY
# Arms: direct-all, direct-16, claims-16, facets-16 (16 = GENERATION_SELECT_N).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-.}"; cd "$REPO_ROOT"
SPLIT="${SPLIT:-13B4}"
OUT="${OUT:-oracle_out_$SPLIT}"
DISTILL_DIR="${DISTILL_DIR:-$OUT}"
EXTRACT_MODEL="${EXTRACT_MODEL:-llama3.3:latest}"
GENERATOR="${GENERATOR:-llama3.3:latest}"
PY="${PY:-python}"
G=scripts/public/shared_scripts/generation
SCH=scripts/public/prompts/schemas
ORA=scripts/private_scripts/hpc/gen_pipeline_test/oracle
# Evidence files. Default = GT-snippet oracle; override CTX_ALL/CTX_16 to point
# at the long-context doc-abstract oracle (oracle_docabs_*_${SPLIT}.jsonl).
CTX_ALL="${CTX_ALL:-$ORA/oracle_all_${SPLIT}.jsonl}"
CTX_16="${CTX_16:-$ORA/oracle_16_${SPLIT}.jsonl}"

export GENERATION_BACKEND=ollama
export LLAMA_API_KEY="${LLAMA_API_KEY:-local-selfhost}"
export GENERATION_NUM_CTX="${GENERATION_NUM_CTX:-16384}"   # deep direct-all prompts

mkdir -p "$OUT" "$DISTILL_DIR"
GEN=(--schemas-dir "$SCH" --model "$GENERATOR" --max-contexts 200 --max-chars-per-context 2000 --temperature 0.0)
CLAIMS="$DISTILL_DIR/distilled_claims16.jsonl"
FACETS="$DISTILL_DIR/distilled_facets16.jsonl"

# ---- Stage 1: distillation (generator-independent; build once, then reuse) ----
if [ -f "$CLAIMS" ] && [ -f "$FACETS" ]; then
  echo "[$(date +%T)] reuse distillation in $DISTILL_DIR (skip extraction)"
else
  echo "[$(date +%T)] EXTRACT claims from ALL GT snippets (extractor=$EXTRACT_MODEL; sequential, slow)"
  $PY $G/extract_claims.py --evidence "$CTX_ALL" --cache "$DISTILL_DIR/claims_cache.jsonl" \
    --model "$EXTRACT_MODEL" --top-n 200 > "$DISTILL_DIR/log_extract.txt" 2>&1
  echo "[$(date +%T)]   cached $(wc -l < "$DISTILL_DIR/claims_cache.jsonl")"
  $PY $G/distil_claims.py --evidence "$CTX_ALL" --cache "$DISTILL_DIR/claims_cache.jsonl" \
    --out "$CLAIMS" --top-n 200 --slots 16 > "$DISTILL_DIR/log_distil.txt" 2>&1
  $PY $G/summarize_facets.py --evidence "$CTX_ALL" --cache "$DISTILL_DIR/claims_cache.jsonl" \
    --out "$DISTILL_DIR/facets_full.jsonl" --summary-cache "$DISTILL_DIR/facets_summary_cache.jsonl" \
    --top-n 200 --model "$EXTRACT_MODEL" --num-ctx 8192 > "$DISTILL_DIR/log_summarize.txt" 2>&1
  $PY $G/select_contexts.py --full "$DISTILL_DIR/facets_full.jsonl" --out "$FACETS" --n 16 >> "$DISTILL_DIR/log_summarize.txt" 2>&1
fi

# ---- Stage 2: generation with GENERATOR (fast; per-generator) ----
echo "[$(date +%T)] GEN direct-all  (generator=$GENERATOR)"
$PY $G/generate_answers.py --input "$CTX_ALL" --output "$OUT/direct_all" "${GEN[@]}" > "$OUT/log_direct_all.txt" 2>&1
echo "[$(date +%T)] GEN direct-16"
$PY $G/generate_answers.py --input "$CTX_16" --output "$OUT/direct_16" "${GEN[@]}" > "$OUT/log_direct_16.txt" 2>&1
echo "[$(date +%T)] GEN claims-16"
$PY $G/generate_answers.py --input "$CLAIMS" --output "$OUT/claims_16" "${GEN[@]}" > "$OUT/log_claims_gen.txt" 2>&1
echo "[$(date +%T)] GEN facets-16"
$PY $G/generate_answers.py --input "$FACETS" --output "$OUT/facets_16" "${GEN[@]}" > "$OUT/log_facets_gen.txt" 2>&1
echo "[$(date +%T)] DONE generator=$GENERATOR -> $OUT"
