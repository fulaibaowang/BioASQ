#!/bin/bash
# Plan B generation matrix on the long-context (full gold abstract) oracle.
# 6 arms, one generator per invocation. Distillation artifacts are prebuilt +
# generator-independent (distill_docabs.sh) and reused from DISTILL_DIR.
#   Direct ladder : direct-10, direct-16, direct-40  (first-K abstracts, gold order)
#   Claims        : claim-50, claim-160 (~direct-10 token budget)
#   Facets        : facet-16 (= all available clusters, ~7-16)
# Env:
#   REPO_ROOT (default .), SPLIT (default 13B4)
#   OUT           answers dir for THIS generator
#   DISTILL_DIR   shared distillation artifacts (claims_cache + distilled_*)
#   CTX_ALL       long-context evidence (default oracle_docabs_all_${SPLIT}.jsonl)
#   GENERATOR     final answer model (default llama3.3:latest)
#   EXTRACT_MODEL used only to (re)build a missing distilled_claims160 (default llama3.3:latest)
#   OLLAMA_URL, LLAMA_API_KEY, GENERATION_THINK (set "false" for gemma)
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-.}"; cd "$REPO_ROOT"
SPLIT="${SPLIT:-13B4}"
OUT="${OUT:-oracle_docabs_out_$SPLIT}"
DISTILL_DIR="${DISTILL_DIR:-/yun/bioasq14_output/oracle_docabs_${SPLIT}_distill}"
EXTRACT_MODEL="${EXTRACT_MODEL:-llama3.3:latest}"
GENERATOR="${GENERATOR:-llama3.3:latest}"
PY="${PY:-python}"
G=scripts/public/shared_scripts/generation
SCH=scripts/public/prompts/schemas
ORA=scripts/private_scripts/hpc/gen_pipeline_test/oracle
CTX_ALL="${CTX_ALL:-$ORA/oracle_docabs_all_${SPLIT}.jsonl}"

export GENERATION_BACKEND=ollama
export LLAMA_API_KEY="${LLAMA_API_KEY:-local-selfhost}"
export GENERATION_NUM_CTX="${GENERATION_NUM_CTX:-16384}"   # fits direct-40 (~13.6k tok w/ answer)
mkdir -p "$OUT"

CLAIMS50="$DISTILL_DIR/distilled_claims50.jsonl"
CLAIMS160="$DISTILL_DIR/distilled_claims160.jsonl"
FACETS16="$DISTILL_DIR/distilled_facets16.jsonl"

# claim-160 is generator-independent; build once if the distill step didn't emit it.
if [ ! -s "$CLAIMS160" ]; then
  echo "[$(date +%T)] build distilled_claims160 (missing)"
  $PY $G/distil_claims.py --evidence "$CTX_ALL" --cache "$DISTILL_DIR/claims_cache.jsonl" \
    --out "$CLAIMS160" --top-n 200 --slots 160 >> "$DISTILL_DIR/log_distil.txt" 2>&1
fi

GEN=(--schemas-dir "$SCH" --model "$GENERATOR" --max-chars-per-context 2000 --temperature 0.0 --timeout 300)

gen () {  # gen <arm_name> <input_jsonl> <max_contexts>
  local name="$1" input="$2" maxc="$3"
  # idempotent: skip an arm already fully generated (85 answers) so re-running
  # only completes the tail left by a timed-out job.
  local done_f; done_f=$(ls "$OUT/$name"/*_answers.jsonl 2>/dev/null | head -1)
  if [ -n "$done_f" ] && [ "$(wc -l < "$done_f")" -ge 85 ]; then
    echo "[$(date +%T)] skip $name (complete: $done_f)"; return; fi
  echo "[$(date +%T)] GEN $name (generator=$GENERATOR, max-contexts=$maxc)"
  $PY $G/generate_answers.py --input "$input" --output "$OUT/$name" \
    --max-contexts "$maxc" "${GEN[@]}" > "$OUT/log_${name}.txt" 2>&1
}

# Direct ladder: first-K abstracts (gold order) from the full-abstract evidence.
gen direct_10 "$CTX_ALL" 10
gen direct_16 "$CTX_ALL" 16
gen direct_40 "$CTX_ALL" 40
# Distilled arms (max-contexts high enough to feed every slot).
gen claims_50  "$CLAIMS50"  200
gen claims_160 "$CLAIMS160" 200
gen facets_16  "$FACETS16"  200
echo "[$(date +%T)] DONE generator=$GENERATOR -> $OUT"
