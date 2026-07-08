#!/bin/bash
# Rescue silently-failed questions (records with an error field = truncated,
# unterminated JSON from temp=0 runaway loops) in each arm, re-running ONLY those
# records via generate_answers with the decoding fix (repeat_penalty breaks the
# loop, num_predict bounds output). Merges in place; other answers untouched.
# Per-arm max-contexts matches the arm so evidence is unchanged.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-.}"; cd "$REPO_ROOT"
: "${OUT:?set OUT to the answers dir}"
GENERATOR="${GENERATOR:-llama3.3:latest}"
PY="${PY:-python}"
G=scripts/public/shared_scripts/generation
SCH=scripts/public/prompts/schemas
export GENERATION_BACKEND=ollama LLAMA_API_KEY="${LLAMA_API_KEY:-local-selfhost}"
# The batch failures were TRANSIENT serve glitches (a fresh single re-run of the
# same prompt/options succeeds), NOT runaway loops. So rescue with the SAME decoding
# as the arms (no repeat_penalty/num_predict override) — just re-run to clear the
# glitch. num_ctx kept high (harmless; needed for direct_40's big prompts).
export GENERATION_NUM_CTX="${GENERATION_NUM_CTX:-32768}"
export GENERATION_REPEAT_PENALTY="${GENERATION_REPEAT_PENALTY:-}"
export GENERATION_NUM_PREDICT="${GENERATION_NUM_PREDICT:-}"

declare -A MC=( [direct_10]=10 [direct_16]=16 [direct_40]=40 [claims_50]=200 [claims_160]=200 [facets_16]=200 )
for a in direct_10 direct_16 direct_40 claims_50 claims_160 facets_16; do
  f=$(ls "$OUT/$a"/*_answers.jsonl 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "[$a] no answers"; continue; }
  nfail=$(grep -c '"error":' "$f" 2>/dev/null || true); nfail=${nfail:-0}
  if [ "$nfail" -eq 0 ]; then echo "[$a] 0 failures, skip"; continue; fi
  echo "[$(date +%T)] [$a] rescuing $nfail (max-contexts=${MC[$a]}, repeat_penalty=$GENERATION_REPEAT_PENALTY, num_predict=$GENERATION_NUM_PREDICT)"
  $PY $G/rescue_failed_generation.py --input "$f" \
    --model "$GENERATOR" --schemas-dir "$SCH" \
    --max-contexts "${MC[$a]}" --max-chars-per-context 2000 --timeout 300 \
    > "$OUT/log_rescue_${a}.txt" 2>&1 || echo "[$a] rescue exited nonzero (see log)"
  nfail2=$(grep -c '"error":' "$f" 2>/dev/null || true); nfail2=${nfail2:-0}
  echo "[$(date +%T)] [$a] failures now: $nfail2"
done
echo "[done] rescue complete for $OUT"
