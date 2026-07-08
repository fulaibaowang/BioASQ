#!/bin/bash
# Plan B, design-agnostic LONG POLE: build the shared distillation artifacts over
# the long-context (full gold abstract) evidence, emit distilled files at several
# claim/facet counts, and MEASURE the realized token budget of every candidate
# arm so we can lock counts before firing the (fast) generation matrix.
#
# Generator-INDEPENDENT: extraction + facet summarization are done once here and
# reused by every generator and every claim-N / facet-M selection.
#
# Env:
#   REPO_ROOT (default .), SPLIT (default 13B4)
#   CTX_ALL       long-context evidence (default oracle_docabs_all_${SPLIT}.jsonl)
#   DISTILL_DIR   where cache + distilled files land
#   EXTRACT_MODEL (default llama3.3:latest)  extractor + facet summarizer
#   CLAIM_SLOTS   space-separated claim counts to emit (default "16 50")
#   FACET_N       space-separated facet counts to emit (default "8 16")
#   OLLAMA_URL, LLAMA_API_KEY
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-.}"; cd "$REPO_ROOT"
SPLIT="${SPLIT:-13B4}"
EXTRACT_MODEL="${EXTRACT_MODEL:-llama3.3:latest}"
PY="${PY:-python}"
G=scripts/public/shared_scripts/generation
ORA=scripts/private_scripts/hpc/gen_pipeline_test/oracle
CTX_ALL="${CTX_ALL:-$ORA/oracle_docabs_all_${SPLIT}.jsonl}"
DISTILL_DIR="${DISTILL_DIR:-/yun/bioasq14_output/oracle_docabs_${SPLIT}_distill}"
CLAIM_SLOTS="${CLAIM_SLOTS:-16 50}"
FACET_N="${FACET_N:-8 16}"

export GENERATION_BACKEND=ollama
export LLAMA_API_KEY="${LLAMA_API_KEY:-local-selfhost}"
export GENERATION_NUM_CTX="${GENERATION_NUM_CTX:-16384}"
mkdir -p "$DISTILL_DIR"
CACHE="$DISTILL_DIR/claims_cache.jsonl"
FACETS_FULL="$DISTILL_DIR/facets_full.jsonl"

# ---- 1. extract claims from every gold abstract (LONG POLE; resume-safe cache) ----
echo "[$(date +%T)] EXTRACT claims (extractor=$EXTRACT_MODEL; sequential over full abstracts)"
$PY $G/extract_claims.py --evidence "$CTX_ALL" --cache "$CACHE" \
  --model "$EXTRACT_MODEL" --top-n 200 > "$DISTILL_DIR/log_extract.txt" 2>&1
echo "[$(date +%T)]   claims_cache lines: $(wc -l < "$CACHE")"

# ---- 2. facet summaries over the same claims (LONG-ish; cached) ----
echo "[$(date +%T)] SUMMARIZE facets"
$PY $G/summarize_facets.py --evidence "$CTX_ALL" --cache "$CACHE" \
  --out "$FACETS_FULL" --summary-cache "$DISTILL_DIR/facets_summary_cache.jsonl" \
  --top-n 200 --model "$EXTRACT_MODEL" --num-ctx 8192 > "$DISTILL_DIR/log_summarize.txt" 2>&1
echo "[$(date +%T)]   facets_full lines: $(wc -l < "$FACETS_FULL")"

# ---- 3. cheap: emit distilled files at each requested count ----
for n in $CLAIM_SLOTS; do
  $PY $G/distil_claims.py --evidence "$CTX_ALL" --cache "$CACHE" \
    --out "$DISTILL_DIR/distilled_claims${n}.jsonl" --top-n 200 --slots "$n" \
    >> "$DISTILL_DIR/log_distil.txt" 2>&1 && echo "[$(date +%T)]   claims-$n written"
done
for m in $FACET_N; do
  $PY $G/select_contexts.py --full "$FACETS_FULL" \
    --out "$DISTILL_DIR/distilled_facets${m}.jsonl" --n "$m" \
    >> "$DISTILL_DIR/log_select.txt" 2>&1 && echo "[$(date +%T)]   facets-$m written"
done

# ---- 4. token-budget measurement (words ~ token/1.3) ----
echo "[$(date +%T)] MEASURE token budgets"
$PY - "$CTX_ALL" "$DISTILL_DIR" <<'PY'
import json, sys, glob, os, statistics as st
ctx_all, dd = sys.argv[1], sys.argv[2]

def load(p): return [json.loads(l) for l in open(p)]
def words(s): return len(str(s).split())

def ctx_words(rec):
    return [words(c.get("text","")) for c in rec.get("contexts", [])]

recs = load(ctx_all)
qid = lambda r: r.get("query_id")

# direct-K budgets: first-K abstracts per question
def direct_budget(K):
    tot = []
    for r in recs:
        w = ctx_words(r)
        tot.append(sum(w[:K]))
    return tot
# greedy-x: max whole abstracts under a word budget (num_ctx 16384 ~ 12000 usable words)
BUD = 12000
def greedy_x():
    xs, ws = [], []
    for r in recs:
        w = ctx_words(r); s = 0; k = 0
        for x in w:
            if s + x > BUD: break
            s += x; k += 1
        xs.append(k); ws.append(s)
    return xs, ws

print(f"\n{'arm':16} {'q':>3} {'ctx/q(med)':>10} {'words/q(med)':>12} {'~tok/q(med)':>11}")
print("-"*58)
for K in (10, 16):
    b = direct_budget(K)
    cq = [min(K, len(ctx_words(r))) for r in recs]
    print(f"{'direct-'+str(K):16} {len(b):>3} {int(st.median(cq)):>10} {int(st.median(b)):>12} {int(st.median(b)*1.3):>11}")
xs, ws = greedy_x()
print(f"{'direct-x(fit12k)':16} {len(ws):>3} {int(st.median(xs)):>10} {int(st.median(ws)):>12} {int(st.median(ws)*1.3):>11}  (x: min {min(xs)} max {max(xs)})")

# distilled arms
for p in sorted(glob.glob(os.path.join(dd, "distilled_*.jsonl"))):
    d = load(p)
    dmap = {r.get("query_id"): r for r in d}
    perq, slots = [], []
    for r in recs:
        rr = dmap.get(qid(r))
        if not rr: continue
        w = ctx_words(rr)
        perq.append(sum(w)); slots.append(len(w))
    if not perq: continue
    name = os.path.basename(p).replace("distilled_","").replace(".jsonl","")
    print(f"{name:16} {len(perq):>3} {int(st.median(slots)):>10} {int(st.median(perq)):>12} {int(st.median(perq)*1.3):>11}")

# per-unit costs to compute budget-matched counts
def unit(pattern):
    ps = glob.glob(os.path.join(dd, pattern))
    if not ps: return None
    d = load(ps[0]); allw=[words(c.get("text","")) for r in d for c in r.get("contexts",[])]
    return st.mean(allw) if allw else None
uc = unit("distilled_claims*.jsonl"); uf = unit("distilled_facets*.jsonl")
abw = st.mean([w for r in recs for w in ctx_words(r)])
print("\n[per-unit words]  abstract ~%.0f | claim ~%.1f | facet ~%.1f" % (abw, uc or 0, uf or 0))
med_dx = int(st.median(ws))
if uc: print("  to budget-match direct-x(~%dw): claim-N with N ~ %d" % (med_dx, round(med_dx/uc)))
if uf: print("  to budget-match direct-x(~%dw): facet-M with M ~ %d" % (med_dx, round(med_dx/uf)))
PY
echo "[$(date +%T)] DONE distillation + measurement -> $DISTILL_DIR"
