REPO_ROOT=/work

# Workflow root (snippet route)
WF=/home/wangy/output/workflow_baseline_full_run_both_routes_gemma
# Cap on docs per question in post_rerank JSON
TOP_K=30   # or whatever you want instead of EVIDENCE_TOP_K
# Corpus for context building (glob or file)
export DOCS_JSONL="/pubmed/jsonl_2026/*.jsonl"   # same as DOCS_JSONL for that run
# Same query JSONs as the pipeline (source your real config, or paste paths)
export TRAIN_JSON="$REPO_ROOT/example/training14b_10pct_sample.json"   # if unused, leave empty: TRAIN_JSON=
export TEST_BATCH_JSONS="$REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B1_golden.json $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B2_golden.json $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B3_golden.json $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B4_golden.json"
# Snippet window / context knobs (optional)
export SNIPPET_WINDOW_SIZE=3
export SNIPPET_CONTEXT_TOP_WINDOWS=2

SCRIPT=scripts/public/shared_scripts/evidence

for TSV in "$WF/snippet_rrf/runs/"*.tsv; do
  [ -f "$TSV" ] || continue
  stem=$(basename "$TSV" .tsv)
  split=${stem#best_rrf_}
  split=${split%%_top*}
  [ -n "$split" ] || continue

  query_json=""
  if [ -n "${TRAIN_JSON:-}" ] && [ "$(basename "$TRAIN_JSON" .json)" = "$split" ]; then
    query_json="$TRAIN_JSON"
  fi
  if [ -z "$query_json" ]; then
    for p in $TEST_BATCH_JSONS; do
      [ -f "$p" ] || continue
      [ "$(basename "$p" .json)" = "$split" ] || continue
      query_json="$p"
      break
    done
  fi
  if [ -z "$query_json" ]; then
    echo "Skip $split: no query JSON (set TRAIN_JSON / TEST_BATCH_JSONS)"
    continue
  fi

  post_json="$WF/snippet_rrf/post_rerank_${split}.json"
  ctx_json="$WF/evidence_snippet/${split}_contexts.json"

  echo "=== $split | top-k=$TOP_K ==="

  python3 "$SCRIPT/post_rerank_json.py" \
    --run-path "$TSV" \
    --query-json "$query_json" \
    --output-path "$post_json" \
    --top-k "$TOP_K"

  python3 "$SCRIPT/build_contexts_from_snippets.py" \
    --post-rerank-json "$post_json" \
    --snippet-windows-dir "$WF/snippet_rerank/windows" \
    --split-name "$split" \
    --corpus-path "$DOCS_JSONL" \
    --output-path "$ctx_json" \
    --window-size "${SNIPPET_WINDOW_SIZE:-3}" \
    --top-windows "${SNIPPET_CONTEXT_TOP_WINDOWS:-2}" 
done