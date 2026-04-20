REPO_ROOT=/work

# Workflow root (snippet route)
WF=/home/wangy/output/workflow_baseline_full_run_both_routes_gemma
# Cap on docs per question in post-rerank JSONL
TOP_K=30   # or whatever you want instead of EVIDENCE_TOP_K
# Corpus for context building (glob or file)
export DOCS_JSONL="/pubmed/jsonl_2026/*.jsonl"   # same as DOCS_JSONL for that run
# Same query JSONLs as the pipeline (source your real config, or paste paths)
export INPUT_JSONL="$REPO_ROOT/example/training14b_10pct_sample.jsonl"   # if unused, leave empty: INPUT_JSONL=

export INPUT_BATCH_JSONLS="$REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B1_golden.jsonl $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B2_golden.jsonl $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B3_golden.jsonl $REPO_ROOT/bioasq_data/Task13BGoldenEnriched/13B4_golden.jsonl"

# Snippet window / context knobs (optional)
export SNIPPET_WINDOW_SIZE=3
export SNIPPET_CONTEXT_TOP_WINDOWS=2

SCRIPT=scripts/public/shared_scripts/evidence

for TSV in "$WF/snippet/snippet_doc_fusion/runs/"*.tsv; do
  [ -f "$TSV" ] || continue
  stem=$(basename "$TSV" .tsv)
  split=${stem#best_rrf_}
  split=${split%%_top*}
  [ -n "$split" ] || continue

  query_jsonl=""
  if [ -n "${INPUT_JSONL:-}" ] && [ "$(basename "$INPUT_JSONL" .jsonl)" = "$split" ]; then
    query_jsonl="$INPUT_JSONL"
  fi
  if [ -z "$query_jsonl" ]; then
    for p in $INPUT_BATCH_JSONLS; do
      [ -f "$p" ] || continue
      [ "$(basename "$p" .jsonl)" = "$split" ] || continue
      query_jsonl="$p"
      break
    done
  fi
  if [ -z "$query_jsonl" ]; then
    echo "Skip $split: no query JSONL (set INPUT_JSONL / INPUT_BATCH_JSONLS)"
    continue
  fi

  post_jsonl="$WF/snippet/snippet_doc_fusion/post_rerank_${split}.jsonl"
  ctx_jsonl="$WF/evidence/evidence_snippet/${split}_contexts.jsonl"

  echo "=== $split | top-k=$TOP_K ==="

  python3 "$SCRIPT/post_rerank_json.py" \
    --run-path "$TSV" \
    --query-jsonl "$query_jsonl" \
    --output-path "$post_jsonl" \
    --top-k "$TOP_K"

  python3 "$SCRIPT/build_contexts_from_snippets.py" \
    --post-rerank-jsonl "$post_jsonl" \
    --snippet-windows-dir "$WF/snippet/snippet_rerank/windows" \
    --split-name "$split" \
    --corpus-path "$DOCS_JSONL" \
    --output-path "$ctx_jsonl" \
    --window-size "${SNIPPET_WINDOW_SIZE:-3}" \
    --top-windows "${SNIPPET_CONTEXT_TOP_WINDOWS:-2}"
done
