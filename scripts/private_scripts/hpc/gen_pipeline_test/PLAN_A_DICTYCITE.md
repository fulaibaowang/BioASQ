# Plan A — reproduction-by-output-diff, run in the dictycite repo

Goal: confirm the new RAG-scripts subtree (with the distillation additions +
the 2 BioASQ bugfixes) leaves **deterministic direct-mode output byte-identical**
to a previously saved run. Do it in **dictycite**, not BioASQ, because:

- BioASQ's named anchor `bioasq14_output/batch_4/` doesn't exist, and the saved
  BioASQ answers were produced by an **unpinned** generator (Mar/Apr runs) → a
  generation diff is meaningless there.
- dictycite runs have **good provenance**: the sbatch copies the config into the
  output dir (`cp $PIPELINE_CONFIG $WORKFLOW_OUTPUT_DIR/`), and retrieval is on a
  **small local index** (`indexes/dicty_bm25_index`, `indexes/dicty_medembed_index`)
  → a full rerun is minutes, not the 12h BioASQ full-PubMed job.

## Anchor (cluster, node frida)
- Repo: `~/dictycite` (HEAD 0eaa4fc, branch main).
- Saved run: `output/workflow_frida_7a_public_goldset_both_routes/`
  - saved config: `config_frida_7a_public_goldset.env` (copied in-place)
  - has `retrieval/{bm25,dense,fusion}`, `rerank/{cross_encoder,post_rerank_fusion,post_rerank_fusion_snippet}`,
    `snippet/`, `evidence/evidence_baseline`, `generation/generation_baseline`.
- Config: `scripts/private_scripts/hpc_scripts/frida/config_frida_7a_public_goldset.env`
  (`WORKFLOW_OUTPUT_DIR=$REPO_ROOT/output/workflow_frida_7a_public_goldset_both_routes`,
  input `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl`).
- Driver: `scripts/private_scripts/hpc_scripts/frida/sbatch_pipeline_full_goldset.sh`.

## Steps
1. **Put the new (patched) subtree into dictycite's checkout.** The diff is only
   meaningful vs the *current* subtree — the one carrying RAG-scripts PR #4
   (`query_type` propagation + zero-slot guard) plus whatever else changed since
   the saved run. Bring dictycite's `scripts/public/shared_scripts/` to that state
   (git subtree pull, or rsync the BioASQ subtree in). Record the before/after
   subtree commit.
2. **Rerun into a fresh dir**, e.g. `WORKFLOW_OUTPUT_DIR=.../workflow_frida_7a_public_goldset_both_routes_REPRO`,
   `GENERATION_MODE` unset (= direct). Point the generator at a standing serve
   (`OLLAMA_URL=http://ana:11434/api/generate`, llama3.3:latest) so generation
   provenance is pinned this time.
3. **Diff (a) retrieval/rerank/fusion run TSVs** old vs REPRO — expect
   **byte-identical** (deterministic; this is the real proof the subtree churn
   didn't move first-stage output). `analysis/compare_result_dirs.py` exists in
   the subtree, or a plain `diff -r` on the `runs/*.tsv`.
4. **Diff (b) generation answers** at temp=0 — expect near-identical; a rare
   question may flip on LLM/server nondeterminism (BioASQ finding 4 saw exactly
   one factoid flip). Since the OLD dictycite generation model may itself be
   unpinned, treat (b) as best-effort; (a) is the load-bearing result.

## Standing resources (as of 2026-07-07, ~6d walltime left)
- `ollama_serve` job on **ana** (A100) and on **aga** (A100) — BOTH serve
  `llama3.3:latest` + `gemma4:31b`. URLs `http://ana:11434/api/generate`,
  `http://aga:11434/api/generate`.
