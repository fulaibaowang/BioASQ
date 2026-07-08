# Generation-pipeline test (reproducibility + claims/facets A/B/C)

Validates the new subtree in two steps:

1. **Reproducibility** — rerun the 13B golden splits with `GENERATION_MODE`
   unset (`direct`, byte-identical to the old pipeline) and confirm retrieval +
   Phase B metrics match [`docs/RESULTS.md`](../../../../docs/RESULTS.md).
2. **New generation** (real test) — **document route**, on **nugget-rich list
   questions** (`nugget_rich_list_qids.txt`: list type, ≥8 gold answer-items).
   The nugget count = `len(exact_answer)`; BioASQ **L_Recall is nugget-recall**,
   so it measures answer comprehensiveness directly. Compare `direct` vs
   `claims` vs `facets`.
   Why not "≥10 gold docs": doc count is evidence breadth, not answer
   complexity. On the ≥10-doc subset the direct pipeline already scores L_F1
   0.34 (little to gain → planning is a no-op). On the nugget-rich subset the
   direct pipeline scores **L_F1 ≈ 0.185, L_Recall ≈ 0.16** — large headroom,
   the regime where facet planning can actually help.
   Subset sizes: ≥8 nuggets = 25 Q (13B4: 10); ≥6 nuggets = 40 Q (13B4: 13).
   The 13B4-only run gives the first signal; expand to the other splits after.
   (`oracle_rich_list_qids.txt` is the older ≥10-doc set, kept for reference.)

**Generators:** only self-hosted **llama3.3:latest** (ollama serve on node `ana`,
`http://ana:11434/api/generate`) is set up for now. Add `gemma4:31b` once its
serve is ready: `GEN_MODELS="llama3.3:latest gemma4:31b"`.

## Files

| File | Purpose |
|------|---------|
| `config_repro_13b4.env` | Base config; full pipeline, `direct` generation, ground truth on. |
| `config_gen_{direct,claims,facets}.env` | Source the base, override `GENERATION_MODE` + output dir. |
| `sbatch_gen_pipeline_test.sh` | HPC driver: repro run → seed → generation modes. Loops splits. |
| `oracle_rich_list_qids.txt` | The 60 subset qids (split, qid, #gold-docs, body). |
| `compare_subset_metrics.py` | Subsets the official per-question phaseB TSV; reports mean L_P/L_R/L_F1 + Rouge and Δ vs `direct`. |

## Baseline already established locally

`compare_subset_metrics.py` run against the existing `report/phaseB_report_perq.tsv`
gives the **old-pipeline (direct)** numbers on the 60-question subset:

```
mode        L_P      L_R     L_F1   R_2_F1  R_SU4_F1   n
direct    0.3698   0.3600   0.3426  0.1106   0.1254   60
```

The new modes must beat `L_F1 = 0.3426` to count as helpful.

## Subset composition

60 list questions with ≥10 gold documents (median 31): 13B1 ×14, 13B2 ×14,
13B3 ×14, 13B4 ×18. Regenerate with the snippet in the working notes.

## Run

```bash
# fast first pass: richest split only
SPLITS=13B4 sbatch scripts/private_scripts/hpc/gen_pipeline_test/sbatch_gen_pipeline_test.sh
# full RESULTS.md reproduction
sbatch scripts/private_scripts/hpc/gen_pipeline_test/sbatch_gen_pipeline_test.sh
```

Then Phase 3: convert each mode's answers to submission JSON, run the official
BioASQ evaluator, and compare per the sbatch tail instructions.

## Notes

- `claims`/`facets` are **ollama-only**; `GENERATION_EXTRACT_MODEL` (extractor)
  is independent of the generator and defaults to `llama3.3:latest`.
- If distilled answers truncate (`incomplete JSON object`), raise
  `GENERATION_NUM_CTX` — never cut slots (see `docs/PARAMETERS.md`).
- Document route gives claim ordering by retrieval rank; the snippet route
  carries CE scores (better claim ordering). Both routes run here.
