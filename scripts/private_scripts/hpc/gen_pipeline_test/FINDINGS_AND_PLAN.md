# New-generation pipeline on BioASQ — findings & cleaner-experiment plan

Investigation of the RAG-scripts subtree's new distillation generation
(`GENERATION_MODE=direct|claims|facets`) as a second test on BioASQ 13B4.
Working notes; results in `scratchpad/oracle_results/`.

## What we ran
- Subtree pulled; new generation validated end-to-end on BioASQ.
- Reproduction job (retrieval→direct generation, 13B4 golden).
- Oracle generation test: 4 arms (**direct-all, direct-16, claims-16, facets-16**)
  × 2 generators (llama3.3, gemma4:31b), GT snippets as evidence, scored with the
  official evaluator (`../Evaluation-Measures`).

## Findings

### 1. Two real port-from-trec-rag bugs (patched locally; must upstream to RAG-scripts)
- `generation/select_contexts.py`: `BM25Okapi([])` ZeroDivisionError when a query
  yields 0 facet slots. Guarded (`n==0 → []`).
- `generation/distil_claims.py` + `summarize_facets.py`: distilled contexts dropped
  `query_type`, so `generate_answers` couldn't pick the list/factoid/yesno schema →
  emitted only prose `ideal_answer`, `exact_answer=null` → **0 on all BioASQ
  structured metrics**. (No-op on trec-rag, which is prose/nugget-scored.) Fixed by
  propagating `query_type`. Also needed `rank_bm25` in the env.

### 2. Oracle GT snippets are the WRONG input to test distillation
Per-snippet length in 13B4: **median 24 words; 55% ≤25w (~1 sentence), 85% ≤40w**,
2% >100w. Snippets are already atomic → claim-extraction has nothing to compress →
distillation can only paraphrase and lose the verbatim strings the metric rewards.
This reconciles the two earlier results:
- 10-example test on **retrieved full abstracts (~200w)** → facets *helped* recall.
- Oracle **1-sentence snippets** → distillation neutral/harmful.
The distillation only has a job when contexts are LONG (many claims per passage,
more than fit a raw prompt).

### 3. gemma4 direct-all was truncation-contaminated (not a real result)
gemma4 with the `think` flag unset emits a **think-chain in the answer output**,
which consumes `num_predict` → the answer JSON truncates ("incomplete JSON object").
On direct-all (deep prompts + long list answers) this failed **6/19 list Qs** →
empty → scored 0. Excluding failures: gemma direct-all list F1 = **0.539** (vs 0.369
with the zeros; vs direct-16 0.417). llama3.3 had **0** truncations.
- Corrects an earlier wrong claim: "more input hurts gemma / input-saturation is
  generator-dependent" — that was the artifact, not a real effect.
- Fix for any future gemma run: raise `num_predict` substantially, and/or send
  `think:false` on the gemma answer call (or use structured/JSON output).

### 4. Reproduction actually reproduces (earlier "not fully" was a misread)
repro doc-route vs `docs/RESULTS.md` (13B4): rerank MAP@10 0.327 vs 0.326; yesno Acc
0.846 = 0.846; list F1 0.322 vs 0.318; summary R-SU4 F1 0.117 vs 0.120 — all within
noise; only factoid Strict 0.364 vs 0.409 = **one factoid Q flipped** (LLM
nondeterminism). Clean pass. BUT this is an *indirect* (metric) comparison — see
Plan A for the direct output-diff the user actually wants.

### 5. Tentative verdict (must be re-tested on long contexts)
On oracle SNIPPETS, distillation never beats the best direct arm (llama3.3 direct-all
list F1 0.498). But that test is unfair (finding 2), and gemma is contaminated
(finding 3). **No firm verdict on "does distillation help BioASQ" until the
long-context oracle runs.**

## Cleaner-experiment plan

### Plan A — Reproduction the right way (output diff, not metric compare)
Goal: rerun a *previous* config with the new subtree + `GENERATION_MODE=direct` and
**diff outputs** against the saved `/output` run — confirming direct is byte-identical
despite all the subtree changes.
- Target: `bioasq14_output/batch_4/` — has its saved `config.env` **and** reference
  answers (`generation/generation_{baseline,snippet}/BioASQ-task14bPhaseA-testset4_answers.jsonl`).
- Steps: rerun `batch_4/config.env` (testset4) new-subtree, direct, generation ON →
  (a) `diff` retrieval run TSVs (bm25/dense/rerank/fusion) — expect **identical**
  (deterministic); (b) `diff` generation answers — expect near-identical at temp=0
  (caveat: LLM/server nondeterminism can flip a rare question, as in finding 4).

### Plan B — Long-context oracle (the fair distillation test)
Goal: give distillation real material — feed **full abstracts of the gold documents**
(~200w each, ≥16–30 per question) instead of 1-sentence snippets.
- Build contexts: for each 13B4 question, contexts = full abstract text of each gold
  PMID. Abstracts are NOT all local (only top-10 retrieved overlap) → fetch from the
  **cluster corpus `/shared/home/yun.wang/biolab/pubmed/jsonl_2026/*.jsonl`** (records
  keyed by PMID). Concrete: one script that scans that corpus ONCE, collects abstract
  text for the union of gold PMIDs across 13B4 (a few thousand), and writes new-schema
  evidence `oracle_docabs_all_13B4.jsonl` (all gold docs/Q) + `oracle_docabs_16_13B4.jsonl`
  (first 16). Run as a cluster job (single corpus pass). (Alternative: NCBI efetch by PMID.)
  Then reuse `run_oracle_arms.sh` pointing SPLIT/CTX at the new evidence.
- Arms: direct-all, direct-16, claims-16, facets-16 (document route only — snippet
  route N/A here; acceptable). Reuse `run_oracle_arms.sh` with the new evidence.
- Generators: llama3.3 (clean); gemma4 only if we fix the truncation (finding 3).
- Score: official evaluator, 4-arm × 4-type table.

## Execution log

### Plan B — REDESIGNED + long pole running 2026-07-07 (long-context oracle, 13B4)
- **Evidence built** (job 109545, CPU, single corpus pass over `/pubmed/jsonl_2026`,
  1334 shards, ~6 min): `oracle/oracle_docabs_all_13B4.jsonl` (85 q, avg **35.8**
  full-abstract contexts/q; median 6389 w/q, max 23478). **3039/3039 gold PMIDs
  resolved, 0 missing.** Builder: `oracle/build_docabs_oracle.py`. Same evidence
  schema the generation stage consumes; contexts ~200w vs snippet oracle's ~24w.
- **First arm design was WRONG (caught before results).** With full abstracts:
  (a) `direct-all` silently truncates at `num_ctx` for **22/85** questions whose
  gold set exceeds the window — never really "all"; (b) `direct-16` (16 full
  abstracts ~3344w) vs `claims-16` (~350w) is **not budget-matched** (direct got
  ~9x the tokens). The `-16`/`-all` labels were inherited from the *snippet*
  oracle where a snippet≈an abstract≈1 sentence, so budget didn't matter. Killed
  jobs 109549/109550.
- **Corrected design (co-designed with user).** Hold nothing arbitrary; report
  realized token budget per arm.
  - **Direct ladder:** `direct-10` (production default), `direct-16`, **`direct-40`**
    (FIXED x=40, not per-q; questions with <40 gold just use all they have — fine).
    x=40 = largest round doc-count fitting num_ctx=16384 (~12.4k tok evidence +
    room for answer, no truncation); 33/51 ≥25-gold questions actually reach 40
    (subset median gold=48, max 123), so direct never sees the full tail → real
    coverage gap vs distillation. `direct-K` = `generate_answers --max-contexts K`
    on `oracle_docabs_all` (gold order); no evidence rebuild.
  - **Distilled:** `claim-50` + `facet-16` (as proposed) **and** budget-matched
    `claim-x`/`facet-y` (x>50, y>16) tuned to a larger budget. `distil_claims`
    has `--budget-words` → can budget-match claims directly; facets are count-only
    (`select_contexts --n`), size measured.
  - **Test subset:** many-gold questions (**≥25 gold docs**, where the 10/16/x
    ladder separates). 13B4 is the densest split: **51/85** ≥25-gold (all types),
    **14** list. List sweet-spot (list & ≥25-gold & ≥8-nuggets) is thin on 13B4
    (**7**); pool all 4 splits → **15** if the 13B4 list signal needs more power
    (expand later, richest-split-first).
- **Long pole RUNNING (job 109554, llama3.3 @ ana):** `oracle/distill_docabs.sh`
  — extract claims over all 3044 abstracts + facet-summarize (generator- and
  count-independent, ~3-6h), emits distilled at sample counts (claims 16/50,
  facets 8/16) and **prints a token-budget table for every candidate arm** →
  `/yun/bioasq14_output/oracle_docabs_13B4_distill`. NEXT: read the table, lock
  final claim/facet counts + budget-match target, then fire the fast generation
  matrix (llama @ ana, gemma4 @ aga with `GENERATION_THINK=false`) via
  `sbatch_oracle.sh` (now honors `CTX_ALL/CTX_16` + `GENERATION_THINK`).
- **Token table (measured, job 109554 done; medians over 85 q):** direct-10
  2161w/2809tok, direct-16 3325w/4322tok, direct-x(fit~12k) 6389w/8305tok;
  claims-50 616w/800tok, claims-16 205w/266tok; facets-16 314w/408tok. Per-unit:
  abstract ~214w, claim ~13.3w, facet ~47.6w. Extraction: 2996/3044 ok, 30 empty,
  18 fail.
- **Facets are clustering-capped, not count-capped:** median **7** facets/q (mean
  10.7, max 71); facet-16 already returns ALL available. 912 facets total, n_members
  median 2 / mean 8.3 / **max 344** — 46% singletons, the rest absorb redundancy
  (many abstracts restate the same finding → near-dup claims collapse at
  `GENERATION_FACET_DIST_THR=0.4`). Extraction yields only ~2-3 claims/abstract
  (median 2), ~96 raw claims/q → 7 facets. On the ≥25-gold subset facets are richer
  (median 8, mean 13). So facet-y>16 is unreachable without loosening clustering
  (declined). Claims scale freely (claim-160 ≈ direct-10 budget).
- **FINAL ARMS LOCKED (6) + MATRIX FIRED 2026-07-07:**
  - direct-10, direct-16, direct-40 (`--max-contexts K` on `oracle_docabs_all`).
  - claims-50 (compression point) + claims-160 (≈direct-10 tokens, equal-budget
    nuggets-vs-raw). facets-16 (all available).
  - Driver `oracle/run_docabs_arms.sh`; sbatch `oracle/sbatch_docabs_arms.sh`.
  - **gemma think probe result:** `unset` = gemma reasons but trace NOT separated
    (finding-3 truncation risk); `think=true` = reasons, trace in separate field,
    `response` stays clean (generate_answers reads only `response`); `think=false`
    = no reasoning. So gemma runs BOTH false and true (ablation).
  - Jobs: **109685** llama @ ana (unset; builds claim-160) → **109686** gemma
    think=false @ aga (afterok llama) → **109687** gemma think=true @ aga (afterok
    109686). Outputs `/yun/bioasq14_output/oracle_docabs_13B4_{llama,gemma_nothink,gemma_think}`.
- **Score when done:** `eval_list_local.py` for the list signal on the ≥25-gold
  subset (and the list sweet-spot); official evaluator (`../Evaluation-Measures`)
  for the 4-arm×4-type table. Report each arm's realized token budget next to its
  score (arms are not equal-budget — distillation is 1/4–1/10 the direct tokens).

### Plan B RESULTS — list L_F1, many-gold subset, 13B4 (2026-07-08)
Scored with `oracle/score_docabs.sh` → `eval_list_local.py` (fixed: split on `\n`
not `splitlines()` — abstract text carries U+2028/VT that fragmented JSONL records).
Subsets: ge25 = list & ≥25 gold docs (n=14); sweet = & ≥8 nuggets (n=7).

| arm (ge25 n=14) | llama | gemma-false | gemma-true |
|---|---|---|---|
| **direct-10** | **0.491** (P.584 R.464) | 0.336 | 0.354 |
| direct-16 | 0.417 | 0.325 | 0.298 |
| direct-40 | 0.351 | 0.304 | 0.248 |
| claims-50 | 0.404 | 0.287 | 0.329 |
| claims-160 | 0.399 (R.418) | 0.293 (R.378) | 0.309 |
| **facets-16** | 0.334 | 0.263 | **0.377** |

sweet (n=7): gemma-true facets-16 = **0.455** (P.551 R.441) vs its direct-10 0.403.

Findings:
1. **direct-10 best for llama/gemma-false; MORE raw context HURTS** (10>16>40 monotone,
   sharpest under gemma-true: direct_40→0.248) — dilution/lost-in-the-middle.
2. **REASONING×FACETS is the one win:** facets-16 is the WORST arm for llama (0.334)
   and gemma-false (0.263) but the BEST for gemma-**true** (0.377, +0.11 vs no-think;
   sweet 0.455 with P AND R up), overtaking direct-10. Reasoning barely helps direct
   (gemma 0.336→0.354) but transforms facets — a reasoning model unpacks dense facet
   summaries into comprehensive lists; raw abstracts already give surface text.
3. **Claims stay below direct** even at equal budget (claims-160≈direct-10 tokens).
   Coverage shows only as a faint recall gain (claims-160 highest recall in gemma).
4. **Best overall system = llama + direct-10 (0.491)**, no distillation/reasoning;
   gemma-true facets (0.377) wins within gemma but not vs llama.

**VERDICT (list metric):** distillation mostly doesn't beat direct top-10 on BioASQ
list answers — EXCEPT facet distillation + a reasoning generator (gemma think=true),
which overtakes direct within gemma. Top-10 raw abstracts already capture the content
at high precision; more context hurts.

### SILENT-ERROR AUDIT (2026-07-08) — direct_40 only
Cause: "Empty response; cannot parse JSON" after 3 retries = num_ctx overflow on
big prompts. In the scored 14-Q subset, confined to **direct_40**: llama 1, gemma-
false 1 (+claims_50 1), gemma-true **5**/14. All other arms (direct_10/16,
claims_50/160, facets_16) = **0 empties** → headline findings clean. Re-ran direct_40
(+ gemma-false claims_50) at **GENERATION_NUM_CTX=32768** (idempotent driver) —
sbatch now honors the env. So "more context hurts" was OVERSTATED for gemma-true
direct_40 (0.248 was 5/14 truncations, not dilution).

### OFFICIAL 4-TYPE EVAL (2026-07-08, full 85 Q; direct_40 pending redo)
`queries_jsonl_to_bioasq_json.py` (cluster, python3) → pull → local
`Evaluation-Measures/run_phaseB_batch.sh` (Java 11). Table (YN_Acc/F_Strict/F_MRR/L_F1/R_SU4_F1):

| arm | YN | F_Str | F_MRR | L_F1 | R_SU4 |
|---|---|---|---|---|---|
| llama direct_10 | .962 | .455 | .515 | **.527** | .157 |
| llama direct_16 | .962 | .455 | .492 | .471 | .150 |
| llama claims_50/160 | .89/.92 | **.500** | .523 | .42/.44 | .14 |
| llama facets_16 | .923 | **.500** | .523 | .396 | .123 |
| gemma-false direct_10 | 1.00 | .455 | .477 | .359 | **.171** |
| gemma-false direct_16 | 1.00 | .455 | .477 | .344 | .163 |
| gemma-false claims_160 | .923 | **.545** | **.568** | .344 | .145 |
| gemma-false facets_16 | .962 | .500 | .523 | .348 | .134 |
| gemma-true direct_10 | 1.00 | .500 | .523 | .409 | .161 |
| gemma-true direct_16 | 1.00 | .500 | .523 | .367 | .142 |
| gemma-true claims_50/160 | .962 | **.545** | **.568** | .38/.39 | .15/.14 |
| gemma-true facets_16 | .962 | .500 | .523 | .351 | .130 |
(direct_40 pending num_ctx=32768 redo; direct_16<direct_10 on list+summary confirms
the "more context hurts" gradient.)

**Per-type verdict (the real story — distillation is TYPE-DEPENDENT):**
- **Factoid: distillation HELPS** — `claims` beats direct on F_Strict & F_MRR in
  every config (gemma-false .455→.545, gemma-true .500→.545). Extracting the exact
  fact from a deduped claim list beats raw abstracts. It's CLAIMS, not facets.
- **Summary: distillation HURTS** (direct R_SU4 .157–.171 vs distill .12–.15) —
  paraphrase loses n-gram overlap ROUGE rewards. Disconfirms the "semantic metric
  will favor distillation" guess.
- **List: direct_10 best on full 85**; the facets+reasoning win was many-gold-subset-
  specific.
- **Yes/no: near-ceiling** (.88–1.00), direct marginally ahead.
Not one verdict but four. Best per type: list→llama direct_10; factoid→gemma claims;
summary→gemma-false direct_10; yesno→any direct.

### FINAL CLEAN 4-TYPE TABLE (2026-07-08, all 18 arms, 0 silent failures)
After: (a) direct_40 rebuilt at num_ctx=32768 (input-truncation fix); (b) ~9 transient
serve-glitch failures cleared by plain re-run (rescue); (c) 3 stubborn gemma-false
direct cells (q680fcb72, immunoglobulins) injected with the model's own verified
answer. `run_rescue.sh`/`sbatch_rescue.sh` added; generate_answers gained env-gated
GENERATION_NUM_PREDICT/REPEAT_PENALTY (unused in the end — failures were transient,
not runaway; repeat_penalty even CAUSED failures, so rescue uses plain decoding).

| arm | YN | F_Str | F_MRR | L_F1 | R_SU4 |
|---|---|---|---|---|---|
| llama d10/d16/d40 | .96/.96/.92 | .455 | .515/.492/.477 | **.527**/.471/.446 | .157/.150/.148 |
| llama claims50/160 | .923 | .500 | .523 | .416/.442 | .142/.137 |
| llama facets16 | .923 | .500 | .523 | .396 | .123 |
| gemma-false d10/16/40 | 1.00 | .455 | .477 | .412/.397/.403 | **.172**/.164/.159 |
| gemma-false claims160 | .923 | **.545** | **.568** | .344 | .145 |
| gemma-false facets16 | .962 | .500 | .523 | .348 | .134 |
| gemma-true d10/16/40 | 1.00 | **.545** | .568/.568/**.580** | .409/.367/.415 | .163/.145/.147 |
| gemma-true claims50/160 | .962 | **.545** | **.568** | .382/.392 | .150/.142 |
| gemma-true facets16 | .962 | .500 | .523 | **.436** | .134 |

**CORRECTIONS vs the pre-fix numbers:**
1. **"More raw context hurts" was mostly a TRUNCATION ARTIFACT — it evaporates once
   clean.** gemma-true list 0.409→0.367→**0.415** (d40≈d10, not the earlier "collapse
   to 0.248" = its 5 num_ctx=16384 empties); gemma-false flat (.412/.397/.403); only
   llama declines mildly (.527/.471/.446). More full-abstract context ≈ flat-to-
   slightly-negative, NOT a strong penalty.
2. **Reasoning×facets list win is cleaner/stronger:** gemma-true facets16 = **0.436**
   (was .351 with 3 failures) > gemma-true direct_10 .409.
Per-type verdict unchanged: factoid→claims AND reasoning help (.455→.545); summary→
distillation hurts, direct best (gemma-false .172); yesno→ceiling; list→llama d10 (.527).
The silent failures were transient ollama serve glitches (a fresh single re-run of the
identical call succeeds), NOT model weakness/truncation/loops.

### WHY distillation barely helps — the task rewards concision, not coverage (2026-07-08)
BioASQ answers are SHORT. Gold ideal_answer per-reference length (13B4, words):
factoid **25**, yesno **29**, list **36**, summary **53** (medians; p90 ~56–133, max
~185–284). exact_answer is tiny: factoid = **1** item, list ≈ **8**. (Each question
ships ~15 reference ideal answers for multi-reference ROUGE — don't confuse with one
answer's length.) Official BioASQ Task B guideline: ideal answer ≤ **~200 words**
(single paragraph); gold sits far below.
- The ideal-answer metric is **ROUGE-SU4 F1** (F-measure): a too-long answer has high
  recall but low PRECISION → F1 drops. Verbosity is penalized (a recall-only ROUGE
  would instead reward dumping text).
- **Distillation optimizes COMPREHENSIVENESS** (extract every claim / cluster every
  facet across all docs) — the wrong target for a task that rewards concise KEY
  evidence. Top-10 abstracts already hold the key evidence; gathering more (direct-40,
  or distilling ~48 docs) adds low-value coverage that lowers precision.
- Reconciles the per-type table: distillation HELPS **factoid** (win is PRECISION —
  a deduped claim list surfaces the one fact cleanly), HURTS **summary** (ROUGE
  precision punishes paraphrase/reorg), neutral/worse on **list**/**yesno** (direct
  already precise). i.e. distillation only helps where the payoff is precision, not
  coverage.

### WHY llama > gemma — it's LIST-ONLY, a style×metric artifact (2026-07-08)
gemma is NOT globally worse: it's ≥ llama on **yesno** (1.00 vs .962), **factoid**
(claims .545 vs .500), **summary** ROUGE (.171 vs .157). It collapses ONLY on **list**
(llama .527 vs gemma .36–.41). Cause (from reading answers, direct_10, ge25):
gemma over-produces **verbose, over-specific, REDUNDANT** list items — e.g. "diseases
treated by fecal transplant": gemma listed C. difficile **6×** as variants ("recurrent
CDI","severe and fulminant CDI",...) + missed 2 gold items → precision 2/8; llama gave
5 terse canonical items matching gold. Radiomics Q: gemma repeats "texture features"/
"statistic-histogram" duplicating items already listed. So list-F1 punishes gemma's
elaborative style (spurious/dup items tank precision, no recall gain — they rephrase
concepts already counted). **Largely a metric artifact** (content usually right, style
mismatched to terse-canonical list scoring) + a REAL dedup/canonicalization defect.
Reconciles trec-rag (gemma wins on prose/writing/recall; loses on terse structured
extraction). TEST to separate artifact vs real: prompt gemma to dedup + use canonical
short entity names (or post-process merge near-dup list items); if the list gap closes,
it's style not knowledge.

### Plan A — RETIRED 2026-07-07 (not needed; proven by construction)
Decision (with user): the end-to-end reproduction diff is unnecessary. There is
NO trustworthy historical anchor anyway — the pipeline never stamps its config
into the output dir (only logs the config *path*), and no saved run pinned its
generator; dictycite has the same generator-unpinned problem. But the git history
settles it more strongly than any run-diff could: `git diff 2b4d8a0..HEAD` (old
subtree → new) shows the entire **document/direct path is unchanged** — `retrieval/`,
`rerank/`, `evidence/build_doc_contexts.py` have **0 changes**, and
`generation/generate_answers.py` changed only by a payload refactor + env-gated
`num_ctx`/`think` hooks that are **no-ops when `GENERATION_NUM_CTX`/`GENERATION_THINK`
are unset** → the direct-mode ollama request is **byte-identical by construction**.
The only code whose behavior actually changed is the **distillation path**
(`extract_claims`/`distil_claims`/`summarize_facets`/`select_contexts` + the two
BioASQ bugfixes) — which is exactly what **Plan B exercises**. So Plan B validates
the only path that could differ; a direct reproduction would only re-confirm the
untouched path. `PLAN_A_DICTYCITE.md` kept for reference but moot.

### Also
- **Upstream the two bug fixes (finding 1) to fulaibaowang/RAG-scripts.** The 3
  patched files (`distil_claims.py`, `summarize_facets.py`, `select_contexts.py`) are
  currently UNCOMMITTED working-tree edits in BioASQ. Design intent (confirmed with
  user): bug 2 keeps `query_type` if present but allows it absent (trec-rag-safe);
  bug 1 guards the zero-slot crash AND warns (stderr, lists qids) so it's not silent.
  **DONE 2026-07-07: PR https://github.com/fulaibaowang/RAG-scripts/pull/4**
  (branch `fix/bioasq-distillation-query-type-and-zeroslot`, +23/-1, 3 files).
  NOTE: the BioASQ subtree still carries these as local edits until the PR merges
  and the subtree is re-pulled.
- Optionally extend Plan B to the other three 13B splits for n.
