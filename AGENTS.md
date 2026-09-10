# AGENTS.md

How to *change* this repository safely. For running it, start with [README.md](README.md) and
[docs/USAGE.md](docs/USAGE.md).

## What this is

**BioASQ Task 14b is a task layer, not a pipeline.** The retrieval-rerank-generate machinery lives
in [fulaibaowang/RAG-scripts](https://github.com/fulaibaowang/RAG-scripts), vendored here as a git
subtree at `scripts/public/shared_scripts/`. What this repo adds is everything that knows the word
*BioASQ*: the corpus (PubMed), the adapters in and out of the official JSON wire format, the
per-question-type answer schemas, and the task data.

So the first question for any change is **which side of the line it falls on**:

| Concern | Lives | Why |
|---|---|---|
| Retrieval, fusion, rerank, snippet windows, evidence, generation | upstream (`shared_scripts/`) | corpus-agnostic; shared with other projects |
| Orchestrator flags, config variables, output layout, JSONL schemas | upstream | same |
| BioASQ JSON ⇄ pipeline JSONL, PubMed parsing, PMID handling, per-type schemas, HyDE query parsing | here | task-specific by definition |
| Task data, configs, results, notebooks | here | ours |

- **Fix it upstream unless the change is genuinely BioASQ-specific.** PMIDs, question types, and
  the BioASQ wire format belong under `scripts/public/{format,data,evidence,query_parsing}/`.
- **Never delete or rename** a config variable, an output path, or a JSONL field because BioASQ
  stopped using it. Another consumer still does.

**Read [RAG-scripts AGENTS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/AGENTS.md)
before touching anything under `shared_scripts/`** — conventions, the stage-skipping gotcha, CI.
It describes upstream `main`, though, and the vendored copy is frozen at an earlier state.

### Which pipeline version

- **`main` is frozen at the state used for the post-working-note analysis**: tree `2f479ef`
  (RAG-scripts, 2026-07-08), the same pipeline as dictycite `v0.3.0`. Check with
  `git rev-parse HEAD:scripts/public/shared_scripts`. RAG-scripts `main` has moved on since, so
  check the vendored tree for what is present rather than assuming upstream's current state.
- **The working-note submissions are tag `v0.1.0`.** Its vendored pipeline is byte-identical to
  RAG-scripts `v0.1.0` (both tree `0bf93ba`) and predates `STAGE1_SOURCE` and the generation
  checkpoint. Reproducing a submission means checking out that tag.
- Don't run `git subtree pull` as routine maintenance — it would move the code out from under
  published results. Pulling deliberately, for a reason, is a different matter — say so in the
  commit message.

## The BioASQ boundary: adapt-in and adapt-out

The pipeline speaks JSONL (`query_id` / `query_text` / `query_type`). Official files are wrapped
JSON (`{"questions":[…]}` with `id` / `body` / `type`). Three adapters are the whole of the task
coupling — script list: [scripts/public/README.md](scripts/public/README.md). Commands:
[docs/USAGE.md](docs/USAGE.md).

```bash
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py --input task.json --output queries.jsonl
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config my_run.env
python3 scripts/public/format/queries_jsonl_to_bioasq_json.py --input …_answers.jsonl --output submission.json
```

What to keep true when editing them:

- **Adapt-in is lossless and additive.** Gold `documents` / `snippets` ride along so the pipeline
  can compute metrics; `query_parse` / `hyde` metadata too. Unknown fields pass through. Adding a
  field must not change what an existing consumer sees.
- **Adapt-out strips pipeline-only fields, it does not invent them.** `_PIPELINE_STRIP_KEYS` in
  `queries_jsonl_to_bioasq_json.py` is the list; `evidence_ids` is deliberately *kept* as
  provenance. Numeric doc ids become `http://www.ncbi.nlm.nih.gov/pubmed/<pmid>` URLs; non-numeric
  ids pass through verbatim.
- **The `documents` cap is 10 by default** (`--max-documents`, `0` for no limit) and `DOC_CAP = 10`
  in the snippet adapter (`scripts/public/evidence/contexts_json_to_bioasq_snippets.py`) — that is
  the BioASQ Phase A submission limit, not a display choice.
- **Snippet offsets are computed against the corpus text, not the context text.** The snippet
  adapter re-reads the PubMed JSONL (`--corpus-path`) and aligns spans the same way snippet contexts
  were built (NLTK sentence splits on the raw abstract). If you change window construction
  upstream, this alignment is what breaks, and it breaks silently into `--allow-fallback-offsets`
  territory.

## The corpus and the indexes

`docno` **is the PMID** — the one place this repo leans on docid shape, and the reason adapt-out
can build PubMed URLs. Upstream treats docids as opaque; keep that assumption out of
`shared_scripts/`.

Build commands: [docs/USAGE.md](docs/USAGE.md). Parse output schema:
`{docno, pmid, type, title, text, mesh_terms, keywords, is_deleted}`.

- **The same JSONL shards are the corpus at three points**: index build, reranker candidate text,
  and evidence/snippet construction (`DOCS_JSONL`). A shard set that disagrees with the index is a
  silent recall hole, not an error.
- **The dense index is sharded** for full PubMed — `DENSE_INDEX_GLOB` wins over `DENSE_INDEX_DIR`
  when both are set. Default embedder is `abhinand/MedEmbed-small-v0.1`.
- **`scripts/public/data/migrate_jsonl_schema.py` exists because the shard schema changed once**
  (`abstract` → `text`, added `type`). It is idempotent and atomic per shard; run it rather than
  hand-editing old shards.
- Full-PubMed indexing is a ~150 GB, ~6 h job — see
  [README.md](README.md#estimated-resource-requirements) before suggesting a rebuild.

## Running a batch

Start from [`bioasq_data/14b/workflow_config_14b_example.env`](bioasq_data/14b/workflow_config_14b_example.env).
Copy it to a private path — configs carry absolute paths and are not committed.

Easy to get wrong:

- **`GENERATION_SCHEMAS_DIR=$REPO_ROOT/scripts/public/prompts/schemas`.** Without it, generation
  falls back to upstream's default prompts and every question gets the generic schema. This is the
  usual cause of answers in the wrong shape.
- **`HAVE_GROUND_TRUTH=0` for Phase A test sets.** Official test sets have no gold `documents`, so
  metrics come back all zero and look like a broken run. Only golden-enriched and training data
  have ground truth.
- **`--dense-query-field query_text,query_text_hyde`** to use HyDE. BM25 stays on `query_text`.

**A stage whose outputs already exist is skipped.** Point `WORKFLOW_OUTPUT_DIR` somewhere new.
Symptoms and the generation checkpoint sidecar:
[RAG-scripts AGENTS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/AGENTS.md).

## Query parsing and HyDE (stage 0, BioASQ-only)

`scripts/public/query_parsing/` runs *before* adapt-in. One LLM pass normalizes the question and
sets a per-question `hyde_enabled` flag — off for numeric/measurement, exact-identifier and other
narrow-target questions, where a hypothetical abstract pulls retrieval away from the answer.
Prompt: [`query_parsing/prompt.md`](scripts/public/query_parsing/prompt.md) (background:
[`MULTI_QUERY_HYDE.md`](scripts/public/query_parsing/MULTI_QUERY_HYDE.md)).

`prepare_query.py` turns `query_parse` into `query_text_normalized` and `query_text_hyde`; adapt-in
applies the same rules in-process when a question already carries a complete `query_parse`, so the
two paths must agree: `bioasq_json_to_queries_jsonl.py` imports `prepare()` rather than
reimplementing it — keep it that way.

## Answer schemas per question type

One file per type in `scripts/public/prompts/schemas/` (`factoid`, `list`, `yesno`, `summary`, plus
`default` for a missing or unknown `query_type`). Generation resolves `<type>.txt` by name, so
**adding a type is adding a file**, not changing code. `query_type` is optional on the wire; never
write code that requires it.

## Verifying a change

There is no unit test framework here. What exists:

```bash
python3 -m compileall -q scripts/public/
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -h
python3 -m compileall -q notebooks/
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input example/13b_golden_50q_sample.json --output /tmp/q.jsonl
```

`.github/workflows/pipeline-smoke.yml` runs the first two on pushes touching
`scripts/public/shared_scripts/**` — no corpus, no GPU. **Pipeline CI is upstream.** A pipeline
change belongs there.

`example/` ships three question samples; everything else there is gitignored local scratch.

## Notebooks

`jupytext` with `formats = "ipynb,py:percent"` (`pyproject.toml`). **The `.py` files are the source
of truth.** Edit the `.py`, then `jupytext --sync notebooks/<name>.py`. Never resolve a conflict by
editing the `.ipynb`.

## Repo map

| Path | Contents |
|---|---|
| `scripts/public/shared_scripts/` | **Subtree from RAG-scripts** — the pipeline |
| `scripts/public/format/` | Adapt-in and adapt-out for the BioASQ JSON wire format |
| `scripts/public/data/` | PubMed XML parsing, shard schema migration, PMID subsetting, question simplification |
| `scripts/public/evidence/` | Contexts → BioASQ `documents` + `snippets` with offsets (Phase A) |
| `scripts/public/query_parsing/` | Normalization, per-question HyDE decision, `query_text_hyde` |
| `scripts/public/prompts/schemas/` | Per-`query_type` answer schemas |
| `bioasq_data/` | Official task data and the example workflow config |
| `example/` | Small committed question samples for local testing |
| `docs/` | BioASQ-oriented runbook, results, task background |
| `notebooks/` | Analysis and ablations (jupytext `.py` is source) |
| `scripts/private_scripts/` | HPC/SLURM job scripts — machine-specific, not runnable as-is |
| `scripts/deprecated/` | Superseded experiments. **Do not build on these** |

## Where the answers are

| Question | Source |
|---|---|
| What flags and env toggles exist? | `run_retrieval_rerank_pipeline.sh --help` |
| What does a knob do, what range is sane? | [vendored docs/PARAMETERS.md](scripts/public/shared_scripts/docs/PARAMETERS.md) |
| What does a run write, and where? | [vendored docs/output.md](scripts/public/shared_scripts/docs/output.md) |
| How do I change the pipeline? | [RAG-scripts AGENTS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/AGENTS.md) |
| Docker, indexes, BioASQ paths, adapters | [docs/USAGE.md](docs/USAGE.md) |
| What did we measure? | [docs/RESULTS.md](docs/RESULTS.md) |
| What is the task, who won last year? | [docs/BioASQ.md](docs/BioASQ.md) |

When `--help` and the docs disagree, `--help` is right — fix BioASQ docs here, pipeline docs
upstream.

## What we know that the code doesn't say

- **BM25 usually beats dense retrieval on BioASQ.** Biomedical questions are entity-heavy and the
  lexical match is often exactly right. Fused (`rrf`) still beats either alone; don't propose
  dropping BM25.
- **HyDE helps dense retrieval, selectively.** Hence the per-question `hyde_enabled` flag rather
  than a global switch.
- **LLM query rewriting did not improve MAP** in the configurations tested —
  `scripts/deprecated/query_rewrite_llm.py`. Treat it as answered, not unexplored.
- **`bge-reranker-v2-m3` at `max_length=512` is the default reranker** because it measurably beat
  the MiniLM cross-encoder. Reranking, not first-stage retrieval, is where the gain is.

## Style

Match the file you are editing: `set -e` bash, stdlib-plus-pinned-deps Python, argparse CLIs, no
framework. There is no linter or formatter config, so consistency with the surrounding code is the
standard.
