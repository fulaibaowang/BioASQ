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

**Read [RAG-scripts AGENTS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/AGENTS.md)
before touching anything under that directory** — it is the pipeline's own operating manual (stage
table, conventions, the stage-skipping gotcha, CI). Everything it says applies here unchanged.
The vendored copy can lag upstream, so read it upstream and check the vendored tree for what is
actually present here.

### The subtree rule

`scripts/public/shared_scripts/` is not ours to edit casually. A commit there propagates upstream
and from upstream into other consumers, so:

- **Fix it upstream unless the change is genuinely BioASQ-specific.** If it needs to know about
  PMIDs, BioASQ question types, or the BioASQ wire format, it does not belong upstream at all —
  write it under `scripts/public/{format,data,evidence,query_parsing}/` instead.
- **Never delete or rename a config variable, an output path, or a JSONL field** because BioASQ
  stopped using it. Another consumer still does.
- Bring upstream changes down with
  `git subtree pull --prefix scripts/public/shared_scripts sharedscripts main --squash`.

## The BioASQ boundary: adapt-in and adapt-out

The pipeline reads and writes **JSONL only**, with `query_id` / `query_text` / `query_type` on the
wire. Official BioASQ files are wrapped JSON (`{"questions": [...]}`) with `id` / `body` / `type`.
Three adapters bridge that, and they are the whole of the task coupling:

| Direction | Script | Turns |
|---|---|---|
| **in** | `scripts/public/format/bioasq_json_to_queries_jsonl.py` | `{"questions":[…]}` → one query object per line |
| **out (answers)** | `scripts/public/format/queries_jsonl_to_bioasq_json.py` | `*_answers.jsonl` → `{"questions":[…]}` with `documents` URLs |
| **out (Phase A evidence)** | `scripts/public/evidence/contexts_json_to_bioasq_snippets.py` | `*_contexts.jsonl` → `documents` + `snippets` with character offsets |

```bash
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py --input task.json --output queries.jsonl
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config my_run.env
python3 scripts/public/format/queries_jsonl_to_bioasq_json.py --input …_answers.jsonl --output submission.json
```

What to keep true when editing them:

- **Adapt-in is lossless and additive.** `documents` and `snippets` (gold, when present) ride along
  untouched so the pipeline can compute metrics; `query_parse` / `hyde` metadata rides along too.
  Unknown fields pass through. Adding a field must not change what an existing consumer sees.
- **Adapt-out strips pipeline-only fields, it does not invent them.** `_PIPELINE_STRIP_KEYS` in
  `queries_jsonl_to_bioasq_json.py` is the list; `evidence_ids` is deliberately *kept* as
  provenance. Numeric doc ids become `http://www.ncbi.nlm.nih.gov/pubmed/<pmid>` URLs; non-numeric
  ids pass through verbatim.
- **The `documents` cap is 10 by default** (`--max-documents`, `0` for no limit) and `DOC_CAP = 10`
  in the snippet adapter — that is the BioASQ Phase A submission limit, not a display choice.
- **Snippet offsets are computed against the corpus text, not the context text.** The snippet
  adapter re-reads the PubMed JSONL (`--corpus-path`) and aligns spans the same way snippet contexts
  were built (NLTK sentence splits on the raw abstract). If you change window construction upstream,
  this alignment is what breaks, and it breaks silently into `--allow-fallback-offsets` territory.

## The corpus and the indexes

`docno` **is the PMID** — the one place this repo leans on docid shape, and the reason adapt-out can
build PubMed URLs. Upstream treats docids as opaque; keep that assumption out of `shared_scripts/`.

The corpus is built once, then reused by every run:

```
PubMed baseline XML.gz  (ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
  → scripts/public/data/parse_pubmed_local.py           → JSONL shards
      {docno, pmid, type, title, text, mesh_terms, keywords, is_deleted}
  → shared_scripts/index/build_bm25_index_from_jsonl_shards.py        → Terrier index   (BM25_INDEX_PATH)
  → shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py  → HNSW index      (DENSE_INDEX_DIR / DENSE_INDEX_GLOB)
```

Facts worth having before you touch indexing (commands: [docs/USAGE.md](docs/USAGE.md)):

- **The same JSONL shards are the corpus at three points**: index build, reranker candidate text,
  and evidence/snippet construction (`DOCS_JSONL`). A shard set that disagrees with the index is a
  silent recall hole, not an error.
- **The dense index is sharded** for full PubMed — `DENSE_INDEX_GLOB` wins over `DENSE_INDEX_DIR`
  when both are set. Default embedder is `abhinand/MedEmbed-small-v0.1`.
- **`scripts/public/data/migrate_jsonl_schema.py` exists because the shard schema changed once**
  (`abstract` → `text`, added `type`). It is idempotent and atomic per shard; run it rather than
  hand-editing old shards.
- Full-PubMed indexing is a ~150 GB, ~6 h job — see the requirements table in
  [README.md](README.md#estimated-resource-requirements) before suggesting a rebuild.

## Running a batch

One entrypoint, one config file:

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config /path/to/my_run.env
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --help    # authoritative flag list
```

Start from [`bioasq_data/14b/workflow_config_14b_example.env`](bioasq_data/14b/workflow_config_14b_example.env)
(BioASQ-shaped: both routes, PubMed paths, `GENERATION_SCHEMAS_DIR` pointed at this repo) or from the
upstream templates in `scripts/public/shared_scripts/conf/`. Copy it to a private path — configs
carry absolute paths and are not committed.

BioASQ-specific settings that are easy to get wrong:

- **`GENERATION_SCHEMAS_DIR=$REPO_ROOT/scripts/public/prompts/schemas`.** Without it, generation
  falls back to the upstream default prompts dir and every question gets the generic schema instead
  of the per-type one. This is the single most common cause of "the answers came back in the wrong
  shape".
- **`HAVE_GROUND_TRUTH=0` for Phase A test sets.** Official test sets have no gold `documents`, so
  metrics come back all zero and look like a broken run. Only golden-enriched and training data have
  ground truth.
- **`--dense-query-field query_text,query_text_hyde`** to use HyDE (see below). BM25 stays on
  `query_text`; HyDE helps dense retrieval, not lexical.

**A stage whose outputs already exist is skipped.** Editing a stage and re-running the same config
changes nothing. Point `WORKFLOW_OUTPUT_DIR` somewhere new — it is the only reliable reset. (Full
explanation, including the generation checkpoint sidecar, in the upstream AGENTS.md.)

## Query parsing and HyDE (stage 0, BioASQ-only)

`scripts/public/query_parsing/` adds fields to questions *before* adapt-in. One LLM pass normalizes
the question and, **per question**, decides whether HyDE helps: it is switched off for
numeric/measurement, exact-identifier and other narrow-target questions, where a hypothetical
abstract pulls retrieval away from the answer. The decision is a `hyde_enabled` flag inside
`query_parse`, and the prompt that makes it is [`query_parsing/prompt.md`](scripts/public/query_parsing/prompt.md)
(background: [`MULTI_QUERY_HYDE.md`](scripts/public/query_parsing/MULTI_QUERY_HYDE.md)).

`prepare_query.py` turns `query_parse` into the `query_text_normalized` and `query_text_hyde` fields;
adapt-in applies the same rules in-process when a question already carries a complete `query_parse`,
so the two paths must stay in agreement — `bioasq_json_to_queries_jsonl.py` imports `prepare()`
rather than reimplementing it. Keep it that way.

## Answer schemas per question type

BioASQ has four question types and each wants a different answer object. One file per type in
`scripts/public/prompts/schemas/` — `factoid.txt`, `list.txt`, `yesno.txt`, `summary.txt`, plus
`default.txt` as the fallback for a missing or unknown `query_type`. Generation resolves
`<type>.txt` by name, so **adding a type is adding a file**, not changing code. `query_type` is
optional on the wire; never write code that requires it.

## Verifying a change

There is no unit test framework here. What exists:

```bash
python3 -m compileall -q scripts/public/            # syntax
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -h    # orchestrator parses
python3 -m compileall -q notebooks/                 # notebooks are .py first (see below)
```

`.github/workflows/pipeline-smoke.yml` runs the first two on every push touching
`scripts/public/shared_scripts/**`. It is deliberately light: no corpus, no index, no GPU.
**The real pipeline CI is upstream** — RAG-scripts runs mock-LLM generation and end-to-end configs
in Docker. A pipeline change belongs there, where it is actually tested.

Round-tripping the adapters on a committed sample is the cheapest real check available here:

```bash
python3 scripts/public/format/bioasq_json_to_queries_jsonl.py \
  --input example/13b_golden_50q_sample.json --output /tmp/q.jsonl
```

`example/` ships three question samples (`13b_golden_50q_sample.json`,
`training14b_10pct_sample.json`, `training14b_3pct_sample.json`); everything else there is
gitignored local scratch.

## Notebooks

`jupytext` with `formats = "ipynb,py:percent"` (`pyproject.toml`). **The `.py` files are the source
of truth**; `.ipynb` is generated. Edit the `.py`, then `jupytext --sync notebooks/<name>.py`. Never
resolve a conflict by editing the `.ipynb`.

## Repo map

| Path | Contents |
|---|---|
| `scripts/public/shared_scripts/` | **Subtree from RAG-scripts** — the pipeline. Its own AGENTS.md governs it |
| `scripts/public/format/` | Adapt-in and adapt-out for the BioASQ JSON wire format |
| `scripts/public/data/` | PubMed XML parsing, shard schema migration, PMID subsetting, question simplification |
| `scripts/public/evidence/` | Contexts → BioASQ `documents` + `snippets` with offsets (Phase A) |
| `scripts/public/query_parsing/` | Normalization, per-question HyDE decision, `query_text_hyde` |
| `scripts/public/prompts/schemas/` | Per-`query_type` answer schemas |
| `bioasq_data/` | Official task data and the example workflow config |
| `example/` | Small committed question samples for local testing |
| `docs/` | BioASQ-oriented runbook, results, task background |
| `notebooks/` | Analysis and ablations (jupytext `.py` is source) |
| `scripts/private_scripts/` | HPC/SLURM job scripts and configs — machine-specific, not runnable as-is |
| `scripts/deprecated/` | Superseded experiments, kept for the record. **Do not build on these** |

## Where the answers are

| Question | Source |
|---|---|
| What flags and env toggles exist? | `run_retrieval_rerank_pipeline.sh --help` |
| What does a knob do, what range is sane? | [RAG-scripts docs/PARAMETERS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/PARAMETERS.md) |
| What does a run write, and where? | [RAG-scripts docs/output.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/docs/output.md) |
| How do I change the pipeline? | [RAG-scripts AGENTS.md](https://github.com/fulaibaowang/RAG-scripts/blob/main/AGENTS.md) |
| Docker, indexes, BioASQ paths, adapters | [docs/USAGE.md](docs/USAGE.md) |
| What did we measure? | [docs/RESULTS.md](docs/RESULTS.md) |
| What is the task, who won last year? | [docs/BioASQ.md](docs/BioASQ.md) |

When `--help` and the docs disagree, `--help` is right — then fix the docs.

## What we know that the code doesn't say

- **BM25 usually beats dense retrieval on BioASQ.** Biomedical questions are entity-heavy and the
  lexical match is often exactly right. Fused (`rrf`) still beats either alone; don't propose
  dropping BM25.
- **HyDE helps dense retrieval, selectively.** Hence the per-question `hyde_enabled` flag rather
  than a global switch.
- **LLM query rewriting did not improve MAP** in the configurations tested — see
  `scripts/deprecated/query_rewrite_llm.py`. Treat it as answered, not unexplored.
- **`bge-reranker-v2-m3` at `max_length=512` is the default reranker** because it measurably beat
  the MiniLM cross-encoder. Reranking, not first-stage retrieval, is where the gain is.

## Style

Match the file you are editing: `set -e` bash, stdlib-plus-pinned-deps Python, argparse CLIs, no
framework. There is no linter or formatter config, so consistency with the surrounding code is the
standard.
