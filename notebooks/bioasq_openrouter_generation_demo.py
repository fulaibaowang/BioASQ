# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dicty (Python 3.14 venv)
#     language: python
#     name: dicty-py314
# ---

# %% [markdown]
# # OpenRouter — four experiments (snippet file: first three + facet→answer)
#
# 1. **Baseline contexts** — first question only in `evidence/evidence_baseline/..._contexts.json`; full BioASQ answer JSON via `build_full_prompt_for_record` (same as [`generate_answers.py`](../scripts/public/shared_scripts/generation/generate_answers.py)).
# 2. **Snippet-route contexts** — questions at **indices 0, 1, 2** in `evidence_snippet/..._contexts.json` (each up to 30×960 passages); same answer path, stress test.
# 3. **Evidence planner** — same three snippet questions; facet-plan JSON only (no BioASQ answer schema).
# 4. **Answer from facet plan** — **first snippet question only**: run planner once, drop `unused_evidence`, then same pipeline [`system.txt`](../scripts/public/shared_scripts/prompts/system.txt) + schema + [`user_base.txt`](../scripts/public/shared_scripts/prompts/user_base.txt) with **`{EVIDENCE_BLOCK}`** = JSON `{{"facets": [...]}}` (not raw passages). System message gets a short clarification that input is structured facets, not raw text.
#
# **Model id:** set `GENERATION_MODEL` in `.env` only — this notebook does **not** substitute a default (so your provider choice, e.g. `google/gemini-2.5-flash-lite:floor`, is never overridden here).
#
# Experiments 1–2 use `build_full_prompt_for_record` → `chat/completions`. Experiment 3 uses `format_evidence_block` + planner template. Experiment 4 chains planner → pipeline-style answer prompt.

# %%
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Notebook lives in repo_root/notebooks/
REPO_ROOT = Path("..").resolve()

env_path = REPO_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Secrets: keep in .env (gitignored) or export before starting Jupyter
GEN_API_KEY = (os.getenv("GEN_API_KEY") or "").strip()
# Run config (can also live in workflow env files)
GEN_API_BASE = (os.getenv("GEN_API_BASE") or "https://openrouter.ai/api/v1").strip().rstrip("/")
# OpenRouter model id: MUST come from env — no hardcoded default (avoids silently changing your model).
MODEL = "google/gemini-2.5-flash-lite:floor"

CONTEXTS_JSON = (
    REPO_ROOT
    / "bioasq14_output"
    / "batch_1"
    / "evidence/evidence_baseline"
    / "BioASQ-task14bPhaseB-testset1_contexts.json"
)
PROMPTS_DIR = REPO_ROOT / "scripts" / "public" / "shared_scripts" / "prompts"

MAX_CONTEXTS = 8
MAX_CHARS_PER_CONTEXT = 1300

REPO_ROOT, CONTEXTS_JSON.exists(), bool(GEN_API_KEY), MODEL or "(GENERATION_MODEL unset)"


# %%
if not GEN_API_KEY:
    raise RuntimeError(
        "Missing GEN_API_KEY. Add it to repo .env (gitignored) or export it before running this notebook."
    )
if not MODEL:
    raise RuntimeError(
        "Missing GENERATION_MODEL in environment or .env (e.g. google/gemini-2.5-flash-lite:floor). "
        "This notebook does not set a default model id."
    )


# %%
# Load generate_answers from an explicit path so a long-lived Jupyter kernel does not keep
# a stale sys.modules["generate_answers"] from before optional-qtype edits.
_gen_dir = REPO_ROOT / "scripts" / "public" / "shared_scripts" / "generation"
if str(_gen_dir) not in sys.path:
    sys.path.insert(0, str(_gen_dir))

_ga_path = _gen_dir / "generate_answers.py"
if not _ga_path.is_file():
    raise FileNotFoundError(f"Expected generate_answers at {_ga_path}")
_spec = importlib.util.spec_from_file_location("generate_answers", _ga_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not create import spec for {_ga_path}")
_ga_mod = importlib.util.module_from_spec(_spec)
sys.modules["generate_answers"] = _ga_mod
_spec.loader.exec_module(_ga_mod)

from generate_answers import (  # noqa: E402
    build_full_prompt_for_record,
    call_llm_openai_compat,
    chat_completions_endpoint,
    extract_first_json_object,
    format_evidence_block,
    format_user_prompt,
    load_contexts_json,
    parse_answer_json_for_type,
    resolve_schema_block,
)


# %%
questions = load_contexts_json(CONTEXTS_JSON)
record = questions[0]
record["id"], record.get("type"), (record.get("body") or "")[:120]


# %%
def split_system_user(combined: str) -> tuple[str, str]:
    """Split output of build_full_prompt_for_record into chat messages."""
    prefix, marker = "[SYSTEM]\n", "\n\n[USER]\n"
    if not combined.startswith(prefix) or marker not in combined:
        raise ValueError("Unexpected combined prompt format from build_full_prompt_for_record")
    body = combined[len(prefix) :]
    system_text, user_text = body.split(marker, 1)
    return system_text.strip(), user_text.strip()


full_prompt = build_full_prompt_for_record(
    record,
    PROMPTS_DIR,
    max_contexts=MAX_CONTEXTS,
    max_chars_per_context=MAX_CHARS_PER_CONTEXT,
)
if not full_prompt:
    raise RuntimeError("Empty prompt (missing prompts dir, question, or contexts?)")

system_text, user_text = split_system_user(full_prompt)

print("=== FULL PROMPT (single block, same shape as Ollama `prompt` field) ===\n")
print(full_prompt)

print("\n--- For OpenAI-style chat API ---")
print("POST", chat_completions_endpoint(GEN_API_BASE))
print("model:", MODEL)
print("system chars:", len(system_text), "user chars:", len(user_text))


# %%
raw = call_llm_openai_compat(
    GEN_API_KEY,
    GEN_API_BASE,
    system_text,
    user_text,
    MODEL,
    timeout=180,
    temperature=0.0,
    top_p=1.0,
)

print("--- Raw model output (truncated to 4000 chars if longer) ---\n")
if len(raw) > 4000:
    print(raw[:4000] + "\n... [truncated]")
else:
    print(raw)


# %%
qtype = (record.get("type") or "").strip().lower()
parsed = parse_answer_json_for_type(raw, qtype, q_id=record.get("id"))
print("--- Parsed / validated ---")
print(json.dumps(parsed, indent=2, ensure_ascii=False))


# %% [markdown]
# ## Experiment 2 — Snippet evidence (many contexts), **three questions**
#
# Questions at **indices 0, 1, 2** in `evidence_snippet/..._contexts.json`. Up to **30** contexts × **960** chars each (snippet-route style). Change `SNIPPET_QUESTION_INDICES` if you want other rows.

# %%
SNIPPET_CONTEXTS_JSON = (
    REPO_ROOT
    / "bioasq14_output"
    / "batch_1"
    / "evidence_snippet"
    / "BioASQ-task14bPhaseB-testset1_contexts.json"
)

# Match pipeline-style snippet caps (see run_retrieval_rerank_pipeline.sh defaults)
SNIPPET_MAX_CONTEXTS = 30
SNIPPET_MAX_CHARS_PER_CONTEXT = 960
# First question + two more (indices 0, 1, 2)
SNIPPET_QUESTION_INDICES = [0, 1, 2]

questions_snip = load_contexts_json(SNIPPET_CONTEXTS_JSON)
_need = max(SNIPPET_QUESTION_INDICES) + 1
if len(questions_snip) < _need:
    raise RuntimeError(
        f"Snippet JSON has {len(questions_snip)} questions; need at least {_need} for indices {SNIPPET_QUESTION_INDICES}"
    )
print("Snippet file:", SNIPPET_CONTEXTS_JSON.name, "| questions in file:", len(questions_snip))
print("Running experiment 2 for indices:", SNIPPET_QUESTION_INDICES)
print("model:", MODEL)


# %%
for qi in SNIPPET_QUESTION_INDICES:
    rec = questions_snip[qi]
    ctxs_i = rec.get("contexts") or []
    n_ctx_i = len(ctxs_i)
    print("\n" + "=" * 72)
    print(f"Experiment 2 | snippet question index={qi} | id={rec.get('id')!r} | type={rec.get('type')!r}")
    print(f"contexts in JSON: {n_ctx_i} | caps: max_contexts={SNIPPET_MAX_CONTEXTS}, max_chars={SNIPPET_MAX_CHARS_PER_CONTEXT}")
    if n_ctx_i > SNIPPET_MAX_CONTEXTS:
        print(
            f"Note: only first {SNIPPET_MAX_CONTEXTS} contexts sent (JSON has {n_ctx_i}). "
            "Raise SNIPPET_MAX_CONTEXTS to include more."
        )
    body_prev = (rec.get("body") or "")[:200]
    print("body (first 200 chars):", body_prev + ("..." if len(rec.get("body") or "") > 200 else ""))

    full_prompt_snip = build_full_prompt_for_record(
        rec,
        PROMPTS_DIR,
        max_contexts=SNIPPET_MAX_CONTEXTS,
        max_chars_per_context=SNIPPET_MAX_CHARS_PER_CONTEXT,
    )
    if not full_prompt_snip:
        print("SKIP: empty prompt")
        continue

    system_snip, user_snip = split_system_user(full_prompt_snip)
    print("combined prompt chars:", len(full_prompt_snip), "| user chars:", len(user_snip))
    print("--- user tail (last 2500 chars) ---")
    print(user_snip[-2500:] if len(user_snip) > 2500 else user_snip)
    if qi == 0:
        print("\n--- FULL combined prompt (index 0 only; very long) ---\n")
        print(full_prompt_snip)
    else:
        print("\n--- (full combined prompt omitted for index > 0; use qi==0 cell pattern if needed) ---")

    raw_snip = call_llm_openai_compat(
        GEN_API_KEY,
        GEN_API_BASE,
        system_snip,
        user_snip,
        MODEL,
        timeout=300,
        temperature=0.0,
        top_p=1.0,
    )
    print("\n--- Raw answer (truncated to 5000 chars) ---")
    if len(raw_snip) > 5000:
        print(raw_snip[:5000] + "\n... [truncated]")
    else:
        print(raw_snip)

    qtype_snip = (rec.get("type") or "").strip().lower()
    try:
        parsed_snip = parse_answer_json_for_type(raw_snip, qtype_snip, q_id=rec.get("id"))
        print("--- Parsed / validated ---")
        print(json.dumps(parsed_snip, indent=2, ensure_ascii=False))
    except Exception as e:
        print("--- Parse / validation FAILED ---")
        print(repr(e))
        print("\nFirst 1200 chars of raw:\n", raw_snip[:1200])


# %% [markdown]
# ## Experiment 3 — Evidence planner (middle stage), **same three questions**
#
# Same indices as experiment 2. The model must **not** answer the question; it returns a **facet plan** JSON (1–3 facets + `unused_evidence`). `evidence_ids` must match the **exact** passage id strings in the evidence block brackets (e.g. `29174931-1`).

# %%
PLANNER_SYSTEM = (
    "You are an evidence planner. Follow every instruction in the user message. "
    "Reply with a single valid JSON object only — no markdown code fences, no commentary."
)

PLANNER_USER_TEMPLATE = """You are an evidence planner.

Do NOT answer the question directly.
Group the evidence into 1 to 5 facets.

Instructions:
- Use the smallest number of facets needed to cover distinct answer-relevant aspects.
- Do not create extra facets just to maximize the count.
- Merge overlapping evidence into one facet.
- Each facet must represent a clearly distinct aspect of the answer.

For each facet, provide:
- facet_id: use "F1", "F2", "F3", ...
- label: a short human-readable facet name
- summary: 1-2 short grounded sentence
- Keep the summary readable and high-level.
- Do NOT try to pack every detail, every number, or every variant into the summary.
- Use supporting_evidence_ids for evidence that directly supports the facet summary.
- Use qualifying_evidence_ids for evidence that limits, qualifies, partially conflicts with, or narrows the facet.
- Use qualifier_note as one short sentence describing the qualification if qualifying_evidence_ids exists
- priority must be either "vital" or "okay".

{{
  "facets": [
    {{
      "facet_id": "F1",
      "label": "string",
      "summary": "string",
      "supporting_evidence_ids": ["E01"],
      "qualifying_evidence_ids": [],
      "qualifier_note": "short note",
      "priority": "vital",
      "exact_answer_candidates": ["string"]
    }}
  ],
  "unused_evidence": [
    {{
      "evidence_id": "passage-id-from-brackets",
      "reason": "peripheral background"
    }}
  ]
}}

Question(type={QTYPE}:
{QUESTION}

Evidence Contexts:
{EVIDENCE_BLOCK}
"""


def validate_planner_json(obj: object) -> list[str]:
    """Return a list of validation warnings (empty if OK)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["root must be a JSON object"]
    facets = obj.get("facets")
    unused = obj.get("unused_evidence")
    if not isinstance(facets, list):
        errs.append("'facets' must be a list")
        return errs
    if len(facets) < 1:
        errs.append("facets must be non-empty (at least one facet)")
    elif len(facets) > 3:
        errs.append(f"expected at most 3 facets per instructions, got {len(facets)}")
    for i, f in enumerate(facets):
        if not isinstance(f, dict):
            errs.append(f"facets[{i}] must be an object")
            continue
        for k in ("facet_id", "label", "summary", "evidence_ids", "evidence_status", "priority"):
            if k not in f:
                errs.append(f"facets[{i}] missing key {k!r}")
        ev = f.get("evidence_ids")
        if ev is not None and not (isinstance(ev, list) and all(isinstance(x, str) for x in ev)):
            errs.append(f"facets[{i}].evidence_ids must be a list of strings")
        st = f.get("evidence_status")
        if st is not None and st not in ("supported", "mixed", "weak"):
            errs.append(f"facets[{i}].evidence_status invalid: {st!r}")
        pr = f.get("priority")
        if pr is not None and pr not in ("vital", "okay"):
            errs.append(f"facets[{i}].priority invalid: {pr!r}")
    if unused is not None:
        if not isinstance(unused, list):
            errs.append("'unused_evidence' must be a list")
        else:
            for j, u in enumerate(unused):
                if not isinstance(u, dict):
                    errs.append(f"unused_evidence[{j}] must be an object")
                    continue
                if "evidence_id" not in u or "reason" not in u:
                    errs.append(f"unused_evidence[{j}] needs evidence_id and reason")
    return errs


# %%
for qi in SNIPPET_QUESTION_INDICES:
    rec = questions_snip[qi]
    ctxs_i = rec.get("contexts") or []
    print("\n" + "=" * 72)
    print(f"Experiment 3 | planner | snippet index={qi} | id={rec.get('id')!r}")

    evidence_block_plan = format_evidence_block(
        ctxs_i,
        SNIPPET_MAX_CONTEXTS,
        SNIPPET_MAX_CHARS_PER_CONTEXT,
    )
    question_plan = (rec.get("body") or "").strip()
    planner_user = PLANNER_USER_TEMPLATE.format(QUESTION=question_plan, EVIDENCE_BLOCK=evidence_block_plan)

    print("planner user chars:", len(planner_user))
    if qi == 0:
        print("\n--- Planner user (first 3500 chars) ---\n")
        print(planner_user[:3500] + ("..." if len(planner_user) > 3500 else ""))
        print("\n--- Planner user (tail 2000 chars) ---\n")
        print(planner_user[-2000:] if len(planner_user) > 2000 else planner_user)
    else:
        print("--- planner user: head 2000 chars ---\n" + planner_user[:2000] + "\n... [tail omitted for brevity] ---")

    raw_plan = call_llm_openai_compat(
        GEN_API_KEY,
        GEN_API_BASE,
        PLANNER_SYSTEM,
        planner_user,
        MODEL,
        timeout=300,
        temperature=0.0,
        top_p=1.0,
    )

    print("\n--- Raw planner output (truncated to 6000 chars) ---")
    if len(raw_plan) > 6000:
        print(raw_plan[:6000] + "\n... [truncated]")
    else:
        print(raw_plan)

    try:
        plan_json_str = extract_first_json_object(raw_plan.strip())
        plan_obj = json.loads(plan_json_str)
    except Exception as e:
        print("--- Failed to extract/parse planner JSON ---")
        print(repr(e))
        print("\nFirst 1500 chars of raw:\n", raw_plan[:1500])
        continue

    warnings = validate_planner_json(plan_obj)
    if warnings:
        print("--- Planner JSON structure warnings ---")
        for w in warnings:
            print(" ", w)
    else:
        print("--- Planner JSON structure: OK (light check) ---")

    print("\n--- Parsed planner JSON ---")
    print(json.dumps(plan_obj, indent=2, ensure_ascii=False))

    known_ids = {str(c.get("id", "")) for c in ctxs_i if c.get("id")}
    for fi, facet in enumerate(plan_obj.get("facets", []) if isinstance(plan_obj, dict) else []):
        if not isinstance(facet, dict):
            continue
        for eid in facet.get("evidence_ids") or []:
            if eid not in known_ids:
                print(f"[warn] facet {fi} references unknown evidence_id: {eid!r}")
    for u in plan_obj.get("unused_evidence", []) if isinstance(plan_obj, dict) else []:
        if isinstance(u, dict) and u.get("evidence_id") not in known_ids:
            print(f"[warn] unused_evidence unknown id: {u.get('evidence_id')!r}")


# %% [markdown]
# ## Experiment 4 — Pipeline answer prompt using **facet JSON** (first snippet question)
#
# 1. Run the **same planner** as experiment 3 on `questions_snip[0]`.
# 2. Keep only `{{"facets": ...}}` — **`unused_evidence` is removed** before generation.
# 3. Load [`system.txt`](../scripts/public/shared_scripts/prompts/system.txt), [`user_base.txt`](../scripts/public/shared_scripts/prompts/user_base.txt), and schema via `resolve_schema_block` (same as [`generate_answers.py`](../scripts/public/shared_scripts/generation/generate_answers.py)).
# 4. Replace the `Evidence Contexts:` line label so the model knows the block is structured JSON, not raw `[id]` passages.
# 5. Call `chat/completions` and parse with `parse_answer_json_for_type`.

# %%
# %%
rec_gen = questions_snip[0]
qtype_gen = (rec_gen.get("type") or "").strip().lower()
question_gen = (rec_gen.get("body") or "").strip()
ctxs_gen = rec_gen.get("contexts") or []

print("Experiment 4 | first snippet question | id=", rec_gen.get("id"), "type=", qtype_gen)

# --- Middle stage (planner) ---
evidence_for_planner = format_evidence_block(
    ctxs_gen,
    SNIPPET_MAX_CONTEXTS,
    SNIPPET_MAX_CHARS_PER_CONTEXT,
)
planner_user_gen = PLANNER_USER_TEMPLATE.format(
    QUESTION=question_gen,
    EVIDENCE_BLOCK=evidence_for_planner,
)
raw_plan_gen = call_llm_openai_compat(
    GEN_API_KEY,
    GEN_API_BASE,
    PLANNER_SYSTEM,
    planner_user_gen,
    MODEL,
    timeout=300,
    temperature=0.0,
    top_p=1.0,
)
plan_gen = json.loads(extract_first_json_object(raw_plan_gen.strip()))
facets_only_gen = {"facets": list(plan_gen.get("facets") or [])}
plan_block_gen = json.dumps(facets_only_gen, ensure_ascii=False, indent=2)
print("--- Facet JSON passed to generation (unused_evidence stripped) ---\n", plan_block_gen[:4000])
if len(plan_block_gen) > 4000:
    print("\n... [truncated; total chars", len(plan_block_gen), "]")

# --- Pipeline prompts: system + schema + user_base ---
system_path_gen = PROMPTS_DIR / "system.txt"
user_base_path_gen = PROMPTS_DIR / "user_base.txt"
system_text_gen = system_path_gen.read_text(encoding="utf-8").strip()
user_base_text_gen = user_base_path_gen.read_text(encoding="utf-8").strip()
schema_block_gen = resolve_schema_block(PROMPTS_DIR / "schemas", rec_gen.get("type") or "")

# Clarify input modality (pipeline system.txt refers to "Evidence Contexts" as passages).
system_text_gen_aug = system_text_gen + (
    "\n\nThe Evidence Contexts section below is a JSON object with a single key `facets` "
    "(structured evidence plan: labels, summaries, evidence_ids per facet). "
    "Raw passage text is not included. Ground your answer only in that JSON."
)

USER_BASE_PLAN_INPUT = user_base_text_gen.replace(
    "Evidence Contexts:",
    "Structured evidence plan (JSON; facets only; unused_evidence omitted from planner output):\n",
)
user_prompt_gen = format_user_prompt(
    USER_BASE_PLAN_INPUT,
    schema_block=schema_block_gen,
    qtype=qtype_gen,
    question=question_gen,
    evidence_block=plan_block_gen,
)

combined_gen = f"[SYSTEM]\n{system_text_gen_aug}\n\n[USER]\n{user_prompt_gen}"
print("\n=== Experiment 4: full combined prompt (system + user) ===\n")
print(combined_gen)


# %%
raw_answer_from_plan = call_llm_openai_compat(
    GEN_API_KEY,
    GEN_API_BASE,
    system_text_gen_aug,
    user_prompt_gen,
    MODEL,
    timeout=300,
    temperature=0.0,
    top_p=1.0,
)
print("--- Raw BioASQ-style answer (truncated to 5000 chars) ---\n")
if len(raw_answer_from_plan) > 5000:
    print(raw_answer_from_plan[:5000] + "\n... [truncated]")
else:
    print(raw_answer_from_plan)


# %%
try:
    parsed_from_plan = parse_answer_json_for_type(
        raw_answer_from_plan, qtype_gen, q_id=rec_gen.get("id")
    )
    print("--- Parsed / validated (experiment 4) ---")
    print(json.dumps(parsed_from_plan, indent=2, ensure_ascii=False))
except Exception as e:
    print("--- Parse / validation FAILED (experiment 4) ---")
    print(repr(e))
    print("\nFirst 1500 chars of raw:\n", raw_answer_from_plan[:1500])

# %%
