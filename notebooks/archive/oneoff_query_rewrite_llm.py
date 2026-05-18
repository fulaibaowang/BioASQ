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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BioASQ query rewrite via LLM
#
# Read a BioASQ golden JSON, send each question **body** to an LLM with a strict two-rewrite prompt, parse "A:" and "B:" lines, then write a new JSON with `body_rewrite_A` and `body_rewrite_B` per question. Python handles all JSON; the LLM only returns the two lines.

# %% [markdown]
# ## 1. Imports and config

# %%
import os
import re
import json
import time
from pathlib import Path

import requests

# Load .env from repo root so LLAMA_API_KEY is set in Jupyter (Cursor/shell often already have it)
try:
    from dotenv import load_dotenv
    _b = Path(".").resolve()
    if not (_b / "bioasq_data").exists():
        _b = _b.parent
    load_dotenv(_b / ".env")
except ImportError:
    pass

# --- API (same pattern as dictycite goldset_llm_labeling) ---
key = os.getenv("QUERY_REWRITE_LLM_API_KEY", os.getenv("LLAMA_API_KEY", "")).strip()
if not key:
    raise ValueError("Missing QUERY_REWRITE_LLM_API_KEY or LLAMA_API_KEY (set in env or .env)")
URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
MODEL = "llama3.3:latest"

# --- Paths ---
base_dir = Path(".").resolve()
if not (base_dir / "bioasq_data").exists():
    base_dir = base_dir.parent  # run from notebooks/
    assert (base_dir / "bioasq_data").exists(), "Run from repo root or notebooks/"

# List of input JSONs; process one file after another. Output = same dir as input, stem_rewrites.json
INPUT_JSONS = [
    base_dir / "example" / "training14b_10pct_sample.json",
    base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B1_golden.json",
    base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B2_golden.json",
    base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B3_golden.json",
    base_dir / "bioasq_data" / "Task13BGoldenEnriched" / "13B4_golden.json",
]

# Optional: delay between API calls (seconds)
SLEEP_BETWEEN_CALLS = 0.5

# %% [markdown]
# ## 2. Prompt template

# %%
PROMPT_TEMPLATE = """You are rewriting biomedical search queries for retrieval and reranking evaluation.

You must generate TWO rewrites for EACH input query:
(A) normalize_only: conservative normalization
(B) normalize_and_enrich: normalization + minimal generic template enrichment

Global rules (apply to both A and B):
1) Preserve the original intent exactly.
2) Do NOT answer the query.
3) Do NOT add new facts, constraints, or assumptions.
4) Do NOT change biomedical entities (drug names, gene names, proteins, diseases, acronyms) unless fixing obvious spacing/punctuation issues.
   - If unsure whether a token is an entity, preserve it exactly.
5) Fix obvious typos in common English/medical words (e.g., chromsome→chromosome, proceedure→procedure, associaated→associated, sumarize→summarize).
6) Keep English.
7) Keep rewrites concise (roughly similar length to original).

Strategy A: normalize_only
- Make the query grammatical and clear.
- Convert telegraphic phrases into a natural question if needed.
- Only minimal edits; do not add extra wording unless required for grammar.

Strategy B: normalize_and_enrich
- Start from A, then optionally add *generic* retrieval-friendly wording that does NOT add facts:
  - For "target" queries: you MAY use "molecular target" or "binds to" phrasing (no specific target added).
  - For "mechanism of action" queries: you MAY add "(MOA)" or "how it works" phrasing (no mechanism added).
  - For list/symptom/complication/presentation: you MAY add "clinical", "complications", "manifestations", "symptoms" if already implied.
  - For summary queries: you MAY normalize to "Please summarize ..." or "What are the key points of ..."
- Do NOT append synonym lists.
- Do NOT expand acronyms unless the expansion appears in the original query.

OUTPUT FORMAT (very strict):
- Output exactly TWO lines.
- Line 1 must start with "A: " and then the rewritten query.
- Line 2 must start with "B: " and then the rewritten query.
- No extra lines, no bullets, no quotes, no JSON.

Now process this query:
{{QUERY}}"""


# %% [markdown]
# ## 3. API call and parser

# %%
def call_llm(prompt: str, timeout: int = 120) -> str:
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "stream": False, "prompt": prompt},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["response"]


def _strip_ab_prefix(s: str, prefix: str) -> str:
    """Return content after prefix (e.g. 'A:' or 'B:'), with flexible space after colon."""
    s = s.strip()
    if not s:
        return s
    # Match "A:" or "A :" (case-insensitive)
    if s.upper().startswith(prefix.upper()):
        rest = s[len(prefix):].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        return rest
    return s


def parse_rewrites(response: str):
    """Extract (rewrite_a, rewrite_b) from LLM response. Stored values must NOT start with 'A:' or 'B:'."""
    rewrite_a = None
    rewrite_b = None
    for line in response.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped.upper().startswith("A:") or (line_stripped.upper().startswith("A") and ":" in line_stripped[:3]):
            # Take everything after "A" and optional ":" and spaces
            rewrite_a = _strip_ab_prefix(line_stripped, "A")
            if rewrite_a and (rewrite_a.strip().upper().startswith("A:") or rewrite_a.strip().upper().startswith("A ")):
                rewrite_a = _strip_ab_prefix(rewrite_a, "A")
                import warnings
                warnings.warn("Parser: A value still started with A:; stripped again")
        elif line_stripped.upper().startswith("B:") or (line_stripped.upper().startswith("B") and ":" in line_stripped[:3]):
            rewrite_b = _strip_ab_prefix(line_stripped, "B")
            if rewrite_b and (rewrite_b.strip().upper().startswith("B:") or rewrite_b.strip().upper().startswith("B ")):
                rewrite_b = _strip_ab_prefix(rewrite_b, "B")
                import warnings
                warnings.warn("Parser: B value still started with B:; stripped again")
    return rewrite_a, rewrite_b


# %% [markdown]
# ## 4. Validation before write

# %%
def validate_rewrites(questions: list, n_original: int) -> tuple[bool, str | None]:
    """Run checks. Returns (True, None) or (False, error_message). Do not write if False."""
    if len(questions) != n_original:
        return False, f"question count changed: was {n_original}, now {len(questions)}"
    for i, q in enumerate(questions):
        body = (q.get("body") or "").strip()
        a = q.get("body_rewrite_A")
        b = q.get("body_rewrite_B")
        qid = q.get("id") or q.get("qid") or i
        # Presence: every question with non-empty body must have both rewrites set (non-None)
        if body:
            if a is None:
                return False, f"body_rewrite_A is None for question with non-empty body (qid={qid}, index={i})"
            if b is None:
                return False, f"body_rewrite_B is None for question with non-empty body (qid={qid}, index={i})"
        # Prefix check: stored values must NOT start with "A:" or "B:"
        if a is not None:
            a_str = str(a).strip()
            if re.match(r"^A\s*:\s*", a_str, re.IGNORECASE):
                return False, f"body_rewrite_A must not start with 'A:' (found at qid={qid}, index={i})"
        if b is not None:
            b_str = str(b).strip()
            if re.match(r"^B\s*:\s*", b_str, re.IGNORECASE):
                return False, f"body_rewrite_B must not start with 'B:' (found at qid={qid}, index={i})"
    return True, None


# %% [markdown]
# ## 5. Load, run loop, validate, write

# %%
n_files = len(INPUT_JSONS)
for file_idx, input_path in enumerate(INPUT_JSONS):
    if not input_path.exists():
        print(f"Skipping (not found): {input_path}")
        continue
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    n_original = len(questions)
    stem = input_path.stem
    output_path = input_path.parent / f"{stem}_rewrites.json"
    parse_errors = []
    print(f"[{file_idx+1}/{n_files}] {input_path.name} ({n_original} questions)")
    for i, q in enumerate(questions):
        body = (q.get("body") or "").strip()
        qid = q.get("id") or q.get("qid") or i
        if not body:
            q["body_rewrite_A"] = ""
            q["body_rewrite_B"] = ""
            continue
        print(f"  [{file_idx+1}/{n_files}] {stem} ({i+1}/{n_original}) calling API...")
        prompt = PROMPT_TEMPLATE.replace("{{QUERY}}", body)
        try:
            raw = call_llm(prompt)
            rewrite_a, rewrite_b = parse_rewrites(raw)
            if rewrite_a is None or rewrite_b is None:
                parse_errors.append((qid, raw[:500] if raw else ""))
                q["body_rewrite_A"] = body
                q["body_rewrite_B"] = body
            else:
                q["body_rewrite_A"] = rewrite_a
                q["body_rewrite_B"] = rewrite_b
        except Exception as e:
            parse_errors.append((qid, str(e)))
            q["body_rewrite_A"] = body
            q["body_rewrite_B"] = body
        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)
    if parse_errors:
        print(f"  Parse/API errors: {len(parse_errors)}")
    ok, err = validate_rewrites(questions, n_original)
    if not ok:
        raise ValueError(f"Validation failed for {input_path}: {err}")
    out = {k: (questions if k == "questions" else data[k]) for k in data}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  Written: {output_path}")

# %% [markdown]
# ## 6. Dry run (optional)
#
# Run on first 2–3 questions to verify prompt and parse without writing.

# %%
# Dry run: first file, first 2 questions only
dry_path = INPUT_JSONS[0] if INPUT_JSONS else None
if not dry_path or not dry_path.exists():
    print("No input file found for dry run")
else:
    with open(dry_path, "r", encoding="utf-8") as f:
        dry_data = json.load(f)
    dry_questions = dry_data["questions"][:5]
    for i, q in enumerate(dry_questions):
        body = (q.get("body") or "").strip()
        if not body:
            print("(empty body)")
            continue
        prompt = PROMPT_TEMPLATE.replace("{{QUERY}}", body)
        print("--- Prompt (last 400 chars) ---")
        print(prompt[-400:])
        print("---")
        raw = call_llm(prompt)
        print("--- Raw response ---")
        print(raw)
        print()

# %%
