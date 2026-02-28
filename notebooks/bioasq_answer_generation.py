# ---
# jupyter:
#   jupytext:
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

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # tqdm is optional
    tqdm = None

# Assume this notebook lives in repo_root/notebook/
REPO_ROOT = Path("..").resolve()

INPUT_JSON = REPO_ROOT / "output" / "workflow_local_10pct_hpc_bge" / "evidence" / "13B1_golden_contexts.json"
OUTPUT_JSON = REPO_ROOT / "output" / "workflow_local_10pct_hpc_bge" / "generation" / "13B1_golden_answers.json"

PROMPTS_DIR = REPO_ROOT / "scripts" / "public" / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.txt"
USER_BASE_PROMPT_PATH = PROMPTS_DIR / "user_base.txt"
SCHEMAS_DIR = PROMPTS_DIR / "schemas"

# Evidence block limits (used by format_evidence_block)
MAX_CONTEXTS = 10
MAX_CHARS_PER_CONTEXT = 5000

REPO_ROOT, INPUT_JSON, OUTPUT_JSON, PROMPTS_DIR, SCHEMAS_DIR


# %%
from dotenv import load_dotenv

# Load .env from repo root if present
env_path = REPO_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_KEY = (os.getenv("QUERY_REWRITE_LLM_API_KEY") or os.getenv("LLAMA_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("Missing QUERY_REWRITE_LLM_API_KEY or LLAMA_API_KEY in environment or .env")

OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
OLLAMA_MODEL = "llama3.3:latest"

def call_llm(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """Call the Ollama endpoint and return the raw text response."""
    import requests

    prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    r = requests.post(
        OLLAMA_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": OLLAMA_MODEL, "stream": False, "prompt": prompt},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")



# %%
# Load prompts from disk
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_text = f.read().strip()

with open(USER_BASE_PROMPT_PATH, "r", encoding="utf-8") as f:
    user_base_text = f.read().strip()

system_text[:200], "...", user_base_text[:200]


# %%
# Load and cache schema blocks per question type (fallback to summary.txt)
SCHEMA_BLOCKS: Dict[str, str] = {}

def get_schema_block(qtype: str) -> str:
    """Load schema for qtype from schemas/{qtype}.txt, fallback to summary.txt."""
    qtype = (qtype or "summary").strip().lower()
    if qtype in SCHEMA_BLOCKS:
        return SCHEMA_BLOCKS[qtype]
    path = SCHEMAS_DIR / f"{qtype}.txt"
    if not path.exists():
        path = SCHEMAS_DIR / "summary.txt"
    with open(path, "r", encoding="utf-8") as f:
        block = f.read().strip()
    SCHEMA_BLOCKS[qtype] = block
    return block

# Preload known types so we see any missing-file errors early
for _q in ("summary", "yesno", "factoid", "list"):
    get_schema_block(_q)
list(SCHEMA_BLOCKS.keys())


# %%
def load_contexts_json(path: Path) -> List[Dict[str, Any]]:
    """Load contexts from JSON: expects {\"questions\": [...]} or a top-level list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError("Input JSON must be a list or an object with 'questions' key")

# Inspect a few input records to confirm schema
_all_input = load_contexts_json(INPUT_JSON)
preview = _all_input[:3]

preview[0]


# %%
def format_evidence_block(
    contexts: List[Dict[str, Any]],
    max_contexts: Optional[int] = None,
    max_chars_per_context: Optional[int] = None,
) -> str:
    """Format contexts into a readable block while preserving IDs.

    Each context is expected to have at least `id` and `text` fields,
    and optionally `doc` (e.g. PubMed URL).
    Caps the number of contexts and truncates each context text using
    MAX_CONTEXTS and MAX_CHARS_PER_CONTEXT when not passed.
    """
    cap = max_contexts if max_contexts is not None else MAX_CONTEXTS
    max_chars = max_chars_per_context if max_chars_per_context is not None else MAX_CHARS_PER_CONTEXT
    lines: List[str] = []
    for ctx in contexts[:cap]:
        cid = str(ctx.get("id", ""))
        text = str(ctx.get("text", "")).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        doc = str(ctx.get("doc", "")).strip()
        header_parts = [cid]
        if doc:
            header_parts.append(doc)
        header = " | ".join(header_parts) if header_parts else "(no id)"
        block = f"[{header}]\n{text}" if text else f"[{header}]"
        lines.append(block)
    return "\n\n".join(lines)

def fill_user_prompt(question: str, evidence_block: str, qtype: str, schema_block: str) -> str:
    """Fill the user_base template with SCHEMA_BLOCK, QTYPE, QUESTION, EVIDENCE_BLOCK."""
    return (
        user_base_text
        .replace("{SCHEMA_BLOCK}", schema_block)
        .replace("{QTYPE}", qtype)
        .replace("{QUESTION}", question)
        .replace("{EVIDENCE_BLOCK}", evidence_block)
    )



# %%
def extract_first_json_object(raw: str) -> str:
    """Extract the first {...} JSON object from the response, ignoring surrounding text."""
    raw = raw.strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found in response; cannot parse JSON")
    depth = 0
    in_string = False
    escape = False
    quote_char = '"'
    i = start
    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if c == quote_char and not escape:
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
        i += 1
    raise ValueError("No matching '}' for first '{'; incomplete JSON object")

def parse_answer_json_for_type(raw: str, qtype: str, q_id: Optional[str] = None) -> Dict[str, Any]:
    """Parse and validate the model JSON output per question type.

    All types: ideal_answer (string), evidence_ids (list of strings).
    yesno: exact_answer in {"yes", "no"}.
    factoid/list: exact_answer is a list of strings.
    summary: no exact_answer required.
    Uses extract_first_json_object so extra text before/after the JSON is ignored.
    """
    raw_stripped = raw.strip()
    if not raw_stripped:
        raise ValueError("Empty response; cannot parse JSON")
    json_str = extract_first_json_object(raw_stripped)
    obj = json.loads(json_str)
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object")

    if "ideal_answer" not in obj or "evidence_ids" not in obj:
        raise ValueError("Model output must contain 'ideal_answer' and 'evidence_ids' keys")

    ideal = obj["ideal_answer"]
    ev_ids = obj["evidence_ids"]

    if not isinstance(ideal, str):
        raise ValueError("'ideal_answer' must be a string")
    if not isinstance(ev_ids, list) or not all(isinstance(x, str) for x in ev_ids):
        raise ValueError("'evidence_ids' must be a list of strings")

    qtype = (qtype or "summary").strip().lower()
    out: Dict[str, Any] = {"ideal_answer": ideal, "evidence_ids": ev_ids}

    if qtype == "yesno":
        if "exact_answer" not in obj:
            raise ValueError("yesno type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, str):
            raise ValueError("yesno exact_answer must be a string")
        if ea.strip().lower() not in ("yes", "no"):
            raise ValueError(f"yesno exact_answer must be 'yes' or 'no', got: {ea!r}")
        out["exact_answer"] = ea.strip().lower()
    elif qtype in ("factoid", "list"):
        if "exact_answer" not in obj:
            raise ValueError(f"{qtype} type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, list):
            raise ValueError(f"{qtype} exact_answer must be a list of strings")
        if not all(isinstance(x, str) for x in ea):
            raise ValueError(f"{qtype} exact_answer list must contain only strings")
        if qtype == "factoid" and len(ea) > 5:
            raise ValueError("factoid exact_answer must have 0-5 items")
        out["exact_answer"] = ea
    # summary or unknown: do not add exact_answer key at all

    return out



# %%
# Single-example trace: one question, full visibility (schema, prompt, raw, parsed, validation)
TRACE_INDEX = 1  # which record in the JSON to run (0 = first)

_trace_objs = load_contexts_json(INPUT_JSON)
if TRACE_INDEX >= len(_trace_objs):
    raise RuntimeError(f"Index {TRACE_INDEX} out of range (len={len(_trace_objs)}) in {INPUT_JSON}")
sample = _trace_objs[TRACE_INDEX]

q_id = sample.get("id")
qtype = sample.get("type", "summary")
question = sample.get("body", "")
contexts = sample.get("contexts", [])

print("--- Input ---")
print("q_id:", q_id)
print("qtype:", qtype)
print("body:", question[:200] + ("..." if len(question) > 200 else ""))

schema_block = get_schema_block(qtype)
schema_file = f"{qtype}.txt" if (SCHEMAS_DIR / f"{qtype}.txt").exists() else "summary.txt (fallback)"
print("\n--- Schema ---")
print("resolved file:", schema_file)
print("schema_block (first 300 chars):", repr(schema_block[:300]) + ("..." if len(schema_block) > 300 else ""))

evidence_block = format_evidence_block(contexts)
print("\n--- Evidence (first 400 chars) ---")
print(evidence_block[:400] + ("..." if len(evidence_block) > 400 else ""))

user_prompt = fill_user_prompt(question, evidence_block, qtype, schema_block)
print("\n--- User prompt (first 500 chars) ---")
print(user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""))

full_prompt = f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_prompt}"
print("\n=== FULL PROMPT FOR LLaMA (copy/paste) ===\n")
print(full_prompt)

raw_response = call_llm(system_text, user_prompt)
print("\n--- Raw LLM response ---")
print(raw_response[:800] + ("..." if len(raw_response) > 800 else ""))

parsed = parse_answer_json_for_type(raw_response, qtype, q_id=q_id)
print("\n--- Parsed (validated) ---")
print(json.dumps(parsed, indent=2))
print("\n--- Validation passed for type:", qtype)

# %%
# Batch processing loop (limited to first 5 questions for testing) with concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

def write_answers_json(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def process_one(idx: int, obj: Dict[str, Any], total: int) -> tuple[int, Dict[str, Any]]:
    """Process a single question; returns (idx, record) for ordering."""
    q_id = obj.get("id")
    qtype = obj.get("type", "summary")
    question = obj.get("body", "") or ""
    contexts = obj.get("contexts", []) or []

    if not question or not contexts:
        rec = {"id": q_id, "body": question, "type": qtype, "ideal_answer": None, "evidence_ids": [], "error": "missing_question_or_contexts"}
        if qtype in ("yesno", "factoid", "list"):
            rec["exact_answer"] = None
        return idx, rec

    raw = None
    try:
        schema_block = get_schema_block(qtype)
        evidence_block = format_evidence_block(contexts)
        user_prompt = fill_user_prompt(question, evidence_block, qtype, schema_block)
        raw = call_llm(system_text, user_prompt)
        parsed = parse_answer_json_for_type(raw, qtype, q_id=q_id)
        rec = {"id": q_id, "body": question, "type": qtype, "ideal_answer": parsed["ideal_answer"], "evidence_ids": parsed["evidence_ids"]}
        if qtype in ("yesno", "factoid", "list"):
            rec["exact_answer"] = parsed.get("exact_answer")
    except Exception as e:
        print(f"[DEBUG] Parse failed for id={q_id} type={qtype}: {e}")
        print(f"[DEBUG] Raw response (first 600 chars): {repr(raw[:600]) if raw else '(empty)'}")
        rec = {"id": q_id, "body": question, "type": qtype, "ideal_answer": None, "evidence_ids": [], "error": str(e)}
        if qtype in ("yesno", "factoid", "list"):
            rec["exact_answer"] = None
    return idx, rec

MAX_EXAMPLES = 5
CONCURRENCY = 2  # tune based on your API (2–4 recommended from parallel test)
all_objs = load_contexts_json(INPUT_JSON)
subset = all_objs[:MAX_EXAMPLES]
total = len(subset)
if total == 0:
    raise RuntimeError("No questions found in input JSON; aborting.")

results_by_idx: Dict[int, Dict[str, Any]] = {}
with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    futs = {ex.submit(process_one, idx, obj, total): idx for idx, obj in enumerate(subset, start=1)}
    completed = as_completed(futs)
    if tqdm is not None:
        completed = tqdm(completed, total=total, desc="Processing")
    for fut in completed:
        idx, rec = fut.result()
        results_by_idx[idx] = rec
        if tqdm is None:
            print(f"Completed {idx}/{total} (id={rec.get('id')}, type={rec.get('type')})")

records_out = [results_by_idx[i] for i in range(1, total + 1)]

successful = [
    r for r in records_out
    if isinstance(r.get("ideal_answer"), str) and r["ideal_answer"].strip()
]
if not successful:
    raise RuntimeError(
        f"No successful non-empty ideal_answer out of {total} questions; not writing {OUTPUT_JSON}."
    )

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
write_answers_json(OUTPUT_JSON, records_out)
print(f"Wrote {len(records_out)} records (successful={len(successful)}) to {OUTPUT_JSON}")
OUTPUT_JSON, len(records_out)

# %%
