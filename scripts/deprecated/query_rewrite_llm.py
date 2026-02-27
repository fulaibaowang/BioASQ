#!/usr/bin/env python3
"""
Deprecated script: LLM query rewrite for BioASQ JSONs.

Reads a list of BioASQ JSON files (each with "questions" and "body" per question),
calls an LLM to produce two rewrites (A: normalize_only, B: normalize_and_enrich),
and writes outputs to a single folder as <stem>_rewrite.json.

Internal experiments with these two rewrite variants, used in downstream reranking
and evaluated with MAP@K, did not show improvements over the baseline queries.
The script is kept here for reference only.

Usage (for reference only):
  python scripts/deprecated/query_rewrite_llm.py --input FILE [FILE ...] [--out_dir DIR]
  Example (all 5 inputs, output to example/query_rewrite/):
  python scripts/deprecated/query_rewrite_llm.py --input example/training14b_10pct_sample.json bioasq_data/Task13BGoldenEnriched/13B1_golden.json bioasq_data/Task13BGoldenEnriched/13B2_golden.json bioasq_data/Task13BGoldenEnriched/13B3_golden.json bioasq_data/Task13BGoldenEnriched/13B4_golden.json

Requires: QUERY_REWRITE_LLM_API_KEY or LLAMA_API_KEY in env or .env at repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import requests

# Repo root (script lives in scripts/private_scripts/)
REPO_ROOT = Path(__file__).resolve().parents[2]

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


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv as _load
        _load(env_path)
    except ImportError:
        # Fallback: read .env manually so it works without python-dotenv
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _strip_ab_prefix(s: str, prefix: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.upper().startswith(prefix.upper()):
        rest = s[len(prefix):].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        return rest
    return s


def parse_rewrites(response: str) -> tuple[str | None, str | None]:
    rewrite_a = None
    rewrite_b = None
    for line in response.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped.upper().startswith("A:") or (
            line_stripped.upper().startswith("A") and ":" in line_stripped[:3]
        ):
            rewrite_a = _strip_ab_prefix(line_stripped, "A")
            if rewrite_a and (
                rewrite_a.strip().upper().startswith("A:")
                or rewrite_a.strip().upper().startswith("A ")
            ):
                rewrite_a = _strip_ab_prefix(rewrite_a, "A")
                warnings.warn("Parser: A value still started with A:; stripped again")
        elif line_stripped.upper().startswith("B:") or (
            line_stripped.upper().startswith("B") and ":" in line_stripped[:3]
        ):
            rewrite_b = _strip_ab_prefix(line_stripped, "B")
            if rewrite_b and (
                rewrite_b.strip().upper().startswith("B:")
                or rewrite_b.strip().upper().startswith("B ")
            ):
                rewrite_b = _strip_ab_prefix(rewrite_b, "B")
                warnings.warn("Parser: B value still started with B:; stripped again")
    return rewrite_a, rewrite_b


def validate_rewrites(questions: list, n_original: int) -> tuple[bool, str | None]:
    if len(questions) != n_original:
        return False, f"question count changed: was {n_original}, now {len(questions)}"
    for i, q in enumerate(questions):
        body = (q.get("body") or "").strip()
        a = q.get("body_rewrite_A")
        b = q.get("body_rewrite_B")
        qid = q.get("id") or q.get("qid") or i
        if body:
            if a is None:
                return False, f"body_rewrite_A is None for question with non-empty body (qid={qid}, index={i})"
            if b is None:
                return False, f"body_rewrite_B is None for question with non-empty body (qid={qid}, index={i})"
        if a is not None:
            a_str = str(a).strip()
            if re.match(r"^A\s*:\s*", a_str, re.IGNORECASE):
                return False, f"body_rewrite_A must not start with 'A:' (found at qid={qid}, index={i})"
        if b is not None:
            b_str = str(b).strip()
            if re.match(r"^B\s*:\s*", b_str, re.IGNORECASE):
                return False, f"body_rewrite_B must not start with 'B:' (found at qid={qid}, index={i})"
    return True, None


def main() -> None:
    _load_dotenv()
    key = os.getenv("QUERY_REWRITE_LLM_API_KEY", os.getenv("LLAMA_API_KEY", "")).strip()
    if not key:
        print(
            "Missing QUERY_REWRITE_LLM_API_KEY or LLAMA_API_KEY. "
            f"Set in env or in {REPO_ROOT / '.env'} (e.g. LLAMA_API_KEY=your_key)",
            file=sys.stderr,
        )
        sys.exit(1)

    URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
    MODEL = "llama3.3:latest"

    ap = argparse.ArgumentParser(description="LLM query rewrite for BioASQ JSONs")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "example" / "query_rewrite",
        help="Output directory; files written as <stem>_rewrite.json",
    )
    ap.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input JSON paths (relative to repo root or absolute)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls",
    )
    args = ap.parse_args()

    input_paths = [p if p.is_absolute() else REPO_ROOT / p for p in args.input]

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = len(input_paths)

    def call_llm(prompt: str, timeout: int = 120) -> str:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "stream": False, "prompt": prompt},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["response"]

    for file_idx, input_path in enumerate(input_paths):
        path = input_path if input_path.is_absolute() else REPO_ROOT / input_path
        if not path.exists():
            print(f"Skipping (not found): {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data["questions"]
        n_original = len(questions)
        stem = path.stem
        output_path = out_dir / f"{stem}_rewrite.json"
        parse_errors = []
        print(f"[{file_idx + 1}/{n_files}] {path.name} ({n_original} questions)")
        for i, q in enumerate(questions):
            body = (q.get("body") or "").strip()
            qid = q.get("id") or q.get("qid") or i
            if not body:
                q["body_rewrite_A"] = ""
                q["body_rewrite_B"] = ""
                continue
            print(f"  [{file_idx + 1}/{n_files}] {stem} ({i + 1}/{n_original}) calling API...")
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
            if args.sleep > 0:
                time.sleep(args.sleep)
        if parse_errors:
            print(f"  Parse/API errors: {len(parse_errors)}", file=sys.stderr)
        ok, err = validate_rewrites(questions, n_original)
        if not ok:
            print(f"Validation failed for {path}: {err}", file=sys.stderr)
            sys.exit(1)
        out = {k: (questions if k == "questions" else data[k]) for k in data}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  Written: {output_path}")


if __name__ == "__main__":
    main()
