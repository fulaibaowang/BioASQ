#!/usr/bin/env python3
"""Adapt-in: BioASQ wrapped JSON {"questions":[...]} -> one JSONL line per question (lossless)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def question_to_line(q: dict) -> dict:
    qid = q.get("id")
    body = q.get("body")
    qtype = q.get("type")
    return {
        "query_id": qid,
        "query_text": body,
        "query_type": qtype,
        "bioasq": q,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="BioASQ JSON file (wrapped questions).")
    p.add_argument("--output", type=Path, required=True, help="Output .jsonl path.")
    args = p.parse_args()
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        print(f"Not a file: {inp}", file=sys.stderr)
        return 1
    if out.suffix.lower() != ".jsonl":
        print("Error: --output must end with .jsonl", file=sys.stderr)
        return 1
    data = json.loads(inp.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list):
        print("Input must be JSON with a 'questions' array.", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in questions:
            if not isinstance(q, dict):
                continue
            f.write(json.dumps(question_to_line(q), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
