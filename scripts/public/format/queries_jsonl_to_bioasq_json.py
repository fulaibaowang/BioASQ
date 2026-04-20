#!/usr/bin/env python3
"""Adapt-out: queries .jsonl -> BioASQ wrapped JSON {"questions":[...]}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def line_to_question(rec: Dict[str, Any]) -> dict:
    if isinstance(rec.get("bioasq"), dict):
        return dict(rec["bioasq"])
    qid = rec.get("query_id", rec.get("id"))
    body = rec.get("query_text", rec.get("body"))
    qtype = rec.get("query_type", rec.get("type"))
    out: Dict[str, Any] = {k: v for k, v in rec.items() if k not in ("query_id", "query_text", "query_type", "bioasq")}
    if qid is not None:
        out["id"] = qid
    if body is not None:
        out["body"] = body
    if qtype is not None:
        out["type"] = qtype
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Queries .jsonl")
    p.add_argument("--output", type=Path, required=True, help="BioASQ JSON output path.")
    p.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    args = p.parse_args()
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        print(f"Not a file: {inp}", file=sys.stderr)
        return 1
    if inp.suffix.lower() != ".jsonl":
        print("Error: --input must be .jsonl", file=sys.stderr)
        return 1
    questions = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                print(f"Skip non-object line: {line[:80]!r}", file=sys.stderr)
                continue
            questions.append(line_to_question(rec))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"questions": questions}
    if args.pretty:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        text = json.dumps(payload, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(questions)} questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
