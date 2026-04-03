#!/usr/bin/env python3
"""
Strip BioASQ question JSON down to id, body, type, and snippet texts only.

Removes per-question `documents` and per-snippet metadata (beginSection,
endSection, document, offsets). Each snippet becomes the string in `text`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _snippet_text(snippet: object) -> str:
    if isinstance(snippet, str):
        return snippet
    if isinstance(snippet, dict):
        t = snippet.get("text")
        return t if isinstance(t, str) else ""
    return ""


def simplify_questions(data: dict) -> dict:
    raw = data.get("questions")
    if not isinstance(raw, list):
        raise ValueError("Input JSON must have a 'questions' array")

    out_questions = []
    for q in raw:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        body = q.get("body")
        qtype = q.get("type")
        snippets_in = q.get("snippets", [])
        if not isinstance(snippets_in, list):
            snippets_in = []
        texts = [_snippet_text(s) for s in snippets_in]
        out_questions.append(
            {
                "id": qid,
                "body": body,
                "type": qtype,
                "snippets": texts,
            }
        )
    return {"questions": out_questions}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        help="BioASQ JSON (e.g. BioASQ-task14bPhaseB-testset2)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here; default: stdout",
    )
    args = p.parse_args()

    path_in = args.input.expanduser().resolve()
    if not path_in.is_file():
        print(f"Not a file: {path_in}", file=sys.stderr)
        return 1

    with open(path_in, encoding="utf-8") as f:
        data = json.load(f)

    simplified = simplify_questions(data)

    out_s = json.dumps(simplified, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        try:
            sys.stdout.write(out_s)
        except BrokenPipeError:
            try:
                sys.stdout.close()
            except Exception:
                pass
            return 0
    else:
        path_out = args.output.expanduser().resolve()
        path_out.parent.mkdir(parents=True, exist_ok=True)
        path_out.write_text(out_s, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
