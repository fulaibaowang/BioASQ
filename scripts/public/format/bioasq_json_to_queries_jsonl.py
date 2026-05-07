#!/usr/bin/env python3
"""Adapt-in: BioASQ wrapped JSON {"questions":[...]} -> JSONL (query_id, query_text; optional query_type, documents, snippets).

Passthrough: when a question includes ``query_parse`` (LLM normalization / HyDE metadata) or legacy ``hyde``,
those objects are copied onto each JSONL line.

When ``query_parse`` is complete (non-empty ``parsed_query``), the same rules as
``scripts/public/query_parsing/prepare_query.py`` are applied in-process so each line also gets
``query_text_normalized`` and ``query_text_hyde`` for ``--dense-query-field query_text,query_text_hyde``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_PUBLIC = Path(__file__).resolve().parent.parent
if str(_SCRIPT_PUBLIC) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PUBLIC))

from query_parsing.prepare_query import prepare


def question_to_line(q: dict) -> dict:
    qid = q.get("id")
    body = q.get("body")
    # Strict pipeline query JSONL: required id + text; optional BioASQ ``type`` / ``documents`` / ``snippets``.
    out: dict = {
        "query_id": qid,
        "query_text": body,
    }
    qt = q.get("type")
    if qt is not None and str(qt).strip():
        out["query_type"] = str(qt).strip()
    docs = q.get("documents")
    if isinstance(docs, list) and docs:
        out["documents"] = list(docs)
    snips = q.get("snippets")
    if isinstance(snips, list) and snips:
        copied = [dict(s) for s in snips if isinstance(s, dict)]
        if copied:
            out["snippets"] = copied
    qp = q.get("query_parse")
    if isinstance(qp, dict):
        out["query_parse"] = dict(qp)
        q_work = dict(q)
        try:
            prepare([q_work], "query_text_hyde", "query_text_normalized", no_fallback=False)
        except ValueError:
            pass
        else:
            if "query_text_normalized" in q_work:
                out["query_text_normalized"] = q_work["query_text_normalized"]
            if "query_text_hyde" in q_work:
                out["query_text_hyde"] = q_work["query_text_hyde"]
    hyde = q.get("hyde")
    if isinstance(hyde, dict):
        out["hyde"] = dict(hyde)
    return out


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
