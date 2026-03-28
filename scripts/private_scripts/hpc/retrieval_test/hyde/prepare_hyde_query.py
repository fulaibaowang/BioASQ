#!/usr/bin/env python3
"""Flatten hyde.hyde_text into a top-level query field for the retrieval pipeline.

Reads a *.hyde_ready.json file ({"questions": [...]}) and adds a new top-level
field (default ``body_hyde``) to every question:

* If ``hyde.enabled`` is true and ``hyde.hyde_text`` is non-empty, the value
  comes from ``hyde.hyde_text``.
* Otherwise the value falls back to the original ``body``.

The output JSON can then be used with ``--query-field body_hyde`` (or
``DENSE_QUERY_FIELD=body_hyde``) so no other pipeline code needs to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def prepare(questions: list[dict], field_name: str) -> tuple[int, int]:
    """Add *field_name* to each question in-place. Returns (n_hyde, n_fallback)."""
    n_hyde = 0
    n_fallback = 0
    for q in questions:
        hyde = q.get("hyde", {})
        if hyde.get("enabled") and hyde.get("hyde_text"):
            q[field_name] = hyde["hyde_text"]
            n_hyde += 1
        else:
            q[field_name] = q["body"]
            n_fallback += 1
    return n_hyde, n_fallback


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if ".hyde_ready." in name:
        return input_path.with_name(name.replace(".hyde_ready.", ".hyde_query."))
    stem = input_path.stem
    return input_path.with_name(f"{stem}.hyde_query{input_path.suffix}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Flatten hyde.hyde_text into a top-level query field.",
    )
    ap.add_argument("input", type=Path, help="Path to *.hyde_ready.json")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: derive from input name, e.g. *.hyde_query.json)",
    )
    ap.add_argument(
        "--field-name",
        default="body_hyde",
        help="Name of the new top-level field (default: body_hyde)",
    )
    args = ap.parse_args(argv)

    input_path: Path = args.input
    if not input_path.is_file():
        ap.error(f"input file not found: {input_path}")

    output_path: Path = args.output or default_output_path(input_path)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if questions is None:
        ap.error("input JSON has no top-level 'questions' key")

    n_hyde, n_fallback = prepare(questions, args.field_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = n_hyde + n_fallback
    print(
        f"Wrote {output_path}  "
        f"({total} questions: {n_hyde} HyDE, {n_fallback} fallback to body)"
    )


if __name__ == "__main__":
    main()
