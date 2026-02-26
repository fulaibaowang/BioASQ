#!/usr/bin/env python3
"""
Clean BioASQ golden JSON files to retain only body, type, and id per question.
Removes documents, snippets, ideal_answer, and any other fields.
"""

import argparse
import json
import sys
from pathlib import Path

# Fields to keep per question
KEEP_KEYS = {"body", "type", "id"}


def clean_question(q: dict) -> dict:
    """Return a new dict with only body, type, id (if present)."""
    return {k: q[k] for k in KEEP_KEYS if k in q}


def clean_golden_json(data: dict) -> dict:
    """Clean the golden JSON: keep only body, type, id for each question."""
    if "questions" not in data:
        raise ValueError("Expected top-level key 'questions'")
    return {
        "questions": [clean_question(q) for q in data["questions"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean BioASQ golden JSON to keep only body, type, and id per question."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file (e.g. 13B1_golden.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON file. Default: input_cleaned.json next to input",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with cleaned JSON",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.in_place and args.output is not None:
        print("Error: cannot use both -o/--output and --in-place", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        out_path = args.input
    elif args.output is not None:
        out_path = args.output
    else:
        stem = args.input.stem
        out_path = args.input.parent / f"{stem}_cleaned.json"

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = clean_golden_json(data)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(cleaned['questions'])} questions to {out_path}")


if __name__ == "__main__":
    main()
