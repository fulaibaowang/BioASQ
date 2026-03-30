#!/usr/bin/env python3
"""Flatten HyDE/facet/listwise fields into top-level query fields for retrieval.

Reads a *.hyde_ready.json file ({"questions": [...]}) and adds a new top-level
field (default ``body_hyde``) to every question.

This repo used to store HyDE metadata under ``hyde``. Newer datasets store it
under ``query_parse`` with keys like ``hyde_enabled`` and ``hyde_text``.
This script supports both shapes, preferring ``query_parse`` when present.

By default it writes:
- ``body_hyde``: HyDE text when enabled and non-empty; else fallback to ``body``
  (or ``null`` with ``--no-fallback``).
- ``body_facet1``, ``body_facet2``, ...: one field per facet in
  ``query_parse.facets`` (or ``[]``).
- ``body_listwise``: the original body plus an optional "coverage hints" block
  based on ``query_parse.listwise_bullets``.

With ``--no-fallback``, queries where HyDE did not produce a different text
get ``null``.  Combined with ``DENSE_QUERY_FIELD=body,body_hyde`` and
``--skip-empty-query-field`` in the pipeline, this avoids re-running
identical queries through retrieval/reranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _get_query_parse(q: dict) -> dict:
    qp = q.get("query_parse")
    if isinstance(qp, dict):
        return qp
    # Back-compat: map old `hyde` shape into a partial query_parse-like view.
    hyde = q.get("hyde")
    if isinstance(hyde, dict):
        return {
            "hyde_enabled": bool(hyde.get("enabled")),
            "hyde_text": hyde.get("hyde_text") or "",
            "facets": [],
            "listwise_bullets": [],
        }
    return {"hyde_enabled": False, "hyde_text": "", "facets": [], "listwise_bullets": []}


def _mk_body_listwise(body: str, bullets: list[str]) -> str:
    bullets = [b.strip() for b in bullets if isinstance(b, str) and b.strip()]
    if not bullets:
        return body
    hints = "\n".join(f"- {b}" for b in bullets)
    return (
        f"{body}\n"
        f"Optional coverage hints:\n"
        f"{hints}\n"
        f"Treat the coverage hints as soft guidance, not hard requirements."
    )


def prepare(
    questions: list[dict],
    field_name: str,
    no_fallback: bool = False,
    include_facets: bool = True,
    include_listwise: bool = True,
) -> tuple[int, int]:
    """Add top-level query fields in-place. Returns (n_hyde, n_fallback)."""
    n_hyde = 0
    n_fallback = 0
    for q in questions:
        body = q.get("body")
        if not isinstance(body, str):
            raise ValueError("each question must have string field 'body'")

        qp = _get_query_parse(q)
        hyde_enabled = bool(qp.get("hyde_enabled"))
        hyde_text = qp.get("hyde_text")
        if isinstance(hyde_text, str) and hyde_text.strip() and hyde_enabled:
            q[field_name] = hyde_text
            n_hyde += 1
        else:
            q[field_name] = None if no_fallback else body
            n_fallback += 1

        if include_facets:
            facets = qp.get("facets") or []
            if not isinstance(facets, list):
                facets = []
            for idx, facet in enumerate(facets, start=1):
                if isinstance(facet, str) and facet.strip():
                    # Facets are already self-contained queries.
                    q[f"body_facet{idx}"] = facet.strip()
                else:
                    q[f"body_facet{idx}"] = None

        if include_listwise:
            bullets = qp.get("listwise_bullets") or []
            if not isinstance(bullets, list):
                bullets = []
            q["body_listwise"] = _mk_body_listwise(body, bullets)
    return n_hyde, n_fallback


def default_output_path(input_path: Path) -> Path:
    """Strip '.hyde_ready' from the filename so the stem matches the original
    data file (important: BM25/Dense/Hybrid all key run files off the stem)."""
    name = input_path.name
    if ".hyde_ready." in name:
        return input_path.with_name(name.replace(".hyde_ready.", "."))
    return input_path


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
        help="Output path (default: strip .hyde_ready from input name to preserve stem)",
    )
    ap.add_argument(
        "--field-name",
        default="body_hyde",
        help="Name of the new top-level field (default: body_hyde)",
    )
    ap.add_argument(
        "--no-facets",
        action="store_true",
        help="Do not generate body_facet1, body_facet2, ... fields.",
    )
    ap.add_argument(
        "--no-listwise",
        action="store_true",
        help="Do not generate body_listwise.",
    )
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        help="Set field to null (instead of body) when HyDE text is absent. "
        "Use with DENSE_QUERY_FIELD=body,body_hyde to skip duplicate queries.",
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

    n_hyde, n_fallback = prepare(
        questions,
        args.field_name,
        no_fallback=args.no_fallback,
        include_facets=not args.no_facets,
        include_listwise=not args.no_listwise,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = n_hyde + n_fallback
    fallback_label = "null (no-fallback)" if args.no_fallback else "fallback to body"
    print(
        f"Wrote {output_path}  "
        f"({total} questions: {n_hyde} HyDE, {n_fallback} {fallback_label})"
    )


if __name__ == "__main__":
    main()
