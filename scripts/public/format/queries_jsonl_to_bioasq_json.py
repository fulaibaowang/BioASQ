#!/usr/bin/env python3
"""Adapt-out: queries .jsonl -> BioASQ wrapped JSON {"questions":[...]}.

Generation JSONL may include ``contexts`` and long ``doc_ids`` lists; by default this
script omits pipeline-only fields (not ``evidence_ids``), and caps ``documents`` to the
first N ids (10). Use ``--max-documents 0`` to keep the full list.

Lines that look like ``generate_answers.py`` output with an ``error`` field or missing
answers emit a **warning to stderr** (answers are still converted, often as null)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PUBMED_DOCUMENT_PREFIX = "http://www.ncbi.nlm.nih.gov/pubmed/"

# Keys not part of BioASQ submission JSON (pipeline / prompt artifacts).
# ``evidence_ids`` is kept when present (snippet/context ids used as provenance).
_PIPELINE_STRIP_KEYS = frozenset(
    {
        "contexts",
        "context_mode",
        "error",
        "doc_snippet_windows",
        "rejected_windows",
        "snippets",
        "query_parse",
        "query_text_normalized",
        "query_text_hyde",
        "hyde",
    }
)


def doc_ids_to_bioasq_documents(doc_ids: List[Any]) -> List[str]:
    """Turn internal doc ids into BioASQ ``documents`` URLs (numeric ids → PubMed URL)."""
    out: List[str] = []
    for x in doc_ids:
        s = str(x).strip()
        if not s:
            continue
        if s.isdigit():
            out.append(f"{PUBMED_DOCUMENT_PREFIX}{s}")
        else:
            out.append(s)
    return out


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
    if qtype is not None and str(qtype).strip():
        out["type"] = str(qtype).strip()
    return out


def _strip_pipeline_fields(q: Dict[str, Any]) -> None:
    for k in _PIPELINE_STRIP_KEYS:
        q.pop(k, None)


def _looks_like_generation_jsonl(rec: Dict[str, Any]) -> bool:
    return "ideal_answer" in rec or "evidence_ids" in rec or "error" in rec


def _ideal_answer_missing(rec: Dict[str, Any]) -> bool:
    v = rec.get("ideal_answer")
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    return False


def _exact_answer_missing(rec: Dict[str, Any]) -> bool:
    qtype = str(rec.get("query_type") or rec.get("type") or "").strip().lower()
    if qtype == "summary":
        return False
    if qtype not in ("yesno", "factoid", "list"):
        return False
    v = rec.get("exact_answer")
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, list):
        return len(v) == 0
    return False


def warn_if_generation_issues(rec: Dict[str, Any]) -> None:
    """Warn on stderr when generation output will yield null or empty answers in BioASQ JSON."""
    if not _looks_like_generation_jsonl(rec):
        return
    qid = rec.get("query_id", rec.get("id", "?"))
    err = rec.get("error")
    if isinstance(err, str) and err.strip():
        msg = err.strip().replace("\n", " ")
        if len(msg) > 400:
            msg = msg[:400] + "..."
        print(
            f"Warning: query_id={qid!r} generation failed; "
            f"BioASQ JSON will omit the error field and may contain null answers: {msg}",
            file=sys.stderr,
        )
        return
    if _ideal_answer_missing(rec) or _exact_answer_missing(rec):
        problems: List[str] = []
        if _ideal_answer_missing(rec):
            problems.append("missing or empty ideal_answer")
        if _exact_answer_missing(rec):
            problems.append("missing or empty exact_answer")
        print(
            f"Warning: query_id={qid!r} {'; '.join(problems)} "
            f"(BioASQ JSON may be incomplete for this question).",
            file=sys.stderr,
        )


def record_to_bioasq_question(rec: Dict[str, Any], *, max_documents: int) -> dict:
    """Merge line_to_question with ``documents`` / ``doc_ids`` / ``docnos`` → BioASQ ``documents`` URLs."""
    q = line_to_question(rec)
    _strip_pipeline_fields(q)
    doc_ids = rec.get("doc_ids") or rec.get("docnos")
    if isinstance(doc_ids, list) and doc_ids:
        ids = list(doc_ids)
        if max_documents > 0:
            ids = ids[:max_documents]
        q["documents"] = doc_ids_to_bioasq_documents(ids)
    elif isinstance(q.get("documents"), list) and q["documents"]:
        # Strict query JSONL stores normalized PMIDs; BioASQ expects PubMed URLs.
        docs = doc_ids_to_bioasq_documents(q["documents"])
        if max_documents > 0:
            docs = docs[:max_documents]
        q["documents"] = docs
    q.pop("doc_ids", None)
    q.pop("docnos", None)
    return q


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Queries .jsonl")
    p.add_argument("--output", type=Path, required=True, help="BioASQ JSON output path.")
    p.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    p.add_argument(
        "--max-documents",
        type=int,
        default=10,
        metavar="N",
        help="Keep at most the first N document ids / URLs in order (default: 10). "
        "Use 0 for no limit.",
    )
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
            warn_if_generation_issues(rec)
            questions.append(record_to_bioasq_question(rec, max_documents=args.max_documents))
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
