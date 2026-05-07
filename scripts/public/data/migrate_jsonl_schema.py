#!/usr/bin/env python3
"""Migrate PubMed JSONL shards to the unified RAG-pipeline schema in place.

Old schema (parse_pubmed_local.py prior to the abstract->text rename):
    {pmid, docno, title, abstract, mesh_terms, keywords, is_deleted}

New schema:
    {docno, pmid, type: "abstract", title, text, mesh_terms, keywords, is_deleted}

The transformation per row:
    - rename "abstract" -> "text"
    - add "type": "abstract" (parallels chunked corpora's body/caption rows)
    - reorder so docno + pmid + type lead each row (cosmetic)

The migration is idempotent: rows that already have a "text" field and a
"type" field are passed through unchanged (so re-running on a partially
migrated dir is safe).

Each shard is rewritten via a *.partial sibling and atomic rename, so a
mid-run interruption never leaves a half-written file in place.

Usage:
    # one shard
    python migrate_jsonl_schema.py --input /data/jsonl_2026/0001.jsonl

    # whole directory
    python migrate_jsonl_schema.py --input-dir /data/jsonl_2026

    # custom glob, dry-run first
    python migrate_jsonl_schema.py --input-dir /data/jsonl_2026 --glob "pubmed*.jsonl" --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

LEAD_FIELDS = ("docno", "pmid", "type", "title", "text")


def _resolve_inputs(input_path: Path | None, input_dir: Path | None, glob: str) -> list[Path]:
    files: list[Path] = []
    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        files.append(input_path)
    if input_dir:
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        files.extend(sorted(input_dir.glob(glob)))
    if not files:
        raise SystemExit("No input .jsonl files matched (use --input or --input-dir).")
    return files


def _migrate_row(rec: dict) -> Tuple[dict, str]:
    """Return (migrated_record, status). status in {'migrated','already','no_text_no_abstract'}.

    'already' means the row already has the new schema (idempotent skip).
    'no_text_no_abstract' means neither field is present — we write text="".
    """
    has_text = "text" in rec
    has_abstract = "abstract" in rec

    if has_text and rec.get("type"):
        return rec, "already"

    new = dict(rec)
    if has_abstract and not has_text:
        new["text"] = new.pop("abstract")
        status = "migrated"
    elif has_text and has_abstract:
        # both fields present (rare/manual edits) — prefer existing text, drop abstract
        new.pop("abstract", None)
        status = "migrated"
    elif not has_text and not has_abstract:
        new["text"] = ""
        status = "no_text_no_abstract"
    else:
        status = "migrated"

    new.setdefault("type", "abstract")

    # Reorder so identifying fields lead.
    ordered = {k: new[k] for k in LEAD_FIELDS if k in new}
    for k, v in new.items():
        if k not in ordered:
            ordered[k] = v
    return ordered, status


def migrate_file(path: Path, dry_run: bool = False) -> dict:
    n_total = 0
    n_migrated = 0
    n_already = 0
    n_empty = 0
    out_path = path.with_suffix(path.suffix + ".partial") if not dry_run else None
    out_fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                new, status = _migrate_row(rec)
                n_total += 1
                if status == "migrated":
                    n_migrated += 1
                elif status == "already":
                    n_already += 1
                elif status == "no_text_no_abstract":
                    n_empty += 1
                if out_fh is not None:
                    out_fh.write(json.dumps(new, ensure_ascii=False) + "\n")
    finally:
        if out_fh is not None:
            out_fh.close()
    if out_path is not None:
        out_path.replace(path)
    return {
        "path": str(path),
        "total": n_total,
        "migrated": n_migrated,
        "already": n_already,
        "no_text_no_abstract": n_empty,
        "wrote": out_path is None,  # True if dry-run only
        "dry_run": dry_run,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="Single .jsonl path.")
    ap.add_argument("--input-dir", type=Path, default=None, help="Directory of .jsonl shards.")
    ap.add_argument("--glob", type=str, default="*.jsonl", help="Glob under --input-dir (default: *.jsonl).")
    ap.add_argument("--dry-run", action="store_true", help="Inspect only; do not rewrite files.")
    args = ap.parse_args()

    files = _resolve_inputs(args.input, args.input_dir, args.glob)
    print(f"[INFO] {len(files)} file(s) to process. dry_run={args.dry_run}", file=sys.stderr)

    total_rows = 0
    total_migrated = 0
    total_already = 0
    total_empty = 0
    for fp in files:
        stats = migrate_file(fp, dry_run=args.dry_run)
        total_rows += stats["total"]
        total_migrated += stats["migrated"]
        total_already += stats["already"]
        total_empty += stats["no_text_no_abstract"]
        print(
            f"[{'DRY' if args.dry_run else 'OK'}] {stats['path']}: "
            f"rows={stats['total']:,} migrated={stats['migrated']:,} "
            f"already={stats['already']:,} empty={stats['no_text_no_abstract']:,}",
            flush=True,
        )

    print(
        f"\n[SUMMARY] rows={total_rows:,} migrated={total_migrated:,} "
        f"already_new_schema={total_already:,} empty(no_text_no_abstract)={total_empty:,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
