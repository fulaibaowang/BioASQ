#!/usr/bin/env python3
"""Filter PubMed JSONL shards to rows whose PMID appears in a given allow-list.

Used to materialise BioASQ subset corpora (e.g. the 3% / 10% training subsets):
given a text file of PMIDs (one per line) and a glob of source JSONL shards,
emit a single subset JSONL containing only the matching rows.

Usage:
    python extract_jsonl_subset_by_pmids.py \
        --pmids subset_pmids.txt \
        --jsonl-glob "/data/pubmed_2026/*.jsonl" \
        --output subset.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, Set


def read_pmids(pmid_path: str) -> Set[str]:
    pmids: Set[str] = set()
    with open(pmid_path, "r", encoding="utf-8") as f:
        for line in f:
            pmid = line.strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def iter_jsonl_paths(jsonl_glob: str) -> Iterable[Path]:
    for fp in sorted(glob.glob(jsonl_glob)):
        yield Path(fp)


def get_pmid(d: dict) -> str:
    return (d.get("pmid") or d.get("docno") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_glob", required=True, help='e.g. "/data/pubmed_jsonl/baseline/*.jsonl"')
    ap.add_argument("--pmid_list", required=True, help='e.g. "/path/to/subset_pmids.txt"')
    ap.add_argument("--output_jsonl", required=True, help='e.g. "/path/to/subset.jsonl"')
    ap.add_argument("--output_missing", default="", help="Optional path to save missing PMIDs.")
    ap.add_argument("--dedup", action="store_true", help="Avoid writing duplicate PMIDs.")
    ap.add_argument("--stop_when_complete", action="store_true", help="Stop when all PMIDs found.")
    args = ap.parse_args()

    target_pmids = read_pmids(args.pmid_list)
    remaining = set(target_pmids)
    seen = set() if args.dedup else None

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for jsonl_path in iter_jsonl_paths(args.jsonl_glob):
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    pmid = get_pmid(d)
                    if not pmid or pmid not in remaining:
                        continue

                    if seen is not None:
                        if pmid in seen:
                            continue
                        seen.add(pmid)

                    out_f.write(line + "\n")
                    written += 1
                    remaining.discard(pmid)

                    if args.stop_when_complete and not remaining:
                        break

            if args.stop_when_complete and not remaining:
                break

    if args.output_missing:
        missing_path = Path(args.output_missing)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with missing_path.open("w", encoding="utf-8") as f:
            for pmid in sorted(remaining):
                f.write(f"{pmid}\n")

    print("DONE")
    print(f"Input PMIDs: {len(target_pmids)}")
    print(f"Written: {written}")
    print(f"Missing: {len(remaining)}")
    print(f"Total lines scanned: {total_lines}")
    print(f"Output: {out_path}")
    if args.output_missing:
        print(f"Missing list: {args.output_missing}")


if __name__ == "__main__":
    main()
