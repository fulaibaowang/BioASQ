#!/usr/bin/env python3
"""
Collect all question IDs from test batch JSON files and write to txt.

Usage:
    python scripts/collect_test_batch_qids.py \
        --test_dir bioasq_data/Task13BGoldenEnriched \
        --output_file example/test_batch_qids.txt
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="Directory with test batch JSONs")
    ap.add_argument("--output_file", required=True, help="Output txt file for qids")
    ap.add_argument("--batch_names", default="13B1_golden,13B2_golden,13B3_golden,13B4_golden",
                    help="Comma-separated batch names (without .json)")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_qids = []
    batch_names = args.batch_names.split(",")

    for batch_name in batch_names:
        batch_path = test_dir / f"{batch_name}.json"
        if not batch_path.exists():
            print(f"[SKIP] {batch_path} not found")
            continue

        with open(batch_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        batch_qids = [str(q["id"]) for q in data.get("questions", [])]
        all_qids.extend(batch_qids)
        print(f"[DONE] {batch_name}: {len(batch_qids)} questions")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for qid in sorted(all_qids):
            f.write(f"{qid}\n")

    print(f"\nTotal: {len(all_qids)} questions")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
