#!/usr/bin/env python3
"""
Subset test batch JSON files by excluding questions already in the training sample.

Usage:
    python bioasq_data/Task13BGoldenEnriched/_subset_test_batches.py \
        --train_sample example/training14b_10pct_sample.json \
        --test_dir bioasq_data/Task13BGoldenEnriched \
        --output_dir bioasq_data/Task13BGoldenEnriched/test_batches_subset
"""

import argparse
import json
from pathlib import Path


def load_train_qids(train_path: Path) -> set[str]:
    """Load question IDs from training sample."""
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(q["id"]) for q in data.get("questions", [])}


def subset_batch_by_qids(batch_path: Path, exclude_qids: set[str]) -> dict:
    """
    Load batch and keep only questions NOT in exclude_qids.
    Returns: {"questions": [...]}
    """
    with open(batch_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_questions = [
        q for q in data.get("questions", [])
        if str(q["id"]) not in exclude_qids
    ]

    return {
        "questions": filtered_questions,
        "original_count": len(data.get("questions", [])),
        "filtered_count": len(filtered_questions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_sample", required=True, help="Path to training sample JSON")
    ap.add_argument("--test_dir", required=True, help="Directory with test batch JSONs")
    ap.add_argument("--output_dir", required=True, help="Output directory for subsets")
    ap.add_argument("--batch_names", default="13B1_golden,13B2_golden,13B3_golden,13B4_golden",
                    help="Comma-separated batch names (without .json)")
    args = ap.parse_args()

    train_path = Path(args.train_sample)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training sample qids
    train_qids = load_train_qids(train_path)
    print(f"Training sample has {len(train_qids)} questions")

    # Process each batch
    batch_names = args.batch_names.split(",")
    for batch_name in batch_names:
        batch_path = test_dir / f"{batch_name}.json"
        if not batch_path.exists():
            print(f"[SKIP] {batch_path} not found")
            continue

        result = subset_batch_by_qids(batch_path, train_qids)
        
        # Save subset
        output_path = output_dir / f"{batch_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"questions": result["questions"]}, f, indent=2, ensure_ascii=False)

        print(f"[DONE] {batch_name}: {result['original_count']} -> {result['filtered_count']} questions")
        print(f"       Saved to: {output_path}")


if __name__ == "__main__":
    main()
