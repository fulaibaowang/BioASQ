#!/usr/bin/env python3
"""
Remove from a training JSON any questions whose id appears in the given test batch JSONs.
Use this to make train and test disjoint so policies that exclude test qids from train
behave consistently and ceiling analysis (e.g. R@2000 on train) doesn't show
zero-recall for queries that only exist in test runs.

Usage:
  python remove_test_qids_from_train_json.py \\
    --train_json example/training14b_10pct_sample.json \\
    --test_jsons bioasq_data/Task13BGoldenEnriched/13B1_golden.json \\
                  bioasq_data/Task13BGoldenEnriched/13B2_golden.json \\
                  bioasq_data/Task13BGoldenEnriched/13B3_golden.json \\
                  bioasq_data/Task13BGoldenEnriched/13B4_golden.json \\
    [--out_json PATH]   # default: overwrite --train_json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "questions" not in data:
        raise KeyError(f"{path} missing top-level 'questions'")
    return data["questions"]


def collect_qids(questions: list[dict]) -> set[str]:
    out: set[str] = set()
    for q in questions:
        qid = q.get("id") or q.get("qid")
        if qid is not None:
            out.add(str(qid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_json", type=Path, required=True, help="Path to training questions JSON")
    ap.add_argument("--test_jsons", type=Path, nargs="+", required=True, help="Paths to test batch JSONs")
    ap.add_argument("--out_json", type=Path, default=None, help="Output path (default: overwrite train_json)")
    args = ap.parse_args()

    train_path = args.train_json.resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Train JSON not found: {train_path}")

    test_qids: set[str] = set()
    for p in args.test_jsons:
        fp = Path(p).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Test JSON not found: {fp}")
        test_qids |= collect_qids(load_questions(fp))

    questions = load_questions(train_path)
    n_before = len(questions)
    kept = [q for q in questions if str(q.get("id") or q.get("qid") or "") not in test_qids]
    n_after = len(kept)
    removed = n_before - n_after

    out_path = (args.out_json or train_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"questions": kept}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Test qids collected: {len(test_qids)}")
    print(f"Train before: {n_before} questions")
    print(f"Removed (ids in test batches): {removed}")
    print(f"Train after: {n_after} questions")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
