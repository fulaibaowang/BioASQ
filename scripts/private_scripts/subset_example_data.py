#!/usr/bin/env python3
"""
Subset example/training and test data for lighter runs.

1. Training: from example/training14b_10pct_sample.json, remove questions that appear in 13B1–13B4
   golden (train-only ~580), then subset that by fraction → example/training14b_3pct_sample.json
2. Golden: randomly sample ~50 questions across 13B1–13B4 golden → example/13b_golden_50q_sample.json
"""

import argparse
import json
import random
from pathlib import Path

# Repo root (script lives in scripts/private_scripts/)
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "example"
TRAINING_10PCT = EXAMPLE_DIR / "training14b_10pct_sample.json"
GOLDEN_DIR = REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched"
GOLDEN_BATCHES = [GOLDEN_DIR / f"13B{i}_golden.json" for i in range(1, 5)]


def _golden_question_ids() -> set[str]:
    """Question IDs that appear in 13B1–13B4 golden (to exclude from training subset)."""
    ids = set()
    for p in GOLDEN_BATCHES:
        if not p.exists():
            raise FileNotFoundError(f"Golden file not found: {p}")
        with open(p) as f:
            data = json.load(f)
        for q in data.get("questions", []):
            qid = q.get("id")
            if qid is not None:
                ids.add(str(qid))
    return ids


def subset_training(fraction: float = 0.25, seed: int = 42) -> None:
    """Subset training14b_10pct_sample.json by fraction; write training14b_3pct_sample.json.
    Excludes any questions that appear in 13B1–13B4 golden (train-only), then samples.
    """
    path_in = TRAINING_10PCT
    path_out = EXAMPLE_DIR / "training14b_3pct_sample.json"
    if not path_in.exists():
        raise FileNotFoundError(f"Training file not found: {path_in}")

    golden_ids = _golden_question_ids()
    with open(path_in) as f:
        data = json.load(f)
    all_questions = data.get("questions", [])
    train_only = [q for q in all_questions if str(q.get("id", "")) not in golden_ids]
    n_total, n_train_only = len(all_questions), len(train_only)
    k = max(1, int(n_train_only * fraction))
    random.seed(seed)
    chosen = random.sample(train_only, k)
    out = {"questions": chosen}
    with open(path_out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        f"Training: {path_in.name} ({n_total} total, {n_train_only} train-only after removing golden overlap) "
        f"-> {path_out.name} ({len(chosen)} questions, fraction={fraction})"
    )


def subset_golden(n_questions: int = 50, seed: int = 42) -> None:
    """Pool 13B1–13B4 golden, randomly sample n_questions; write 13b_golden_50q_sample.json."""
    path_out = EXAMPLE_DIR / "13b_golden_50q_sample.json"
    all_questions = []
    for p in GOLDEN_BATCHES:
        if not p.exists():
            raise FileNotFoundError(f"Golden file not found: {p}")
        with open(p) as f:
            data = json.load(f)
        qs = data.get("questions", [])
        all_questions.extend(qs)
    n_total = len(all_questions)
    k = min(n_questions, n_total)
    random.seed(seed)
    chosen = random.sample(all_questions, k)
    out = {"questions": chosen}
    with open(path_out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Golden: 13B1–13B4 ({n_total} questions) -> {path_out.name} ({len(chosen)} questions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-fraction", type=float, default=0.25,
                    help="Fraction of training 10pct to keep (default 0.25 ≈ 3pct)")
    ap.add_argument("--golden-n", type=int, default=50,
                    help="Number of questions to sample from 13B1–13B4 (default 50)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--training-only", action="store_true", help="Only subset training")
    ap.add_argument("--golden-only", action="store_true", help="Only subset golden")
    args = ap.parse_args()

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.golden_only:
        subset_training(fraction=args.training_fraction, seed=args.seed)
    if not args.training_only:
        subset_golden(n_questions=args.golden_n, seed=args.seed)


if __name__ == "__main__":
    main()
