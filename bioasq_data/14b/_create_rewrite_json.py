#!/usr/bin/env python3
"""Create BioASQ JSON with rewritten queries as additional body* fields."""

import json
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "bioasq_data" / "14b"

INPUT_JSON = DATA_DIR / "BioASQ-task14bPhaseA-testset1"
INPUT_CSV = DATA_DIR / "bioasq_80_rewrite_screening_v3_with_rerank_column.csv"
OUTPUT_JSON = DATA_DIR / "BioASQ-task14bPhaseA-testset1-rewrite"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

rewrites = {}
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        num = int(row["num"])
        rewrites[num] = {
            "smart_rewrite": row.get("smart_rewrite", "").strip(),
            "rerank_rewrite": row.get("rerank_rewrite", "").strip(),
        }

questions = data["questions"]
if len(questions) != len(rewrites):
    print(
        f"WARNING: JSON has {len(questions)} questions but CSV has {len(rewrites)} rows",
        file=sys.stderr,
    )

for i, q in enumerate(questions):
    original_body = q["body"]
    row = rewrites.get(i + 1, {})

    smart = row.get("smart_rewrite", "")
    rerank = row.get("rerank_rewrite", "")

    q["body_smart_rewrite"] = smart if smart else original_body
    q["body_rerank_rewrite"] = rerank if rerank else original_body

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(questions)} questions to {OUTPUT_JSON}")
for i, q in enumerate(questions[:3]):
    print(f"\n--- Question {i+1} ---")
    print(f"  body:               {q['body'][:80]}...")
    print(f"  body_smart_rewrite: {q['body_smart_rewrite'][:80]}...")
    print(f"  body_rerank_rewrite:{q['body_rerank_rewrite'][:80]}...")
