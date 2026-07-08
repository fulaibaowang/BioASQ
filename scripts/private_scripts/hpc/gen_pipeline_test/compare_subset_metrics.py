#!/usr/bin/env python3
"""Compare generation modes on the oracle-rich list-question subset.

Reuses the OFFICIAL BioASQ evaluator output (the per-question phaseB TSV,
same format as report/phaseB_report_perq.tsv) rather than reimplementing the
metric. For each mode you pass its per-question TSV; this script restricts to
the subset qids (oracle_rich_list_qids.txt) and reports mean L_P/L_R/L_F1 and
Rouge, plus the delta vs the `direct` baseline.

Usage:
  python compare_subset_metrics.py \
    --qids oracle_rich_list_qids.txt \
    --mode direct=/path/to/direct_phaseB_report_perq.tsv \
    --mode claims=/path/to/claims_phaseB_report_perq.tsv \
    --mode facets=/path/to/facets_phaseB_report_perq.tsv
"""
import argparse, csv, statistics as st
from pathlib import Path

METRICS = ["L_P", "L_R", "L_F1", "R_2_F1", "R_SU4_F1"]


def load_qids(path):
    ids = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # manifest columns: split \t qid \t ... (qid is column 2)
        ids.add(line.split("\t")[1].strip())
    return ids


def load_tsv(path, qids):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qid = r["question_id"].strip()
            if qid in qids:
                rows[qid] = r
    return rows


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", required=True)
    ap.add_argument("--mode", action="append", required=True,
                    help="name=path/to/per_question.tsv (repeatable)")
    args = ap.parse_args()

    qids = load_qids(args.qids)
    modes = {}
    for spec in args.mode:
        name, _, path = spec.partition("=")
        modes[name] = load_tsv(path, qids)

    print(f"Subset: {len(qids)} oracle-rich list questions\n")
    header = f"{'mode':10} " + " ".join(f"{m:>9}" for m in METRICS) + "   n"
    print(header)
    print("-" * len(header))
    base = {}
    for name, rows in modes.items():
        means = {}
        for m in METRICS:
            vals = [fnum(rows[q].get(m)) for q in rows]
            vals = [v for v in vals if v is not None]
            means[m] = st.mean(vals) if vals else float("nan")
        if not base:
            base = means
        cells = " ".join(f"{means[m]:9.4f}" for m in METRICS)
        print(f"{name:10} {cells}   {len(rows)}")
        if name != next(iter(modes)):
            deltas = " ".join(f"{means[m]-base[m]:+9.4f}" for m in METRICS)
            print(f"{'  Δ vs '+next(iter(modes)):10} {deltas}")


if __name__ == "__main__":
    main()
