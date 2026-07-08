#!/usr/bin/env python3
"""Local BioASQ list-metric scorer for the generation A/B/C test.

Computes mean list Precision/Recall/F1 (the official BioASQ list scoring:
synonym-group matching, case-insensitive) over a qid subset, for one or more
answer files, and prints the delta vs the first (baseline) file.

L_Recall here == nugget-recall: fraction of gold answer-items recovered.

Usage:
  python eval_list_local.py --gold <golden.json> --qids <manifest.txt> \
    --mode direct=answers_direct.jsonl --mode claims=answers_claims.jsonl \
    --mode facets=answers_facets.jsonl
"""
import argparse, json, re
from pathlib import Path

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def groups_to_sets(ea):
    """exact_answer -> list of sets of normalized synonyms (one set per item)."""
    out = []
    if not isinstance(ea, list):
        return out
    for item in ea:
        if isinstance(item, list):
            out.append({norm(x) for x in item if str(x).strip()})
        elif str(item).strip():
            out.append({norm(item)})
    return [g for g in out if g]

def load_answers(path):
    p = Path(path)
    txt = p.read_text().strip()
    recs = []
    if txt.startswith("{") and '"questions"' in txt[:200]:
        recs = json.loads(txt)["questions"]
    else:
        # split ONLY on \n: abstract text can contain exotic Unicode line
        # separators (U+2028, VT, ...) that str.splitlines() would break a
        # JSON record on, mid-string.
        for line in txt.split("\n"):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    out = {}
    for r in recs:
        qid = str(r.get("id") or r.get("query_id"))
        out[qid] = groups_to_sets(r.get("exact_answer"))
    return out

def prf(sys_items, gold_items):
    if not gold_items:
        return None
    # a system item matches a gold group if any synonym overlaps
    def matches(a, b):
        return bool(a & b)
    matched_sys = sum(1 for s in sys_items if any(matches(s, g) for g in gold_items))
    matched_gold = sum(1 for g in gold_items if any(matches(s, g) for s in sys_items))
    P = matched_sys / len(sys_items) if sys_items else 0.0
    R = matched_gold / len(gold_items)
    F = (2 * P * R / (P + R)) if (P + R) else 0.0
    return P, R, F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--mode", action="append", required=True, help="name=answers_file")
    args = ap.parse_args()

    gold_all = json.load(open(args.gold))["questions"]
    gold = {q["id"]: groups_to_sets(q.get("exact_answer")) for q in gold_all
            if q.get("type") == "list"}
    qids = [l.split("\t")[1].strip() for l in Path(args.qids).read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    qids = [q for q in qids if q in gold]

    print(f"Subset: {len(qids)} list questions\n")
    hdr = f"{'mode':10} {'L_P':>8} {'L_R':>8} {'L_F1':>8}   n"
    print(hdr); print("-" * len(hdr))
    base = None
    for spec in args.mode:
        name, _, path = spec.partition("=")
        ans = load_answers(path)
        Ps = Rs = Fs = 0.0; n = 0
        for qid in qids:
            r = prf(ans.get(qid, []), gold[qid])
            if r:
                Ps += r[0]; Rs += r[1]; Fs += r[2]; n += 1
        m = (Ps / n, Rs / n, Fs / n) if n else (0, 0, 0)
        print(f"{name:10} {m[0]:8.4f} {m[1]:8.4f} {m[2]:8.4f}   {n}")
        if base is None:
            base = m
        else:
            print(f"{'  Δ':10} {m[0]-base[0]:+8.4f} {m[1]-base[1]:+8.4f} {m[2]-base[2]:+8.4f}")

if __name__ == "__main__":
    main()
