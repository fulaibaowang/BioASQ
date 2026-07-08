#!/usr/bin/env python3
"""Build the LONG-CONTEXT oracle evidence for the distillation test (Plan B).

Unlike the GT-*snippet* oracle (oracle_all_*.jsonl, ~1-sentence contexts), this
feeds distillation real material: one context per GOLD document = that PubMed
record's full abstract (title + abstract body, ~200w). This is the fair test of
claims/facets distillation (snippets are already atomic -> nothing to compress;
see FINDINGS_AND_PLAN.md finding 2).

Single pass over the corpus shards (records keyed by ``pmid``; ``docno==pmid``
for ``type==abstract`` rows), collecting the union of gold PMIDs across the
split. Emits the SAME evidence schema the generation stage already consumes:

    {"query_id","query_text","query_type",
     "contexts":[{"id":"<pmid>-a0","text":"<title>\\n<abstract>","doc_id":"<pubmed url>"}, ...]}

Two outputs:
  * oracle_docabs_all_<SPLIT>.jsonl  -- all gold docs per question
  * oracle_docabs_<N>_<SPLIT>.jsonl  -- first N gold docs per question (golden order)
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

PMID_RE = re.compile(r"/(\d+)/?$")


def pmid_of(url: str):
    m = PMID_RE.search(str(url))
    return m.group(1) if m else None


def load_gold(golden_json: Path):
    """Return list of (qid, body, qtype, [gold_pmids in golden order])."""
    qs = json.load(open(golden_json))["questions"]
    out = []
    for q in qs:
        pmids, seen = [], set()
        for u in (q.get("documents") or []):
            p = pmid_of(u)
            if p and p not in seen:
                seen.add(p)
                pmids.append(p)
        out.append((q["id"], q.get("body", ""), q.get("type", ""), pmids))
    return out


def scan_corpus(corpus_glob: str, needed: set):
    """One pass over corpus shards -> {pmid: (title, abstract_text)} for needed pmids.

    Prefer ``type==abstract`` rows; ``docno==pmid`` there. Fall back to any row
    whose ``pmid`` matches if no abstract row is seen.
    """
    paths = sorted(glob.glob(corpus_glob))
    if not paths:
        sys.exit(f"[build_docabs] no shards matched: {corpus_glob}")
    found = {}          # pmid -> (title, text) from a type==abstract row
    fallback = {}       # pmid -> (title, text) from any other row
    n_files = len(paths)
    for i, path in enumerate(paths, 1):
        if len(found) >= len(needed):
            break  # every needed pmid has an abstract row; stop early
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = str(obj.get("pmid") or "")
                if p not in needed:
                    continue
                title = obj.get("title") or ""
                text = obj.get("text") or ""
                if isinstance(title, list):
                    title = " ".join(str(t) for t in title)
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                rec = (str(title).strip(), str(text).strip())
                if obj.get("type") == "abstract":
                    found[p] = rec
                elif p not in fallback:
                    fallback[p] = rec
        if i % 100 == 0 or i == n_files:
            print(f"[build_docabs] scanned {i}/{n_files} shards; "
                  f"abstracts={len(found)}/{len(needed)}", flush=True)
    for p, rec in fallback.items():
        found.setdefault(p, rec)
    return found


def ctx_text(title: str, text: str) -> str:
    if title and text:
        return f"{title}\n{text}"
    return title or text


def build_record(qid, body, qtype, pmids, pmid2text):
    contexts = []
    for p in pmids:
        rec = pmid2text.get(p)
        if not rec:
            continue  # gold pmid absent from corpus -> skip (warned in main)
        t = ctx_text(*rec)
        if not t:
            continue
        contexts.append({
            "id": f"{p}-a0",
            "text": t,
            "doc_id": f"http://www.ncbi.nlm.nih.gov/pubmed/{p}",
        })
    return {
        "query_id": qid,
        "query_text": body,
        "query_type": qtype,
        "contexts": contexts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=Path,
                    help="Task13B*_golden.json for the split")
    ap.add_argument("--corpus-glob", required=True,
                    help="e.g. /pubmed/jsonl_2026/*.jsonl")
    ap.add_argument("--out-all", required=True, type=Path)
    ap.add_argument("--out-topn", required=True, type=Path)
    ap.add_argument("--topn", type=int, default=16)
    args = ap.parse_args()

    gold = load_gold(args.golden)
    union = set()
    for _, _, _, pmids in gold:
        union.update(pmids)
    print(f"[build_docabs] {len(gold)} questions; {len(union)} union gold PMIDs")

    pmid2text = scan_corpus(args.corpus_glob, union)
    missing = union - set(pmid2text.keys())
    print(f"[build_docabs] resolved {len(pmid2text)}/{len(union)} PMIDs "
          f"({len(missing)} missing from corpus)")
    if missing:
        sample = ", ".join(sorted(missing)[:20])
        print(f"[build_docabs] WARNING missing PMIDs (first 20): {sample}",
              file=sys.stderr)

    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    n_all_ctx = n_topn_ctx = 0
    empty = []
    with open(args.out_all, "w") as fa, open(args.out_topn, "w") as ft:
        for qid, body, qtype, pmids in gold:
            r_all = build_record(qid, body, qtype, pmids, pmid2text)
            r_topn = build_record(qid, body, qtype, pmids[:args.topn], pmid2text)
            n_all_ctx += len(r_all["contexts"])
            n_topn_ctx += len(r_topn["contexts"])
            if not r_all["contexts"]:
                empty.append(qid)
            fa.write(json.dumps(r_all, ensure_ascii=False) + "\n")
            ft.write(json.dumps(r_topn, ensure_ascii=False) + "\n")
    print(f"[build_docabs] wrote {args.out_all} "
          f"(avg {n_all_ctx/max(len(gold),1):.1f} ctx/q)")
    print(f"[build_docabs] wrote {args.out_topn} "
          f"(avg {n_topn_ctx/max(len(gold),1):.1f} ctx/q, cap {args.topn})")
    if empty:
        print(f"[build_docabs] WARNING {len(empty)} questions have 0 contexts: "
              f"{', '.join(empty[:10])}", file=sys.stderr)


if __name__ == "__main__":
    main()
