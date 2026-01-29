#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable

import pyterrier as pt


def iter_docs(jsonl_glob: str) -> Iterable[Dict]:
    """
    Stream documents from many JSONL shards.

    Filters:
      - is_deleted must be False (or missing)
      - abstract must be non-empty

    Yields:
      {"docno": pmid, "text": title + "\\n\\n" + abstract}
    """
    for fp in sorted(glob.glob(jsonl_glob)):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                if bool(d.get("is_deleted", False)):
                    continue

                pmid = (d.get("pmid") or d.get("docno") or "").strip()
                if not pmid:
                    continue

                abstract = (d.get("abstract") or "").strip()
                if not abstract:
                    # <--- abstract-only indexing, as requested
                    continue

                title = (d.get("title") or "").strip()
                text = (title + "\n\n" + abstract).strip()

                yield {"docno": pmid, "text": text}


def build_index(index_path: str, jsonl_glob: str, overwrite: bool, threads: int):
    os.makedirs(index_path, exist_ok=True)

    # Meta sizes are fixed-width in Terrier. Keep them small.
    # We only need docno; text is stored in the direct index, not meta.
    indexer = pt.IterDictIndexer(
        index_path,
        text_attrs=["text"],
        meta={"docno": 32},  # PMID fits easily
        overwrite=overwrite,
        threads=threads,
    )

    indexref = indexer.index(iter_docs(jsonl_glob))
    return indexref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_glob", required=True, help='e.g. "/data/pubmed_jsonl/baseline/*.jsonl"')
    ap.add_argument("--index_path", required=True, help='e.g. "/data/terrier_indexes/pubmed_baseline_bm25"')
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not pt.started():
        pt.init()
    
    # Allow duplicate PMIDs in meta index (useful for multi-threaded indexing)
    import jnius
    jnius.autoclass('java.lang.System').setProperty('metaindex.compressed.reverse.allow.duplicates', 'true')

    indexref = build_index(
        index_path=args.index_path,
        jsonl_glob=args.jsonl_glob,
        overwrite=args.overwrite,
        threads=args.threads,
    )

    print("DONE")
    print("IndexRef:", indexref)


if __name__ == "__main__":
    main()
