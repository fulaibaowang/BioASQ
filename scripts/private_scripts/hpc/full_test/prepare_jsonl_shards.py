#!/usr/bin/env python3
"""
Prepare one shard of JSONL files for dense index building: shuffle a glob of
JSONL paths with a fixed seed, split into N shards, and for the given shard
index create a directory of symlinks plus a manifest file.
"""
from __future__ import annotations

import argparse
import glob
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shuffle JSONL list (fixed seed), split into N shards, create one shard dir with symlinks and manifest."
    )
    ap.add_argument("--jsonl_glob", required=True, help='Glob for JSONL files, e.g. "/pubmed/jsonl_2026/*.jsonl"')
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    ap.add_argument("--n_shards", type=int, required=True, help="Number of shards (e.g. 10)")
    ap.add_argument("--shard_index", type=int, required=True, help="This shard index in 0..n_shards-1")
    ap.add_argument("--out_shard_dir", required=True, help="Output directory for this shard (symlinks + manifest)")
    args = ap.parse_args()

    if not 0 <= args.shard_index < args.n_shards:
        raise SystemExit(f"--shard_index must be in [0, {args.n_shards}), got {args.shard_index}")

    files = sorted(glob.glob(args.jsonl_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.jsonl_glob}")

    random.seed(args.seed)
    random.shuffle(files)

    chunk_size = (len(files) + args.n_shards - 1) // args.n_shards
    start = args.shard_index * chunk_size
    end = min(start + chunk_size, len(files))
    chunk = files[start:end]

    os.makedirs(args.out_shard_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_shard_dir, "manifest.txt")
    with open(manifest_path, "w") as mf:
        for fp in chunk:
            mf.write(fp + "\n")
            # Symlink: basename in out_shard_dir -> absolute path
            target = os.path.abspath(fp)
            link_name = os.path.join(args.out_shard_dir, os.path.basename(fp))
            if os.path.lexists(link_name):
                os.remove(link_name)
            os.symlink(target, link_name)

    print(f"[prepare_jsonl_shards] shard {args.shard_index}/{args.n_shards} -> {args.out_shard_dir}: {len(chunk)} files, manifest={manifest_path}")


if __name__ == "__main__":
    main()
