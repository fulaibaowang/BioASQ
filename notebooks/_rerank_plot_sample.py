"""Shared helpers for rerank diagnostic notebooks (canonical stratified query sample)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Must stay in sync with notebooks that use this for cross-notebook comparability.
DEFAULT_N_GOLD_BINS: list[tuple[int, int, str]] = [
    (0, 0, "0"),
    (1, 3, "1-3"),
    (4, 10, "4-10"),
    (11, 9999, ">10"),
]


def sample_queries_for_diagnostic_plots(
    df_pos: pd.DataFrame,
    *,
    sample_size: int = 48,
    rng_seed: int = 42,
    n_gold_bins: Sequence[tuple[int, int, str]] | None = None,
) -> list[tuple[str, str]]:
    """Stratified sample of (split, qid) with n_gold > 0; same logic as rerank_cutoff_01_adaptive_gap_oracle.

    *df_pos* must have columns ``split``, ``qid``, ``n_gold``.
    """
    if n_gold_bins is None:
        n_gold_bins = DEFAULT_N_GOLD_BINS
    rng = np.random.default_rng(rng_seed)
    sampled: list[tuple[str, str]] = []
    per_bin = max(1, sample_size // len(n_gold_bins))
    for lo, hi, _ in n_gold_bins:
        pool = df_pos[(df_pos["n_gold"] >= lo) & (df_pos["n_gold"] <= hi)]
        n_take = min(per_bin, len(pool))
        if n_take > 0:
            chosen = pool.sample(n=n_take, random_state=int(rng.integers(1 << 31)))
            sampled.extend(zip(chosen["split"], chosen["qid"]))

    if len(sampled) < sample_size:
        already = set(sampled)
        rest = df_pos[~df_pos.apply(lambda r: (r["split"], r["qid"]) in already, axis=1)]
        n_extra = min(sample_size - len(sampled), len(rest))
        if n_extra > 0:
            extra = rest.sample(n=n_extra, random_state=int(rng.integers(1 << 31)))
            sampled.extend(zip(extra["split"], extra["qid"]))

    return sampled[:sample_size]
