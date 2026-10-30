# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Row sampling for the analysis subset (independent of stability resampling)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("input_shuffling_v2")


def apply_sampling(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["sampling"]
    use = str(s.get("use", "all")).lower()
    if use == "all":
        return df

    seed = int(cfg["run"]["seed"]) + int(s.get("seed_offset", 0) or 0)
    replace = bool(s.get("replace", False))

    if use == "sample_n":
        n = int(s["sample_n"])
    elif use == "sample_frac":
        n = int(round(float(s["sample_frac"]) * len(df)))
    else:
        raise ValueError(f"Unsupported sampling.use: {use}")
    n = min(n, len(df)) if not replace else n

    stratify_by = s.get("stratify_by")
    if stratify_by:
        col = cfg["schema"].get("target_column") if stratify_by == "y" else stratify_by
        if col not in df.columns:
            raise ValueError(f"sampling.stratify_by column not found: {col}")
        frac = n / len(df)
        rng_state = np.random.default_rng(seed)
        sampled = (
            df.groupby(col, group_keys=False, dropna=False)
            .apply(lambda g: g.sample(
                n=max(1, int(round(frac * len(g)))), replace=replace,
                random_state=int(rng_state.integers(0, 2**31 - 1))))
        )
        out = sampled.reset_index(drop=True)
    else:
        out = df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)

    logger.info(f"Sampling: {use} -> {len(out):,} of {len(df):,} rows (seed={seed})")
    return out
