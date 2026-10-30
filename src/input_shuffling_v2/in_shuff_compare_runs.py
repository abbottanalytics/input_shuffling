# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-model importance comparison across input-shuffling runs.

Merges importance_feature_summary.csv from several run directories into a
feature x model rank table with a consensus ordering, plus a pairwise
Spearman rank-correlation matrix between models.

Usage (from Analytics Projects/):

  # explicit run dirs
  PYTHONPATH=input_shuffling_v2/src:. python3 -m input_shuffling_v2.in_shuff_compare_runs \
      input_shuffling_v2/runs/..._logistic__20260716_103703 \
      input_shuffling_v2/runs/..._knn__20260716_131544 \
      --output comparison.csv

  # or glob patterns; for each pattern the LATEST matching run is used
  PYTHONPATH=input_shuffling_v2/src:. python3 -m input_shuffling_v2.in_shuff_compare_runs \
      --latest "input_shuffling_v2/runs/*knime_target_b_*" \
      --output knime_target_b_comparison.csv

Model labels come from run.name in each run's config_used.yaml (with the
common prefix trimmed), so columns read e.g. "logistic", "knn", "rf".
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # scipy ships with sklearn; fall back just in case
    spearmanr = None


def _label_for(run_dir: Path, used: set[str]) -> str:
    name = None
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        import yaml
        name = ((yaml.safe_load(cfg_path.read_text()) or {}).get("run") or {}).get("name")
    if not name:
        name = re.sub(r"__\d{8}_\d{6}$", "", run_dir.name)
    label = name
    n = 2
    while label in used:
        label = f"{name}_{n}"
        n += 1
    used.add(label)
    return label


def _trim_common_prefix(labels: list[str]) -> list[str]:
    if len(labels) < 2:
        return labels
    prefix = os.path.commonprefix(labels)
    # trim only to a separator boundary so we don't cut mid-word
    cut = max(prefix.rfind("_"), prefix.rfind("-")) + 1
    trimmed = [lab[cut:] or lab for lab in labels]
    return trimmed if len(set(trimmed)) == len(trimmed) else labels


def _resolve_run_dirs(args) -> list[Path]:
    run_dirs: list[Path] = []
    for d in args.run_dirs:
        p = Path(d)
        if not (p / "importance_feature_summary.csv").exists():
            raise FileNotFoundError(f"No importance_feature_summary.csv in {p}")
        run_dirs.append(p)
    for pattern in args.latest or []:
        matches = {}
        for d in sorted(glob.glob(pattern)):
            p = Path(d)
            if not (p / "importance_feature_summary.csv").exists():
                continue
            base = re.sub(r"__\d{8}_\d{6}$", "", p.name)
            matches[base] = p  # sorted() => the latest timestamp wins
        if not matches:
            raise FileNotFoundError(f"No runs with importance output match: {pattern}")
        run_dirs.extend(matches.values())
    if len(run_dirs) < 2:
        raise SystemExit("Need at least two runs to compare.")
    return run_dirs


def build_comparison(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    used: set[str] = set()
    labels = [_label_for(d, used) for d in run_dirs]
    labels = _trim_common_prefix(labels)

    merged = None
    stats = {}
    for run_dir, label in zip(run_dirs, labels):
        imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
        stats[label] = imp["ranking_stat"].iloc[0]
        part = imp[["feature", "rank", "ranking_value"]].rename(
            columns={"rank": f"rank_{label}", "ranking_value": f"value_{label}"})
        merged = part if merged is None else merged.merge(part, on="feature", how="outer")

    rank_cols = [f"rank_{lab}" for lab in labels]
    merged["mean_rank"] = merged[rank_cols].mean(axis=1)
    merged["median_rank"] = merged[rank_cols].median(axis=1)
    merged["rank_range"] = merged[rank_cols].max(axis=1) - merged[rank_cols].min(axis=1)
    merged = merged.sort_values("mean_rank").reset_index(drop=True)
    merged.insert(1, "consensus_rank", np.arange(1, len(merged) + 1))

    # pairwise Spearman on ranks (complete pairs only)
    corr = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            pair = merged[[f"rank_{a}", f"rank_{b}"]].dropna()
            if spearmanr is not None and len(pair) > 2:
                rho = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
            else:
                rho = pair.corr(method="spearman").iloc[0, 1]
            corr.loc[a, b] = corr.loc[b, a] = rho

    if len(set(stats.values())) > 1:
        print(f"NOTE: runs use different ranking stats: {stats}", file=sys.stderr)
    return merged, corr, labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare importance rankings across runs")
    ap.add_argument("run_dirs", nargs="*", help="Run directories to compare")
    ap.add_argument("--latest", action="append",
                    help="Glob pattern; the latest run per run-name is used (repeatable)")
    ap.add_argument("--output", default=None, help="Write the merged table to this CSV")
    ap.add_argument("--top", type=int, default=None,
                    help="Print only the top N consensus features")
    args = ap.parse_args(argv)

    run_dirs = _resolve_run_dirs(args)
    merged, corr, labels = build_comparison(run_dirs)

    rank_view = merged[["feature", "consensus_rank", "mean_rank", "rank_range"]
                       + [f"rank_{lab}" for lab in labels]]
    if args.top:
        rank_view = rank_view.head(args.top)

    pd.set_option("display.width", 200)
    print("\n=== Feature ranks by model (sorted by consensus) ===")
    print(rank_view.to_string(index=False))
    print("\n=== Pairwise Spearman rank correlation ===")
    print(corr.round(3).to_string())

    if args.output:
        merged.to_csv(args.output, index=False)
        corr_path = re.sub(r"\.csv$", "", args.output) + "__spearman.csv"
        corr.to_csv(corr_path)
        print(f"\nWrote: {args.output}\nWrote: {corr_path}")


if __name__ == "__main__":
    main()
