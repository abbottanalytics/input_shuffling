# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run input shuffling on the sample donor data and check against the stored answers.

Trains three models on the TRAIN rows of sample_donor_data.csv, then runs input
shuffling on the TEST rows for each of them plus the true data-generating model
("oracle"). Results are compared with the expected answers in expected/.

    python examples/sample_donor/in_shuff_run_sample.py                   # run + check
    python examples/sample_donor/in_shuff_run_sample.py --write-expected  # regenerate answers

Outputs go to examples/sample_donor/output/ (not committed).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

try:
    import input_shuffling_v2  # noqa: F401  (pip-installed)
except ImportError:
    sys.path.insert(0, str(EXAMPLE_DIR.parents[1] / "src"))  # source checkout

from input_shuffling_v2.in_shuff_engine import run_engine
from input_shuffling_v2.in_shuff_importance import build_feature_summary, record_stat_matrix
from in_shuff_sample_data import FEATURE_COLUMNS, load_sample_data, true_response_probability

RANDOM_STATE = 2026
N_SHUFFLES = 50
TOP_FRACTION = 0.30
VALUE_DECIMALS = 8
REL_TOLERANCE = 1e-6

OUTPUT_DIR = EXAMPLE_DIR / "output"
EXPECTED_DIR = EXAMPLE_DIR / "expected"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models(train: pd.DataFrame) -> dict:
    """Fit the sample models and return {name: scorer(DataFrame) -> P(TARGET_B=1)}."""
    X, y = train[FEATURE_COLUMNS], train["TARGET_B"]
    fitted = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=1),
        "gbm": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
    }
    scorers = {"oracle": true_response_probability}
    for name, model in fitted.items():
        model.fit(X, y)
        scorers[name] = lambda df, m=model: m.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return scorers


# ---------------------------------------------------------------------------
# Input shuffling
# ---------------------------------------------------------------------------

def run_input_shuffling(name: str, scorer, test: pd.DataFrame) -> pd.DataFrame:
    """Shuffle every feature N_SHUFFLES times; return overall + top-decile importance."""
    cfg = {
        "run": {"seed": RANDOM_STATE},
        "shuffling": {"n_shuffles": N_SHUFFLES},
        "importance": {"eps": 0.01, "ranking_stat": "mean_record_iqr"},
        "performance": {"shuffle_batch": 10, "n_jobs": 1},
        "schema": {"prediction_columns": [name]},
    }
    X_base = test[FEATURE_COLUMNS].reset_index(drop=True)
    y = test["TARGET_B"].to_numpy()
    baseline = scorer(X_base)
    feature_items = [(f, i) for i, f in enumerate(FEATURE_COLUMNS)]

    results = run_engine(X_base, feature_items, scorer, baseline, cfg,
                         metric_fns={"roc_auc": lambda s: roc_auc_score(y, s)})

    summary = build_feature_summary(results, cfg)
    keep = ["feature", "rank", "mean_record_iqr", "median_record_iqr",
            "mean_record_std", "mean_abs_delta", "fraction_abs_delta_gt_eps"]
    out = summary[keep].copy()

    # Top-decile importance: same shuffles, restricted to the top 30% of baseline scores
    record_iqr = record_stat_matrix(results, "iqr")
    top_mask = baseline >= np.quantile(baseline, 1 - TOP_FRACTION)
    top_iqr = record_iqr[top_mask].mean()
    out["top30_mean_record_iqr"] = out["feature"].map(top_iqr)
    out["top30_rank"] = out["top30_mean_record_iqr"].rank(ascending=False, method="first").astype(int)

    # Secondary: AUC drop (classic permutation-importance view)
    baseline_auc = roc_auc_score(y, baseline)
    auc_drop = {r.feature: baseline_auc - r.metric_values["roc_auc"].mean() for r in results}
    out["auc_drop"] = out["feature"].map(auc_drop)

    out.insert(0, "model", name)
    out["baseline_auc"] = baseline_auc
    out["n_records"] = len(X_base)
    out["n_top30_records"] = int(top_mask.sum())
    out["n_shuffles"] = N_SHUFFLES
    num_cols = out.select_dtypes("float").columns
    out[num_cols] = out[num_cols].round(VALUE_DECIMALS)
    return out


# ---------------------------------------------------------------------------
# Comparison with expected answers
# ---------------------------------------------------------------------------

def compare_with_expected(name: str, actual: pd.DataFrame) -> tuple[bool, str]:
    path = EXPECTED_DIR / f"importance_{name}.csv"
    if not path.exists():
        return False, f"{name}: no expected file ({path.name})"
    expected = pd.read_csv(path)
    merged = expected.merge(actual, on="feature", suffixes=("_exp", "_act"))
    if len(merged) != len(expected) or len(merged) != len(actual):
        return False, f"{name}: feature lists differ"

    rank_mismatch = merged.loc[merged["rank_exp"] != merged["rank_act"], "feature"].tolist()
    rel = (np.abs(merged["mean_record_iqr_act"] - merged["mean_record_iqr_exp"])
           / np.maximum(np.abs(merged["mean_record_iqr_exp"]), 1e-12))
    max_rel = float(rel.max())
    if not rank_mismatch and max_rel <= REL_TOLERANCE:
        return True, f"{name}: MATCH (max relative diff in mean_record_iqr = {max_rel:.2g})"
    detail = f"ranks differ for {rank_mismatch}" if rank_mismatch else "ranks match"
    return False, f"{name}: DIFFERS — {detail}; max relative diff = {max_rel:.3g}"


def rank_comparison_table(all_results: pd.DataFrame) -> pd.DataFrame:
    """Overall rank of each feature for every model, ordered by the oracle ranking."""
    table = all_results.pivot(index="feature", columns="model", values="rank")
    order = ["oracle", "logistic", "random_forest", "gbm"]
    return table[order].sort_values("oracle")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-expected", action="store_true",
                        help="overwrite expected/ with this run's results")
    args = parser.parse_args(argv)

    df = load_sample_data()
    train, test = df[df["split_flag"] == "TRAIN"], df[df["split_flag"] == "TEST"]
    print(f"Sample data: {len(train)} TRAIN rows, {len(test)} TEST rows, "
          f"{len(FEATURE_COLUMNS)} features, {N_SHUFFLES} shuffles per feature")

    scorers = build_models(train)
    OUTPUT_DIR.mkdir(exist_ok=True)

    frames = []
    for name, scorer in scorers.items():
        t0 = time.perf_counter()
        result = run_input_shuffling(name, scorer, test)
        result.to_csv(OUTPUT_DIR / f"importance_{name}.csv", index=False)
        frames.append(result)
        print(f"  {name:<14} AUC {result['baseline_auc'].iloc[0]:.3f}   "
              f"shuffled in {time.perf_counter() - t0:.1f}s")

    all_results = pd.concat(frames, ignore_index=True)
    ranks = rank_comparison_table(all_results)
    ranks.to_csv(OUTPUT_DIR / "rank_comparison.csv")
    print("\nOverall rank by model (ranked by mean_record_iqr; oracle = true model):")
    print(ranks.to_string())

    if args.write_expected:
        EXPECTED_DIR.mkdir(exist_ok=True)
        for frame in frames:
            frame.to_csv(EXPECTED_DIR / f"importance_{frame['model'].iloc[0]}.csv", index=False)
        ranks.to_csv(EXPECTED_DIR / "rank_comparison.csv")
        meta = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                "random_state": RANDOM_STATE, "n_shuffles": N_SHUFFLES,
                "top_fraction": TOP_FRACTION}
        (EXPECTED_DIR / "expected_versions.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nWrote expected answers to {EXPECTED_DIR}")
        return 0

    print("\nComparison with expected answers:")
    all_ok = True
    for frame in frames:
        ok, msg = compare_with_expected(frame["model"].iloc[0], frame)
        all_ok &= ok
        print(f"  {msg}")
    if not all_ok:
        print("\nDifferences in random_forest/gbm can come from a different scikit-learn "
              "version; see expected/expected_versions.json.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
