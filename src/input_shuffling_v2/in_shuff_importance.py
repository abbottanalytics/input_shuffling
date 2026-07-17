"""Build the feature-importance summary table from engine results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .in_shuff_engine import FeatureResult

SUMMARY_COLUMNS = [
    "model", "feature", "rank", "ranking_stat", "ranking_value",
    "mean_record_variance", "median_record_variance",
    "mean_record_std", "median_record_std",
    "mean_record_iqr", "median_record_iqr",
    "mean_record_mad", "median_record_mad",
    "mean_abs_delta", "median_abs_delta", "p90_abs_delta",
    "fraction_abs_delta_gt_eps",
    "n_rows", "n_shuffles", "eps",
]


def build_feature_summary(results: list[FeatureResult], cfg: dict) -> pd.DataFrame:
    ranking_stat = cfg["importance"]["ranking_stat"]
    pred_col = cfg["schema"]["prediction_columns"][0]

    rows = []
    for r in results:
        row = {"model": pred_col, "feature": r.feature, **r.summary}
        row["ranking_stat"] = ranking_stat
        row["ranking_value"] = row[ranking_stat]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("ranking_value", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[SUMMARY_COLUMNS]


def record_stat_matrix(results: list[FeatureResult], stat: str) -> pd.DataFrame:
    """(n_rows x n_features) matrix of a per-record dispersion stat."""
    return pd.DataFrame({r.feature: r.record_stats[stat] for r in results})
