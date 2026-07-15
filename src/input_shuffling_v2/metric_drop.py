"""Classic permutation importance: metric degradation under shuffling.

For each feature and each shuffle, the model metric (AUC, logloss, ...) is
recomputed on the shuffled predictions. The drop relative to the baseline
metric summarizes how much model performance depends on that feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import FeatureResult
from .metrics import METRIC_HIGHER_IS_BETTER


def build_metric_drop_table(
    results: list[FeatureResult],
    baseline_metric_values: dict,
    cfg: dict,
) -> pd.DataFrame:
    md = cfg["metric_drop"]
    pred_col = cfg["schema"]["prediction_columns"][0]
    include = (md.get("summary_stats") or {}).get("include", ["mean", "std", "se", "median", "iqr"])

    rows = []
    for r in results:
        for metric, values in r.metric_values.items():
            baseline = baseline_metric_values.get(metric)
            higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
            mean_val = float(np.mean(values))
            # Positive drop always means "shuffling hurt the model"
            drop = (baseline - mean_val) if higher_is_better else (mean_val - baseline)

            row = {
                "model": pred_col,
                "feature": r.feature,
                "metric": metric,
                "baseline_value": baseline,
                "shuffled_mean": mean_val,
                "drop": float(drop),
                "n_shuffles": len(values),
            }
            if "std" in include:
                row["shuffled_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if "se" in include:
                row["shuffled_se"] = (row.get("shuffled_std", float(np.std(values, ddof=1)))
                                      / np.sqrt(len(values))) if len(values) > 1 else 0.0
            if "median" in include:
                row["shuffled_median"] = float(np.median(values))
            if "iqr" in include:
                row["shuffled_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Rank features by the configured ranking metric's drop
    ranking_metric = (md.get("ranking_metric") or {}).get("name", "roc_auc")
    ranks = (
        out[out["metric"] == ranking_metric]
        .sort_values("drop", ascending=False)
        .reset_index(drop=True)
    )
    rank_map = {f: i + 1 for i, f in enumerate(ranks["feature"])}
    out["ranking_metric"] = ranking_metric
    out["rank"] = out["feature"].map(rank_map)
    return out.sort_values(["rank", "metric"]).reset_index(drop=True)
