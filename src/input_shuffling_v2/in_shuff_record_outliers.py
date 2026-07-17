"""Record-level instability: which records' predictions are most destabilized
by shuffling, and which features drive that instability.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .in_shuff_engine import FeatureResult

logger = logging.getLogger("input_shuffling_v2")

_STAT_KEY = {"variance": "variance", "std": "std", "iqr": "iqr", "mad": "mad"}
_AGG_FNS = {
    "mean": lambda m: m.mean(axis=1),
    "median": lambda m: m.median(axis=1),
    "max": lambda m: m.max(axis=1),
    "p90": lambda m: m.quantile(0.90, axis=1),
}


def compute_record_outliers(
    df: pd.DataFrame,
    results: list[FeatureResult],
    cfg: dict,
    baseline: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_outliers_df, full_instability_df).

    full_instability_df has one row per record: id, instability score, and the
    per-feature variability stat for every feature.
    """
    ro = cfg["record_outliers"]
    schema = cfg["schema"]
    stat = str(ro.get("variability_stat", "iqr")).lower()
    if stat not in _STAT_KEY:
        raise ValueError(f"Unsupported record_outliers.variability_stat: {stat}")
    agg = str(ro.get("aggregation_across_features", "mean")).lower()
    if agg not in _AGG_FNS:
        raise ValueError(f"Unsupported record_outliers.aggregation_across_features: {agg}")

    stat_matrix = pd.DataFrame({r.feature: r.record_stats[_STAT_KEY[stat]] for r in results})
    instability = _AGG_FNS[agg](stat_matrix)

    id_col = schema.get("id_column")
    pred_col = schema["prediction_columns"][0]
    target_col = schema.get("target_column")

    base = pd.DataFrame({
        "record_instability": instability,
        "instability_stat": stat,
        "instability_aggregation": agg,
        "baseline_prediction": baseline,
    })
    if id_col and id_col in df.columns:
        base.insert(0, id_col, df[id_col].to_numpy())
    if target_col and target_col in df.columns:
        base["target"] = df[target_col].to_numpy()

    full = pd.concat([base, stat_matrix.add_prefix(f"{stat}__")], axis=1)

    out_cfg = ro.get("output_top_outliers") or {}
    top_n = int(out_cfg.get("top_n", 200))
    top = full.sort_values("record_instability", ascending=False).head(top_n).copy()

    if out_cfg.get("include_meta", True):
        meta_cols = [c for c in (schema.get("meta_columns") or []) if c in df.columns]
        for c in meta_cols:
            top[c] = df[c].to_numpy()[top.index]

    tf_cfg = out_cfg.get("include_top_features") or {}
    if tf_cfg.get("enabled", True):
        top_k = min(int(tf_cfg.get("top_k", 5)), stat_matrix.shape[1])
        sub = stat_matrix.loc[top.index]
        order = np.argsort(-sub.to_numpy(), axis=1)[:, :top_k]
        features = np.array(stat_matrix.columns)
        for i in range(top_k):
            top[f"top_feature_{i + 1}"] = features[order[:, i]]
            top[f"top_feature_{i + 1}_{stat}"] = np.take_along_axis(
                sub.to_numpy(), order[:, i:i + 1], axis=1).ravel()

    top = top.drop(columns=[c for c in top.columns if c.startswith(f"{stat}__")])
    top.insert(len(base.columns) - (0 if id_col is None else 0), "outlier_rank",
               np.arange(1, len(top) + 1))

    logger.info(f"Record outliers: top {len(top)} of {len(full):,} records by {agg}({stat}).")
    return top.reset_index(drop=True), full
