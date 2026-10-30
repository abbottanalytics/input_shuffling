# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bucketed (segment-level) importance.

Bucket generation supports prediction quantiles, top/bottom percent slices,
and categorical/quantile meta-column buckets (same spec format as v1).

Two computation modes (bucketing.mode):

* "within_bucket" (v1-compatible, default): re-run the shuffle engine inside
  each bucket, permuting feature values only among the bucket's rows. This
  preserves the bucket-conditional marginal distribution but costs a full
  engine pass per bucket.

* "aggregate": reuse the full-data per-record dispersion stats and aggregate
  them over each bucket's rows. Much faster (no rescoring) and answers
  "which globally-shuffled variables destabilize this segment most".
  In this mode median_abs_delta / p90_abs_delta are quantiles of the
  per-record mean |delta| within the bucket (not pooled across shuffles).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .in_shuff_engine import FeatureResult, run_engine, seed_context_for
from .in_shuff_importance import SUMMARY_COLUMNS, build_feature_summary

logger = logging.getLogger("input_shuffling_v2")

BUCKET_META_COLUMNS = [
    "bucket_spec", "bucket_by", "bucket_column", "bucket_id", "bucket_label",
    "n_rows_bucket", "bucket_value_column",
    "bucket_value_min", "bucket_value_max", "bucket_value_mean", "bucket_value_std",
    "bucket_value_p25", "bucket_value_median", "bucket_value_p75",
]


def _numeric_bucket_stats(series: pd.Series) -> dict:
    if len(series) == 0:
        return {k: None for k in BUCKET_META_COLUMNS[7:]}
    return {
        "bucket_value_min": float(series.min()),
        "bucket_value_max": float(series.max()),
        "bucket_value_mean": float(series.mean()),
        "bucket_value_std": float(series.std()),
        "bucket_value_p25": float(series.quantile(0.25)),
        "bucket_value_median": float(series.quantile(0.50)),
        "bucket_value_p75": float(series.quantile(0.75)),
    }


def generate_buckets(df: pd.DataFrame, cfg: dict, pred_col: str) -> list[dict]:
    if not cfg["bucketing"]["enabled"]:
        return []

    specs = cfg["bucketing"].get("specs", [])
    include_all = cfg["bucketing"].get("always_include_all", True)
    stats_enabled = (cfg["bucketing"].get("bucket_stats") or {}).get("enabled", True)

    buckets = []

    def add_bucket(spec_name, by, column, bucket_id, label, mask, value_series, value_column):
        stats = _numeric_bucket_stats(value_series.loc[mask]) if stats_enabled else {}
        buckets.append({
            "bucket_spec": spec_name, "bucket_by": by, "bucket_column": column,
            "bucket_id": str(bucket_id), "bucket_label": str(label),
            "n_rows_bucket": int(mask.sum()), "bucket_value_column": value_column,
            **stats, "mask": mask,
        })

    if include_all:
        add_bucket("all", "all", "all", "all", "all",
                   pd.Series(True, index=df.index), df[pred_col], pred_col)

    for spec in specs:
        name, by, scheme = spec["name"], spec["by"], spec["scheme"]

        if by == "prediction":
            s = df[pred_col]
            if scheme["type"] == "quantile":
                q = int(scheme["q"])
                labels = [f"Q{i + 1:02d}" for i in range(q)]
                b = pd.qcut(s.rank(method="first"), q=q, labels=labels)
                for bucket_id in labels:
                    add_bucket(name, by, pred_col, bucket_id, bucket_id,
                               (b == bucket_id), s, pred_col)
            elif scheme["type"] == "top_pct":
                pct = float(scheme["pct"])
                if scheme.get("include_ties", False):
                    mask = s >= s.quantile(1 - pct)
                else:
                    k = max(1, int(round(len(s) * pct)))
                    selected = s.sort_values(ascending=False, kind="stable").index[:k]
                    mask = pd.Series(False, index=df.index)
                    mask.loc[selected] = True
                add_bucket(name, by, pred_col, "top_pct", f"top_{pct:.0%}", mask, s, pred_col)
            elif scheme["type"] == "bottom_pct":
                pct = float(scheme["pct"])
                if scheme.get("include_ties", False):
                    mask = s <= s.quantile(pct)
                else:
                    k = max(1, int(round(len(s) * pct)))
                    selected = s.sort_values(ascending=True, kind="stable").index[:k]
                    mask = pd.Series(False, index=df.index)
                    mask.loc[selected] = True
                add_bucket(name, by, pred_col, "bottom_pct", f"bottom_{pct:.0%}", mask, s, pred_col)
            else:
                raise ValueError(f"Unsupported prediction bucket scheme: {scheme['type']}")

        elif by == "meta":
            for meta_col in spec["columns"]:
                s = df[meta_col]
                if scheme["type"] == "categories":
                    min_count = int(scheme.get("min_count", 1))
                    other_label = scheme.get("other_label", "__OTHER__")
                    vc = s.value_counts(dropna=False)
                    keep = set(vc[vc >= min_count].index)
                    collapsed = s.where(s.isin(keep), other_label).astype(str)
                    for bucket_id in collapsed.drop_duplicates():
                        mask = collapsed == bucket_id
                        # Summarize the model prediction inside categorical buckets
                        add_bucket(name, by, meta_col, bucket_id, bucket_id,
                                   mask, df[pred_col], pred_col)
                elif scheme["type"] == "quantile":
                    q = int(scheme["q"])
                    labels = [f"Q{i + 1:02d}" for i in range(q)]
                    b = pd.qcut(s.rank(method="first"), q=q, labels=labels)
                    for bucket_id in labels:
                        add_bucket(name, by, meta_col, bucket_id, bucket_id,
                                   (b == bucket_id), s, meta_col)
                else:
                    raise ValueError(f"Unsupported meta bucket scheme: {scheme['type']}")
        else:
            raise ValueError(f"Unsupported bucket 'by': {by}")

    logger.info(f"Generated {len(buckets)} buckets (including 'all' if enabled).")
    return buckets


def _aggregate_bucket_summary(
    results: list[FeatureResult], mask: np.ndarray, cfg: dict
) -> pd.DataFrame:
    """Fast bucketed summary: aggregate full-data per-record stats over bucket rows."""
    ranking_stat = cfg["importance"]["ranking_stat"]
    pred_col = cfg["schema"]["prediction_columns"][0]
    eps = float(cfg["importance"]["eps"])

    rows = []
    for r in results:
        rs = {k: v[mask] for k, v in r.record_stats.items()}
        row = {
            "model": pred_col, "feature": r.feature,
            "mean_record_variance": float(np.mean(rs["variance"])),
            "median_record_variance": float(np.median(rs["variance"])),
            "mean_record_std": float(np.mean(rs["std"])),
            "median_record_std": float(np.median(rs["std"])),
            "mean_record_iqr": float(np.mean(rs["iqr"])),
            "median_record_iqr": float(np.median(rs["iqr"])),
            "mean_record_mad": float(np.mean(rs["mad"])),
            "median_record_mad": float(np.median(rs["mad"])),
            "mean_abs_delta": float(np.mean(rs["mean_abs_delta"])),
            # Quantiles of per-record mean |delta| (see module docstring)
            "median_abs_delta": float(np.median(rs["mean_abs_delta"])),
            "p90_abs_delta": float(np.quantile(rs["mean_abs_delta"], 0.90)),
            "fraction_abs_delta_gt_eps": float(np.mean(rs["frac_gt_eps"])),
            "n_rows": int(mask.sum()),
            "n_shuffles": r.summary["n_shuffles"],
            "eps": eps,
        }
        row["ranking_stat"] = ranking_stat
        row["ranking_value"] = row[ranking_stat]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("ranking_value", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[SUMMARY_COLUMNS]


def compute_bucketed_importance(
    df: pd.DataFrame,
    cfg: dict,
    feature_items: list[tuple[str, int]],
    scorer,
    baseline: np.ndarray,
    full_results: list[FeatureResult],
) -> pd.DataFrame:
    """Compute the per-bucket importance table (excluding the 'all' bucket)."""
    pred_col = cfg["schema"]["prediction_columns"][0]
    mode = cfg["bucketing"].get("mode", "within_bucket")
    buckets = generate_buckets(df, cfg, pred_col=pred_col)

    bucketed_rows = []
    for bucket in buckets:
        if bucket["bucket_id"] == "all":
            continue
        mask = bucket["mask"]
        n_bucket = int(mask.sum())
        if n_bucket == 0:
            continue

        logger.info(
            f"Bucket {bucket['bucket_spec']} | {bucket['bucket_column']}={bucket['bucket_id']} "
            f"| rows={n_bucket:,} | mode={mode}")

        if mode == "aggregate":
            imp = _aggregate_bucket_summary(full_results, mask.to_numpy(), cfg)
        else:
            df_bucket = df.loc[mask]
            baseline_bucket = baseline[mask.to_numpy()]
            ctx = seed_context_for(
                f"{bucket['bucket_spec']}|{bucket['bucket_column']}|{bucket['bucket_id']}")
            bucket_results = run_engine(
                df_bucket[[f for f, _ in feature_items]], feature_items, scorer,
                baseline_bucket, cfg, seed_context=ctx, collect_record_stats=False,
                progress_prefix=f" [{bucket['bucket_spec']}:{bucket['bucket_id']}]",
            )
            imp = build_feature_summary(bucket_results, cfg)

        for col in BUCKET_META_COLUMNS:
            imp[col] = bucket.get(col)
        bucketed_rows.append(imp)

    if not bucketed_rows:
        return pd.DataFrame()

    out = pd.concat(bucketed_rows, ignore_index=True)
    return out[BUCKET_META_COLUMNS + SUMMARY_COLUMNS]
