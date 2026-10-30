# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baseline metrics, target distribution, thresholding, and the metric registry
used by metric_drop (classic permutation importance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Thresholding (binary classification)
# ---------------------------------------------------------------------------

def compute_threshold(y: np.ndarray, scores: np.ndarray, threshold_cfg: dict, positive_label=1):
    """Return (method_used, threshold, computed_prior, optimize_metric_used)."""
    method = threshold_cfg.get("method", "fixed_or_prior")
    fixed_threshold = threshold_cfg.get("fixed_threshold")

    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("thresholding.method=fixed requires fixed_threshold")
        return "fixed", float(fixed_threshold), None, None

    prior = float(np.mean(y == positive_label))

    if method == "prior":
        return "prior", prior, prior, None

    if method == "fixed_or_prior":
        if fixed_threshold is not None:
            return "fixed", float(fixed_threshold), None, None
        return "prior", prior, prior, None

    if method == "maximize_metric":
        metric_name = threshold_cfg.get("optimize_metric", "f1")
        search = threshold_cfg.get("search") or {}
        grid_size = int(search.get("grid_size", 200))
        lo, hi = search.get("range", [0.0, 1.0])
        grid = np.linspace(float(lo), float(hi), grid_size)

        y_bin = (y == positive_label).astype(int)
        best_t, best_val = 0.5, -np.inf
        for t in grid:
            pred = (scores >= t).astype(int)
            if metric_name == "accuracy":
                val = accuracy_score(y_bin, pred)
            elif metric_name == "f1":
                val = f1_score(y_bin, pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported optimize_metric: {metric_name}")
            if val > best_val:
                best_val, best_t = val, float(t)
        return "maximize_metric", best_t, prior, metric_name

    raise ValueError(f"Unsupported threshold method: {method}")


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def lift_at_decile(y_bin: np.ndarray, scores: np.ndarray, decile: int = 1) -> float:
    """Cumulative lift in the top `decile` tenths of the score distribution.

    lift = response rate among the top decile*10% scored records
           / overall response rate.
    """
    n = len(scores)
    k = max(1, int(round(n * decile / 10.0)))
    order = np.argsort(-scores, kind="stable")
    top = y_bin[order[:k]]
    overall = y_bin.mean()
    if overall == 0:
        return float("nan")
    return float(top.mean() / overall)


def make_metric_fns(cfg: dict, y: np.ndarray, threshold: float) -> dict:
    """Return {metric_name: fn(scores) -> float} for the configured target type.

    `y` and `threshold` are bound so metric_drop can evaluate each shuffled
    prediction vector with a single call.
    """
    schema = cfg["schema"]
    target_type = str(schema.get("target_type", "binary")).lower()
    positive_label = schema.get("positive_label", 1)
    md = cfg["metric_drop"]
    names = (md.get("metrics") or {}).get(target_type, [])

    fns = {}
    if target_type == "binary":
        y_bin = (y == positive_label).astype(int)
        decile = int((md.get("lift_settings") or {}).get("decile", 1) or 1)
        registry = {
            "roc_auc": lambda s: float(roc_auc_score(y_bin, s)),
            "logloss": lambda s: float(log_loss(y_bin, np.clip(s, 1e-15, 1 - 1e-15))),
            "accuracy": lambda s: float(accuracy_score(y_bin, (s >= threshold).astype(int))),
            "f1": lambda s: float(f1_score(y_bin, (s >= threshold).astype(int), zero_division=0)),
            "lift_at_decile": lambda s: lift_at_decile(y_bin, s, decile=decile),
        }
    elif target_type == "regression":
        registry = {
            "rmse": lambda s: float(np.sqrt(mean_squared_error(y, s))),
            "mae": lambda s: float(mean_absolute_error(y, s)),
            "r2": lambda s: float(r2_score(y, s)),
        }
    else:
        raise NotImplementedError(f"metric_drop not implemented for target_type={target_type}")

    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown metric_drop metric '{name}' for target_type={target_type}")
        fns[name] = registry[name]
    return fns


# Direction: True if larger values indicate a better model
METRIC_HIGHER_IS_BETTER = {
    "roc_auc": True, "accuracy": True, "f1": True, "lift_at_decile": True, "r2": True,
    "logloss": False, "rmse": False, "mae": False,
}


# ---------------------------------------------------------------------------
# Baseline outputs
# ---------------------------------------------------------------------------

def compute_baseline_target_distribution(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    schema = cfg["schema"]
    target_col = schema.get("target_column")
    target_type = schema.get("target_type", "binary")

    if not target_col or target_col not in df.columns:
        return pd.DataFrame()

    y = df[target_col]
    if target_type in {"binary", "multiclass"}:
        counts = y.value_counts(dropna=False)
        return pd.DataFrame({
            "target_column": target_col,
            "target_type": target_type,
            "class_label": counts.index.astype(str),
            "n": counts.values,
            "frac": counts.values / len(df),
            "n_rows_total": len(df),
        })

    stats = {
        "n_rows": len(y),
        "missing_n": int(y.isna().sum()),
        "missing_frac": float(y.isna().mean()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "p25": float(y.quantile(0.25)),
        "median": float(y.quantile(0.50)),
        "p75": float(y.quantile(0.75)),
        "max": float(y.max()),
    }
    return pd.DataFrame({
        "target_column": target_col,
        "target_type": target_type,
        "stat": list(stats.keys()),
        "value": list(stats.values()),
    })


def compute_baseline_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    schema = cfg["schema"]
    pred_col = schema["prediction_columns"][0]
    target_col = schema.get("target_column")
    target_type = schema.get("target_type", "binary")
    positive_label = schema.get("positive_label", 1)

    if not target_col or target_col not in df.columns:
        return pd.DataFrame()

    y = df[target_col].to_numpy()
    scores = df[pred_col].to_numpy()
    rows = []

    def add(metric, value, method=None, threshold=None, prior=None, opt=None):
        rows.append({
            "model": pred_col, "metric": metric, "value": value, "n_rows": len(df),
            "threshold_method_used": method, "threshold_value_used": threshold,
            "computed_prior": prior, "optimize_metric_used": opt,
        })

    if target_type == "binary":
        y_bin = (y == positive_label).astype(int)
        add("roc_auc", float(roc_auc_score(y_bin, scores)))
        add("logloss", float(log_loss(y_bin, np.clip(scores, 1e-15, 1 - 1e-15))))

        threshold_cfg = cfg["metric_drop"].get("thresholding") or {}
        method_used, threshold, prior, opt = compute_threshold(
            y, scores, threshold_cfg, positive_label=positive_label)
        pred_bin = (scores >= threshold).astype(int)
        add("accuracy", float(accuracy_score(y_bin, pred_bin)), method_used, threshold, prior, opt)
        add("f1", float(f1_score(y_bin, pred_bin, zero_division=0)), method_used, threshold, prior, opt)

    elif target_type == "regression":
        add("rmse", float(np.sqrt(mean_squared_error(y, scores))))
        add("mae", float(mean_absolute_error(y, scores)))
        add("r2", float(r2_score(y, scores)))

    return pd.DataFrame(rows)
