"""Data loading and schema checks."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("input_shuffling_v2")


def load_scored_data(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"]["path"])
    fmt = str(cfg["data"].get("format", "")).lower()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    cols = (cfg["data"].get("read_options") or {}).get("columns")
    logger.info(f"Loading scored data from {path}")

    if fmt == "parquet" or path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=cols)
    elif fmt == "csv" or path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=cols)
    else:
        raise ValueError(f"Unsupported data format: {fmt}")

    # Legacy compatibility: alias "score" -> derived prediction column name
    pred_col = cfg["schema"]["prediction_columns"][0]
    if pred_col not in df.columns and "score" in df.columns:
        logger.warning(
            f"Prediction column '{pred_col}' not found; using legacy 'score' column as '{pred_col}'.")
        df = df.copy()
        df[pred_col] = df["score"]

    logger.info(f"Loaded scored data shape: {df.shape}")
    return df


def validate_prediction_scores(scores, cfg: dict, source: str) -> None:
    """Enforce that model scores are real-valued with variance.

    Input shuffling ranks features by dispersion of scores, and buckets by
    score quantiles — both are undefined on hard {0,1} class labels. For
    classification the score column must be the probability (or any
    real-valued score), never the predicted label.
    """
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"{source}: {n_bad:,} missing/non-finite score values.")

    uniq = np.unique(scores)
    target_type = str(cfg["schema"].get("target_type", "binary")).lower()

    if len(uniq) == 1:
        raise ValueError(
            f"{source}: scores are constant ({uniq[0]}) — no variance to rank with.")

    if target_type in {"binary", "multiclass"} and set(uniq) <= {0.0, 1.0}:
        raise ValueError(
            f"{source}: scores contain only {{0, 1}} values — this looks like a "
            "predicted class-label column, not a probability. Input shuffling needs "
            "a real-valued score (thresholded labels lose the rank ordering that "
            "dispersion stats and quantile bucketing depend on). Point "
            "schema.prediction_columns at the model's probability/score column.")

    if len(uniq) < 5:
        logger.warning(
            f"{source}: only {len(uniq)} distinct score values — quantile buckets "
            "will be coarse (typical for heavily pruned trees).")


# Columns never treated as features when inferring from the data file
_STRUCTURAL_COLUMNS = {"score", "model_id", "is_test", "rownum"}


def infer_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Last-resort feature inference: every data column that is not the id,
    target, prediction, meta, bucketing, or an explicit ignore column.

    Used only when schema.feature_columns is empty AND the scoring plugin
    provides no feature list. List columns to exclude (e.g. other models'
    probability columns in a multi-model export) under schema.ignore_columns.
    """
    schema = cfg["schema"]
    excluded = set(_STRUCTURAL_COLUMNS)
    excluded.update(schema.get("prediction_columns") or [])
    excluded.update(schema.get("meta_columns") or [])
    excluded.update(schema.get("ignore_columns") or [])
    if schema.get("id_column"):
        excluded.add(schema["id_column"])
    if schema.get("target_column"):
        excluded.add(schema["target_column"])
    for spec in (cfg.get("bucketing") or {}).get("specs", []):
        if spec.get("by") == "meta":
            excluded.update(spec.get("columns", []))

    features = [c for c in df.columns if c not in excluded]
    if not features:
        raise ValueError(
            "Feature inference found no candidate columns. List the inputs "
            "explicitly under schema.feature_columns.")
    logger.warning(
        f"Inferred {len(features)} feature columns from the data file (no explicit "
        f"schema.feature_columns and no model metadata): {features}. "
        "Verify this list — add any non-input columns (e.g. other models' score "
        "columns) to schema.ignore_columns.")
    return features


def validate_schema_consistency(cfg: dict) -> None:
    """Catch column-role conflicts before any scoring happens."""
    schema = cfg["schema"]
    features = set(schema["feature_columns"])
    pred_cols = set(schema.get("prediction_columns") or [])
    target = schema.get("target_column")
    ignore = set(schema.get("ignore_columns") or [])

    overlap = features & pred_cols
    if overlap:
        raise ValueError(
            f"Prediction column(s) listed as features: {sorted(overlap)}. "
            "A model's score column must never be shuffled as an input.")
    if target and target in features:
        raise ValueError(
            f"Target column '{target}' is listed as a feature — remove it "
            "(shuffling the target is not input shuffling).")
    overlap = features & ignore
    if overlap:
        raise ValueError(
            f"Columns appear in both feature_columns and ignore_columns: "
            f"{sorted(overlap)} — resolve the conflict.")


def ensure_required_columns(df: pd.DataFrame, cfg: dict) -> None:
    schema = cfg["schema"]

    required = list(schema["feature_columns"])
    if cfg["scoring"].get("baseline", {}).get("use_existing_predictions", True):
        required.extend(schema["prediction_columns"])
    if schema.get("id_column"):
        required.append(schema["id_column"])
    if cfg["metric_drop"]["enabled"] or schema.get("target_column"):
        tgt = schema.get("target_column")
        if cfg["metric_drop"]["enabled"] and not tgt:
            raise ValueError("metric_drop.enabled=true requires schema.target_column")
        if tgt and tgt in df.columns:
            required.append(tgt)
    if cfg["bucketing"]["enabled"]:
        for spec in cfg["bucketing"].get("specs", []):
            if spec["by"] == "meta":
                required.extend(spec.get("columns", []))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    logger.info("Required columns check passed.")
