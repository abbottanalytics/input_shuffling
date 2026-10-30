# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generic scoring plugin for any joblib/pickle sklearn-like model.

This is the default plugin when a config has no scoring.plugin section, so a
project needs nothing beyond a saved model file:

  - model.joblib (or .pkl): an estimator or Pipeline with predict_proba(),
    decision_function(), or predict()

Feature columns are resolved in this order:
  1. schema.feature_columns in the YAML (always wins if non-empty)
  2. model_info.json next to the model file (if present)
  3. the model's own feature_names_in_ attribute (sklearn >= 1.0)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

_CACHE: dict[str, Any] = {}


def _load_model(model_path: str) -> Any:
    resolved = str(Path(model_path).expanduser().resolve())
    if resolved not in _CACHE:
        if not Path(resolved).exists():
            raise FileNotFoundError(f"Model file not found: {resolved}")
        _CACHE[resolved] = joblib.load(resolved)
    return _CACHE[resolved]


def _feature_names_from_model(model: Any) -> list[str] | None:
    for obj in (model, *(getattr(model, "steps", []) and [model.steps[0][1]] or [])):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def model_info(*, model_path: str) -> dict:
    """Return model metadata: model_info.json if present, else model introspection."""
    info_path = Path(model_path).expanduser().resolve().parent / "model_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))

    model = _load_model(model_path)
    info: dict = {"source": "introspected", "model_class": type(model).__name__}
    names = _feature_names_from_model(model)
    if names:
        info["feature_columns"] = names
    return info


def predict(
    df_features: pd.DataFrame,
    *,
    model_path: str,
    feature_columns: Optional[Sequence[str]] = None,
    clip_eps: Optional[float] = None,
) -> np.ndarray:
    """Score and return a 1-D array: P(positive) for classifiers, else predictions."""
    model = _load_model(model_path)

    if feature_columns is None:
        feature_columns = model_info(model_path=model_path).get("feature_columns")
    if feature_columns:
        missing = [c for c in feature_columns if c not in df_features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for scoring: {missing}")
        X = df_features.loc[:, list(feature_columns)]
    else:
        X = df_features

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"predict_proba returned unexpected shape: {proba.shape}")
        p = proba[:, 1].astype(float)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        p = 1.0 / (1.0 + np.exp(-scores))
    elif hasattr(model, "predict"):
        if hasattr(model, "classes_"):
            raise TypeError(
                "Classifier has no predict_proba or decision_function; predict() "
                "would return hard class labels, which input shuffling cannot use — "
                "it needs a real-valued score with variance (probability, margin, "
                "etc.) to measure dispersion and build quantile buckets.")
        p = np.asarray(model.predict(X), dtype=float)
    else:
        raise TypeError(
            "Model does not support predict_proba, decision_function, or predict.")

    if clip_eps is not None:
        eps = float(clip_eps)
        p = np.clip(p, eps, 1.0 - eps)
    return p
