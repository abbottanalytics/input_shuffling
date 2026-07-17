"""Scoring plugin for tests: a linear model whose coefficients live in a JSON file."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def predict(df_features: pd.DataFrame, *, model_path: str,
            feature_columns=None, clip_eps=None) -> np.ndarray:
    spec = json.loads(Path(model_path).read_text())
    cols = feature_columns or list(spec["coefs"].keys())
    p = np.full(len(df_features), float(spec.get("intercept", 0.0)))
    for c in cols:
        p = p + float(spec["coefs"][c]) * df_features[c].to_numpy(dtype=float)
    if clip_eps is not None:
        p = np.clip(p, clip_eps, 1 - clip_eps)
    return p


def model_info(*, model_path: str) -> dict:
    spec = json.loads(Path(model_path).read_text())
    return {"feature_columns": list(spec["coefs"].keys())}
