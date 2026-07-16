"""Scoring plugin loading — same interface as v1.

A plugin module exposes:
  predict(df_features, *, model_path, feature_columns=None, clip_eps=None) -> np.ndarray
  model_info(*, model_path) -> dict   (optional)
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
import pandas as pd


def load_scoring_plugin(cfg: dict) -> dict:
    plugin_cfg = cfg["scoring"]["plugin"]
    module = importlib.import_module(plugin_cfg["module"])

    function_name = plugin_cfg["function"]
    if not hasattr(module, function_name):
        raise AttributeError(f"Scoring module {plugin_cfg['module']} has no function {function_name}")
    predict_fn: Callable[..., Any] = getattr(module, function_name)

    info_fn = None
    info_function_name = plugin_cfg.get("info_function")
    if info_function_name:
        if not hasattr(module, info_function_name):
            raise AttributeError(
                f"Scoring module {plugin_cfg['module']} has no function {info_function_name}")
        info_fn = getattr(module, info_function_name)

    return {"module": module, "predict_fn": predict_fn, "info_fn": info_fn}


def make_scorer(cfg: dict, predict_fn) -> Callable[[pd.DataFrame], np.ndarray]:
    """Bind config (model path, feature order, preprocessing, clipping, row
    batching) into one callable."""
    from .transforms import load_preprocessing

    schema = cfg["schema"]
    feature_cols = schema["feature_columns"]
    model_path = cfg["scoring"]["model"]["path"]

    pred_clip_cfg = schema.get("prediction_clip") or {}
    clip_eps = pred_clip_cfg.get("eps") if pred_clip_cfg.get("enabled", False) else None
    row_batch = cfg["scoring"].get("batch_size")
    transforms = load_preprocessing(cfg)

    def score(X: pd.DataFrame) -> np.ndarray:
        # Preprocessing (e.g. a KNIME Normalizer PMML) runs after shuffling,
        # before scoring: shuffle in raw units, score in model units.
        for t in transforms:
            X = t.transform(X)
        if row_batch:
            parts = []
            for start in range(0, len(X), int(row_batch)):
                parts.append(predict_fn(
                    X.iloc[start:start + int(row_batch)],
                    model_path=model_path, feature_columns=feature_cols, clip_eps=clip_eps))
            return np.concatenate(parts)
        return np.asarray(predict_fn(
            X, model_path=model_path, feature_columns=feature_cols, clip_eps=clip_eps),
            dtype=float)

    return score
