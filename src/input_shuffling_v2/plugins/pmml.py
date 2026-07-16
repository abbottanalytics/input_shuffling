"""Scoring plugin for PMML models (e.g. exported from KNIME).

Usage in a config:

  scoring:
    plugin:
      module: "input_shuffling_v2.plugins.pmml"
      function: "predict"
      info_function: "model_info"
    model:
      path: "path/to/model.pmml"

Works with any PMML the PMML4S engine supports: logistic/linear regression,
decision trees, random forest / GBM ensembles, neural networks, SVM, etc.
Requires the `pypmml` package (pip install pypmml) and a Java runtime —
both present wherever KNIME runs.

Output column selection:
  * classification PMML: probability of the positive class. The positive
    label defaults to "1"; override with the INPUT_SHUFFLING_POSITIVE_LABEL
    environment variable if your target uses different labels.
  * regression PMML: the predicted value.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

_CACHE: dict[str, Any] = {}


def _load_model(model_path: str):
    from pypmml import Model

    resolved = str(Path(model_path).expanduser().resolve())
    if resolved not in _CACHE:
        if not Path(resolved).exists():
            raise FileNotFoundError(f"PMML file not found: {resolved}")
        _CACHE[resolved] = Model.fromFile(resolved)
    return _CACHE[resolved]


def model_info(*, model_path: str) -> dict:
    model = _load_model(model_path)
    return {
        "source": "pmml",
        "model_class": model.modelElement,
        "function_name": model.functionName,        # classification | regression
        "feature_columns": list(model.inputNames),  # from the PMML MiningSchema
        "target_fields": list(model.targetNames),
        "output_fields": [f.name for f in model.outputFields],
    }


def _pick_output_column(model, result: pd.DataFrame) -> pd.Series:
    """Choose the score column from PMML4S output."""
    if model.functionName == "regression":
        for col in result.columns:
            if col.startswith("predicted") or col.startswith("Predicted"):
                return result[col]
        return result.iloc[:, 0]

    # classification: probability of the positive class.
    # PMML tools name this differently: "probability(1)" (PMML4S default),
    # "P (1)" (KNIME ensemble exports), etc. Try the declared OutputFields
    # first, then common name patterns.
    positive_label = os.environ.get("INPUT_SHUFFLING_POSITIVE_LABEL", "1")

    for f in model.outputFields:
        if (getattr(f, "feature", None) == "probability"
                and str(getattr(f, "value", "")) == positive_label
                and f.name in result.columns):
            return result[f.name]

    candidates = (
        f"probability({positive_label})", f"probability_{positive_label}",
        f"P ({positive_label})", f"P({positive_label})",
    )
    for cand in candidates:
        if cand in result.columns:
            return result[cand]

    prob_cols = [c for c in result.columns
                 if c.lower().startswith(("probability", "p (", "p("))]
    if len(prob_cols) == 1:
        return result[prob_cols[0]]
    raise ValueError(
        f"Could not identify the probability of class '{positive_label}' in PMML "
        f"output columns {list(result.columns)}. Set INPUT_SHUFFLING_POSITIVE_LABEL "
        "to the positive class label used in the PMML.")


def predict(
    df_features: pd.DataFrame,
    *,
    model_path: str,
    feature_columns: Optional[Sequence[str]] = None,
    clip_eps: Optional[float] = None,
) -> np.ndarray:
    model = _load_model(model_path)

    cols = list(feature_columns) if feature_columns else list(model.inputNames)
    missing = [c for c in cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for scoring: {missing}")

    result = model.predict(df_features.loc[:, cols])
    p = _pick_output_column(model, result).to_numpy(dtype=float)

    if clip_eps is not None:
        eps = float(clip_eps)
        p = np.clip(p, eps, 1.0 - eps)
    return p
