"""Scoring plugin for k-nearest-neighbor models (e.g. KNIME's kNN node).

kNN has no exportable model equation — the "model" IS the training set plus
the distance settings. This plugin's model file is a small JSON spec:

  {
    "type": "knn",
    "train_data": "kddcup1998/data/raw/target_b_knime_train.csv",
    "target_column": "TARGET_B",
    "positive_label": 1,
    "feature_columns": null,          // null => all non-target numeric columns
    "n_neighbors": 21,                // KNIME's k
    "weights": "distance",            // "distance" if KNIME's 'weight by distance'
    "metric": "euclidean",
    "preprocessing_pmml": ["all_to_0_1.pmml", "dummies_to_03_07.pmml"],
                                      // optional but strongly recommended:
                                      // kNN is distance-based, so features
                                      // should be on a common scale. A single
                                      // path or an ordered list of PMMLs,
                                      // applied in order — identically to the
                                      // reference set (at load) and to queries.
    "train_data_normalized": false,   // true if train_data was exported AFTER
                                      // normalization: the chain then applies
                                      // to queries only.
    "reference_sample_n": null,       // optional: subsample the reference set
    "seed": 12345
  }

Point scoring.model.path at this JSON. Relative train_data /
preprocessing_pmml paths are resolved against the cwd first, then against
the spec file's directory.

Note: sklearn's distance weighting (1/d) may differ in detail from KNIME's;
expect close but not bit-identical probabilities — the runner's scorer
consistency check reports the actual agreement with your stored score column.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("input_shuffling_v2")

_CACHE: dict[str, dict] = {}


def _resolve(path_str: str, spec_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute() or p.exists():
        return p
    alt = spec_dir / p
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Path not found (tried cwd and spec dir): {path_str}")


def _load(model_path: str) -> dict:
    resolved = str(Path(model_path).expanduser().resolve())
    if resolved in _CACHE:
        return _CACHE[resolved]

    from sklearn.neighbors import KNeighborsClassifier

    spec_path = Path(resolved)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_dir = spec_path.parent

    train_path = _resolve(spec["train_data"], spec_dir)
    if train_path.suffix.lower() in {".parquet", ".pq"}:
        train = pd.read_parquet(train_path)
    else:
        train = pd.read_csv(train_path)

    target_col = spec["target_column"]
    positive_label = spec.get("positive_label", 1)
    feature_cols = spec.get("feature_columns")
    if not feature_cols:
        feature_cols = [c for c in train.columns
                        if c != target_col
                        and pd.api.types.is_numeric_dtype(train[c])]
        logger.info(f"knn_lookup: using {len(feature_cols)} numeric feature columns "
                    "from the training data (spec.feature_columns not set)")

    transforms = []
    pre = spec.get("preprocessing_pmml")
    if pre:
        from ..transforms import PMMLNormalizer
        paths = [pre] if isinstance(pre, str) else list(pre)
        transforms = [PMMLNormalizer(_resolve(p, spec_dir)) for p in paths]
        if spec.get("train_data_normalized", False):
            logger.info("knn_lookup: train_data is already normalized; "
                        "preprocessing chain applies to queries only")
        else:
            for t in transforms:
                train = t.transform(train)

    sample_n = spec.get("reference_sample_n")
    if sample_n and int(sample_n) < len(train):
        train = train.sample(n=int(sample_n), random_state=int(spec.get("seed", 12345)))
        logger.info(f"knn_lookup: reference set subsampled to {len(train):,} rows")

    X = train[feature_cols].to_numpy(dtype=float)
    y = (train[target_col] == positive_label).astype(int).to_numpy()
    if np.isnan(X).any():
        raise ValueError("knn_lookup: training features contain missing values — "
                         "impute before export, kNN distances are undefined on NaN.")

    knn = KNeighborsClassifier(
        n_neighbors=int(spec.get("n_neighbors", 5)),
        weights=spec.get("weights", "uniform"),
        metric=spec.get("metric", "euclidean"),
        n_jobs=-1,
    ).fit(X, y)
    pos_idx = int(np.where(knn.classes_ == 1)[0][0])

    entry = {"knn": knn, "feature_columns": feature_cols, "transforms": transforms,
             "pos_idx": pos_idx, "n_reference": len(train), "spec": spec}
    _CACHE[resolved] = entry
    logger.info(f"knn_lookup: fitted k={knn.n_neighbors} ({spec.get('weights','uniform')} "
                f"weights) on {len(train):,} reference rows, {len(feature_cols)} features")
    return entry


def model_info(*, model_path: str) -> dict:
    entry = _load(model_path)
    return {
        "source": "knn_lookup",
        "model_class": "KNeighborsClassifier",
        "feature_columns": list(entry["feature_columns"]),
        "n_reference_rows": entry["n_reference"],
        "n_neighbors": entry["knn"].n_neighbors,
        "weights": entry["spec"].get("weights", "uniform"),
    }


def predict(
    df_features: pd.DataFrame,
    *,
    model_path: str,
    feature_columns: Optional[Sequence[str]] = None,
    clip_eps: Optional[float] = None,
) -> np.ndarray:
    entry = _load(model_path)
    cols = list(feature_columns) if feature_columns else entry["feature_columns"]
    missing = [c for c in cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for scoring: {missing}")

    X = df_features
    for t in entry["transforms"]:
        X = t.transform(X)
    X = X.loc[:, cols].to_numpy(dtype=float)

    p = entry["knn"].predict_proba(X)[:, entry["pos_idx"]].astype(float)
    if clip_eps is not None:
        eps = float(clip_eps)
        p = np.clip(p, eps, 1.0 - eps)
    return p
