# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the scoring.preprocessing hook (normalizer PMML applied before
scoring) and the knn_lookup plugin (training set as the model artifact).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from input_shuffling_v2.in_shuff_transforms import PMMLNormalizer  # noqa: E402


def write_normalizer_pmml(path: Path, ranges: dict[str, tuple[float, float]]) -> None:
    """Emit a KNIME-style min-max Normalizer PMML."""
    fields = "\n".join(
        f'''    <DerivedField name="{c}*" optype="continuous" dataType="double">
      <NormContinuous field="{c}">
        <LinearNorm orig="{lo}" norm="0.0"/>
        <LinearNorm orig="{hi}" norm="1.0"/>
      </NormContinuous>
    </DerivedField>''' for c, (lo, hi) in ranges.items())
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" version="4.2">
  <Header/>
  <DataDictionary numberOfFields="{len(ranges)}"/>
  <TransformationDictionary>
{fields}
  </TransformationDictionary>
</PMML>''')


# ---------------------------------------------------------------------------
# PMMLNormalizer
# ---------------------------------------------------------------------------

def test_normalizer_minmax_and_extrapolation(tmp_path):
    pmml = tmp_path / "norm.pmml"
    write_normalizer_pmml(pmml, {"a": (10.0, 20.0)})
    t = PMMLNormalizer(pmml)

    df = pd.DataFrame({"a": [10.0, 15.0, 20.0, 25.0, 5.0], "b": [1, 2, 3, 4, 5]})
    out = t.transform(df)
    np.testing.assert_allclose(out["a"], [0.0, 0.5, 1.0, 1.5, -0.5])  # extrapolates
    assert (out["b"] == df["b"]).all()          # untouched column
    assert (df["a"].iloc[0] == 10.0)            # original frame not mutated


# ---------------------------------------------------------------------------
# scoring.preprocessing end-to-end: model trained on normalized features,
# data file holds raw features (the KNIME MLP situation)
# ---------------------------------------------------------------------------

def test_preprocessing_hook_end_to_end(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from input_shuffling_v2.in_shuff_cli import main

    rng = np.random.default_rng(2)
    n = 1000
    raw = pd.DataFrame({
        "big": rng.uniform(1000, 5000, n),      # deliberately unscaled
        "small": rng.uniform(0, 1, n),
    })
    ranges = {c: (raw[c].min(), raw[c].max()) for c in raw.columns}
    scaled = raw.copy()
    for c, (lo, hi) in ranges.items():
        scaled[c] = (raw[c] - lo) / (hi - lo)

    y = (rng.random(n) < 1 / (1 + np.exp(-(3 * scaled["big"] - 1.5)))).astype(int)
    model = LogisticRegression().fit(scaled, y)   # trained on NORMALIZED features

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    norm_path = tmp_path / "normalizer.pmml"
    write_normalizer_pmml(norm_path, ranges)

    flat = raw.copy()                              # data file holds RAW features
    flat["y"] = y
    flat["y_prob"] = model.predict_proba(scaled)[:, 1]
    flat["record_id"] = np.arange(n)
    data_path = tmp_path / "flat.parquet"
    flat.to_parquet(data_path, index=False)

    cfg = {
        "run": {"name": "prep_hook", "output_dir": str(tmp_path / "runs")},
        "data": {"path": str(data_path)},
        "schema": {"id_column": "record_id", "target_column": "y",
                   "target_type": "binary",
                   "feature_columns": ["big", "small"],
                   "prediction_columns": ["y_prob"]},
        "shuffling": {"n_shuffles": 8},
        "scoring": {
            "model": {"path": str(model_path)},
            "preprocessing": [{"path": str(norm_path)}],
        },
        "reporting": {"write_parquet": False, "write_markdown_report": False,
                      "logging": {"level": "WARNING"}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = main(["--config", str(cfg_path)])
    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert imp["feature"].iloc[0] == "big"
    assert imp["ranking_value"].iloc[0] > 0


# ---------------------------------------------------------------------------
# knn_lookup plugin
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def knn_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("knn")
    rng = np.random.default_rng(9)
    n_train, n_test = 1500, 700

    def make(n):
        X = pd.DataFrame({"strong": rng.normal(size=n), "weak": rng.normal(size=n),
                          "noise": rng.normal(size=n)})
        y = (rng.random(n) < 1 / (1 + np.exp(-3 * X["strong"] - 0.8 * X["weak"]))).astype(int)
        return X, y

    X_train, y_train = make(n_train)
    train = X_train.copy()
    train["TARGET"] = y_train
    train_path = tmp / "train.parquet"
    train.to_parquet(train_path, index=False)

    spec = {
        "type": "knn",
        "train_data": str(train_path),
        "target_column": "TARGET",
        "positive_label": 1,
        "feature_columns": ["strong", "weak", "noise"],
        "n_neighbors": 25,
        "weights": "distance",
        "metric": "euclidean",
        "seed": 1,
    }
    spec_path = tmp / "knn_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    X_test, y_test = make(n_test)
    return tmp, spec_path, train, X_test, y_test


def test_knn_plugin_matches_sklearn(knn_setup):
    from sklearn.neighbors import KNeighborsClassifier
    from input_shuffling_v2.plugins.in_shuff_knn_lookup import predict, model_info

    tmp, spec_path, train, X_test, _ = knn_setup
    info = model_info(model_path=str(spec_path))
    assert info["feature_columns"] == ["strong", "weak", "noise"]
    assert info["n_reference_rows"] == len(train)

    ref = KNeighborsClassifier(n_neighbors=25, weights="distance").fit(
        train[["strong", "weak", "noise"]], train["TARGET"])
    expected = ref.predict_proba(X_test)[:, 1]
    p = predict(X_test, model_path=str(spec_path))
    np.testing.assert_allclose(p, expected)


def test_knn_end_to_end_shuffling(knn_setup):
    from input_shuffling_v2.in_shuff_cli import main

    tmp, spec_path, train, X_test, y_test = knn_setup
    flat = X_test.copy()
    flat["y"] = y_test
    flat["record_id"] = np.arange(len(X_test))
    data_path = tmp / "scored.parquet"
    flat.to_parquet(data_path, index=False)

    cfg = {
        "run": {"name": "knn_e2e", "output_dir": str(tmp / "runs")},
        "data": {"path": str(data_path)},
        "schema": {"id_column": "record_id", "target_column": "y",
                   "target_type": "binary", "feature_columns": [],
                   "prediction_columns": []},
        "shuffling": {"n_shuffles": 6},
        "scoring": {
            "plugin": {"module": "input_shuffling_v2.plugins.in_shuff_knn_lookup",
                       "function": "predict", "info_function": "model_info"},
            "model": {"path": str(spec_path)},
            "baseline": {"use_existing_predictions": False},  # no stored scores
        },
        "reporting": {"write_parquet": False, "write_markdown_report": False,
                      "logging": {"level": "WARNING"}},
    }
    cfg_path = tmp / "cfg_knn.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = main(["--config", str(cfg_path)])
    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert imp["feature"].iloc[0] == "strong"
    vals = imp.set_index("feature")["ranking_value"]
    assert vals["noise"] < vals["strong"]


def test_normalizer_chain_two_steps(tmp_path):
    """Dean's kNN chain: all vars to [0,1], then dummies to {0.3, 0.7}."""
    step1 = tmp_path / "all_to_0_1.pmml"
    write_normalizer_pmml(step1, {"cont": (0.0, 200.0), "dummy": (0.0, 1.0)})
    step2 = tmp_path / "dummies_to_03_07.pmml"
    write_normalizer_pmml(step2, {"dummy": (0.0, 1.0)})
    # rewrite step2 norms to 0.3/0.7
    step2.write_text(step2.read_text()
                     .replace('norm="0.0"', 'norm="0.3"')
                     .replace('norm="1.0"', 'norm="0.7"'))

    df = pd.DataFrame({"cont": [0.0, 100.0, 200.0], "dummy": [0.0, 1.0, 1.0]})
    out = df
    for t in (PMMLNormalizer(step1), PMMLNormalizer(step2)):
        out = t.transform(out)
    np.testing.assert_allclose(out["cont"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["dummy"], [0.3, 0.7, 0.7])
