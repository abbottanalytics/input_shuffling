"""KNIME-style workflow test: PMML models + one flat file with inputs,
target, and score columns for multiple models.

Simulates what KNIME exports: each model saved as PMML (PMML Writer node),
and a single table holding the model-ready inputs plus per-model
probability/prediction columns. One input-shuffling config per model,
each naming its own prediction column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pypmml = pytest.importorskip("pypmml")
pytest.importorskip("nyoka")


def _make_data(rng, n=1200):
    X = pd.DataFrame({
        "strong": rng.normal(size=n),
        "weak": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    signal = 2.0 * X["strong"] + 0.5 * X["weak"]
    return X, signal


@pytest.fixture(scope="module")
def classification_setup(tmp_path_factory):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from nyoka import skl_to_pmml

    tmp = tmp_path_factory.mktemp("pmml_clf")
    rng = np.random.default_rng(11)
    X, signal = _make_data(rng)
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-signal))).astype(int)
    features = list(X.columns)

    flat = X.copy()
    flat["y"] = y
    flat["record_id"] = np.arange(len(X))

    models = {
        "lr": Pipeline([("model", LogisticRegression())]),
        "tree": Pipeline([("model", DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=50, random_state=0))]),
    }
    paths = {}
    for name, pipe in models.items():
        pipe.fit(X, y)
        pmml_path = tmp / f"{name}.pmml"
        skl_to_pmml(pipe, features, "y", str(pmml_path))
        flat[f"P_y1_{name}"] = pipe.predict_proba(X)[:, 1]
        paths[name] = pmml_path

    data_path = tmp / "knime_export.parquet"
    flat.to_parquet(data_path, index=False)
    return tmp, data_path, paths, features


def _config(tmp, data_path, pmml_path, pred_col, features, target_type, name, **extra_schema):
    return {
        "run": {"name": name, "output_dir": str(tmp / "runs"), "seed": 7},
        "data": {"path": str(data_path)},
        "schema": {
            "id_column": "record_id",
            "target_column": "y",
            "target_type": target_type,
            "feature_columns": features,
            "prediction_columns": [pred_col],
            **extra_schema,
        },
        "shuffling": {"n_shuffles": 8},
        "importance": {"enabled": True, "eps": 0.01, "ranking_stat": "mean_record_iqr"},
        "scoring": {
            "plugin": {"module": "input_shuffling_v2.plugins.pmml",
                       "function": "predict", "info_function": "model_info"},
            "model": {"path": str(pmml_path)},
            "baseline": {"use_existing_predictions": True},
        },
        "reporting": {"write_parquet": False, "write_markdown_report": False,
                      "logging": {"level": "WARNING"}},
    }


def test_pmml_scores_match_sklearn(classification_setup):
    """pypmml must reproduce the stored (sklearn) probabilities."""
    from input_shuffling_v2.plugins.pmml import predict, model_info

    tmp, data_path, paths, features = classification_setup
    flat = pd.read_parquet(data_path)

    info = model_info(model_path=str(paths["lr"]))
    assert info["function_name"] == "classification"
    assert set(info["feature_columns"]) == set(features)

    p = predict(flat, model_path=str(paths["lr"]), feature_columns=features)
    np.testing.assert_allclose(p, flat["P_y1_lr"].to_numpy(), atol=1e-6)


@pytest.mark.parametrize("model_name", ["lr", "tree"])
def test_classification_multi_model_flat_file(classification_setup, model_name):
    """One flat file, several models: one config per model/prediction column."""
    from input_shuffling_v2.cli import main

    tmp, data_path, paths, features = classification_setup
    cfg = _config(tmp, data_path, paths[model_name], f"P_y1_{model_name}",
                  features, "binary", f"pmml_{model_name}", positive_label=1)
    cfg_path = tmp / f"config_{model_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = main(["--config", str(cfg_path)])
    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert imp["feature"].iloc[0] == "strong"
    assert imp.set_index("feature")["ranking_value"]["noise"] <= \
        imp.set_index("feature")["ranking_value"]["weak"]
    assert imp["model"].iloc[0] == f"P_y1_{model_name}"


def test_regression_pmml_flat_file(tmp_path_factory):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from nyoka import skl_to_pmml
    from input_shuffling_v2.cli import main

    tmp = tmp_path_factory.mktemp("pmml_reg")
    rng = np.random.default_rng(23)
    X, signal = _make_data(rng)
    y = signal + rng.normal(0, 0.3, len(X))
    features = list(X.columns)

    flat = X.copy()
    flat["y"] = y
    flat["record_id"] = np.arange(len(X))

    for name, pipe in {
        "lin": Pipeline([("model", LinearRegression())]),
        "tree": Pipeline([("model", DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=30, random_state=0))]),
    }.items():
        pipe.fit(X, y)
        skl_to_pmml(pipe, features, "y", str(tmp / f"{name}.pmml"))
        flat[f"pred_{name}"] = pipe.predict(X)

    data_path = tmp / "knime_export.parquet"
    flat.to_parquet(data_path, index=False)

    for name in ("lin", "tree"):
        cfg = _config(tmp, data_path, tmp / f"{name}.pmml", f"pred_{name}",
                      features, "regression", f"pmml_reg_{name}")
        cfg["importance"]["eps"] = 0.1
        cfg_path = tmp / f"config_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        run_dir = main(["--config", str(cfg_path)])
        imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
        assert imp["feature"].iloc[0] == "strong", f"{name}: {list(imp['feature'])}"
