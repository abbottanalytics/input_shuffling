"""Column-role handling and score validation.

Contract: one real-valued score column per model per config; every other
column in the data file is ignored unless given a role (feature, target, id,
meta, bucket). When features must be inferred from the data file,
schema.ignore_columns excludes non-inputs (e.g. other models' score columns).
"""
from __future__ import annotations

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

from input_shuffling_v2.in_shuff_io import validate_prediction_scores, validate_schema_consistency  # noqa: E402
from input_shuffling_v2.in_shuff_bucketing import generate_buckets  # noqa: E402


def _base_cfg(**schema_extra):
    return {"schema": {"target_column": "y", "target_type": "binary",
                       "feature_columns": ["a", "b"],
                       "prediction_columns": ["y_prob"], **schema_extra}}


# ---------------------------------------------------------------------------
# validate_prediction_scores
# ---------------------------------------------------------------------------

def test_label_column_rejected():
    labels = np.array([0.0, 1.0, 1.0, 0.0, 1.0] * 50)
    with pytest.raises(ValueError, match="class-label column"):
        validate_prediction_scores(labels, _base_cfg(), source="test")


def test_constant_scores_rejected():
    with pytest.raises(ValueError, match="constant"):
        validate_prediction_scores(np.full(100, 0.3), _base_cfg(), source="test")


def test_real_valued_probabilities_pass():
    rng = np.random.default_rng(0)
    validate_prediction_scores(rng.random(100), _base_cfg(), source="test")


def test_regression_binary_values_allowed():
    # {0,1}-valued predictions are only suspicious for classification
    cfg = _base_cfg(target_type="regression")
    cfg["schema"]["target_type"] = "regression"
    values = np.array([0.0, 1.0] * 50)
    validate_prediction_scores(values, cfg, source="test")  # no raise


# ---------------------------------------------------------------------------
# validate_schema_consistency
# ---------------------------------------------------------------------------

def test_prediction_column_as_feature_rejected():
    cfg = _base_cfg()
    cfg["schema"]["feature_columns"] = ["a", "y_prob"]
    with pytest.raises(ValueError, match="never be shuffled"):
        validate_schema_consistency(cfg)


def test_target_as_feature_rejected():
    cfg = _base_cfg()
    cfg["schema"]["feature_columns"] = ["a", "y"]
    with pytest.raises(ValueError, match="[Tt]arget"):
        validate_schema_consistency(cfg)


# ---------------------------------------------------------------------------
# End-to-end: multi-model flat file, inference + ignore_columns
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_file_setup(tmp_path_factory):
    """A KNIME-style flat file with two models' score columns, scored by a
    model saved WITHOUT feature names (numpy-fit), forcing inference."""
    from sklearn.linear_model import LogisticRegression

    tmp = tmp_path_factory.mktemp("colhandling")
    rng = np.random.default_rng(5)
    n = 800
    X = pd.DataFrame({"strong": rng.normal(size=n), "weak": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * X["strong"]))).astype(int)

    # Fit on numpy so the model has no feature_names_in_ -> inference kicks in
    model = LogisticRegression().fit(X.to_numpy(), y)
    model_path = tmp / "model.joblib"
    joblib.dump(model, model_path)

    flat = X.copy()
    flat["y"] = y
    flat["record_id"] = np.arange(n)
    flat["y_prob"] = model.predict_proba(X.to_numpy())[:, 1]
    flat["P_y1_other_model"] = rng.random(n)   # a second model's scores
    flat["analyst_notes"] = "check"            # junk column
    data_path = tmp / "flat.parquet"
    flat.to_parquet(data_path, index=False)
    return tmp, data_path, model_path


def _run_cfg(tmp, data_path, model_path, **schema_extra):
    return {
        "run": {"name": "colhandling", "output_dir": str(tmp / "runs")},
        "data": {"path": str(data_path)},
        "schema": {"id_column": "record_id", "target_column": "y",
                   "target_type": "binary", "feature_columns": [],
                   "prediction_columns": ["y_prob"], **schema_extra},
        "shuffling": {"n_shuffles": 6},
        "scoring": {"model": {"path": str(model_path)}},
        "reporting": {"write_parquet": False, "write_markdown_report": False,
                      "logging": {"level": "WARNING"}},
    }


def test_inference_with_ignore_columns(flat_file_setup):
    from input_shuffling_v2.in_shuff_cli import main

    tmp, data_path, model_path = flat_file_setup
    cfg = _run_cfg(tmp, data_path, model_path,
                   ignore_columns=["P_y1_other_model", "analyst_notes"])
    cfg_path = tmp / "cfg_ok.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = main(["--config", str(cfg_path)])
    used = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    assert sorted(used["schema"]["feature_columns"]) == ["strong", "weak"]

    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert list(imp["feature"]) == ["strong", "weak"]


def test_label_prediction_column_fails_fast(flat_file_setup):
    from input_shuffling_v2.in_shuff_cli import main

    tmp, data_path, model_path = flat_file_setup
    flat = pd.read_parquet(data_path)
    flat["y_label"] = (flat["y_prob"] >= 0.5).astype(float)
    bad_path = tmp / "flat_with_label.parquet"
    flat.to_parquet(bad_path, index=False)

    cfg = _run_cfg(tmp, bad_path, model_path,
                   prediction_columns=["y_label"],
                   ignore_columns=["P_y1_other_model", "analyst_notes", "y_prob"])
    cfg_path = tmp / "cfg_label.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="class-label column"):
        main(["--config", str(cfg_path)])


def test_prediction_pct_buckets_are_exact_size_by_default():
    df = pd.DataFrame({"score": [1.0] * 50 + list(np.linspace(0.49, 0.0, 50))})
    cfg = {
        "bucketing": {"enabled": True, "always_include_all": False,
                      "specs": [{"name": "top5", "by": "prediction",
                                 "scheme": {"type": "top_pct", "pct": 0.05}},
                                {"name": "bottom5", "by": "prediction",
                                 "scheme": {"type": "bottom_pct", "pct": 0.05}}]},
    }
    buckets = generate_buckets(df, cfg, pred_col="score")
    counts = {b["bucket_spec"]: b["n_rows_bucket"] for b in buckets}
    assert counts == {"top5": 5, "bottom5": 5}


def test_prediction_pct_buckets_can_include_ties():
    df = pd.DataFrame({"score": [1.0] * 50 + list(np.linspace(0.49, 0.0, 50))})
    cfg = {
        "bucketing": {"enabled": True, "always_include_all": False,
                      "specs": [{"name": "top5", "by": "prediction",
                                 "scheme": {"type": "top_pct", "pct": 0.05,
                                            "include_ties": True}}]},
    }
    buckets = generate_buckets(df, cfg, pred_col="score")
    assert buckets[0]["n_rows_bucket"] == 50
