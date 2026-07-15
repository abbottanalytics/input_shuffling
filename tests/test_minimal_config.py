"""The minimal-infrastructure path: a bare joblib sklearn model + a data file,
no custom plugin, no model_info.json, no feature_columns in the YAML.
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


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from sklearn.linear_model import LogisticRegression

    tmp = tmp_path_factory.mktemp("minimal")
    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame({
        "strong": rng.normal(size=n),
        "weak": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    logit = 2.5 * X["strong"] + 0.6 * X["weak"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    model = LogisticRegression().fit(X, y)  # keeps feature_names_in_
    model_path = tmp / "model.joblib"
    joblib.dump(model, model_path)

    # Scored dataset: features + target + a legacy-style "score" column
    df = X.copy()
    df["y"] = y
    df["score"] = model.predict_proba(X)[:, 1]
    data_path = tmp / "scored.parquet"
    df.to_parquet(data_path, index=False)

    # The literal minimum config: data, schema basics, model path. No plugin,
    # no model_info.json, no feature_columns.
    cfg = {
        "run": {"name": "minimal", "output_dir": str(tmp / "runs")},
        "data": {"path": str(data_path)},
        "schema": {"target_column": "y", "target_type": "binary",
                   "feature_columns": [], "prediction_columns": []},
        "shuffling": {"n_shuffles": 10},
        "scoring": {"model": {"path": str(model_path)}},
        "reporting": {"write_parquet": False, "logging": {"level": "WARNING"}},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from input_shuffling_v2.cli import main
    return main(["--config", str(cfg_path)])


def test_runs_and_ranks_correctly(run_dir):
    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert list(imp["feature"]) == ["strong", "weak", "noise"]
    assert imp["ranking_stat"].iloc[0] == "mean_record_iqr"  # default kicked in


def test_feature_columns_resolved_from_model(run_dir):
    cfg_used = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    assert cfg_used["schema"]["feature_columns"] == ["strong", "weak", "noise"]
    assert "sklearn_joblib" in cfg_used["scoring"]["plugin"]["module"]


def test_legacy_score_alias_and_derived_pred_col(run_dir):
    cfg_used = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    assert cfg_used["schema"]["prediction_columns"] == ["y_prob"]
    # baseline metrics were computed from the aliased score column
    bm = pd.read_csv(run_dir / "baseline_metrics.csv")
    auc = bm[bm["metric"] == "roc_auc"]["value"].iloc[0]
    assert auc > 0.85


def test_report_written(run_dir):
    assert (run_dir / "report.md").exists()
