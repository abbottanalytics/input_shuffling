"""End-to-end and unit validation on synthetic linear data.

The classic sanity check from the input-shuffling presentations: three
N(20, 5) inputs, target = 0.5*var1 + 0.2*var2 + 0.3*var3. Shuffling
influence must rank var1 > var3 > var2 (proportional to |coef|*std).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
TESTS = Path(__file__).resolve().parent
for p in (str(SRC), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from input_shuffling_v2.in_shuff_engine import _reduce_delta_matrix  # noqa: E402


COEFS = {"var1": 0.5, "var2": 0.2, "var3": 0.3}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Build synthetic data + config, run the CLI once, return the run dir."""
    tmp = tmp_path_factory.mktemp("synthetic")

    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({v: rng.normal(20, 5, n) for v in COEFS})
    pred = sum(c * df[v] for v, c in COEFS.items())
    df["target"] = pred + rng.normal(0, 0.1, n)
    df["target_pred"] = pred
    df["record_id"] = np.arange(n)
    df["segment"] = rng.choice(["A", "B"], n)
    data_path = tmp / "scored.parquet"
    df.to_parquet(data_path, index=False)

    model_path = tmp / "model.json"
    model_path.write_text(json.dumps({"coefs": COEFS, "intercept": 0.0}))

    cfg = {
        "version": "2.0",
        "run": {"name": "synthetic", "output_dir": str(tmp / "runs"), "seed": 42},
        "data": {"path": str(data_path), "format": "parquet"},
        "schema": {
            "id_column": "record_id",
            "target_column": "target",
            "target_type": "regression",
            "feature_columns": list(COEFS),
            "meta_columns": ["segment"],
            "prediction_columns": ["target_pred"],
        },
        "shuffling": {"n_shuffles": 20},
        "importance": {"enabled": True, "eps": 0.01, "ranking_stat": "mean_record_iqr"},
        "scoring": {
            "plugin": {"module": "in_shuff_synthetic_plugin", "function": "predict",
                       "info_function": "model_info"},
            "model": {"path": str(model_path)},
            "baseline": {"use_existing_predictions": True},
        },
        "metric_drop": {"enabled": True,
                        "metrics": {"regression": ["rmse", "mae", "r2"]},
                        "ranking_metric": {"name": "rmse"}},
        "record_outliers": {"enabled": True},
        "bucketing": {
            "enabled": True, "mode": "aggregate",
            "specs": [
                {"name": "pred_quartile", "by": "prediction", "columns": "ALL_MODELS",
                 "scheme": {"type": "quantile", "q": 4}},
                {"name": "by_segment", "by": "meta", "columns": ["segment"],
                 "scheme": {"type": "categories", "min_count": 10}},
            ],
        },
        "stability": {"enabled": True,
                      "resampling": {"method": "bootstrap", "B": 30},
                      "rankings_to_check": ["importance", "metric_drop"]},
        "performance": {"n_jobs": 1, "shuffle_batch": 7},
        "reporting": {"write_parquet": False, "write_csv": True,
                      "write_markdown_report": True,
                      "logging": {"level": "WARNING"}},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from input_shuffling_v2.in_shuff_cli import main
    return main(["--config", str(cfg_path)])


def test_importance_ranking_matches_coefficients(run_dir):
    imp = pd.read_csv(run_dir / "importance_feature_summary.csv")
    assert list(imp["feature"]) == ["var1", "var3", "var2"]
    assert imp["ranking_stat"].iloc[0] == "mean_record_iqr"
    # Influence should be roughly proportional to |coef| (equal stds):
    vals = imp.set_index("feature")["mean_record_std"]
    ratio_31 = vals["var3"] / vals["var1"]
    assert abs(ratio_31 - 0.6) < 0.05  # 0.3/0.5


def test_metric_drop_ranking(run_dir):
    md = pd.read_csv(run_dir / "metric_drop.csv")
    rmse = md[md["metric"] == "rmse"].sort_values("rank")
    assert list(rmse["feature"]) == ["var1", "var3", "var2"]
    assert (rmse["drop"] > 0).all()  # shuffling must degrade RMSE
    # r2 baseline should be ~1 for the exact model
    r2 = md[md["metric"] == "r2"]
    assert r2["baseline_value"].iloc[0] > 0.99


def test_bucketed_output(run_dir):
    b = pd.read_csv(run_dir / "importance_feature_summary__bucketed.csv")
    # 4 quartiles + 2 segments, 3 features each
    assert set(b["bucket_spec"]) == {"pred_quartile", "by_segment"}
    assert len(b) == (4 + 2) * 3
    # var1 should dominate every bucket
    top = b[b["rank"] == 1]
    assert (top["feature"] == "var1").all()


def test_record_outliers_output(run_dir):
    ro = pd.read_csv(run_dir / "record_outliers.csv")
    assert len(ro) == 200
    assert ro["record_instability"].is_monotonic_decreasing
    assert "record_id" in ro.columns
    assert ro["top_feature_1"].iloc[0] in {"var1", "var2", "var3"}


def test_stability_outputs(run_dir):
    ss = pd.read_csv(run_dir / "stability_summary.csv")
    assert {"importance", "metric_drop"} <= set(ss["ranking"])
    spearman = ss[(ss["ranking"] == "importance")
                  & (ss["metric"] == "mean_pairwise_spearman")]["value"].iloc[0]
    assert spearman > 0.95  # clean linear signal => very stable ranking

    fr = pd.read_csv(run_dir / "stability_feature_ranks.csv")
    imp_ranks = fr[fr["ranking"] == "importance"].set_index("feature")
    assert imp_ranks.loc["var1", "consensus_rank"] == 1


def test_report_written(run_dir):
    report = (run_dir / "report.md").read_text()
    assert "Feature Importance" in report
    assert "var1" in report


def test_reduce_delta_matrix_math():
    delta = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])  # M=3 shuffles, N=2 records
    record_stats, summary = _reduce_delta_matrix(delta, eps=0.5)
    assert record_stats["variance"][0] == pytest.approx(4.0)   # var([1,3,5], ddof=1)
    assert record_stats["iqr"][0] == pytest.approx(2.0)        # q75-q25 of [1,3,5]
    assert record_stats["mad"][0] == pytest.approx(2.0)
    assert record_stats["variance"][1] == 0.0
    assert summary["mean_abs_delta"] == pytest.approx(9.0 / 6.0)
    assert summary["fraction_abs_delta_gt_eps"] == pytest.approx(0.5)


def test_determinism_across_batch_sizes(run_dir, tmp_path):
    """Same seeds must give identical deltas regardless of shuffle_batch."""
    from input_shuffling_v2.in_shuff_engine import shuffle_one_feature
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"var1": rng.normal(size=200), "var2": rng.normal(size=200)})
    baseline = 0.5 * X["var1"].to_numpy() + 0.2 * X["var2"].to_numpy()

    def scorer(df):
        return 0.5 * df["var1"].to_numpy() + 0.2 * df["var2"].to_numpy()

    kw = dict(n_shuffles=9, seed=42, seed_context=0, eps=0.01)
    r1 = shuffle_one_feature(X, "var1", 0, scorer, baseline, shuffle_batch=1, **kw)
    r2 = shuffle_one_feature(X, "var1", 0, scorer, baseline, shuffle_batch=4, **kw)
    np.testing.assert_allclose(r1.record_stats["iqr"], r2.record_stats["iqr"])
    assert r1.summary["mean_record_std"] == pytest.approx(r2.summary["mean_record_std"])


def test_scorer_mismatch_raises_by_default(tmp_path):
    """Stored baseline scores must match the scoring plugin unless explicitly downgraded."""
    n = 100
    df = pd.DataFrame({
        "var1": np.linspace(0, 1, n),
        "var2": np.linspace(1, 2, n),
    })
    true_pred = 0.5 * df["var1"] + 0.2 * df["var2"]
    df["target"] = true_pred
    df["target_pred"] = true_pred + 0.25
    data_path = tmp_path / "bad_scored.parquet"
    df.to_parquet(data_path, index=False)

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"coefs": {"var1": 0.5, "var2": 0.2}}))
    cfg = {
        "run": {"name": "bad_baseline", "output_dir": str(tmp_path / "runs"), "seed": 42},
        "data": {"path": str(data_path), "format": "parquet"},
        "schema": {"target_column": "target", "target_type": "regression",
                   "feature_columns": ["var1", "var2"],
                   "prediction_columns": ["target_pred"]},
        "shuffling": {"n_shuffles": 2},
        "scoring": {"plugin": {"module": "in_shuff_synthetic_plugin",
                               "function": "predict",
                               "info_function": "model_info"},
                    "model": {"path": str(model_path)}},
        "metric_drop": {"enabled": False, "ranking_metric": {"name": "rmse"}},
        "reporting": {"write_parquet": False, "write_markdown_report": False,
                      "logging": {"level": "WARNING"}},
    }
    cfg_path = tmp_path / "bad_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from input_shuffling_v2.in_shuff_cli import main
    with pytest.raises(ValueError, match="Scorer consistency check"):
        main(["--config", str(cfg_path)])


def test_stability_seed_is_process_stable():
    from input_shuffling_v2.in_shuff_engine import FeatureResult
    from input_shuffling_v2.in_shuff_stability import compute_stability

    cfg = {
        "run": {"seed": 123},
        "importance": {"ranking_stat": "mean_record_iqr"},
        "metric_drop": {"enabled": False},
        "stability": {"enabled": True, "rankings_to_check": ["importance"],
                      "resampling": {"B": 20, "bootstrap": {"sample_frac": 1.0}},
                      "metrics": {"spearman": {"enabled": True},
                                  "kendall_tau": {"enabled": False},
                                  "top_k_jaccard": {"enabled": False},
                                  "bottom_k_jaccard": {"enabled": False},
                                  "pairwise_order_consistency": {"enabled": False}},
                      "consensus_ranking": {"method": "median_rank"}},
    }
    results = [
        FeatureResult("a", {}, {"iqr": np.array([3.0, 2.0, 1.0])}),
        FeatureResult("b", {}, {"iqr": np.array([2.0, 1.0, 0.5])}),
        FeatureResult("c", {}, {"iqr": np.array([1.0, 0.5, 0.1])}),
    ]

    s1, r1 = compute_stability(results, None, cfg)
    s2, r2 = compute_stability(results, None, cfg)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(r1, r2)
