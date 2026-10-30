"""Config loading, validation, and defaulting for input_shuffling_v2.

Accepts both the v0.5 and v0.6 YAML schemas used by the v1 tool. Missing
optional sections get sensible defaults so older configs run unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("input_shuffling_v2")

ALLOWED_RANKING_STATS = {
    "mean_record_variance",
    "median_record_variance",
    "mean_record_std",
    "median_record_std",
    "mean_record_iqr",
    "median_record_iqr",
    "mean_record_mad",
    "median_record_mad",
    "mean_abs_delta",
    "median_abs_delta",
    "p90_abs_delta",
    "fraction_abs_delta_gt_eps",
}

DEFAULT_RANKING_STAT = "mean_record_iqr"

# Built-in generic sklearn/joblib scorer, used when scoring.plugin is omitted
DEFAULT_PLUGIN = {
    "module": "input_shuffling_v2.plugins.in_shuff_sklearn_joblib",
    "function": "predict",
    "info_function": "model_info",
}


def _derive_prediction_column(cfg: dict) -> str:
    schema = cfg["schema"]
    target_col = schema.get("target_column")
    target_type = str(schema.get("target_type", "binary")).lower()
    if not target_col:
        raise ValueError("schema.target_column is required to derive the prediction column name")
    if target_type == "binary":
        return f"{target_col}_prob"
    return f"{target_col}_pred"


def _setdefaults(d: dict, defaults: dict) -> dict:
    """Recursively fill missing keys in d from defaults (in place)."""
    for k, v in defaults.items():
        if k not in d or d[k] is None:
            d[k] = v
        elif isinstance(v, dict) and isinstance(d[k], dict):
            _setdefaults(d[k], v)
    return d


def apply_defaults(cfg: dict) -> dict:
    _setdefaults(cfg, {
        "run": {"name": "input_shuffling", "output_dir": "runs", "seed": 12345},
        "sampling": {"use": "all", "sample_n": None, "sample_frac": None,
                     "stratify_by": None, "replace": False, "seed_offset": 0},
        "shuffling": {"n_shuffles": 50, "shuffle_scope": "global",
                      "shuffle_method": "permute", "seed_strategy": "deterministic"},
        "importance": {"enabled": True, "eps": 0.01, "ranking_stat": DEFAULT_RANKING_STAT},
        "scoring": {"baseline": {"use_existing_predictions": True,
                                  "verify_scorer": True,
                                  "scorer_mismatch": "error",
                                  "scorer_tolerance": 1e-6}},
        "metric_drop": {
            "enabled": False,
            "metrics": {
                "binary": ["roc_auc", "logloss", "f1", "accuracy", "lift_at_decile"],
                "regression": ["rmse", "mae", "r2"],
            },
            "thresholding": {"enabled": True, "method": "fixed_or_prior",
                             "fixed_threshold": None, "optimize_metric": "f1",
                             "search": {"grid_size": 200, "range": [0.0, 1.0]}},
            "ranking_metric": {"name": "roc_auc", "direction": "higher_is_better"},
            "lift_settings": {"decile": 1},
            "summary_stats": {"include": ["mean", "std", "se", "median", "iqr"]},
        },
        "record_outliers": {
            "enabled": False,
            "variability_stat": "iqr",
            "aggregation_across_features": "mean",
            "output_top_outliers": {
                "enabled": True, "top_n": 200, "include_meta": True,
                "include_top_features": {"enabled": True, "top_k": 5},
            },
        },
        "bucketing": {"enabled": False, "always_include_all": True,
                      "mode": "within_bucket",
                      "bucket_stats": {"enabled": True}, "specs": []},
        "stability": {
            "enabled": False,
            "resampling": {"method": "bootstrap", "B": 100,
                           "bootstrap": {"sample_frac": 1.0, "stratify_by": None}},
            "rankings_to_check": ["importance"],
            "metrics": {
                "top_k_jaccard": {"enabled": True, "k": 10},
                "bottom_k_jaccard": {"enabled": True, "k": 10},
                "spearman": {"enabled": True},
                "kendall_tau": {"enabled": True},
                "pairwise_order_consistency": {"enabled": True, "Z": 0.90},
            },
            "consensus_ranking": {"method": "median_rank", "tie_breaker": "mean_value"},
        },
        "reporting": {"write_parquet": True, "write_csv": True,
                      "write_markdown_report": True,
                      "report_sections": ["baseline_metrics", "baseline_target_distribution",
                                          "importance", "metric_drop", "record_outliers",
                                          "bucketing", "stability"],
                      "logging": {"level": "INFO"}},
        "performance": {"n_jobs": 1, "shuffle_batch": 10},
    })
    return cfg


def load_config(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")

    for section in ("schema", "data", "scoring"):
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")
    if "model" not in cfg["scoring"] or "path" not in cfg["scoring"]["model"]:
        raise ValueError(
            "scoring.model.path is required — input shuffling rescores the model on "
            "shuffled data, so a loadable model artifact is the one hard requirement.")

    # Default to the built-in generic sklearn scorer when no plugin is given.
    # (Only default the whole block: a partial custom plugin spec is an error,
    # not something to silently fill in.)
    if not cfg["scoring"].get("plugin"):
        logger.info("scoring.plugin omitted; using built-in generic plugin "
                    f"{DEFAULT_PLUGIN['module']}")
        cfg["scoring"]["plugin"] = dict(DEFAULT_PLUGIN)

    apply_defaults(cfg)
    schema = cfg["schema"]

    # NOTE: empty schema.feature_columns is allowed here; the CLI resolves it
    # via the scoring plugin's info function (model_info.json or model
    # introspection) after the plugin is loaded.

    # Auto-derive prediction column name if omitted
    if not schema.get("prediction_columns"):
        derived = _derive_prediction_column(cfg)
        schema["prediction_columns"] = [derived]
        logger.info(f"schema.prediction_columns is empty; derived prediction column: {derived}")

    if len(schema["prediction_columns"]) != 1:
        raise ValueError(
            "v2 supports exactly one prediction column per config. "
            f"Found {len(schema['prediction_columns'])}.")

    shuffling = cfg["shuffling"]
    supported_shuffle_options = {
        "shuffle_scope": ("global",),
        "shuffle_method": ("permute",),
        "seed_strategy": ("deterministic",),
    }
    for key, allowed in supported_shuffle_options.items():
        value = shuffling.get(key)
        if value not in allowed:
            raise ValueError(
                f"Unsupported shuffling.{key}={value!r}. "
                f"v2 currently supports only {allowed}.")
    if int(shuffling["n_shuffles"]) < 1:
        raise ValueError("shuffling.n_shuffles must be >= 1")

    if not cfg["importance"].get("enabled", True):
        raise ValueError(
            "importance.enabled=false is not supported by this CLI because the "
            "shuffle engine and downstream outputs require importance results.")

    ranking_stat = cfg["importance"]["ranking_stat"]
    if ranking_stat not in ALLOWED_RANKING_STATS:
        raise ValueError(
            f"Unsupported importance.ranking_stat={ranking_stat}. "
            f"Allowed: {sorted(ALLOWED_RANKING_STATS)}")

    mode = cfg["bucketing"].get("mode", "within_bucket")
    if mode not in {"within_bucket", "aggregate"}:
        raise ValueError(f"bucketing.mode must be within_bucket|aggregate, got {mode}")

    baseline_cfg = cfg["scoring"].get("baseline") or {}
    mismatch_action = str(baseline_cfg.get("scorer_mismatch", "error")).lower()
    if mismatch_action not in {"error", "warn"}:
        raise ValueError("scoring.baseline.scorer_mismatch must be 'error' or 'warn'")
    if float(baseline_cfg.get("scorer_tolerance", 1e-6)) < 0:
        raise ValueError("scoring.baseline.scorer_tolerance must be >= 0")

    target_type = str(schema.get("target_type", "binary")).lower()
    if cfg["metric_drop"]["enabled"] and target_type == "regression":
        ranking_metric = (cfg["metric_drop"].get("ranking_metric") or {}).get("name")
        if ranking_metric == "roc_auc":
            raise ValueError(
                "metric_drop.ranking_metric.name='roc_auc' is not valid for "
                "regression. Use rmse, mae, or r2.")

    if cfg["metric_drop"]["enabled"] and not schema.get("target_column"):
        raise ValueError("metric_drop.enabled=true requires schema.target_column")

    return cfg


def dump_config_used(cfg: dict, out_path: Path) -> None:
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
