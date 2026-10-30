"""Input Shuffling v2 command-line entry point.

Usage:
  PYTHONPATH=input_shuffling_v2/src:. \
  python3 -m input_shuffling_v2.in_shuff_cli --config input_shuffling_v2/configs/<config>.yaml
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .in_shuff_bucketing import compute_bucketed_importance
from .in_shuff_config import dump_config_used, load_config
from .in_shuff_engine import run_engine
from .in_shuff_importance import build_feature_summary
from .in_shuff_io import (ensure_required_columns, infer_feature_columns, load_scored_data,
                 validate_prediction_scores, validate_schema_consistency)
from .in_shuff_metric_drop import build_metric_drop_table
from .in_shuff_metrics import (compute_baseline_metrics, compute_baseline_target_distribution,
                      compute_threshold, make_metric_fns)
from .in_shuff_record_outliers import compute_record_outliers
from .in_shuff_report import write_markdown_report
from .in_shuff_sampling import apply_sampling
from .in_shuff_scoring import load_scoring_plugin, make_scorer
from .in_shuff_stability import compute_stability


def setup_logger(cfg: dict) -> logging.Logger:
    level_name = ((cfg.get("reporting") or {}).get("logging") or {}).get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("input_shuffling_v2")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def make_run_dir(cfg: dict) -> Path:
    # __file__ = input_shuffling_v2/src/input_shuffling_v2/cli.py
    # parents[2] = the input_shuffling_v2 project root
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / cfg["run"]["output_dir"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{cfg['run']['name']}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write(df: pd.DataFrame | None, run_dir: Path, name: str, cfg: dict) -> None:
    if df is None or len(df) == 0:
        return
    rep = cfg["reporting"]
    if rep.get("write_csv", True):
        df.to_csv(run_dir / f"{name}.csv", index=False)
    if rep.get("write_parquet", False):
        try:
            df.to_parquet(run_dir / f"{name}.parquet", index=False)
        except Exception as exc:  # parquet writing is best-effort
            logging.getLogger("input_shuffling_v2").warning(
                f"Could not write {name}.parquet: {exc}")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Input shuffling variable importance (v2)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    t_start = time.time()
    cfg = load_config(Path(args.config))
    logger = setup_logger(cfg)
    run_dir = make_run_dir(cfg)
    logger.info(f"Run directory: {run_dir}")
    dump_config_used(cfg, run_dir / "config_used.yaml")

    schema = cfg["schema"]
    pred_col = schema["prediction_columns"][0]
    target_col = schema.get("target_column")

    # ---- Load data, resolve feature columns ------------------------------
    # Resolution order: explicit schema.feature_columns -> model metadata
    # (model_info.json / PMML MiningSchema / feature_names_in_) -> inference
    # from the data file (all columns not claimed by another role; see
    # schema.ignore_columns). Any other columns in the file are ignored.
    plugin = load_scoring_plugin(cfg)
    df = load_scored_data(cfg)

    if not schema.get("feature_columns"):
        model_path = cfg["scoring"]["model"]["path"]
        info = plugin["info_fn"](model_path=model_path) if plugin["info_fn"] else {}
        feature_cols = info.get("feature_columns")
        if feature_cols:
            logger.info(f"Resolved {len(feature_cols)} feature columns via scoring plugin")
        else:
            feature_cols = infer_feature_columns(df, cfg)
        schema["feature_columns"] = list(feature_cols)
        dump_config_used(cfg, run_dir / "config_used.yaml")
    feature_cols = schema["feature_columns"]

    validate_schema_consistency(cfg)
    scorer = make_scorer(cfg, plugin["predict_fn"])

    ensure_required_columns(df, cfg)
    df = apply_sampling(df, cfg).reset_index(drop=True)
    logger.info(
        f"Analysis set: {len(df):,} rows, {len(feature_cols)} features, "
        f"{cfg['shuffling']['n_shuffles']} shuffles, "
        f"ranking_stat={cfg['importance']['ranking_stat']}")

    baseline_cfg = cfg["scoring"].get("baseline", {})
    if baseline_cfg.get("use_existing_predictions", True):
        baseline = df[pred_col].to_numpy(dtype=float)
        validate_prediction_scores(baseline, cfg, source=f"prediction column '{pred_col}'")

        # Sanity check: the scoring plugin should reproduce the stored scores.
        # A mismatch usually means the wrong model file, a probability-vs-label
        # mixup, or a feature misalignment.
        if baseline_cfg.get("verify_scorer", True):
            n_check = min(len(df), 2000)
            recomputed = scorer(df[feature_cols].head(n_check))
            max_diff = float(np.max(np.abs(recomputed - baseline[:n_check])))
            tolerance = float(baseline_cfg.get("scorer_tolerance", 1e-6))
            mismatch_action = str(baseline_cfg.get("scorer_mismatch", "error")).lower()
            if mismatch_action not in {"error", "warn"}:
                raise ValueError(
                    "scoring.baseline.scorer_mismatch must be 'error' or 'warn'")
            if max_diff > tolerance:
                msg = (
                    f"Scorer consistency check: recomputed scores differ from the "
                    f"stored '{pred_col}' column (max |diff| = {max_diff:.6g} over "
                    f"{n_check} rows). Check that scoring.model.path is the model "
                    "that produced this column. Set "
                    "scoring.baseline.scorer_mismatch: warn to continue anyway, "
                    "or disable with scoring.baseline.verify_scorer: false.")
                if mismatch_action == "error":
                    raise ValueError(msg)
                logger.warning(msg)
            else:
                logger.info("Scorer consistency check passed "
                            f"(max |diff| = {max_diff:.2g} over {n_check} rows).")
    else:
        logger.info("Recomputing baseline predictions via scoring plugin...")
        baseline = scorer(df[feature_cols])
        validate_prediction_scores(baseline, cfg, source="recomputed baseline scores")
        # Make the recomputed scores available under the prediction column name
        # for baseline metrics, prediction bucketing, and outputs.
        df[pred_col] = baseline

    # ---- Baseline outputs ------------------------------------------------
    logger.info("Computing baseline target distribution and metrics...")
    baseline_target_dist = compute_baseline_target_distribution(df, cfg)
    _write(baseline_target_dist, run_dir, "baseline_target_distribution", cfg)
    baseline_metrics = compute_baseline_metrics(df, cfg)
    _write(baseline_metrics, run_dir, "baseline_metrics", cfg)

    # ---- Metric functions for metric_drop --------------------------------
    metric_fns = None
    y = None
    threshold = None
    baseline_metric_values = None
    if cfg["metric_drop"]["enabled"]:
        y = df[target_col].to_numpy()
        _, threshold, _, _ = compute_threshold(
            y, baseline, cfg["metric_drop"].get("thresholding") or {},
            positive_label=schema.get("positive_label", 1))
        metric_fns = make_metric_fns(cfg, y, threshold)
        baseline_metric_values = {name: fn(baseline) for name, fn in metric_fns.items()}
        logger.info(f"metric_drop enabled: {sorted(metric_fns)} (threshold={threshold:.6g})")

    # ---- Core shuffle engine ---------------------------------------------
    feature_items = [(f, i) for i, f in enumerate(feature_cols)]
    logger.info("Running shuffle engine on full analysis set...")
    results = run_engine(
        df[feature_cols], feature_items, scorer, baseline, cfg,
        metric_fns=metric_fns, y=y, threshold=threshold,
    )

    importance = build_feature_summary(results, cfg)
    _write(importance, run_dir, "importance_feature_summary", cfg)
    logger.info("Top 5 features by %s: %s", cfg["importance"]["ranking_stat"],
                ", ".join(importance["feature"].head(5)))

    # ---- Metric drop -------------------------------------------------------
    metric_drop_table = None
    if cfg["metric_drop"]["enabled"]:
        logger.info("Building metric_drop table...")
        metric_drop_table = build_metric_drop_table(results, baseline_metric_values, cfg)
        _write(metric_drop_table, run_dir, "metric_drop", cfg)

    # ---- Record outliers ---------------------------------------------------
    outliers_top = None
    if cfg["record_outliers"]["enabled"]:
        logger.info("Computing record-level instability outliers...")
        outliers_top, outliers_full = compute_record_outliers(df, results, cfg, baseline)
        _write(outliers_top, run_dir, "record_outliers", cfg)
        if cfg["reporting"].get("write_parquet", False):
            outliers_full.to_parquet(run_dir / "record_instability_full.parquet", index=False)

    # ---- Bucketed importance ------------------------------------------------
    bucketed = None
    if cfg["bucketing"]["enabled"]:
        logger.info(f"Computing bucketed importance (mode={cfg['bucketing'].get('mode')})...")
        bucketed = compute_bucketed_importance(
            df, cfg, feature_items, scorer, baseline, results)
        _write(bucketed, run_dir, "importance_feature_summary__bucketed", cfg)

    # ---- Stability ------------------------------------------------------------
    stability_summary = stability_ranks = None
    if cfg["stability"]["enabled"]:
        logger.info("Computing ranking stability under resampling...")
        stability_summary, stability_ranks = compute_stability(
            results, baseline_metric_values, cfg)
        _write(stability_summary, run_dir, "stability_summary", cfg)
        _write(stability_ranks, run_dir, "stability_feature_ranks", cfg)

    # ---- Report -------------------------------------------------------------
    if cfg["reporting"].get("write_markdown_report", True):
        path = write_markdown_report(
            run_dir, cfg,
            baseline_metrics=baseline_metrics,
            baseline_target_dist=baseline_target_dist,
            importance=importance,
            metric_drop_table=metric_drop_table,
            bucketed=bucketed,
            record_outliers_top=outliers_top,
            stability_summary=stability_summary,
            stability_ranks=stability_ranks,
        )
        logger.info(f"Wrote report: {path}")

    logger.info(f"Wrote outputs to: {run_dir} (elapsed {time.time() - t_start:.1f}s)")
    return run_dir


if __name__ == "__main__":
    main()
