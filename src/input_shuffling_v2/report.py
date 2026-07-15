"""Markdown run report (reporting.write_markdown_report)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _md_table(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = ".6g") -> str:
    """Small dependency-free markdown table renderer."""
    if df is None or df.empty:
        return "_(no data)_\n"
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return "" if v is None else str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    run_dir: Path,
    cfg: dict,
    *,
    baseline_metrics: pd.DataFrame | None = None,
    baseline_target_dist: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
    metric_drop_table: pd.DataFrame | None = None,
    bucketed: pd.DataFrame | None = None,
    record_outliers_top: pd.DataFrame | None = None,
    stability_summary: pd.DataFrame | None = None,
    stability_ranks: pd.DataFrame | None = None,
) -> Path:
    sections = cfg["reporting"].get("report_sections", [])
    schema = cfg["schema"]
    pred_col = schema["prediction_columns"][0]
    ranking_stat = cfg["importance"]["ranking_stat"]

    out = []
    out.append(f"# Input Shuffling Report — {cfg['run']['name']}\n")
    out.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"- Data: `{cfg['data']['path']}`")
    out.append(f"- Model: `{cfg['scoring']['model']['path']}`")
    out.append(f"- Prediction column: `{pred_col}`")
    out.append(f"- Shuffles per feature: {cfg['shuffling']['n_shuffles']}")
    out.append(f"- Ranking statistic: `{ranking_stat}`")
    out.append(f"- Seed: {cfg['run']['seed']}\n")

    if "baseline_metrics" in sections and baseline_metrics is not None and not baseline_metrics.empty:
        out.append("## Baseline Model Metrics\n")
        out.append(_md_table(baseline_metrics[["metric", "value", "n_rows"]]))

    if ("baseline_target_distribution" in sections
            and baseline_target_dist is not None and not baseline_target_dist.empty):
        out.append("## Target Distribution\n")
        out.append(_md_table(baseline_target_dist))

    if "importance" in sections and importance is not None and not importance.empty:
        out.append("## Feature Importance (Input Shuffling)\n")
        out.append(f"Features ranked by `{ranking_stat}` — the average per-record dispersion "
                   "of predictions under repeated shuffling of each input.\n")
        cols = ["rank", "feature", "ranking_value",
                "mean_record_iqr", "mean_record_std", "mean_record_variance",
                "mean_abs_delta", "fraction_abs_delta_gt_eps"]
        cols = [c for c in cols if c in importance.columns]
        out.append(_md_table(importance[cols], max_rows=25))

    if "metric_drop" in sections and metric_drop_table is not None and not metric_drop_table.empty:
        ranking_metric = metric_drop_table["ranking_metric"].iloc[0]
        out.append("## Metric Drop (Classic Permutation Importance)\n")
        out.append(f"Features ranked by mean `{ranking_metric}` degradation across shuffles.\n")
        pivot = metric_drop_table[metric_drop_table["metric"] == ranking_metric]
        cols = ["rank", "feature", "baseline_value", "shuffled_mean", "drop", "shuffled_std"]
        cols = [c for c in cols if c in pivot.columns]
        out.append(_md_table(pivot.sort_values("rank")[cols], max_rows=25))

    if "record_outliers" in sections and record_outliers_top is not None and not record_outliers_top.empty:
        out.append("## Record-Level Instability Outliers\n")
        out.append("Records whose predictions are most destabilized by shuffling "
                   "(candidates for review / data-quality checks).\n")
        show_cols = [c for c in record_outliers_top.columns
                     if not c.startswith("top_feature_") or c.endswith(("_1", "_2", "_3"))]
        out.append(_md_table(record_outliers_top[show_cols], max_rows=15))

    if "bucketing" in sections and bucketed is not None and not bucketed.empty:
        out.append("## Bucketed Importance\n")
        out.append(f"Top 3 features per bucket (mode: `{cfg['bucketing'].get('mode')}`).\n")
        top3 = (bucketed[bucketed["rank"] <= 3]
                [["bucket_spec", "bucket_id", "n_rows_bucket", "rank", "feature", "ranking_value"]])
        out.append(_md_table(top3, max_rows=80))

    if "stability" in sections and stability_summary is not None and not stability_summary.empty:
        out.append("## Ranking Stability\n")
        out.append(_md_table(stability_summary))
        if stability_ranks is not None and not stability_ranks.empty:
            out.append("\n### Rank Distribution (top 15 by consensus)\n")
            cols = ["ranking", "feature", "consensus_rank", "median_rank",
                    "std_rank", "p05_rank", "p95_rank"]
            out.append(_md_table(
                stability_ranks[stability_ranks["consensus_rank"] <= 15][cols], max_rows=40))

    path = run_dir / "report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
