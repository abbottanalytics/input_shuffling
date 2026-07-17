"""Core input-shuffling engine.

For each feature: permute its values n_shuffles times (all other columns
fixed), rescore the model, and reduce the (n_shuffles x n_rows) prediction
deltas to per-record dispersion statistics and per-feature summaries.

Performance notes
-----------------
* Shuffles are scored in batches: `shuffle_batch` shuffled copies of the
  dataset are stacked and scored with a single plugin call, amortizing
  per-call overhead (pipeline column transforms, etc.).
* Features can be scored in parallel processes (`performance.n_jobs`).
  Each worker re-imports the scoring plugin and loads the model once.
* Seeding is deterministic per (seed, context, feature_index, shuffle_index),
  so results are identical regardless of batching or parallelism.
"""
from __future__ import annotations

import logging
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("input_shuffling_v2")

# Per-record dispersion statistics computed across shuffles
RECORD_STAT_KEYS = ("variance", "std", "iqr", "mad", "mean_abs_delta", "frac_gt_eps")


@dataclass
class FeatureResult:
    feature: str
    summary: dict
    record_stats: dict = field(default_factory=dict)   # {stat: np.ndarray of shape (n_rows,)}
    metric_values: dict = field(default_factory=dict)  # {metric: np.ndarray of shape (n_shuffles,)}


def seed_context_for(label: str) -> int:
    """Stable integer context so bucket-level reshuffles get distinct streams."""
    return zlib.crc32(label.encode("utf-8"))


def _reduce_delta_matrix(delta: np.ndarray, eps: float) -> tuple[dict, dict]:
    """Reduce (M, N) delta matrix to per-record stats and a per-feature summary."""
    n_shuffles, _ = delta.shape

    if n_shuffles > 1:
        variance = np.var(delta, axis=0, ddof=1)
    else:
        variance = np.zeros(delta.shape[1], dtype=float)
    std = np.sqrt(variance)

    q75 = np.quantile(delta, 0.75, axis=0)
    q25 = np.quantile(delta, 0.25, axis=0)
    iqr = q75 - q25

    med = np.median(delta, axis=0)
    mad = np.median(np.abs(delta - med), axis=0)

    abs_delta = np.abs(delta)
    mean_abs_delta = abs_delta.mean(axis=0)
    frac_gt_eps = (abs_delta > eps).mean(axis=0)

    record_stats = {
        "variance": variance, "std": std, "iqr": iqr, "mad": mad,
        "mean_abs_delta": mean_abs_delta, "frac_gt_eps": frac_gt_eps,
    }

    pooled_abs = abs_delta.ravel()
    summary = {
        "mean_record_variance": float(np.mean(variance)),
        "median_record_variance": float(np.median(variance)),
        "mean_record_std": float(np.mean(std)),
        "median_record_std": float(np.median(std)),
        "mean_record_iqr": float(np.mean(iqr)),
        "median_record_iqr": float(np.median(iqr)),
        "mean_record_mad": float(np.mean(mad)),
        "median_record_mad": float(np.median(mad)),
        "mean_abs_delta": float(np.mean(pooled_abs)),
        "median_abs_delta": float(np.median(pooled_abs)),
        "p90_abs_delta": float(np.quantile(pooled_abs, 0.90)),
        "fraction_abs_delta_gt_eps": float(np.mean(pooled_abs > eps)),
        "n_rows": delta.shape[1],
        "n_shuffles": n_shuffles,
        "eps": eps,
    }
    return record_stats, summary


def shuffle_one_feature(
    X_base: pd.DataFrame,
    feature: str,
    feature_index: int,
    scorer,
    baseline: np.ndarray,
    *,
    n_shuffles: int,
    seed: int,
    seed_context: int,
    eps: float,
    shuffle_batch: int,
    metric_fns: dict | None = None,
    collect_record_stats: bool = True,
) -> FeatureResult:
    n_rows = len(X_base)
    col_values = X_base[feature].to_numpy()
    delta = np.empty((n_shuffles, n_rows), dtype=float)
    metric_values = {name: np.empty(n_shuffles, dtype=float) for name in (metric_fns or {})}

    batch = max(1, int(shuffle_batch))
    for start in range(0, n_shuffles, batch):
        ks = list(range(start, min(start + batch, n_shuffles)))

        shuffled_cols = [
            np.random.default_rng([seed, seed_context, feature_index, k]).permutation(col_values)
            for k in ks
        ]
        X_batch = pd.concat([X_base] * len(ks), ignore_index=True)
        X_batch[feature] = np.concatenate(shuffled_cols)

        p = scorer(X_batch).reshape(len(ks), n_rows)
        delta[ks[0]:ks[-1] + 1] = p - baseline[None, :]

        if metric_fns:
            for i, k in enumerate(ks):
                for name, fn in metric_fns.items():
                    metric_values[name][k] = fn(p[i])

    record_stats, summary = _reduce_delta_matrix(delta, eps)
    return FeatureResult(
        feature=feature,
        summary=summary,
        record_stats=record_stats if collect_record_stats else {},
        metric_values=metric_values,
    )


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def _parallel_worker(args: dict) -> list[FeatureResult]:
    """Runs in a child process: rebuild scorer + metric fns, score a feature chunk."""
    from .in_shuff_scoring import load_scoring_plugin, make_scorer
    from .in_shuff_metrics import make_metric_fns

    cfg = args["cfg"]
    plugin = load_scoring_plugin(cfg)
    scorer = make_scorer(cfg, plugin["predict_fn"])

    metric_fns = None
    if args["collect_metrics"]:
        metric_fns = make_metric_fns(cfg, args["y"], args["threshold"])

    results = []
    for feature, feature_index in args["feature_items"]:
        results.append(shuffle_one_feature(
            args["X_base"], feature, feature_index, scorer, args["baseline"],
            n_shuffles=args["n_shuffles"], seed=args["seed"],
            seed_context=args["seed_context"], eps=args["eps"],
            shuffle_batch=args["shuffle_batch"], metric_fns=metric_fns,
            collect_record_stats=args["collect_record_stats"],
        ))
    return results


def run_engine(
    X_base: pd.DataFrame,
    feature_items: list[tuple[str, int]],
    scorer,
    baseline: np.ndarray,
    cfg: dict,
    *,
    seed_context: int = 0,
    metric_fns: dict | None = None,
    y: np.ndarray | None = None,
    threshold: float | None = None,
    collect_record_stats: bool = True,
    progress_prefix: str = "",
) -> list[FeatureResult]:
    """Run the shuffle engine over `feature_items` = [(feature_name, stable_index), ...]."""
    n_shuffles = int(cfg["shuffling"]["n_shuffles"])
    seed = int(cfg["run"]["seed"])
    eps = float(cfg["importance"]["eps"])
    perf = cfg.get("performance") or {}
    shuffle_batch = int(perf.get("shuffle_batch", 10) or 10)
    n_jobs = int(perf.get("n_jobs", 1) or 1)

    X_base = X_base.reset_index(drop=True)
    total = len(feature_items)

    if n_jobs <= 1:
        results = []
        for j, (feature, feature_index) in enumerate(feature_items, start=1):
            logger.info(f"[shuffle]{progress_prefix} feature {j}/{total}: {feature}")
            results.append(shuffle_one_feature(
                X_base, feature, feature_index, scorer, baseline,
                n_shuffles=n_shuffles, seed=seed, seed_context=seed_context,
                eps=eps, shuffle_batch=shuffle_batch, metric_fns=metric_fns,
                collect_record_stats=collect_record_stats,
            ))
        return results

    # Parallel: split features into n_jobs chunks, one task per chunk
    chunks = [feature_items[i::n_jobs] for i in range(n_jobs)]
    chunks = [c for c in chunks if c]
    logger.info(f"[shuffle]{progress_prefix} scoring {total} features "
                f"in {len(chunks)} parallel workers (n_jobs={n_jobs})")
    task_args = [{
        "cfg": cfg, "X_base": X_base, "baseline": baseline,
        "feature_items": chunk, "n_shuffles": n_shuffles, "seed": seed,
        "seed_context": seed_context, "eps": eps, "shuffle_batch": shuffle_batch,
        "collect_metrics": metric_fns is not None, "y": y, "threshold": threshold,
        "collect_record_stats": collect_record_stats,
    } for chunk in chunks]

    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(pool.map(_parallel_worker, task_args))

    # Restore the original feature order
    by_name = {r.feature: r for results in chunk_results for r in results}
    return [by_name[feature] for feature, _ in feature_items]
