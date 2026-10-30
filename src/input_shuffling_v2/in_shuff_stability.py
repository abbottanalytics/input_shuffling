"""Stability of importance rankings under resampling.

For the shuffle-based importance ranking, stability is assessed by
bootstrap-resampling *records*: per-record dispersion stats are re-aggregated
over each bootstrap sample, giving a full re-ranking per resample without any
rescoring.

For the metric_drop ranking, per-shuffle metric values are bootstrap-resampled
over *shuffles* (row-level bootstrap would require storing every shuffled
prediction vector).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .in_shuff_engine import FeatureResult, seed_context_for

logger = logging.getLogger("input_shuffling_v2")

# ranking_stat -> (record stat key, aggregation)
_RANKING_STAT_MAP = {
    "mean_record_variance": ("variance", np.mean),
    "median_record_variance": ("variance", np.median),
    "mean_record_std": ("std", np.mean),
    "median_record_std": ("std", np.median),
    "mean_record_iqr": ("iqr", np.mean),
    "median_record_iqr": ("iqr", np.median),
    "mean_record_mad": ("mad", np.mean),
    "median_record_mad": ("mad", np.median),
    "mean_abs_delta": ("mean_abs_delta", np.mean),
    "median_abs_delta": ("mean_abs_delta", np.median),
    "p90_abs_delta": ("mean_abs_delta", lambda a, **kw: np.quantile(a, 0.90, **kw)),
    "fraction_abs_delta_gt_eps": ("frac_gt_eps", np.mean),
}


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Rank 1 = largest value."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _bootstrap_importance_ranks(
    results: list[FeatureResult], cfg: dict, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (ranks B x p, values B x p, feature names)."""
    st = cfg["stability"]
    B = int((st.get("resampling") or {}).get("B", 100))
    frac = float(((st.get("resampling") or {}).get("bootstrap") or {}).get("sample_frac", 1.0))

    ranking_stat = cfg["importance"]["ranking_stat"]
    stat_key, agg = _RANKING_STAT_MAP[ranking_stat]

    features = [r.feature for r in results]
    # (n_rows x p) matrix of the relevant per-record stat
    M = np.column_stack([r.record_stats[stat_key] for r in results])
    n_rows = M.shape[0]
    n_sample = max(1, int(round(frac * n_rows)))

    ranks = np.empty((B, len(features)), dtype=int)
    values = np.empty((B, len(features)), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n_rows, size=n_sample)
        vals = agg(M[idx], axis=0)
        values[b] = vals
        ranks[b] = _ranks_desc(vals)
    return ranks, values, features


def _bootstrap_metric_drop_ranks(
    results: list[FeatureResult], baseline_metric_values: dict, cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    from .in_shuff_metrics import METRIC_HIGHER_IS_BETTER

    st = cfg["stability"]
    B = int((st.get("resampling") or {}).get("B", 100))
    metric = (cfg["metric_drop"].get("ranking_metric") or {}).get("name", "roc_auc")

    features = [r.feature for r in results]
    if not results or metric not in results[0].metric_values:
        return None
    baseline = baseline_metric_values[metric]
    sign = 1.0 if METRIC_HIGHER_IS_BETTER.get(metric, True) else -1.0
    # (n_shuffles x p) matrix of per-shuffle drops
    D = np.column_stack([sign * (baseline - r.metric_values[metric]) for r in results])
    n_shuffles = D.shape[0]

    ranks = np.empty((B, len(features)), dtype=int)
    values = np.empty((B, len(features)), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n_shuffles, size=n_shuffles)
        vals = D[idx].mean(axis=0)
        values[b] = vals
        ranks[b] = _ranks_desc(vals)
    return ranks, values, features


def _pairwise_spearman(ranks: np.ndarray) -> float:
    """Mean pairwise Spearman = mean off-diagonal Pearson correlation of rank rows."""
    corr = np.corrcoef(ranks)
    n = corr.shape[0]
    return float((corr.sum() - n) / (n * (n - 1)))


def _pairwise_kendall(ranks: np.ndarray, max_pairs: int = 2000,
                      rng: np.random.Generator | None = None) -> float:
    B = ranks.shape[0]
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if len(pairs) > max_pairs and rng is not None:
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[s] for s in sel]
    taus = [kendalltau(ranks[i], ranks[j]).statistic for i, j in pairs]
    return float(np.nanmean(taus))


def _pairwise_topk_jaccard(ranks: np.ndarray, k: int, bottom: bool = False) -> float:
    membership = (ranks > ranks.shape[1] - k) if bottom else (ranks <= k)
    m = membership.astype(float)
    inter = m @ m.T
    union = 2 * k - inter
    with np.errstate(invalid="ignore"):
        jac = inter / union
    n = m.shape[0]
    off_diag_sum = np.nansum(jac) - np.trace(jac)
    return float(off_diag_sum / (n * (n - 1)))


def _order_consistency(ranks: np.ndarray, consensus_rank: np.ndarray, Z: float, k: int) -> dict:
    B, p = ranks.shape
    # freq[i, j] = fraction of resamples where feature i outranks (is more important than) j
    freq = np.zeros((p, p))
    for b in range(B):
        freq += (ranks[b][:, None] < ranks[b][None, :])
    freq /= B

    iu = np.triu_indices(p, k=1)
    stable = (freq[iu] >= Z) | (freq[iu] <= 1 - Z)

    top_set = set(np.where(consensus_rank <= k)[0])
    bottom_set = set(np.where(consensus_rank > p - k)[0])
    top_pair = np.array([(i in top_set and j in top_set) for i, j in zip(*iu)])
    bottom_pair = np.array([(i in bottom_set and j in bottom_set) for i, j in zip(*iu)])

    out = {
        "pct_stable_pairs": float(stable.mean()),
        "pct_ambiguous_pairs": float(1 - stable.mean()),
    }
    if top_pair.any():
        out["pct_stable_top_pairs"] = float(stable[top_pair].mean())
    if bottom_pair.any():
        out["pct_stable_bottom_pairs"] = float(stable[bottom_pair].mean())
    return out


def _consensus(ranks: np.ndarray, values: np.ndarray, cfg: dict) -> np.ndarray:
    method = (cfg["stability"].get("consensus_ranking") or {}).get("method", "median_rank")
    central = np.median(ranks, axis=0) if method == "median_rank" else np.mean(ranks, axis=0)
    # Tie-break by mean ranking value (higher value -> better rank)
    mean_vals = values.mean(axis=0)
    order = np.lexsort((-mean_vals, central))
    consensus = np.empty(ranks.shape[1], dtype=int)
    consensus[order] = np.arange(1, ranks.shape[1] + 1)
    return consensus


def compute_stability(
    results: list[FeatureResult],
    baseline_metric_values: dict | None,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stability_summary, feature_rank_distribution)."""
    st = cfg["stability"]
    seed = int(cfg["run"]["seed"])
    metrics_cfg = st.get("metrics") or {}

    summary_rows, feature_rows = [], []

    for ranking_name in st.get("rankings_to_check", ["importance"]):
        rng = np.random.default_rng([seed, 987654321, seed_context_for(ranking_name)])

        if ranking_name == "importance":
            ranks, values, features = _bootstrap_importance_ranks(results, cfg, rng)
            resample_axis = "records"
        elif ranking_name == "metric_drop":
            if not cfg["metric_drop"]["enabled"] or baseline_metric_values is None:
                logger.info("stability: metric_drop ranking requested but metric_drop disabled; skipping")
                continue
            packed = _bootstrap_metric_drop_ranks(results, baseline_metric_values, cfg, rng)
            if packed is None:
                continue
            ranks, values, features = packed
            resample_axis = "shuffles"
        else:
            raise ValueError(f"Unknown ranking in stability.rankings_to_check: {ranking_name}")

        k_top = int((metrics_cfg.get("top_k_jaccard") or {}).get("k", 10))
        k_bottom = int((metrics_cfg.get("bottom_k_jaccard") or {}).get("k", 10))
        consensus = _consensus(ranks, values, cfg)

        def add(metric, value):
            summary_rows.append({
                "ranking": ranking_name, "resample_axis": resample_axis,
                "B": ranks.shape[0], "metric": metric, "value": value,
            })

        if (metrics_cfg.get("spearman") or {}).get("enabled", True):
            add("mean_pairwise_spearman", _pairwise_spearman(ranks))
        if (metrics_cfg.get("kendall_tau") or {}).get("enabled", True):
            add("mean_pairwise_kendall_tau", _pairwise_kendall(ranks, rng=rng))
        if (metrics_cfg.get("top_k_jaccard") or {}).get("enabled", True):
            add(f"mean_pairwise_top_{k_top}_jaccard", _pairwise_topk_jaccard(ranks, k_top))
        if (metrics_cfg.get("bottom_k_jaccard") or {}).get("enabled", True):
            add(f"mean_pairwise_bottom_{k_bottom}_jaccard",
                _pairwise_topk_jaccard(ranks, k_bottom, bottom=True))
        poc = metrics_cfg.get("pairwise_order_consistency") or {}
        if poc.get("enabled", True):
            Z = float(poc.get("Z", 0.90))
            for metric, value in _order_consistency(ranks, consensus, Z, k_top).items():
                add(f"{metric}_Z{Z:g}", value)

        for i, feature in enumerate(features):
            feature_rows.append({
                "ranking": ranking_name,
                "feature": feature,
                "consensus_rank": int(consensus[i]),
                "median_rank": float(np.median(ranks[:, i])),
                "mean_rank": float(np.mean(ranks[:, i])),
                "std_rank": float(np.std(ranks[:, i], ddof=1)) if ranks.shape[0] > 1 else 0.0,
                "min_rank": int(ranks[:, i].min()),
                "max_rank": int(ranks[:, i].max()),
                "p05_rank": float(np.quantile(ranks[:, i], 0.05)),
                "p95_rank": float(np.quantile(ranks[:, i], 0.95)),
                "mean_ranking_value": float(values[:, i].mean()),
            })

    summary = pd.DataFrame(summary_rows)
    feature_dist = pd.DataFrame(feature_rows)
    if not feature_dist.empty:
        feature_dist = feature_dist.sort_values(
            ["ranking", "consensus_rank"]).reset_index(drop=True)
    return summary, feature_dist
