# Input Shuffling v2

Model-agnostic variable importance via input shuffling — a clean rewrite of the
v1 `input_shuffling` tool implementing the full v0.6 config specification.

## The algorithm

1. Score the model on the analysis set (baseline predictions).
2. For each input variable: randomly permute its values M times (all other
   columns fixed), rescore after each shuffle.
3. For each record, compute the dispersion of predictions across shuffles
   (IQR, std dev, variance, MAD).
4. Aggregate dispersion across records → one influence number per variable.
5. Rank descending.

The default ranking statistic is **`mean_record_iqr`** (IQR as primary metric,
SD as complementary diagnostic — see the book-chapter recommendation).

## Install

```bash
git clone https://github.com/abbottanalytics/input_shuffling.git
cd input_shuffling
pip install -e .            # add [pmml] for KNIME/PMML models, [parquet] for parquet I/O, [dev] for pytest
```

## Try it: sample data with known answers

```bash
python examples/sample_donor/in_shuff_run_sample.py
```

Synthetic donor data with a known true model, three fitted models
(logistic, random forest, GBM), and the expected importance rankings. See
[examples/sample_donor/README.md](examples/sample_donor/README.md).

## Use from Python (notebooks)

Any callable that maps a DataFrame to a 1-D array of scores can be shuffled —
a fitted model, a pipeline, or a combined score such as
P(response) × predicted amount:

```python
from input_shuffling_v2.in_shuff_engine import run_engine
from input_shuffling_v2.in_shuff_importance import build_feature_summary, record_stat_matrix

cfg = {
    "run": {"seed": 42},
    "shuffling": {"n_shuffles": 50},
    "importance": {"eps": 0.01, "ranking_stat": "mean_record_iqr"},
    "performance": {"shuffle_batch": 10, "n_jobs": 1},
    "schema": {"prediction_columns": ["score"]},
}
scorer = lambda df: model.predict_proba(df[features])[:, 1]
X = test_df[features].reset_index(drop=True)
baseline = scorer(X)

results = run_engine(X, [(f, i) for i, f in enumerate(features)], scorer, baseline, cfg)
importance = build_feature_summary(results, cfg)      # ranked table
record_iqr = record_stat_matrix(results, "iqr")       # records x features
```

- `n_jobs` must be 1 with an in-memory scorer (parallel workers rebuild the
  scorer from the config's plugin).
- Pass `metric_fns={"name": fn}` (fn: scores → float) to get a per-shuffle
  metric in each result's `metric_values`, e.g. AUC or net revenue.
- Importance for a segment (e.g. top 30% of scores) is the mean of
  `record_iqr` over that segment's rows.

## Command line (config-driven)

```bash
python -m input_shuffling_v2.in_shuff_cli --config configs/minimal_example.yaml
```

Relative `run.output_dir` resolves against the repository root in a source
checkout, else against the current directory. Outputs land in
`<output_dir>/<name>__<timestamp>/`.

The other files in `configs/` are the author's project configs and point at
data that is not in this repository; use them as templates.
`run_input_shuffling_v2.sh` is the author's wrapper for that layout.

## What v2 adds over v1

| Capability | v1 | v2 |
|---|---|---|
| Shuffle importance (record variance/IQR of deltas) | yes | yes, + std and MAD |
| Baseline metrics / target distribution | yes | yes |
| Bucketed importance | yes (reshuffle per bucket) | two modes: `within_bucket` (v1-compatible) and `aggregate` (no rescoring — fast) |
| `metric_drop` (classic permutation importance: AUC/logloss/F1/accuracy/lift drops) | config stub | implemented |
| `record_outliers` (most shuffle-unstable records + driving features) | config stub | implemented |
| `stability` (bootstrap rank stability: Spearman, Kendall, top/bottom-k Jaccard, pairwise order consistency, consensus ranking) | config stub | implemented |
| `sampling` (sample_n / sample_frac, stratified) | config stub | implemented |
| Markdown run report (`report.md`) | config flag only | implemented |
| Parquet outputs | config flag only | implemented |
| Performance | one scoring call per shuffle | batched scoring (`performance.shuffle_batch`), optional parallel features (`performance.n_jobs`) |
| Determinism | seed per feature | seed per (feature, shuffle) — identical results regardless of batching/parallelism |

## Outputs

- `importance_feature_summary.csv` — the main ranking (same columns as v1 plus
  record std/MAD stats)
- `importance_feature_summary__bucketed.csv` — per-segment rankings
- `metric_drop.csv` — metric degradation per feature × metric, ranked by the
  configured metric's drop
- `record_outliers.csv` — top-N most unstable records with their top driving
  features; `record_instability_full.parquet` has all records
- `stability_summary.csv` / `stability_feature_ranks.csv` — how reproducible
  the ranking is under bootstrap resampling
- `baseline_metrics.csv`, `baseline_target_distribution.csv`, `config_used.yaml`
- `report.md` — human-readable summary of all of the above

## Minimum requirements (no infrastructure needed)

Input shuffling has exactly two hard requirements:

1. **A dataset with the feature columns** (baseline scores can be a column in
   the data, or recomputed from the model).
2. **A model artifact that can score a DataFrame** — the algorithm shuffles a
   column and *rescores* the model, so stored scores alone are not enough.

Nothing else. When `scoring.plugin` is omitted, the built-in generic plugin
(`input_shuffling_v2.plugins.in_shuff_sklearn_joblib`) loads any joblib/pickle
sklearn-like model (`predict_proba` / `decision_function` / `predict`) and
resolves feature names from `model_info.json` if present, else from the
model's `feature_names_in_`, else from an explicit `schema.feature_columns`
list. See `configs/minimal_example.yaml` for the smallest config that runs.

## KNIME (or any PMML) models

Models built outside Python work via PMML: export each model from KNIME with
a **PMML Writer** node, and export one flat file (CSV/parquet) holding the
model-ready inputs, the target, and each model's score column. Then point a
config at the PMML plugin:

```yaml
schema:
  feature_columns: []                  # read from the PMML MiningSchema
  prediction_columns: ["P_y1_lr"]      # this model's score column in the flat file
scoring:
  plugin:
    module: "input_shuffling_v2.plugins.in_shuff_pmml"
    function: "predict"
    info_function: "model_info"
  model:
    path: "path/to/model.pmml"
```

Multiple models in one flat file = one config per model, each naming its own
`prediction_columns` and PMML path. Classification PMML returns
P(positive class) (positive label defaults to "1"; override with the
`INPUT_SHUFFLING_POSITIVE_LABEL` env var); regression PMML returns the
predicted value. Requires `pip install pypmml` and a Java runtime (present
wherever KNIME runs).

KNIME lessons learned (validated on a real 6-model TARGET_B export):

- **Bundle preprocessing into the PMML.** If the workflow normalizes inputs
  (e.g. before an MLP), the learner-only PMML omits that step and scores
  garbage on raw features (a saturated network returns a constant). Feed the
  Normalizer's PMML into the learner's optional PMML in-port (or combine via
  PMML transformation nodes) so the written PMML carries the NormContinuous
  fields. Models trained directly on the exported columns (logistic, tree,
  NB, GBM, RF) reproduce KNIME's stored probabilities to ~1e-9 or better.
- **kNN cannot be exported to PMML** (instance-based) — use the
  `knn_lookup` plugin instead: export the training partition and describe
  the node's settings in a JSON spec (see "Preprocessing and kNN models").
- The tool's scorer-consistency check catches both problems at startup by
  comparing plugin scores against the stored column.
- **Runtime scales with model scoring cost**: logistic/tree/NB score 47k
  rows in ~1s (a 31-feature x 10-shuffle run takes minutes); a large random
  forest at ~20s/pass takes hours single-process — use `performance.n_jobs`
  and/or `sampling` for big ensembles.

## Preprocessing and kNN models

- **`scoring.preprocessing`** — a per-model list of PMML transformation files
  (e.g. a KNIME Normalizer export) applied to the features after shuffling
  and before scoring. Use when a model was trained on transformed inputs but
  the data file holds raw values. Shuffling always operates on the raw
  columns, preserving each variable's true marginal distribution.
- **kNN models** (`input_shuffling_v2.plugins.in_shuff_knn_lookup`) — kNN has no
  exportable equation; the model artifact is a JSON spec pointing at the
  exported training set plus distance settings (`n_neighbors`, `weights`,
  `metric`, optional `preprocessing_pmml` applied to both the reference set
  and queries, optional `reference_sample_n`). See
  `configs/knime/target_b_knn_spec.json`.

## Column contract

- **One score column per model per config.** The score must be real-valued
  with variance — a probability for classification, never the {0,1}
  predicted-label column. The tool errors on label-like or constant score
  columns, and warns when a column has very few distinct values.
- **Every other column in the data file is ignored** unless given a role
  (feature, target, id, meta, bucket column).
- When feature columns are **inferred from the data file** (no explicit list
  and no model metadata), list non-inputs — e.g. other models' score
  columns — under `schema.ignore_columns`. This is a flat-file-only
  concern: anything built through the modelkit pipeline carries its input
  list in `model_info.json`, and KNIME PMML files carry theirs in the
  MiningSchema, so inference (and `ignore_columns`) never comes into play.
- A score or target column listed as a feature is an error, and by default
  the tool verifies the scoring plugin reproduces the stored score column on
  a sample (`scoring.baseline.verify_scorer: false` to disable).

## Design notes

- **Config compatibility**: v0.5/v0.6 YAML configs and the v1 scoring-plugin
  interface (`predict(df, *, model_path, feature_columns, clip_eps)`) work
  unchanged. Missing sections get defaults.
- **Bucketing modes**: `within_bucket` permutes feature values only among a
  bucket's rows (bucket-conditional marginal — v1 semantics, costs a full
  engine pass per bucket). `aggregate` reuses the full-data per-record
  dispersion stats and aggregates them per bucket (answers "which globally
  shuffled variables destabilize this segment"; no rescoring). In aggregate
  mode `median_abs_delta`/`p90_abs_delta` are quantiles of per-record mean
  |delta| rather than pooled across shuffles.
- **Stability resampling axes**: the importance ranking is bootstrapped over
  *records* (exact re-aggregation of per-record stats — no rescoring). The
  metric_drop ranking is bootstrapped over *shuffles*, since row-level
  bootstrap would require storing every shuffled prediction vector.

## Tests

```bash
python -m pytest -v        # from the repository root
```

Synthetic linear-model validation (ranking must match |coef|·std), stats math,
determinism across batch sizes, and end-to-end output checks.

## License

Copyright (C) 2026 Dean Abbott / Abbott Analytics.

Input Shuffling is free software under the
[GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later).
You may use, modify, and share it. If you distribute it, or offer it to
others as a hosted service, you must release your complete source under the
same license and keep the attribution in [NOTICE](NOTICE).

**Commercial licenses are available.** To use Input Shuffling in a
proprietary product or service without releasing its source, contact
dean@abbottanalytics.com.

If you use Input Shuffling in published work, please cite it; see
[CITATION.cff](CITATION.cff).
