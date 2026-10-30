# Sample donor example

A small, self-contained example with known answers: synthetic donor-response
data, three fitted models, and the true data-generating model, all run through
input shuffling.

## Files

| File | Purpose |
|---|---|
| `sample_donor_data.csv` | 4,000 synthetic donors (2,377 TRAIN / 1,623 TEST), 9 inputs, `TARGET_B`, `TRUE_PROB` |
| `in_shuff_sample_data.py` | Generates the CSV and defines the true model (`true_response_probability`) |
| `in_shuff_run_sample.py` | Trains the models, runs input shuffling, compares with `expected/` |
| `expected/` | The answers: one importance table per model, `rank_comparison.csv`, and the library versions that produced them |

## Run it

From the repository root (after `pip install -e .`, or from a plain checkout):

```bash
python examples/sample_donor/in_shuff_run_sample.py
```

It takes about 15 seconds. Results go to `examples/sample_donor/output/`, and
the script reports MATCH or DIFFERS for each model against `expected/`.
`--write-expected` regenerates the answers.

## The true model

```
logit P(TARGET_B = 1) = -3.1
    - 0.08  * (RECENCY - 12)          strong, negative
    + 0.70  * log(FREQUENCY)          strong, positive
    + 0.45  * log(LASTGIFT / 10)      moderate, positive
    + 0.18  * HOMEOWNER * WEALTH      interaction only, no main effects
    + 0.012 * (AGE - 55)              weak
```

`AVGGIFT` (correlated with `LASTGIFT`, r = 0.85), `NOISE_A` and `NOISE_B` have
no effect. The response rate is 16.5%.

## Models

| Model | Settings | Test AUC |
|---|---|---|
| oracle | the true model above | 0.766 |
| logistic | StandardScaler + LogisticRegression | 0.741 |
| random_forest | 200 trees, min_samples_leaf=10 | 0.708 |
| gbm | GradientBoostingClassifier, 150 trees, depth 3, lr 0.05 | 0.726 |

## The answers

Overall rank by `mean_record_iqr` (50 shuffles, 1,623 TEST records):

| Feature | oracle | logistic | random_forest | gbm |
|---|---|---|---|---|
| RECENCY | 1 | 1 | 1 | 1 |
| HOMEOWNER | 2 | 2 | 3 | 2 |
| WEALTH | 3 | 3 | 2 | 3 |
| FREQUENCY | 4 | 4 | 5 | 5 |
| LASTGIFT | 5 | 5 | 4 | 4 |
| AGE | 6 | 7 | 8 | 9 |
| AVGGIFT | 7 | 6 | 6 | 6 |
| NOISE_A | 8 | 9 | 7 | 7 |
| NOISE_B | 9 | 8 | 9 | 8 |

What to notice:

- **The oracle gives exactly zero** to AVGGIFT, NOISE_A and NOISE_B (their
  order among themselves is a tie). Shuffling an input the model ignores never
  moves a score.
- **Every fitted model ranks AVGGIFT above AGE**, although AVGGIFT has no true
  effect. The models borrow part of LASTGIFT's signal through the correlated
  AVGGIFT. Input shuffling reports what the *model* depends on, not what
  causes the response.
- **The interaction is found.** HOMEOWNER and WEALTH have no main effects, yet
  they rank 2–3 for every model, including logistic regression, which has
  only main effects and approximates the interaction with two of them.
- Each table also has `top30_rank` (same shuffles, top 30% of baseline scores
  only) and `auc_drop` (the classic permutation-importance view) for
  comparison.

## Reproducibility

The oracle and logistic answers depend only on the data and the seeds.
Random forest and GBM answers can shift slightly with a different
scikit-learn version; `expected/expected_versions.json` records the versions
used to make them.
