# Input Shuffling: the method

Input shuffling measures how much a trained model depends on each of its
inputs. Take one input and scramble its values across the records, keeping
every other value the same. That detaches the input's values from the rest of
each record. Then rescore the model and measure how much each record's
prediction moves. Do that many times. If the model relies heavily on the
input, shuffling it changes the predictions a lot; an input the model ignores
doesn't move them at all.

The result is a distribution of predictions for every record and every input.
Everything else in this package (the variable ranking, the segment rankings,
the unstable-record report, the stability checks) is a summary of those
distributions.

One advantage of input shuffling is that it works with any algorithm. It
doesn't interpret the model's structure (coefficients or splits), only its
behavior.

This page explains what the method measures, how it differs from permutation
importance and SHAP, and where its limits are. The [README](../README.md)
covers installing and running it.

## History

Input shuffling grew out of John Elder's *target shuffling* (Elder, 2014;
Elder Research, n.d.). Target shuffling tests whether a model's apparent
accuracy is real: shuffle the target column, rerun the whole modeling process,
and repeat many times. If models built on random targets often do as well as
the real one, the real result is probably luck.

I first saw John present target shuffling at Predictive Analytics World. I
found it fascinating, and such a good idea that I kicked myself for not having
thought of it already. The more I considered it, the more I wondered whether
the same move could measure variable importance: shuffle an input instead of
the target, and leave the trained model alone. I first presented the idea at
the KNIME Fall Summit in September 2016 (Abbott, 2016), before I called it
input shuffling.

## The algorithm

0. **Build your model.** Build the model you plan to deploy, or a candidate
   for deployment.
1. **Score the data.** Run the trained model on an analysis set (usually a
   test or holdout sample) to get each record's baseline prediction.
2. **Shuffle one input.** Randomly reorder that input's values across the
   records, leaving every other column as it is. The input keeps its own
   distribution (same values, same frequencies, same missing values), but each
   record now carries another record's value.
3. **Rescore.** Run the model on the shuffled data. Each record gets a new
   prediction; the difference from its baseline is that record's *delta* for
   this shuffle.
4. **Repeat** steps 2 and 3 M times (typically 10 to 50) with different
   random orders. Each record now has M deltas for this input.
5. **Summarize each record.** Compute the spread of the record's M deltas.
   The interquartile range (IQR) is the default; standard deviation, variance
   and median absolute deviation (MAD) are also computed.
6. **Summarize across records.** Average (or take the median of) the
   per-record spreads to get one influence number for the input.
7. **Do this for every input and rank them.**

The model is never refit. Only its predictions are used, so the method works
on any model that can score a table: scikit-learn, XGBoost, neural nets,
PMML exports from KNIME or other tools, kNN lookup tables, or a foundation
model like TabICL.

## Why the spread of each record's predictions

Most shuffling methods ask one question per input: how much does a
performance metric drop when the input is scrambled? Input shuffling also asks:
**how unstable does each record's prediction become?**

That choice has four consequences.

- **No labels are needed.** The spread is computed from predictions alone, so
  the method runs on data whose outcomes aren't known yet, such as this
  month's scoring file.
- **The answer exists for every record, not just for the file.** The
  per-record spreads can be summarized over the whole file, over a segment
  (the top 10% of scores, one region, the donors a campaign will actually
  mail), or not at all, one record at a time.
- **The spread measures dependence, not just shift.** A record whose
  prediction swings widely across shuffles depends on that input, even if
  the swings average out to no change in its mean prediction.
- **Scale is handled by the shuffle.** Every input is replaced by draws from
  its own distribution, and every spread is measured in the same units, the
  model's prediction. A dollar amount, a count and a category code can be
  compared directly without normalizing the inputs.

### Choosing the statistic

| Statistic | What it emphasizes | When to use it |
|---|---|---|
| IQR (default) | The middle of the record's delta distribution; resistant to a few extreme shuffles | Ranking variables, especially for probabilities |
| Standard deviation | All the deltas, including large ones | A complementary check; ties to classical regression results (below) |
| Variance | Same as SD, squared | Compatibility with v1 |
| MAD | The typical distance from the median delta | Very skewed predictions |

The per-record spreads can then be aggregated by mean or median, and the
package also reports the mean absolute delta, the 90th percentile of the
absolute delta, and the share of deltas above a tolerance. These measure
different things, and an input can rank differently under each. Reporting IQR
for the ranking and SD alongside it is the recommended default.

### A check on linear models

For a linear model, shuffling input *x* changes a record's prediction by
β(x′ − x), where x′ is the shuffled value. Across shuffles, the spread of that
change is proportional to |β| × SD(x), the size of the coefficient times the
spread of the input. That is the standardized coefficient. When the inputs
aren't strongly correlated, it ranks the inputs the same way the t-values do.
So on a well-behaved linear model, input shuffling reproduces the classical
answer, and the test suite checks exactly this on synthetic data.

The method doesn't change for nonlinear models. Only the scoring function
does. When the classical tools stop applying (skewed inputs, interactions,
models with no coefficients), the same measurement still does.

## Compared with permutation importance

Permutation importance (Breiman, 2001; generalized as "model reliance" by
Fisher, Rudin and Dominici, 2019) shares the core loop: shuffle an input,
rescore, repeat. It then measures how much a performance metric (AUC,
accuracy, log loss) gets worse.

| | Permutation importance | Input shuffling |
|---|---|---|
| What is measured | The drop in a metric over the whole file | The spread of each record's predictions |
| Needs the actual outcomes | Yes | No |
| Result | One number per input | A distribution per input and per record |
| Segments (top 10%, a region) | Recompute the metric within the segment | Summarize the same per-record spreads within the segment; no rescoring |
| Sensitivity to the metric | The ranking depends on which metric you pick | The ranking is in prediction units; metric drops can be reported alongside |

The two answer related but different questions. Permutation importance asks
how much worse the model gets without the input. Input shuffling asks how
much the model's output depends on it. An input can move predictions a lot
while barely changing AUC, for example when it reorders records within the
top of the list, and that reordering is exactly what a mailing or a campaign
cares about.

This package computes both. `metric_drop` gives the classic permutation
importance for AUC, log loss, F1, accuracy and lift from the same shuffles.

## Compared with SHAP

SHAP (Lundberg and Lee, 2017) assigns each input a share of each record's
prediction, based on Shapley values from game theory. The shares are signed
and add up to the difference between the record's prediction and the
average prediction.

| | SHAP | Input shuffling |
|---|---|---|
| Ranking the inputs for the whole model | Good (mean \|SHAP\|) | Excellent: the method's main output |
| Inputs that matter for one record | Excellent | Computed today, not yet reported; expected good to excellent (see below) |
| Direction for one record (did this input push the score up or down?) | Yes | Not yet reported; can be computed from the same shuffles |
| Contributions add up to the prediction | Yes, by construction | No; an interaction shows up in the spread of each input involved |
| Works on any model that scores | Fast for trees (TreeSHAP); slow and approximate otherwise (KernelSHAP) | Yes, at the same cost for every model |
| Needs the actual outcomes | No | No |
| Segments of the file | Average SHAP within the segment | Built in (score bands, top and bottom percentiles, any column) |
| Cost | Grows quickly with the number of inputs for non-tree models | Inputs × shuffles × rows scored; parallel; subsamples well |
| Correlated inputs | Affected: credit is split or assigned to unrealistic combinations | Affected: credit is split (see Limits) |
| Theory | Shapley axioms | Reduces to standardized coefficients for linear models; otherwise empirical |

### Record-level explanations: what exists and what's next

Input shuffling already produces, for every record, the spread of its
prediction under each input. Ranking the inputs within one record gives that
record's own importance list. Today the package reports it only for the
records with the most unstable predictions (`record_outliers.csv`, with each
record's top driving inputs). The full record × input table is written to
`record_instability_full.parquet`.

Three pieces would turn that into a full record-level explanation:

1. **A report for any record or set of records**, not just the most unstable:
   the inputs ranked by their spread for that record, with the record's
   actual values beside them.
2. **Direction.** The baseline prediction minus the mean of the shuffled
   predictions shows whether the record's actual value pushes its score above
   or below what a typical value would. The deltas are already computed; this
   is a matter of keeping their mean.
3. **Enough shuffles per record.** Ten shuffles are plenty for a ranking
   averaged over 2,000 records, but noisy for one record. Record-level
   reports need more shuffles (30 to 50), or a bootstrap interval on each
   record's spread.

Within a record, no normalization is needed: every input's spread is in the
same prediction units. Comparing *across* records is different. A record
scored at 0.01 can't move as far as one scored at 0.30, so cross-record
comparisons should use each input's share of the record's total spread, or
compute the spreads on the log-odds scale.

### Where SHAP keeps an advantage

SHAP's contributions add up exactly to the prediction. That makes it the
standard for adverse-action reason codes ("these four factors lowered your
score by this much"), where regulators expect the parts to sum to the whole.
Input shuffling measures each input's influence including its interactions,
so the influences don't sum to anything. That's a different and often more
honest description of a nonlinear model, but it isn't an additive
decomposition.

### Where input shuffling has the advantage

The question in applied work is usually not "how does this input contribute
mathematically?" but "where in the list does this input matter?" Input
shuffling answers that directly. Its spreads are in prediction units, so they
map onto movements in the ranked list. It can summarize the same spreads for
the top 5% of the list, the bottom 5%, or each score decile without
rescoring. And it runs at the same cost whether the model is logistic
regression, a 117 MB random forest exported to PMML, or a transformer.

## Limits

- **Correlated inputs split the credit.** If two inputs carry the same
  information, shuffling one leaves the other in place, so each looks less
  important than the pair is. A low rank means "this model doesn't need this
  input given the others," not "this input carries no signal." Shuffling
  correlated inputs together, as a group, measures the pair. Permutation
  importance and SHAP have the same problem; conditional versions exist for
  each (for example Strobl et al., 2008).
- **Shuffling creates unrealistic records.** A shuffled record can pair a
  20-year-old with 40 years of giving history. The model is scored on
  combinations it never saw, and its predictions there can be arbitrary
  (Hooker, Mentch and Zhou, 2021). Large spreads on an input that is tightly
  tied to others are worth checking against a grouped shuffle.
- **It describes the model, not the world.** A high rank means the model
  uses the input. Whether the model *should* use it is a separate question:
  leaks and artifacts rank high too, which makes the method a good leak
  detector.
- **Distance-based models reflect their scaling.** For kNN and other
  distance-based models, the ranking partly reflects how the inputs were
  scaled before distances were computed, since that sets how much each one
  can move a record's neighbors.

## Cost

Each run scores the model once per input per shuffle, on the rows sampled:
inputs × shuffles × rows. Each shuffle is independent, so the work runs in
parallel and in batches, and seeded per input and shuffle, so results are
identical however the work is split.

Sampling works well for rankings. In the TabICL study on KDD Cup 1998 data,
shuffling 2,000 test rows 10 times gave the same ranking as shuffling all
47,706 rows 50 times: Spearman correlations of 1.00 for logistic regression
and 0.99 for XGBoost, with the same ten top inputs. The `stability` option
measures how reproducible a ranking is by bootstrapping the records.

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Elder, J. (2014). The peril of vast search (and how target shuffling can
  save science). Talk at Predictive Analytics World for Healthcare, Boston,
  October 2014. [Video (members only)](https://www.predictiveanalyticsworld.com/machinelearningtimes/the-peril-of-vast-search)
- Elder, J. (n.d.). *Evaluate the validity of your discovery with target
  shuffling*. White paper, Elder Research.
  [Link](https://www.elderresearch.com/white-papers/evaluate-the-validity-of-your-discovery-with-target-shuffling/)
- Elder Research (n.d.). Target shuffling process.
  [Link](https://www.elderresearch.com/resource/innovations/target-shuffling-process/)
- Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but
  many are useful: Learning a variable's importance by studying an entire
  class of prediction models simultaneously. *Journal of Machine Learning
  Research*, 20(177), 1–81.
- Hooker, G., Mentch, L., and Zhou, S. (2021). Unrestricted permutation
  forces extrapolation: Variable importance requires at least one more model,
  or there is no free variable importance. *Statistics and Computing*, 31, 82.
- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *Advances in Neural Information Processing Systems 30*.
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., and Zeileis, A.
  (2008). Conditional variable importance for random forests. *BMC
  Bioinformatics*, 9, 307.

### Talks on input shuffling

- Abbott, D. (2016). Measuring variable importance with target shuffling.
  KNIME Fall Summit, September 16, 2016.
  [Video](https://www.youtube.com/watch?v=OlHW7frH3ug) ·
  [Slides](https://www.knime.com/sites/default/files/inline-images/abbott-variable_importance_using_randomization_v3.pdf)
- Abbott, D. (2017). Variable importance using input shuffling. Predictive
  Analytics World, Berlin.
- Abbott, D. (2023). Variable importance using input shuffling or SHAP.
