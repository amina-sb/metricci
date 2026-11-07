# metricci

Confidence intervals and paired significance tests for classification metrics
that have no convenient closed form: ROC AUC, F1 and lift at k.

A test-set metric is an estimate from a finite sample. "AUC 0.83" on 200 rows
can easily mean anything from 0.76 to 0.89. `metricci` puts the interval next
to the number, and gives you a paired test for "is model A actually better than
model B on this test set?"

```python
import numpy as np
from metricci import auc_ci, paired_diff_ci, paired_permutation_test, roc_auc

rng = np.random.default_rng(0)
y = np.r_[np.ones(60, int), np.zeros(140, int)]
score_a = y * 1.4 + rng.normal(size=200)
score_b = y * 1.0 + rng.normal(size=200)

print(auc_ci(y, score_a, random_state=0))
# 0.8667 (95% CI 0.8062-0.9110, bca)

print(paired_diff_ci(roc_auc, y, score_a, score_b, random_state=0))
# 0.1012 (95% CI 0.0126-0.1942, bca)

print(paired_permutation_test(roc_auc, y, score_a, score_b, random_state=0).pvalue)
# 0.0368
```

Install from source:

```bash
pip install git+https://github.com/amina-sb/metricci
```

## What it does

| function | what you get |
|---|---|
| `auc_ci`, `f1_ci`, `lift_ci` | bootstrap CI for one model's metric, resampled within each class |
| `hanley_mcneil_ci` | the closed-form Wald interval for AUC (Hanley & McNeil, 1982), as a baseline |
| `paired_diff_ci` | bootstrap CI for `metric(A) - metric(B)`, both scored on the same resampled rows |
| `paired_permutation_test` | p-value for "A and B are interchangeable", swapping their outputs row by row |
| `bootstrap_ci` | the general engine: percentile, basic or BCa interval for any statistic of paired arrays |
| `roc_auc`, `f1`, `lift_at_k` | the metrics themselves, accepting one sample `(n,)` or a stack of resamples `(k, n)` |

All bootstrap intervals default to **BCa** (bias-corrected and accelerated;
Efron, 1987), with the acceleration estimated by the jackknife.

## Decisions that matter

**Resampling is stratified by class.** Every resample keeps the data's numbers
of positives and negatives. Without that, a small or imbalanced test set gives
some resamples with no positives, where AUC is undefined; dropping those
resamples quietly biases the interval. The stratified interval describes the
metric for a test set of this composition, which is the question usually being
asked.

**BCa counts ties as half.** F1 and other count-based metrics take few distinct
values, so many bootstrap resamples land exactly on the point estimate. The
textbook bias correction counts only resamples strictly below the estimate and
is then biased by up to the share of ties. Counting ties as half is identical
for continuous statistics and correct for discrete ones.

**The permutation test swaps ranks, not raw scores.** If model A outputs
probabilities and model B outputs margins on a wider scale, swapping their raw
outputs row by row builds score vectors that mix the two scales. The null
distribution is then dominated by the scale mismatch, and the test almost
never rejects: in the simulation below, swapping raw scores never rejected
under the null (rate 0.000) and detected a real difference only half the time,
against 0.965 when the scores are ranked first. AUC and lift depend only on
ranks, so ranking first leaves the observed statistic unchanged. Hard 0/1
predictions are not ranked.

**Paired, not overlapping.** Two models scored on the same test set have
strongly correlated metrics. Comparing two separate intervals for overlap
ignores that correlation and is badly underpowered; `paired_diff_ci` resamples
rows once and scores both models on each resample.

## Evidence

Every number below is produced by `scripts/evidence.py`, which CI runs with
`--check` and fails if this table changes. Data are binormal: positives score
N(1, 1), negatives N(0, 1), so the true AUC is Φ(1/√2) and the true F1 at a
fixed threshold has a closed form. A correct 95% interval covers the truth in
about 95% of data sets. "Range over 5 seeds" is the coverage in each of five
independent blocks of 80 data sets.

<!-- evidence:start -->
**AUC**, true value 0.7602. Nominal 95% intervals, 400 simulated data sets per row, 999 resamples each.

| data | method | coverage (95% CI) | range over 5 seeds | mean width |
|---|---|---|---|---|
| n = 40 (12 pos) | Hanley-McNeil (baseline) | 0.945 (0.918-0.963) | 0.887-0.975 | 0.338 |
| n = 40 (12 pos) | percentile | 0.935 (0.906-0.955) | 0.887-1.000 | 0.315 |
| n = 40 (12 pos) | basic | 0.897 (0.864-0.924) | 0.812-0.938 | 0.315 |
| n = 40 (12 pos) | BCa | 0.970 (0.948-0.983) | 0.925-1.000 | 0.329 |
| n = 200 (60 pos) | Hanley-McNeil (baseline) | 0.965 (0.942-0.979) | 0.938-0.988 | 0.155 |
| n = 200 (60 pos) | percentile | 0.938 (0.909-0.957) | 0.912-0.988 | 0.142 |
| n = 200 (60 pos) | basic | 0.932 (0.904-0.953) | 0.887-0.988 | 0.142 |
| n = 200 (60 pos) | BCa | 0.953 (0.927-0.969) | 0.925-0.988 | 0.144 |

**F1** at threshold 0.5, true value 0.5735 (both sizes have a positive rate of 0.3).

| data | method | coverage (95% CI) | range over 5 seeds | mean width |
|---|---|---|---|---|
| n = 40 (12 pos) | percentile | 0.938 (0.909-0.957) | 0.912-0.975 | 0.359 |
| n = 40 (12 pos) | basic | 0.932 (0.904-0.953) | 0.887-0.963 | 0.359 |
| n = 40 (12 pos) | BCa | 0.958 (0.933-0.973) | 0.912-0.988 | 0.360 |
| n = 200 (60 pos) | percentile | 0.948 (0.921-0.965) | 0.900-0.988 | 0.162 |
| n = 200 (60 pos) | basic | 0.950 (0.924-0.967) | 0.900-0.988 | 0.162 |
| n = 200 (60 pos) | BCa | 0.958 (0.933-0.973) | 0.900-0.988 | 0.162 |

**Paired permutation test** for an AUC difference, 200 data sets per row, n = 200, 499 permutations, rejection rate at alpha = 0.05.

| scenario | swapped values | rejection rate (95% CI) |
|---|---|---|
| equal models (size) | raw scores | 0.000 (0.000-0.019) |
| equal models (size) | ranks (default) | 0.055 (0.031-0.096) |
| A better by 0.5 SD (power) | raw scores | 0.505 (0.436-0.574) |
| A better by 0.5 SD (power) | ranks (default) | 0.965 (0.930-0.983) |
<!-- evidence:end -->

What this shows and what it does not:

- At n = 200 every interval is close to nominal. The bootstrap intervals are
  slightly narrower than Hanley-McNeil.
- At n = 40 (12 positives) the **basic** interval under-covers (about 0.90).
  BCa over-covers a little for AUC, so it is the cautious choice. The
  percentile interval sits between them.
- Hanley-McNeil does well here because binormal data are close to its
  assumptions. It is a baseline, not a straw man.
- These are binormal simulations at two sample sizes and one effect size.
  Heavily imbalanced data (1% positives) and AUC near 1 are not covered yet.
  Both are listed in the roadmap.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                 # includes simulation tests marked "slow"
pytest -q -m "not slow"   # quick subset
ruff check . && ruff format --check . && mypy
python scripts/evidence.py
```

The test suite checks the bootstrap against `scipy.stats.bootstrap`, the
metrics against scikit-learn, the permutation test against
`scipy.stats.permutation_test`, and interval coverage and test size by
simulation. Property-based tests (hypothesis) cover metric bounds, symmetry and
invariance.

## References

- Efron, B. (1987). Better bootstrap confidence intervals. *JASA* 82(397), 171-185.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Hanley, J. A. & McNeil, B. J. (1982). The meaning and use of the area under a
  receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36.

## Licence

MIT
