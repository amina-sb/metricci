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
| `delong_ci`, `delong_test` | DeLong's nonparametric AUC interval and the paired DeLong test (DeLong et al., 1988) |
| `paired_diff_ci` | bootstrap CI for `metric(A) - metric(B)`, both scored on the same resampled rows |
| `paired_permutation_test` | p-value for "A and B are interchangeable", swapping their outputs row by row |
| `bootstrap_ci` | the general engine: percentile, basic or BCa interval for any statistic of paired arrays |
| `roc_auc`, `f1`, `lift_at_k` | the metrics themselves, accepting one sample `(n,)` or a stack of resamples `(k, n)` |

AUC and F1 intervals default to **BCa** (bias-corrected and accelerated;
Efron, 1987), with the acceleration estimated by the jackknife. Lift defaults to
the percentile interval, for the reason below.

API documentation: https://amina-sb.github.io/metricci/

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

**Lift uses the percentile interval, and BCa warns at an edge.** Lift at k is
capped at 1 / base rate, which it reaches when every top-k row is positive. In
those data sets no resample can go above the estimate. A large share of
resamples tie it, BCa's bias correction pushes both quantile levels up, and the
interval collapses onto the cap. In the simulation below BCa covers the true
lift 0.733 of the time at n = 40, against 0.920 for the percentile interval.
`bootstrap_ci` raises a `BootstrapWarning` whenever BCa is asked for and the
estimate is the minimum or maximum of the bootstrap distribution with a quarter
or more of resamples tied to it.

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
| n = 40 (12 pos) | DeLong (baseline) | 0.920 (0.889-0.943) | 0.850-0.963 | 0.324 |
| n = 40 (12 pos) | percentile | 0.935 (0.906-0.955) | 0.887-1.000 | 0.315 |
| n = 40 (12 pos) | basic | 0.897 (0.864-0.924) | 0.812-0.938 | 0.315 |
| n = 40 (12 pos) | BCa | 0.970 (0.948-0.983) | 0.925-1.000 | 0.329 |
| n = 200 (60 pos) | Hanley-McNeil (baseline) | 0.965 (0.942-0.979) | 0.938-0.988 | 0.155 |
| n = 200 (60 pos) | DeLong (baseline) | 0.938 (0.909-0.957) | 0.925-0.975 | 0.144 |
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

**AUC in harder regimes**: rare positives (separation 1.0) and a near-perfect model (separation 3). 400 data sets per row.

| data | method | coverage (95% CI) | range over 5 seeds | mean width |
|---|---|---|---|---|
| n = 1,000, 10 pos, AUC 0.760 | Hanley-McNeil (baseline) | 0.965 (0.942-0.979) | 0.938-0.988 | 0.338 |
| n = 1,000, 10 pos, AUC 0.760 | DeLong (baseline) | 0.870 (0.833-0.899) | 0.800-0.925 | 0.289 |
| n = 1,000, 10 pos, AUC 0.760 | percentile | 0.868 (0.831-0.897) | 0.800-0.925 | 0.271 |
| n = 1,000, 10 pos, AUC 0.760 | BCa | 0.892 (0.858-0.919) | 0.838-0.925 | 0.285 |
| n = 1,000, 50 pos, AUC 0.760 | Hanley-McNeil (baseline) | 0.990 (0.975-0.996) | 0.975-1.000 | 0.157 |
| n = 1,000, 50 pos, AUC 0.760 | DeLong (baseline) | 0.965 (0.942-0.979) | 0.938-0.988 | 0.135 |
| n = 1,000, 50 pos, AUC 0.760 | percentile | 0.965 (0.942-0.979) | 0.938-0.988 | 0.132 |
| n = 1,000, 50 pos, AUC 0.760 | BCa | 0.963 (0.939-0.977) | 0.950-0.975 | 0.134 |
| n = 200, 60 pos, AUC 0.983 | Hanley-McNeil (baseline) | 0.978 (0.958-0.988) | 0.963-1.000 | 0.039 |
| n = 200, 60 pos, AUC 0.983 | DeLong (baseline) | 0.880 (0.844-0.908) | 0.800-0.938 | 0.028 |
| n = 200, 60 pos, AUC 0.983 | percentile | 0.902 (0.869-0.928) | 0.838-0.938 | 0.028 |
| n = 200, 60 pos, AUC 0.983 | BCa | 0.943 (0.915-0.961) | 0.925-0.963 | 0.032 |

**Lift at k = 0.1**, true value 2.3452 (positive rate 0.3).

| data | method | coverage (95% CI) | range over 5 seeds | mean width |
|---|---|---|---|---|
| n = 40 (12 pos) | percentile | 0.920 (0.889-0.943) | 0.900-0.938 | 2.411 |
| n = 40 (12 pos) | basic | 0.690 (0.643-0.733) | 0.637-0.812 | 2.411 |
| n = 40 (12 pos) | BCa | 0.733 (0.687-0.774) | 0.688-0.838 | 2.265 |
| n = 200 (60 pos) | percentile | 0.955 (0.930-0.971) | 0.938-0.963 | 1.283 |
| n = 200 (60 pos) | basic | 0.912 (0.881-0.936) | 0.887-0.950 | 1.283 |
| n = 200 (60 pos) | BCa | 0.943 (0.915-0.961) | 0.900-0.975 | 1.290 |

**Paired permutation test** for an AUC difference, 200 data sets per row, n = 200, 499 permutations, rejection rate at alpha = 0.05.

| scenario | swapped values | rejection rate (95% CI) |
|---|---|---|
| equal models (size) | raw scores | 0.000 (0.000-0.019) |
| equal models (size) | ranks (default) | 0.055 (0.031-0.096) |
| A better by 0.5 SD (power) | raw scores | 0.505 (0.436-0.574) |
| A better by 0.5 SD (power) | ranks (default) | 0.965 (0.930-0.983) |
<!-- evidence:end -->

What this shows and what it does not:

- At n = 200 every AUC interval is close to nominal. The bootstrap and DeLong
  intervals are slightly narrower than Hanley-McNeil.
- At n = 40 (12 positives) the **basic** interval under-covers (0.897) and
  DeLong does too (0.920). BCa over-covers a little (0.970), so it is the
  cautious choice.
- **With only 10 positives** (1% of 1,000 rows) every interval except
  Hanley-McNeil under-covers: DeLong 0.870, percentile 0.868, BCa 0.892. The
  sample size that matters for AUC is the number of positives, not the number
  of rows, and 10 is too few for any of these methods to be trusted at face
  value. Hanley-McNeil reaches 0.965 only by being the widest interval. With 50
  positives all four methods are at or above nominal.
- **When the true AUC is near 1** (0.983) DeLong (0.880) and the percentile
  interval (0.902) under-cover. BCa (0.943) is the only one near nominal
  without the excess width of Hanley-McNeil.
- **Lift**: the percentile interval is close to nominal at both sizes. Basic
  and BCa fail badly at n = 40 for the reason given above.
- Rows within one block are computed on the same simulated data sets, so their
  coverages move together. Read differences between methods within a block,
  not small differences between blocks.
- Hanley-McNeil does well on binormal data because it is close to its
  assumptions. It is a baseline, not a straw man.
- All results are from binormal simulations. Other score distributions, and
  metrics other than AUC, F1 and lift, are untested.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q -m "not evidence"   # full suite except the README regeneration
pytest -q -m "not slow and not evidence"   # quick subset
python scripts/evidence.py --check   # several minutes
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
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). Comparing the
  areas under two or more correlated receiver operating characteristic curves:
  a nonparametric approach. *Biometrics* 44(3), 837-845.
- Hanley, J. A. & McNeil, B. J. (1982). The meaning and use of the area under a
  receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36.
- Sun, X. & Xu, W. (2014). Fast implementation of DeLong's algorithm for
  comparing the areas under correlated receiver operating characteristic
  curves. *IEEE Signal Processing Letters* 21(11), 1389-1393.

## Licence

MIT
