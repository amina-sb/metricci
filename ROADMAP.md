# Roadmap

## Phase 1 — Bootstrap engine
- [x] Percentile, basic and BCa intervals for any statistic of paired arrays
- [x] Stratified resampling
- [x] Stacked `(k, n)` evaluation path for fast metrics
- [x] Reference tests against `scipy.stats.bootstrap`

## Phase 2 — Metrics and one-call intervals
- [x] ROC AUC, F1, lift at k (single sample or stacked resamples)
- [x] `auc_ci`, `f1_ci`, `lift_ci`, stratified by class
- [x] Hanley-McNeil closed-form baseline
- [x] Property tests and scikit-learn reference tests

## Phase 3 — Comparing two models
- [x] Paired bootstrap interval for a metric difference
- [x] Paired permutation test with rank transform
- [x] Size and power measured by simulation

## Phase 4 — Evidence beyond the easy case
- [x] Coverage at heavy class imbalance (1% and 5% positives)
- [x] Coverage when the true AUC is near 1
- [x] DeLong interval and test as a second closed-form baseline
- [x] Coverage for lift at k
- [x] Percentile default for lift; warning when BCa sits on a bootstrap edge
- [ ] An interval that reaches nominal coverage with ~10 positives

## Phase 5 — More metrics
- [ ] Precision and recall at a threshold, average precision
- [ ] Brier score and log loss
- [ ] Regression metrics (MAE, RMSE, R²)

## Phase 6 — Distribution
- [x] Documentation site
- [ ] First PyPI release
