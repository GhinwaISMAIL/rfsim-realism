# Phase 3M: Offline SINR Dynamics Development Result

## Scope

Phase 3M tested whether a constrained first-order SINR response model improves
complete-execution forward prediction beyond the Version 1 static translator.
It was conducted entirely offline and is explicitly post hoc development. It
does not authorize Version 2 inverse commands, hardware execution, or final
validation claims.

The development set contains three complete Test 1 hardware replays and the
single exploratory Test 6 replay. Test 6 was not used to fit Version 1, but is
now development evidence for any Version 2 model.

## Version 1 sign-convention erratum

The frozen Phase 3L configuration described clipping error as original target
minus projected feasible target. The implemented analyzer columns use the
opposite convention:

\[
e_t^{\mathrm{clip}}=x'_t-x_t.
\]

This implemented convention is canonical because it preserves:

\[
e_t^{\mathrm{total}}
=e_t^{\mathrm{clip}}+e_t^{\mathrm{dynamic}}.
\]

The erratum changes no Version 1 data, thresholds, decisions, or absolute-error
metrics. It only clarifies the sign of signed clipping-error columns.

## Compared forward models

All models used projected feasible SINR as their input. Predictions were
recursive and open loop; held-out observations were not used as lagged inputs.
Four leave-one-complete-execution-out folds were evaluated with equal execution
weight.

| Model | Definition |
|---|---|
| Static | \(\hat{s}_t=a+bq_t\) |
| Memory only | \(\hat{s}_t=\alpha\hat{s}_{t-1}+(1-\alpha)q_t\) |
| Combined | \(\hat{s}_t=\alpha\hat{s}_{t-1}+(1-\alpha)(a+bq_t)\) |

Here, \(q_t\) is the projected feasible SINR supplied to the frozen translator.

## Cross-validation results

| Model | Mean MAE (dB) | Mean p95 absolute error (dB) | Worst absolute error (dB) | Mean absolute residual lag-1 correlation |
|---|---:|---:|---:|---:|
| Static | 1.581 | 4.030 | 7.919 | 0.620 |
| Memory only | 1.404 | 3.766 | 6.908 | 0.700 |
| Combined | 1.415 | 3.718 | 6.675 | 0.702 |

The dynamic candidates reduced average and worst errors in every held-out fold.
Their fitted memory parameters were stable:

- Memory only: mean \(\alpha=0.503\), fold range 0.025.
- Combined: mean \(\alpha=0.488\), fold range 0.033.
- Combined: mean \(b=0.967\), fold range 0.042.

However, neither candidate passed every predeclared gate:

- Memory only improved mean MAE by 0.178 dB (11.2%), but improved mean p95
  error by only 6.6%, below the required 8%.
- Combined improved mean MAE by 0.166 dB (10.5%), but improved mean p95 error
  by 7.7%, also below the required 8%.
- Both candidates increased, rather than reduced, the mean absolute residual
  lag-1 correlation by approximately 0.08.

The gates are retained unchanged. The near-threshold p95 result is not rounded
up or reinterpreted as a pass.

## Static SINR variability reference

The Phase 3G factorial and boundary data contain 39 executions and 585 steady
one-second observations. Their diagnostic variability was:

- Median within-execution SINR standard deviation: 0.499 dB.
- Mean within-execution SINR standard deviation: 0.514 dB.
- 95th percentile within-execution SINR standard deviation: 0.692 dB.
- Median within-execution mean absolute deviation: 0.404 dB.
- Pooled between-execution standard deviation of repeated-state means: 0.283 dB.

These values are a diagnostic variability reference, not proof of a universal
irreducible error floor.

## Decision

No first-order candidate is authorized for inverse command design. Version 1
remains the supported deterministic emulator. The Phase 3M result shows that a
roughly half-weight one-step memory term improves average prediction, but the
remaining residual dependence is not adequately represented by the tested
first-order models.

No POWDER reservation is required. Any richer dynamic model would require a
new, separately frozen development protocol and a newly designated untouched
session for final evaluation.
