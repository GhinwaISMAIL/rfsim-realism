# Phase 3G boundary and trace-support diagnosis

## Outcome

The failed Phase 3G SINR uncertainty gate is primarily an extrapolation problem,
not evidence of unstable boundary executions and not evidence that the local
gain/noise controls are non-identifiable.

The frozen diagnosis is:

> `targeted_outer_envelope_validation_extension_required`

Direct replay remains unauthorized. Test 6 was not accessed, ABC remains
unauthorized, and no reservation is requested by this offline result.

## Boundary attribution

The `(gain=0 dB, noise=-17 dB)` state produced 73.715% of the bootstrap maximum
SINR errors. Its full 95% absolute-error interval reaches 3.7578 dB, but the
boundary-observation-only upper limit is 1.4815 dB. The corresponding
central-model-only upper limit is 3.7375 dB. This shows that uncertainty in the
central fit is amplified at the distant boundary; execution variability at the
boundary is not the main cause.

The opposite extreme, `(-18, -35)`, produced 20.79% of the maximum SINR errors.
The two central-near boundary states together produced less than 6%.

## Curvature check

A quadratic response improves the already small relative-RSRP residuals, but it
does not improve the scientifically limiting SINR response:

| Model | Central state LOSO SINR RMSE | Boundary maximum SINR error |
| --- | ---: | ---: |
| Bilinear | 0.3456 dB | 1.2536 dB |
| Quadratic | 0.4414 dB | 2.3061 dB |

The quadratic model is therefore not supported as the remedy. The bilinear
response remains the preferred development model.

## Test 1 support

The empirically updated inverse mapping was applied to all 305 designated Test 1
rows, with a maximum numerical inversion residual below `1e-14 dB`.

| Region | Supported rows | Fraction |
| --- | ---: | ---: |
| Central factorial rectangle | 84 / 305 | 27.54% |
| Convex hull of all 13 tested states | 221 / 305 | 72.46% |
| Tested operational gain/noise limits | 299 / 305 | 98.03% |

Restricting replay to the central rectangle would discard 221 rows and is not a
defensible representation of the designated trace. Six rows lie slightly beyond
the tested operational limits and must remain unsupported or be handled by a
pre-frozen clipping rule.

Using all 13 already observed states as development data reduces the trace's 95th
percentile prediction leverage from 9.9398 to 0.5752. Its state-wise
leave-one-state-out errors remain small: maximum 0.1218 dB for relative RSRP and
0.8139 dB for SINR. This supports using the four former boundary states as
development data in a new protocol, while preserving the original Phase 3G
result as an immutable failed gate.

## Next experiment

The next protocol should use the 13 existing states for development and collect
new held-out validation states along the uncovered Test 1 envelope. The offline
geometry identifies seven preliminary states:

| Gain (dB) | Effective noise (dB) |
| ---: | ---: |
| -14 | -34 |
| -2 | -28 |
| 0 | -21 |
| -6 | -18 |
| -10 | -20 |
| -14 | -23 |
| -18 | -27 |

These states are preliminary, not frozen, and not authorized for execution. The
next step is to freeze the validation thresholds, ordering, safety rules,
clipping policy, execution count, and runner/profile provenance. A POWDER
reservation should be requested only after that freeze.
