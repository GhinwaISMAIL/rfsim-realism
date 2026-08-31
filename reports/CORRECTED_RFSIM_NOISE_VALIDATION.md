# Corrected RFsim Noise Validation

## Decision

The corrected RFsim noise control passed the frozen validation protocol on the pinned OAI revision and POWDER configuration. All 15 executions were valid, every randomized repetition showed a monotonic SINR response, and the original UE image was restored successfully.

This result validates a simulator control. It does not identify environmental noise power or select a calibrated fixed-noise value.

## Measured response

| Commanded noise (dB) | Mean execution-median SS-SINR (dB) | Between-execution SD (dB) | Mean integer SS-RSRP (dBm) |
| ---: | ---: | ---: | ---: |
| -60 | 47.89 | 0.37 | -97.00 |
| -40 | 39.38 | 0.18 | -97.00 |
| -30 | 30.40 | 0.26 | -97.00 |
| -25 | 25.21 | 0.06 | -97.00 |
| -20 | 19.91 | 0.18 | -96.67 |

All three repetitions were strictly monotonic. The adjacent `-25` to `-20 dB` change reduced mean execution-median SINR by 5.30 dB; its execution-level 95% bootstrap interval was -5.48 to -5.15 dB. Similar separation was observed for every adjacent pair.

The high-SINR response begins to compress at `-60 dB`, where the reported SINR is near 48 dB. From `-40` through `-20 dB`, the response is close to an equal-and-opposite mapping between commanded noise and reported SINR.

## UPV development comparison

Only the five frozen development folds were used. The untouched Test 6 session was not opened.

- UPV route-summary SINR spans 10.78 to 22.11 dB, with a median of 15.50 dB.
- The lowest validated RFsim state has a mean execution-median SINR of 19.91 dB, leaving a 4.41 dB gap to the UPV development median.
- UPV relative-RSRP route summaries have a 5th-to-95th-percentile span of 9.98 dB.
- Noise-only RFsim RSRP has a corresponding span of only 0.022 dB.

Therefore, noise control alone cannot reproduce the target radio-condition process. It provides a reliable SINR control axis, while scalar gain remains necessary for relative-RSRP variation. A fixed-noise value is not selected from this experiment.

## Next gate

The next step is offline: freeze a bounded gain/noise replay protocol using development data and the validated response. No new reservation should be requested yet. If that protocol requires controls outside the validated range, a separate safety/localization experiment must be frozen first, with at least 30 minutes' reservation notice.

Absolute RSRP calibration, environmental-noise inference, the gain/noise inverse mapping, final Test 6 validation, and ABC remain unauthorized.
