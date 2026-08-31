# Phase 3G bounded gain/noise response result

## Campaign validity

The frozen campaign completed all 45 planned execution units and collected 675
synchronized RSRP/SINR observations. Every execution remained attached, retained
all 15 usable paired samples, and achieved a ping success fraction of 1.0. No
critical PBCH, PUSCH, random-access, radio-link, RLC, UE-restart, or gNB-restart
failure was recorded.

The runner restored the original UE image, re-established attachment, left the
gNB untouched, and restored the downlink AWGN controls to gain 0 dB and effective
noise -30 dB. The final Test 6 payload was not accessed. Direct trace replay and
ABC remained unauthorized.

The archived hardware evidence has SHA-256:

`3764e3bc400287df9fb008c6363daedaceb60dc7f890e6551a6b88ada8275408`

## Central crossed response

The response was fitted to execution medians from the 27 central factorial
executions. Gain and noise were centred at -10 dB and -25 dB. The local Jacobian
is:

\[
J =
\begin{bmatrix}
0.9671 & 0.0349 \\
1.0409 & -0.9990
\end{bmatrix},
\]

where rows are relative RSRP and SINR and columns are gain and effective noise.

The execution-level bootstrap results are:

| Response | Control | Estimate | 95% interval | Frozen range |
| --- | ---: | ---: | ---: | ---: |
| Relative RSRP | Gain | 0.9671 | 0.9656 to 0.9686 | 0.8 to 1.2 |
| Relative RSRP | Noise | 0.0349 | 0.0339 to 0.0358 | -0.2 to 0.2 |
| SINR | Gain | 1.0409 | 0.9820 to 1.1017 | 0.5 to 1.5 |
| SINR | Noise | -0.9990 | -1.0379 to -0.9596 | -1.2 to -0.7 |

The Jacobian condition number is 2.6306 with a 95% interval of 2.4961 to
2.7805, below the frozen maximum of 10. The gain/noise interaction coefficients
are -0.0076 for relative RSRP and 0.0095 for SINR; their intervals remain inside
the frozen absolute limit of 0.1.

These results support local separation of the two controls in the central
factorial region. Gain primarily controls relative RSRP. Both gain and effective
noise affect SINR, but the crossed response remains well conditioned.

## Boundary validation

The four held-out boundary states all remained operational. Using the central
factorial model, the maximum state-mean point errors were:

- Relative RSRP: 0.6088 dB, below the 1 dB limit.
- SINR: 1.2536 dB, below the 2 dB limit.

The 95% bootstrap interval for the maximum relative-RSRP error was 0.5538 to
0.6679 dB and passed. The corresponding SINR interval was 0.2892 to 3.7578 dB.
Its upper bound exceeds the frozen 2 dB limit.

The uncertainty increases mainly because the most distant boundary points
extrapolate beyond the central `[-12, -8]` dB gain and `[-28, -22]` dB noise
grid. With only three execution units per state, uncertainty in the fitted SINR
slopes and interaction is amplified at those points.

## Decision

The point-estimate coefficient, condition-number, interaction, and boundary
gates all pass. Their uncertainty gates also pass except for the held-out SINR
boundary-error gate.

The frozen decision is therefore:

> `gain_noise_mapping_not_identifiable_in_target_region`

This wording applies to the complete target control envelope, not the central
factorial region. The central response is locally identifiable and behaves in
the intended directions, but the current evidence is not precise enough to
authorize direct Test 1 trace replay over the full envelope.

The next step is to revise and freeze a targeted response design closer to the
required boundary region, or narrow the replay envelope before execution. No
direct replay, final Test 6 access, absolute-RSRP calibration, cross-session
claim, physical-channel reconstruction, or ABC inference is authorized by this
result.
