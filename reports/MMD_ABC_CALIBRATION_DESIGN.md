# MMD–ABC calibration design

## Status

This document predeclares the first likelihood-free RFsim calibration experiment.
It was written before any new calibration execution was requested. No new execution
may be interpreted as calibration evidence unless it follows a versioned
configuration and immutable execution plan derived from this design.

The completed anchor-based work remains historical evidence. The discrete-state MMD
screen on `feature/distribution-calibration` remains a diagnostic over eight executed
states. Neither result is an ABC posterior.

## Feasibility decision

A literal reproduction of the simulation budget in Bharti, Briol, and Pedersen is not
feasible when one simulator call is a complete POWDER execution. Their reported
starting point uses 2,000 proposals per iteration, accepts 100 proposals, and runs
several sequential iterations. Five to ten iterations would require 10,000 to 20,000
simulator calls.

At 180 seconds of treatment time per call, this is 500 to 1,000 hours before setup,
verification, archive, recovery, and reservation overhead. The completed TDL-B
campaign advanced at roughly one execution every 5.5 to 6 minutes, which raises the
single-worker elapsed estimate to about 38 to 83 continuous days for one execution
per proposal. Independent repetitions would multiply that cost. Consecutive seconds
inside one execution are not substitutes for independent simulator calls.

The implemented workflow is therefore asynchronous and execution-bank based. It:

1. creates a deterministic proposal and repetition plan;
2. waits for independently archived simulator executions;
3. treats every complete execution as one stochastic simulator draw;
4. computes MMD without pooling repetitions into a deterministic state;
5. forms a rejection-ABC posterior only when predeclared sample and effective-sample
   gates pass; and
6. otherwise reports an underpowered pilot without calling it a posterior result.

POWDER is suitable for sparse design points and final posterior-predictive validation.
It is not suitable for the inner loop of paper-scale PMC-ABC. A validated local
x86-64 OAI harness or a surrogate for the execution-level discrepancy will be needed
before a well-resolved continuous posterior is practical.

## 1. Parameter vector

The first inferential vector is deliberately one-dimensional:

\[
\theta = (\mathrm{ploss}).
\]

The following quantities are fixed by the first configuration:

| Quantity | First-stage value | Reason |
| --- | ---: | --- |
| RFsim family | `TDL_B` | Hold the model family fixed while numeric controls are identified. |
| `noise_power_dB` | `-30` | Attachment-safe baseline with existing repeated evidence. |
| TDL delay spread | 30 ns | Boot-time value used by the repaired TDL images; it is not verified by the runtime helper. |
| `forgetf` | `0` | Existing campaign value; its main expected effect is temporal dependence, which is not the first selection target. |
| `offset` | `0` samples | Timing/alignment control, not an initial propagation-distribution parameter. |
| `riceanf` | unchanged | TDL tap powers define this family; no active, identifiable first-stage role has been established. |
| `aoa` and `randaoa` | unchanged | The current one-antenna calibration does not identify an angle-of-arrival effect. |
| topology | one gNB, one stationary UE | Required isolation stage. |

This restriction is not a claim that `noise_power_dB` is unimportant. It is an
identifiability and cost decision. A two-dimensional vector
`(ploss, noise_power_dB)` may be declared only after the one-dimensional response,
replicate variance, and operational boundary are measured. `forgetf` belongs in a
later temporal-calibration stage, where transitions, autocorrelation, and dwell
statistics are targets.

## 2. Channel-family decision

TDL-B remains fixed for the first implementation. Earlier screening made it the
strongest family under the discontinued pointwise rule, which is sufficient to make
it a reasonable first family to investigate but not sufficient to accept it as
realistic. Model-family comparison is a later outer loop. Mixing family selection
with continuous-parameter inference would confound model misspecification with
parameter uncertainty and multiply the execution budget.

## 3. Ranges and priors

The pilot prior is uniform on:

\[
\mathrm{ploss} \sim \mathcal U(-10, 0)\ \mathrm{dB}.
\]

This is an operational prior, not a physical-distance prior. Both ends have archived
TDL-B executions at `noise_power_dB=-30`; the range avoids the much weaker `-15` and
`-20` states, whose observed RSRP is mostly below the retained real scenarios and
whose repetitions were unstable. The range must not be interpreted as a calibrated
path-loss exponent or a distance model.

The initial deterministic design uses eight values spanning both bounds and three
independent executions at each value. Interior values are proposals,
not presumed-safe results. Every run must pass the existing attachment, readback,
clock, archive, and measurement-count gates.

No rectangular two-dimensional prior is declared yet. In particular, the archived
points do not establish that every combination in
`ploss in [-10, 0]` and `noise_power_dB in [-30, -7]` is safe: the joint extension
already showed unstable behavior around `(-10, -10)` and `(-5, -10)`.

## 4. Simulator output

The primary simulator output is the settled, verified joint sample:

\[
X_{\theta,k} = \{(\mathrm{SS\mbox{-}RSRP}_t,
                    \mathrm{SS\mbox{-}RSRQ}_t)\}_{t=1}^{m_{\theta,k}},
\]

from complete execution `k`. The matching real target is the joint RSRP/RSRQ sample
from one stationary scenario. Real SNR and OAI SS-SINR remain separate diagnostics.
Packet outcomes are retained for later traffic validation but do not enter the first
radio-distribution discrepancy.

Only rows that pass the channel schedule, state success, readback, transition, model
family, execution identity, and minimum-sample gates are accepted.

## 5. Raw samples and summaries

MMD is computed on the raw joint RSRP/RSRQ samples after a reference-derived linear
transform. Low-dimensional summaries are also saved, but they have three diagnostic
roles only:

- detecting obvious model misspecification by comparing real summaries with the
  ranges generated by the simulator bank;
- explaining whether a mismatch is driven by location, spread, or dependence; and
- assessing parameter sensitivity and identifiability.

Summaries do not replace the primary raw-sample MMD and are not combined into an
unreviewed scalar score.

## 6. Metric scaling and kernel bandwidth

The old 3 dB and 2 dB support tolerances are not used.

The two radio coordinates are centered and whitened using the covariance of all
eligible real observations. A small eigenvalue floor is predeclared for numerical
stability. This makes the metric dimensionless while retaining the real-data
correlation structure.

The primary RBF length scale is fixed before looking at simulated candidates. It uses
the median heuristic on a deterministic subset of the whitened pooled real reference:

\[
\ell = \sqrt{\operatorname{median}_{i<j}\|z_i-z_j\|^2/2}.
\]

The same transform and length scale are used for every proposal and every scenario.
Sensitivity results are also computed at `0.5 * ell` and `2 * ell`; proposal
acceptance uses only the predeclared primary scale. The implementation uses the
unbiased empirical MMD-squared estimator from the paper and clips negative finite-
sample estimates to zero only for ABC weighting. Raw estimates remain in the output.

## 7. Evaluation and repetition budget

Paper-scale rejection or PMC-ABC should target at least 100 accepted samples. At a
5 percent acceptance fraction this requires at least 2,000 independent simulator
draws per iteration. That is the target for a cheap local simulator, not for POWDER.

The first POWDER experiment is a sensitivity and stochasticity pilot:

- eight `ploss` values;
- three independent complete executions per value;
- 24 calibration executions;
- 180 seconds per execution;
- at least 120 verified settled samples per execution; and
- no claim of a resolved ABC posterior from those 24 executions.

The pilot requires 72 minutes of treatment time and approximately 2.2 to 2.4 hours at
the historical campaign cadence. It determines whether the discrepancy changes
smoothly with `ploss`, whether within-value variance is smaller than between-value
variation, and whether the one-dimensional prior reaches the real scenarios.

## 8. Where simulations can run

The current workstation has archived observations and orchestration code, but it does
not currently provide a validated callable OAI RFsim simulator that reproduces the
same UE measurement path. PHY-only simulators are not interchangeable with the
end-to-end SS-RSRP/SS-RSRQ observations.

A future local harness is acceptable only if it uses the pinned OAI revision, repaired
TDL implementation, same gNB/UE configuration and reporting offset, and reproduces a
small set of POWDER controls within predeclared complete-execution variability. Local
throughput is useful only after that equivalence gate passes.

## 9. Budget-control method

The decision sequence is:

1. run the one-dimensional pilot;
2. quantify execution-level variance and parameter sensitivity;
3. stop if the reachable summary ranges exclude the targets or if the discrepancy is
   dominated by run-to-run variance;
4. otherwise build and validate a local harness, or fit a heteroscedastic surrogate to
   execution-level MMD values;
5. use sequential proposals only after the emulator/harness passes held-out checks;
6. keep new POWDER executions for sparse design updates and final validation.

Plain Bayesian optimization can find a minimum but does not by itself produce the
required scenario distribution. Neural simulation-based inference has a still larger
simulation requirement and is not the first choice. A surrogate-assisted ABC or
Bayesian optimization for likelihood-free inference is the closest defensible method
when only expensive POWDER calls are available.

The code implements deterministic rejection-ABC over an execution bank and refuses to
label an underpowered bank as established inference. Sequential or surrogate proposals
can be added without changing the data contract.

## 10. Run-to-run variability

Each complete execution is one stochastic draw from `P_theta`. Repetitions are never
pooled before discrepancy calculation. ABC operates on execution-level discrepancies,
so a parameter value that occasionally matches and often fails receives the
corresponding mixed weight rather than a favorable pooled average.

Outputs include per-parameter replicate counts, mean and standard deviation of MMD,
within-value variation, and complete execution identifiers. Confidence intervals must
use execution-level or block-aware resampling, never independent one-second rows.

## 11. Identifiability

The pilot reports, without automatically declaring success:

- the number of unique parameter values represented among accepted draws;
- posterior standard deviation relative to the prior standard deviation;
- posterior mass near each prior boundary;
- weighted parameter covariance and correlation for multi-parameter extensions;
- between-value versus within-value discrepancy variation;
- monotonicity and local flat regions in execution-level discrepancy; and
- stability of all conclusions across the bandwidth sensitivity values.

Parameters with broad, boundary-concentrated, multimodal, or strongly correlated
posteriors are not individually identifiable. If multiple controls yield equivalent
radio distributions, the deliverable should be a scenario-generation policy over that
equivalence set rather than a falsely precise point estimate.

## 12. Held-out posterior-predictive validation

Validation executions must be collected after inference and may not be added back to
the calibration bank before the validation decision. For each predeclared target
scenario:

1. select posterior candidate controls by weighted posterior quantiles or draws;
2. execute at least three new complete repetitions per selected control;
3. apply the unchanged row-quality gates, real-reference transform, kernel bandwidth,
   and MMD estimator;
4. compare held-out execution-level MMD values with the calibration posterior-
   predictive discrepancy distribution;
5. report real and simulated joint plots plus marginal Wasserstein diagnostics;
6. test summary-range coverage and bandwidth sensitivity; and
7. retain failures and outages as rejected evidence.

A candidate is not accepted merely because its held-out mean is close. The held-out
distribution and its execution-to-execution variability must be consistent with the
predeclared posterior-predictive interval. Temporal correlation and dwell behavior are
separate later validation gates.

## 13. Reused and replaced implementation

Reusable parts of `distribution_calibration.py` are:

- immutable input and checksum handling;
- real and RFsim column contracts;
- applied-control and quality-gate verification;
- scenario grouping;
- complete-execution identity;
- marginal Wasserstein diagnostics; and
- atomic checksummed output creation.

The following parts are replaced or extended:

- fixed 3 dB/2 dB coordinate scaling is replaced by reference covariance whitening;
- fixed bandwidth 1 is replaced by a frozen real-reference median heuristic;
- biased MMD is replaced by the paper's unbiased estimator for inference;
- pooled state rankings are replaced by execution-level proposal discrepancies;
- nearest-state labels are replaced by rejection weights and posterior summaries;
- repeatability becomes part of the stochastic simulator rather than a separate veto;
- explicit inference sufficiency, misspecification, sensitivity, and identifiability
  diagnostics are added; and
- deterministic proposal manifests support future unexecuted controls without
  pretending that they have already been observed.

## 14. Smallest defensible first experiment

Generate the configured 24-execution TDL-B `ploss` pilot and run it only after a new
reservation is available and a single pilot point passes all operational gates. Do not
start with two parameters, a new family, multiple UEs, or changed traffic.

The decision after those 24 executions is one of three outcomes:

- **stop for misspecification:** the real scenario summaries lie outside the simulator
  envelope or MMD remains large throughout the prior;
- **stop for stochastic instability:** within-value variation is comparable to or
  larger than the parameter response; or
- **continue:** the response is identifiable enough to justify a validated local
  harness or a small surrogate-assisted sequential design.

The pilot is not allowed to revive pointwise nearest-anchor coverage, and it is not
allowed to be reported as a completed continuous ABC posterior.

## Literature and implementation basis

- A. Bharti, F.-X. Briol, and T. Pedersen, “A General Method for Calibrating
  Stochastic Radio Channel Models with Kernels,” *IEEE Transactions on Antennas and
  Propagation*, 2022, DOI `10.1109/TAP.2021.3083761`.
- OAI RFsim channel-model documentation and the pinned OAI revision recorded by the
  experiment profile.
- The configured-fading repairs and TDL delay-spread configuration on the current
  `feature/rfsim-rsrp-calibration` profile branch.
