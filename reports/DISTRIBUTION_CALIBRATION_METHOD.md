# Distribution-based RFsim calibration

## Status and scope

This document defines the method that follows the completed anchor-based RFsim
campaign. The earlier implementation and its TDL-B result remain preserved on
the `feature/rf-distribution-analysis-v1` branch at commit `5ae1859`. This new
work is isolated on `feature/distribution-calibration` so the evidence, code,
and conclusions of the first approach remain reproducible.

The immediate scope is deliberately narrow:

- one gNB and one stationary UE;
- one fixed RFsim family per campaign;
- verified, complete RFsim executions;
- real stationary traces represented by their joint RSRP-RSRQ samples; and
- radio-distribution calibration before traffic or topology generalization.

No physical distance control is assumed. The current POWDER profile does not
expose one at runtime, and the UE and gNB are not physically separated in RFsim.

## What the previous approach established

The first campaign created a reliable experimental pipeline and screened AWGN,
TDL-A, TDL-B, TDL-C, and EVA. It identified TDL-B as the strongest single
family under a pointwise rule: a real observation was called supported if an
executed RFsim anchor was within 3 dB in RSRP and 2 dB in RSRQ. The retained
TDL-B anchors supported 856 of 2,977 observations, or 28.75 percent.

That result remains useful as a baseline, but it has three scientific limits:

1. the 3 dB and 2 dB values create a hard boundary and were not learned from a
   formal measurement-error model;
2. nearest-anchor coverage considers individual points rather than the complete
   shape of each scenario distribution; and
3. passing a pointwise support rule does not establish that repeated RFsim runs
   have the same distribution.

The new approach does not erase or reinterpret that result. It asks a different
question: which executed RFsim state produces the empirical joint distribution
closest to each real stationary scenario, and is that state repeatable enough
to justify a new validation experiment?

## Scientific model

Let a verified RFsim control state be

\[
u = (\text{family}, \text{ploss}, \text{noise\_power\_dB}, \ldots).
\]

For real scenario \(s\), the observed samples are

\[
Y_s^{\mathrm{real}} = \{(r_t, q_t)\}_{t=1}^{n_s},
\]

where \(r_t\) is RSRP and \(q_t\) is RSRQ. For an executed RFsim state
\(u\), the corresponding samples from complete repetition \(k\) are

\[
Y_{u,k}^{\mathrm{sim}} = \{(\hat r_t, \hat q_t)\}_{t=1}^{m_{u,k}}.
\]

The first implementation ranks only controls that were executed and passed the
existing quality gates. It does not interpolate between controls and does not
claim support for an unobserved state.

## Primary distribution distance

RSRP and RSRQ are expressed on different numerical scales, so the comparison
uses dimensionless coordinates:

\[
z(r,q) = \left(\frac{r}{a_r}, \frac{q}{a_q}\right).
\]

The version-1 configuration uses \(a_r=3\) dB and \(a_q=2\) dB. These values
balance the two coordinates; unlike the previous method, they are not gates
that label an observation supported or unsupported. They remain a declared
modeling choice and require sensitivity analysis before the final result.

The primary discrepancy is the biased squared maximum mean discrepancy with a
Gaussian radial-basis kernel:

\[
\widehat{\operatorname{MMD}}_b^2(X,Y)
= \frac{1}{n^2}\sum_{i,i'} k(x_i,x_{i'})
+ \frac{1}{m^2}\sum_{j,j'} k(y_j,y_{j'})
- \frac{2}{nm}\sum_{i,j} k(x_i,y_j),
\]

\[
k(x,y)=\exp\left(-\frac{\lVert x-y\rVert^2}{2h^2}\right).
\]

The configured bandwidth is \(h=1\). Lower MMD means that the two empirical
joint distributions are more similar under the declared scaling and kernel.
MMD uses the complete joint RSRP-RSRQ cloud, so it responds to differences in
location, spread, dependence, and multimodality rather than only mean values.

### Adaptation of the published method

The cited kernel-calibration paper combines MMD with approximate Bayesian
computation to infer a posterior distribution over channel-model parameters.
Version 1 here implements the preceding empirical minimum-distance screen: it
computes MMD for the eight discrete RFsim states for which verified data already
exist. It does not claim to implement an ABC posterior. That later step would
require a callable simulator design, declared priors, and enough new independent
executions to evaluate many proposed controls. Keeping the first step discrete
prevents an unexecuted interpolation from being presented as evidence.

## Diagnostic distances

For interpretability, the workflow also computes a one-dimensional quantile
approximation to the first Wasserstein distance for each selected metric:

\[
W_1(X,Y) \approx \frac{1}{L}\sum_{\ell=1}^{L}
\left|Q_X(p_\ell)-Q_Y(p_\ell)\right|.
\]

These values remain in dB and show whether a poor joint match is driven mainly
by RSRP or RSRQ. Real SNR and OAI SS-SINR are summarized only as a diagnostic
proxy because their definitions are not assumed to be equivalent.

## Complete-execution repeatability

Samples from the two executions at one control state are never randomly split
into training and test rows. The workflow compares complete repetitions:

\[
D_{u,k,k'} = \operatorname{MMD}^2
\left(Y_{u,k}^{\mathrm{sim}},Y_{u,k'}^{\mathrm{sim}}\right).
\]

Marginal Wasserstein distances are reported for the same pair. This exposes a
state that has similar means but unstable distribution shapes. Version 1
reports the evidence without inventing an acceptance threshold after seeing the
results. A repeatability decision rule and distance sensitivity analysis must be
declared before a candidate is accepted.

## Version-1 workflow

1. Read the immutable real observations from the private scenario catalog.
2. Read the selected TDL-B campaign manifest and completed campaign state.
3. Require two distinct complete executions for every retained state.
4. Verify model family, execution identity, applied `ploss` and
   `noise_power_dB`, channel quality, and transition exclusion for every row.
5. Summarize every real scenario and every executed RFsim state.
6. Compute the joint MMD and marginal Wasserstein distances for every
   scenario-state pair.
7. Rank the observed states independently for every real scenario.
8. Compare complete repetitions at each RFsim state.
9. Save the code revision, tracked-worktree status, source hashes, settings,
   limitations, tables, and output checksums.

The command is:

```bash
make distribution-calibrate
```

## First diagnostic execution

The preserved inputs contain 2,977 real observations in 18 stationary
scenarios and 2,895 verified RFsim observations from 16 executions at eight
TDL-B control states. The first version therefore produces 144 scenario-state
comparisons and eight complete-repetition comparisons.

The complete-repetition diagnostics are:

| `ploss` | `noise_power_dB` | joint MMD² | RSRP W1 (dB) | RSRQ W1 (dB) | scenarios ranked first |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -20 | -30 | 1.355 | 4.748 | 0.672 | 0 |
| -15 | -30 | 1.067 | 1.496 | 2.403 | 0 |
| -10 | -30 | 0.004 | 0.574 | 0.254 | 0 |
| -5 | -30 | 0.358 | 2.235 | 0.471 | 5 |
| -5 | -7 | 0.119 | 0.898 | 0.499 | 1 |
| 0 | -20 | 1.351 | 4.765 | 0.511 | 2 |
| 0 | -10 | 1.369 | 5.533 | 0.342 | 6 |
| 0 | -5 | 0.775 | 3.014 | 0.397 | 4 |

The final column is descriptive only. It makes the central problem visible:
some states that look closest to several real scenarios are also among the
least stable across repeated RFsim executions. A closest-state count cannot be
used as a calibration decision without the repeatability analysis.

This execution is not yet a final calibration result. In particular, several
states that were retained by the older anchor procedure show large
complete-execution discrepancies under the distribution metrics. That finding
is useful: it prevents the new approach from treating every old safe state as a
stable stochastic generator. The next methodological decision must define a
repeatability rule and test whether the ranking is stable across reasonable
RSRP/RSRQ scales and kernel bandwidths.

Version 1 also compares empirical sample distributions without using their time
order. That is appropriate for the first stationary marginal-distribution
screen, but consecutive one-second radio samples can be dependent. Confidence
intervals or hypothesis tests must therefore use complete executions or a
block-aware resampling method rather than treating every second as independent.
Temporal realism, transitions, and dwell behavior remain a separate later
validation layer.

## Validation rules

A state can become a realism candidate only after all of the following:

1. both existing executions pass the immutable data and channel-quality gates;
2. complete-execution repeatability is acceptable under a predeclared rule;
3. the selected state remains competitive under distance sensitivity analysis;
4. a new POWDER execution, not used to choose the state, reproduces the target
   distribution within a predeclared criterion; and
5. any later claim about traffic, UE count, cell count, or scheduler behavior is
   validated separately under that changed condition.

Distribution similarity is evidence of observable equivalence for the selected
metrics. It is not proof that RFsim reproduces the underlying propagation
physics.

## Relationship to regression and RSSI channel estimation

A log-distance regression can estimate a path-loss exponent when transmitter
power is known and measurements include meaningful transmitter-receiver
distances. Residual variance can then describe a shadowing term. The
expectation-maximization RSSI method discussed during the method review extends
that setting to packet loss, censoring, noise, and interference.

That family of methods remains a possible future physical-model layer, but the
current data do not expose the required physical distances, per-packet
censoring threshold, and packet-loss observation model. Applying the regression
now would therefore estimate an intercept/control response, not an identifiable
physical path-loss exponent. The likelihood-free method is suitable for the
current simulator interface because it requires samples from RFsim rather than
an analytical likelihood or distance parameter.

## Literature basis

- A. Bharti, F.-X. Briol, and T. Pedersen, “A General Method for Calibrating
  Stochastic Radio Channel Models with Kernels,” *IEEE Transactions on Antennas
  and Propagation*, 2022, DOI
  [10.1109/TAP.2021.3083761](https://doi.org/10.1109/TAP.2021.3083761). This work
  establishes likelihood-free calibration of stochastic radio-channel models
  using MMD-based discrepancies.
- A. Bharti, R. Adeogun, and T. Pedersen, “Learning Parameters of Stochastic
  Radio Channel Models from Summaries,” *IEEE Open Journal of Antennas and
  Propagation*, 2020, DOI
  [10.1109/OJAP.2020.2989814](https://doi.org/10.1109/OJAP.2020.2989814). This
  supports simulator-based inference when direct likelihood evaluation is not
  practical.
- E. Bernton, P. E. Jacob, M. Gerber, and C. P. Robert, “On Parameter
  Estimation with the Wasserstein Distance,” *Information and Inference*, 2019,
  DOI [10.1093/imaiai/iaz003](https://doi.org/10.1093/imaiai/iaz003).
  Wasserstein distance is used here as an interpretable marginal diagnostic,
  not as the sole calibration objective.
- The reviewed RSSI expectation-maximization paper is retained as a possible
  future extension for a dataset containing distance, packet loss, censoring,
  and interference observations:
  [arXiv:1504.01072](https://arxiv.org/abs/1504.01072).

## Preserved artifacts

- historical technical report: `reports/COMPLETE_PROJECT_TECHNICAL_REPORT.md`;
- completed anchor-based branch: `feature/rf-distribution-analysis-v1`;
- completed anchor-based commit: `5ae1859`;
- distribution-calibration configuration:
  `configs/distribution_calibration_tdl_b_v1.yaml`; and
- private version-1 result directory:
  `model_runs/ucc_static_tdl_b_distribution_calibration_v1` in the private data
  repository.
