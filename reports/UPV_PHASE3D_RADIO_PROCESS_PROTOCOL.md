# UPV Phase 3D Radio-Condition Process Protocol

## Objective

Phase 3D replaces the former ABC and physical path-loss objective. Its frozen
research question is:

> Can a stochastic radio-condition model learned from one-cell measurements,
> when translated into RFsim scalar-gain and effective-noise controls,
> reproduce predefined marginal, joint, and temporal properties of held-out
> relative RSRP and SINR in OAI?

The target is a statistical radio-condition process. It is not a channel
impulse response, a multipath reconstruction, or a physical propagation model.
Attachment behaviour is outside the validation target because the UPV files do
not provide labelled failed attachment attempts.

## Data roles and access boundary

Corrected ASUS Tests 1 through 5 are the development corpus. Development uses
leave-one-complete-session-out cross-validation. Every preprocessing estimate,
spatial mean, model parameter, bandwidth, and comparison threshold is fitted
inside the training side of each fold.

`Test_6/Test_6_ASUS.csv` is the pre-designated final evaluation session. Phase
3D development code must verify that it is present in the frozen archive but
must not read its payload. Opening it requires a separate post-freeze command
and an explicit recorded authorization.

Phase 1 previously parsed all files for inventory, completeness, GPS, and
global-range diagnostics. The final session is therefore untouched by Phase 3D
model selection and threshold tuning, but it is not described as never parsed.
All S25 sessions remain outside development and final model selection; they are
reserved for an optional later device-transfer analysis.

## Preprocessing

The paired observation is

\[
\mathbf X_t=
\begin{bmatrix}
\Delta\mathrm{RSRP}_t\\
\mathrm{SINR}_t
\end{bmatrix}.
\]

Measurements are aggregated into one-second medians without interpolation.
Rows missing either feature are dropped. Gaps longer than two seconds split a
sequence and are never bridged. Relative RSRP is defined by subtracting the
complete-session median after aggregation; this removes the unresolved absolute
OAI-to-NEMO RSRP origin.

Route conditioning uses 0.05-wide route-fraction bins. Within each training
fold, the route-dependent mean is the median of the contributing per-session
bin medians. A bin requires support from at least three training sessions.
Unsupported holdout positions are excluded and reported; interpolation and
spatial extrapolation are prohibited.

## Candidate processes

The frozen comparison contains one-state and two-state versions of three joint
emission families:

- Multivariate Gaussian in the dB domain.
- Multivariate Student-t in the dB domain with five degrees of freedom.
- Gamma relative-power for de-logged relative RSRP combined with a Gaussian
  SINR emission under a shared state.

The Gamma candidate models

\[
Y_t=10^{\Delta R_t/10}
\]

as a positive relative-power ratio. It does not treat a centred dB value as
linear power and does not assume that SINR is Gamma-distributed.

Two-state candidates use a shared latent state for the paired observation. They
are fitted from twenty deterministic initializations. Labels are aligned by
increasing expected relative RSRP and retain no causal or physical names. Any
fit with a state occupancy below five percent, an expected dwell below 1.25
rows, a non-finite likelihood, or an incomplete fold is rejected.

## Block-bootstrap baseline

The baseline resamples synchronized five-row blocks from training residual
sequences. A source block never crosses a session boundary. The two radio
features are never sampled independently. The generated residual process is
combined with the training-only route mean for the held-out route coordinates.

## Predefined evaluation

For each fold and generator, 100 synthetic traces are produced with frozen
seeds. Joint distribution error is the nonnegative biased RBF MMD squared after
training-only robust scaling. Temporal error is the mean of:

- Feature-wise ACF RMSE at 1, 2, 3, 5, and 10 seconds.
- Feature-wise one-step increment Wasserstein distance after training scaling.

Parametric predictive performance is the held-out log likelihood per supported
row. A two-state candidate must improve the median score over the matching
one-state emission by at least 0.01 nats per row.

A two-state HMM is better than the block bootstrap only if it reduces both
median joint MMD and median temporal error by at least ten percent, wins each
primary metric in at least four of five folds, satisfies the predictive-score
requirement, and passes every state-stability rule.

The block baseline is supported only if it falls below the training-session
pairwise 90th-percentile reference for joint and temporal error in at least four
of five folds. If neither gate passes, the model, conditioning variables,
targets, or dataset must be revised.

## Decision boundary

Phase 3D can select a stochastic process for later RFsim translation. It cannot
authorize ABC, final validation, a physical channel claim, or a POWDER
reservation. A future reservation requires a separately frozen two-dimensional
gain/noise response design, independent repetitions, executable failure rules,
bounded controls, and a rollback plan. The reservation notice must be issued at
least 30 minutes before the experiment is needed.
