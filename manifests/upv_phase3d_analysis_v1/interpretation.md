# Phase 3D Development Result

The frozen development decision is `radio_process_model_revision_required`.
No stochastic process is selected, final evaluation remains unauthorized, and
the POWDER reservation gate remains closed.

## Evidence

All five corrected ASUS development sessions passed the source, paired-row,
serving-PCI, sequence-length, and spatial-support gates. Every fold retained
100 percent of its positions under training-only route conditioning. The locked
`Test_6/Test_6_ASUS.csv` payload was not opened.

The synchronized five-row moving-block bootstrap passed the fold-specific joint
distribution reference in three of five folds and the temporal reference in
zero of five folds. It therefore failed the predefined four-of-five support
rule.

Every two-state fit passed the frozen occupancy and dwell-time rules. Compared
with the matching one-state emission, median held-out predictive log-score
improvements were:

- Gamma-Gaussian: 0.3031 nats per supported row.
- Student-t: 0.1872 nats per supported row.
- Gaussian: 0.1313 nats per supported row.

This supports a repeatable two-state statistical structure in the development
data. It does not establish that the states have causal or physical meanings.

None of the two-state models satisfied the generative gate. Relative to paired
block replay, median temporal error was 78.2 percent worse for Gamma-Gaussian,
89.9 percent worse for Student-t, and 107.9 percent worse for Gaussian. Joint
MMD improvement was also below the frozen ten-percent margin for every model.

## Interpretation

The negative decision is specifically a temporal-generation failure, not a
failed HMM optimization or a rejection of all statistical structure. The
state-conditioned emissions are independent within a state and do not preserve
the residual short-range dynamics. Conversely, the frozen five-row block
baseline breaks dependence at block joins and cannot directly preserve all
predefined ACF lags through ten seconds. Its zero-of-five temporal support result
shows that it is not yet an adequate replay process.

The next revision remains offline. It should estimate an admissible block
length from training-only correlation diagnostics and compare that baseline
with an autoregressive joint model and an autoregressive HMM. A hidden
semi-Markov model is justified only if the revised diagnostics show that state
dwell distributions, rather than within-state autocorrelation, remain the
limiting error. Test 6 must remain locked throughout that revision.

No gain/noise response experiment, final evaluation, ABC analysis, or POWDER
reservation is authorized by this result.
