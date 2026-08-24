# RFsim Realism

This project calibrates OAI RFsim channel controls against real-world radio
measurements and validates the resulting scenarios with packet and radio
observations from the POWDER testbed.

The first reference source is the UCC 5G production dataset. Raw source files
remain unchanged and outside Git. A pinned source record, archive checksum,
curation policy, and deterministic manifest make the analysis reproducible.

## Current scientific direction

The completed anchor-based campaign is preserved as a baseline. Its best
single-family result was TDL-B, for which the old pointwise support rule covered
28.75 percent of the retained real observations. That percentage is not treated
as the final definition of RF realism.

The current approach compares complete empirical distributions instead. For
each stationary real trace, it ranks only RFsim states that were actually
executed and passed the data-quality gates. The primary comparison is joint
RSRP-RSRQ maximum mean discrepancy (MMD); marginal Wasserstein distances and
complete-execution repeatability are retained as diagnostics. This is a
likelihood-free calibration workflow: it does not require RFsim to expose a
physical UE distance parameter or an analytically tractable likelihood.

The method, assumptions, validation rules, and literature basis are recorded in
`reports/DISTRIBUTION_CALIBRATION_METHOD.md`. The older technical report remains
an unchanged historical record of the testbed and anchor campaign.

## Setup

Open this directory in VS Code and install the recommended Python and Jupyter
extensions when prompted. Then run:

```bash
make setup
make fetch-ucc
make curate-static
make static-report
make test
```

Select `.venv/bin/python` as the VS Code interpreter and notebook kernel.
Open `notebooks/01_static_dataset_review.ipynb` to inspect the selected traces,
quality flags, and radio distributions. The notebook reads the deterministic
manifest; it does not define the selection logic.

## UPV prior-support protocol

The first UPV phase prepares an external connected-UE target without treating
the mobile route as a stationary OAI measurement. It freezes the source archive,
records the provisional ASUS Test 1/Test 2 relabelling, reconstructs cumulative
route distance, assigns 10/15/20 metre bins, and locks one calibration region
plus four spatial-validation regions using geometry, sample count, and dwell
time only.

After placing the pinned archive described in
`sources/upv_remote_driving_n40_v1.json` under `data/raw/`, run:

```bash
make prepare-upv
```

The result is a data-engineering and prior-support input. It must not be passed
to ABC inference until the existing RFsim bank demonstrates joint support and
the sensitivity analysis supports local parameter identifiability.

Phase 2 is a diagnostic existing-bank comparison, not ABC. The frozen
specification in `configs/upv_support_v1.yaml` uses RSRP then SINR, one-second
median aggregation, a balanced robust reference mixture, one global RBF
bandwidth, execution-level candidate ranking, and traversal-conditional block
bootstrap intervals. Run it only after `make prepare-upv`:

```bash
make upv-support
```

The analysis records the Phase 1 Git SHA in every JSON manifest, verifies both
route-distance and Euclidean validation separation, retains the original ASUS
filename interpretation as a sensitivity analysis, and limits derivatives to
the two marginal sweeps supported by the current RFsim grid. With only two
executions per state, no result from this target may be described as an ABC
posterior or final calibration.

Phase 3A audits measurement equivalence before any support extension. It traces
the exact OAI SS-RSRP formula, the profile's RFsim reporting-offset patch, the
configured SSB power fields, and the UPV radio-column completeness. The audit
requires clean checkouts at the revisions recorded by the RFsim executions:

```bash
make upv-measurement-audit \
  UPV_AUDIT_OAI_SOURCE=/path/to/openairinterface5g \
  UPV_AUDIT_PROFILE_SOURCE=/path/to/oai-5g-ric
```

The Phase 2 files remain an immutable snapshot. Future ABC uses a nonnegative
biased MMD-squared V-statistic under `upv-support-v2`; clipping negative
unbiased estimates is prohibited. The first `{0,+2.5}` positive-`ploss`
experiment, if later authorized, is a safety and interaction probe rather than
a final support-extension design. Do not request POWDER until
`reservation_gate_v2.json` opens and the exact experiment is frozen.

The next protocol can be prepared offline without calculating new distances:

```bash
make upv-support-v2-plan
```

This writes a non-executable plan. It keeps the measurement branch unresolved,
records the 30-minute reservation-notice rule, and freezes the six-state
positive-`ploss` safety probe as 18 minimum or 30 preferred executions.

## Current static reference catalog

The official `v1.0.0` archive contains 23 static traces: 8 Amazon Prime,
5 Download, and 10 Netflix. The current policy retains 18 dynamic static
windows for later replay, 4 steady windows as calibration anchors, and
quarantines 1 trace whose static label conflicts with its reported speed.

Every chosen window is 180 seconds, 5G-only, and single-cell. Android duplicate
timestamps are resolved by keeping the row with the most complete radio
measurements. Missing seconds remain explicit; they are not silently filled.

## Calibration order

1. Preserve the real stationary traces as joint, time-ordered observations.
2. Accept only complete RFsim executions with verified applied controls and
   valid measurement timing.
3. Measure distributional repeatability between complete repetitions at every
   executed RFsim state.
4. Compare each real trace with each observed safe state using joint RSRP-RSRQ
   MMD and marginal Wasserstein diagnostics.
5. Rank observed states without claiming that an unexecuted control pair is
   supported.
6. Test any proposed new control on POWDER as a new held-out execution before
   adding it to the calibration corpus.
7. Expand to other traffic loads, UE counts, cells, or dynamic channels only
   after the stationary one-cell, one-UE result is repeatable.

CQI, MCS, BLER, throughput, loss, and latency are observations. They are not
used as direct RFsim controls.

The numeric RSRP and RSRQ scales in the calibration configuration balance the
two coordinates; they are not hard support thresholds. A sensitivity analysis
must be completed before a final scientific claim is made.

## Automated AWGN sweep

The campaign uses one immutable 180-second execution for each treatment and
repetition. Every execution has a 45-second baseline lead, a 90-second
treatment segment, and a 45-second baseline tail. Only the final 45 seconds of
the treatment segment are used as the settled comparison window.

Generate the deterministic plan with:

```bash
make sweep-plan
```

The runner checks the core/cell NTP offset spread, the dashboard preflight,
live baseline readback, channel verification, xApp shutdown, packet clock
validity, radio clock lag, immutable checksums, and treatment-segment training
eligibility. Campaign state is written atomically under
`data/calibration_runs/`, so an interrupted campaign resumes without repeating
successful points.

Run one pilot point before a full campaign:

```bash
PYTHONPATH=src uv run --locked rfsim-realism sweep-run \
  --config configs/awgn_calibration_v1.yaml \
  --run-dir /absolute/path/to/traffic_profiles/run_name \
  --dashboard-repo /absolute/path/to/traffic-generation-dashboard \
  --state data/calibration_runs/awgn_campaign.json \
  --point noise_power_dB-m25-r1
```

## UCC static AWGN grid

The UCC matching campaign uses complete 180-second static executions over
`ploss = [0, -3, -7]` and `noise_power_dB = [-2, 0, 2, 4]`, with two
executions per point. Both controls are set and read back before traffic starts.
Dataset Contract V2 stores the requested and applied pair in one segment row,
and rejects the row if either value disagrees.

Generate the deterministic 24-point plan:

```bash
make grid-plan
```

Run one point before continuing the resumable campaign:

```bash
PYTHONPATH=src uv run --locked rfsim-realism grid-run \
  --config configs/ucc_static_grid_v1.yaml \
  --run-dir /absolute/path/to/traffic_profiles/run_name \
  --dashboard-repo /absolute/path/to/traffic-generation-dashboard \
  --state data/calibration_runs/ucc_static_grid_v1.json \
  --point ploss-p0_noise-m2-r1
```

The runner restores `ploss = 0` and `noise_power_dB = -30` after every
execution, including failed quality gates.

## Completed anchor-based baseline

The following mapping and family-screening sections record the completed first
approach. They are retained for provenance, comparison, and reproducibility.
They do not overwrite the current distribution-based method.

### First static mapping

The first mapping is deliberately restricted to the eight observed safe AWGN
states. It uses the two verified applied controls as conditional pre-run inputs
and treats RSRP, RSRQ, SINR, latency, loss, and received throughput as outputs.
It does not use post-run radio observations in the input matrix.

Application targets are reconstructed from `packet_outcomes`: segment p95 is
the direct 0.95 quantile of valid received-packet latency, and DL remains
separate from uncontrolled UL. Each execution is held out in turn and predicted
from the other execution at the same verified state. This measures repeatability
inside the safe grid; it is not evidence of generalization to an unobserved
control state.

Run the local mapping after the private Dataset V2 repository is available:

```bash
make static-map
```

The command verifies the dataset checksums and writes checksummed derived
artifacts under `data/model_runs/`. Candidate matching uses RSRP and RSRQ and
ranks only observed states. UCC SNR versus OAI SS-SINR is retained as a
diagnostic proxy because the measurement definitions are not assumed equal.
No POWDER reservation is required until candidate states are ready for new
held-out validation executions.

### TDL-B calibration result

The first TDL-B campaign retained seven control states with two complete
executions per state. Against the 2,977 complete RF observations in the 18
eligible UCC scenarios, those states represented 28.75 percent of observations,
compared with 16.33 percent for the retained AWGN mapping.

A joint-control extension then tested `(-5, -10)`, `(-10, -10)`, and
`(-5, -7)` as `(ploss, noise_power_dB)` pairs, twice each. The `(-10, -10)`
pair is boundary evidence because its repetitions changed from 99.77 percent
packet loss to zero loss. The `(-5, -10)` pair remained connected but its two
mean RSRP measurements differed by 5.44 dB. Only the repeatable `(-5, -7)` pair
was added to the conservative mapping.

The resulting eight-state TDL-B mapping still represents exactly 856 of 2,977
observations, or 28.75 percent. The additional state changes nearest-state error
values but adds no new representable observation. RSRQ remains the limiting
metric: the retained TDL-B states remain near -10 to -11 dB while many uncovered
real observations are near -13 to -16 dB. Further TDL-B points in this region
are therefore unlikely to expand support; the next calibration should compare
a different channel family using the same one-cell, one-UE procedure.

### TDL-C family decision

The TDL-C comparison retained three repeatable states from sixteen complete
executions. Under the same RSRP and RSRQ tolerances used for TDL-B, TDL-C
represents 247 of 2,977 real observations, or 8.30 percent. TDL-B represents
856 observations, or 28.75 percent, and is therefore the selected single
family for the first realistic traffic-generation dataset.

Of the 247 TDL-C-supported observations, 121 are also supported by TDL-B. The
126 observations unique to TDL-C are Netflix observations and are all mapped
to `(ploss = 0, noise_power_dB = -5)`. Combining the families would increase
support from 28.75 to 32.99 percent, but that 4.23 percentage-point gain does
not replace the simpler single-family design. The TDL-C campaign remains useful
as complementary evidence and the `(0, -5)` state can be revisited in a later
multi-family extension.

Reproduce the checksummed comparison with:

```bash
make family-compare
```

The comparison verifies both input bundles, requires identical real
observations, reports overlap and per-state contributions, and selects the
single family with the larger representable count. Its output remains under
`data/model_runs/` and belongs in the private data repository.

### TDL-A family screening

The TDL-A screen completed eight executions across four control states, with
two repetitions per state. Three states met the 3 dB RSRP and 2 dB RSRQ
repeatability limits: `(-5, -30)`, `(0, -30)`, and `(0, -10)` as
`(ploss, noise_power_dB)`. The `(0, -5)` pair remains boundary evidence
because its execution-level mean RSRP values differed by 4.82 dB.

The conservative TDL-A mapping represents 792 of 2,977 real observations, or
26.60 percent. This is greater than AWGN's 16.33 percent and TDL-C's 8.30
percent. However, all 792 TDL-A-supported observations are also supported by
TDL-B. TDL-B represents 64 additional observations, for 856 total, so TDL-B
remains the selected single family for the first realistic dataset.

The TDL-A mapping, support catalog, and pairwise family comparisons are under
`data/model_runs/`. The campaign plan, safe-state selection, and mapping
configuration are under `manifests/` and `configs/`.

### EVA family screening

The EVA screen completed eight executions across four control states, with two
repetitions per state. Three states met the 3 dB RSRP and 2 dB RSRQ
repeatability limits: `(-5, -30)`, `(0, -10)`, and `(0, -5)` as
`(ploss, noise_power_dB)`. The `(0, -30)` pair was excluded because its
execution-level mean RSRP values differed by 16.82 dB.

The conservative EVA mapping represents 717 of 2,977 real observations, or
24.08 percent. This exceeds AWGN's 16.33 percent and TDL-C's 8.30 percent, but
is below TDL-A's 26.60 percent and TDL-B's 28.75 percent. EVA adds 54
observations beyond TDL-A, producing a combined support of 846 observations,
or 28.42 percent.

Every EVA-supported observation is already supported by TDL-B. TDL-B supports
139 additional observations, so adding EVA would not extend the selected
single-family design. EVA remains useful as independent family-screening
evidence, while TDL-B remains the selected family for realistic dataset
generation.

## Real RF scenario catalog

Build a replay-ready catalog directly from the dynamic static UCC measurement
windows:

```bash
make rf-distribution
```

Each eligible trace window becomes one selectable RF scenario. The analyzer
preserves the observed RSRP, RSRQ, and SNR triplets, their order, dwell times,
and consecutive-second transitions. It never constructs a scenario by sampling
the three metrics independently. Missing seconds remain explicit gaps and no
transition is counted across them.

The catalog contains exact scenario sequences, an equal-trace selection weight,
pooled-time and equal-trace joint distributions, application-conditioned joint
distributions, transitions, and dwell statistics. The same selected scenario
can therefore be reused while UE count, traffic, scheduler settings, or network
configuration changes.

An empirical RFsim mapping can be added as an optional capability annotation:

```bash
make rf-distribution \
  RF_DISTRIBUTION_MAPPING=data/model_runs/ucc_static_awgn_safe_v2_mapping_v1 \
  RF_DISTRIBUTION_OUTPUT=data/model_runs/ucc_static_real_rf_catalog_v1_with_support
```

This annotation reports which real RF states the validated controls can replay.
It does not remove unsupported measurements, alter scenario probabilities, or
turn RFsim support into the definition of realism. UCC SNR and OAI SS-SINR stay
separate because their measurement definitions are not assumed equivalent.

The checksummed output is written under `data/model_runs/`, which remains local
and ignored by Git. These local artifacts may contain source-derived data and
must be published only to the private data repository.

## Distribution calibration

Run the first TDL-B distribution comparison after the private data repository
is available next to this repository:

```bash
make distribution-calibrate
```

The command verifies the selected campaign records, applied controls, model
family, quality flags, and minimum sample counts. It then writes:

- per-scenario summaries of the real observations;
- per-state summaries of the RFsim observations;
- a complete ranking of every executed state for every real scenario;
- complete-execution repeatability comparisons; and
- a manifest containing method settings, limitations, source hashes, and output
  checksums.

The first version is deliberately diagnostic. A low MMD identifies a closer
observed distribution, not a validated physical channel model. Candidate
rankings cannot be accepted until repetition instability, distance sensitivity,
and new held-out POWDER executions have been evaluated. The output belongs in
the private data repository because it is derived from the real reference
measurements.

## MMD–ABC implementation

The next calibration stage is predeclared in
`reports/MMD_ABC_CALIBRATION_DESIGN.md`. A paper-scale PMC-ABC loop is not
operationally feasible when every simulator proposal requires a complete POWDER
execution. The implemented workflow therefore separates deterministic proposal
planning, independently archived simulator executions, execution-level MMD–ABC
inference, and held-out posterior-predictive planning.

Generate the first 24-execution sensitivity and stochasticity pilot plan with:

```bash
make mmd-abc-plan
```

The pilot keeps TDL-B fixed, infers only `ploss` over the predeclared operational
range, and fixes `noise_power_dB=-30`. It uses three complete repetitions at each
of eight proposed values. Planned values are not treated as evidence until they
have been executed and passed the existing quality gates.

The inference command consumes a completed or partial execution bank:

```bash
PYTHONPATH=src uv run --locked rfsim-realism mmd-abc-infer \
  --real-observations /private/path/real_rf_observations.csv \
  --executions-root /private/path/executions \
  --proposal-plan manifests/mmd_abc_tdl_b_ploss_pilot_v1.json \
  --campaign-state /private/path/campaign_state.json \
  --config configs/mmd_abc_tdl_b_ploss_pilot_v1.yaml \
  --output /private/path/model_runs/tdl_b_ploss_mmd_abc_pilot_v1
```

The primary metric uses pooled-real covariance whitening, a real-reference median
kernel bandwidth, and unbiased joint RSRP/RSRQ MMD². Complete repetitions remain
separate stochastic simulator draws. The pilot configuration is explicitly marked
underpowered and cannot claim an established posterior. A posterior-predictive plan
can be generated only from an output that passes the predeclared ABC sample-size,
unique-parameter, and effective-sample gates. All real-derived inference and
validation outputs belong in the private data repository.

## Repository boundaries

- OAI configuration and channel-control helpers remain in `oai-5g-ric`.
- MGEN traffic generation and execution remain in `multimodal-traffic-digital-twins`.
- Dataset archiving and UI remain in `traffic-generation-dashboard`.
- This repository owns reference-trace curation, calibration plans, fitted
  mappings, comparisons, and validation reports.

All imported Powder executions must record the source repository commits,
profile commit, RFsim model, applied controls, UTC timing, and artifact
checksums before they are accepted for calibration.
