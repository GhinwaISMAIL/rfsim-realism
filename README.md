# RFsim Realism

This project calibrates OAI RFsim channel controls against real-world radio
measurements and validates the resulting scenarios with packet and radio
observations from the POWDER testbed.

The first reference source is the UCC 5G production dataset. Raw source files
remain unchanged and outside Git. A pinned source record, archive checksum,
curation policy, and deterministic manifest make the analysis reproducible.

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

## Current static reference catalog

The official `v1.0.0` archive contains 23 static traces: 8 Amazon Prime,
5 Download, and 10 Netflix. The current policy retains 18 dynamic static
windows for later replay, 4 steady windows as calibration anchors, and
quarantines 1 trace whose static label conflicts with its reported speed.

Every chosen window is 180 seconds, 5G-only, and single-cell. Android duplicate
timestamps are resolved by keeping the row with the most complete radio
measurements. Missing seconds remain explicit; they are not silently filled.

## Calibration order

1. Capture trustworthy UE-side RSRP, RSRQ, and SINR with UTC timestamps.
2. Run one cell and one UE with the AWGN model.
3. Sweep RFsim path gain (`ploss`) and `noise_power_dB` separately using the
   checked configuration. Negative `ploss` values attenuate the signal.
4. Learn the mapping from applied RFsim controls to observed radio statistics.
5. Validate the mapping on held-out static reference traces.
6. Compare TDL channel families after the AWGN mapping is stable.

CQI, MCS, BLER, throughput, loss, and latency are observations. They are not
used as direct RFsim controls.

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

## First static mapping

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

## TDL-B calibration result

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

## TDL-C family decision

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

## TDL-A family screening

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

## EVA family screening

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

## Repository boundaries

- OAI configuration and channel-control helpers remain in `oai-5g-ric`.
- MGEN traffic generation and execution remain in `multimodal-traffic-digital-twins`.
- Dataset archiving and UI remain in `traffic-generation-dashboard`.
- This repository owns reference-trace curation, calibration plans, fitted
  mappings, comparisons, and validation reports.

All imported Powder executions must record the source repository commits,
profile commit, RFsim model, applied controls, UTC timing, and artifact
checksums before they are accepted for calibration.
