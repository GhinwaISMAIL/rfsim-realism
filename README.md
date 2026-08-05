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
6. Add TDL channel families only after the AWGN mapping is stable.

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

## Repository boundaries

- OAI configuration and channel-control helpers remain in `oai-5g-ric`.
- MGEN traffic generation and execution remain in `multimodal-traffic-digital-twins`.
- Dataset archiving and UI remain in `traffic-generation-dashboard`.
- This repository owns reference-trace curation, calibration plans, fitted
  mappings, comparisons, and validation reports.

All imported Powder executions must record the source repository commits,
profile commit, RFsim model, applied controls, UTC timing, and artifact
checksums before they are accepted for calibration.
