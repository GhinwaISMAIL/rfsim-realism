# Data directories

- `raw/` stores checksum-verified external source archives.
- `curated/` stores deterministic tables generated from the source manifest.
- `calibration_runs/` stores imported immutable POWDER execution archives.
- `model_runs/` stores checksummed derived mappings, distribution comparisons,
  repeatability diagnostics, and calibration manifests.

The contents of these directories are excluded from normal Git history. Source
records, selection manifests, fitted model metadata, and checksums are tracked.
Any output derived from the real reference measurements is published only to
the private data repository.

## UPV remote-driving protocol

Place the checksum-pinned Zenodo file at
`data/raw/upv_remote_driving_n40_v1.zip`, then run `make prepare-upv`.
The deterministic output under `data/curated/upv_protocol_v1/` contains:

- a SHA-256 inventory of the source archive and every archive member;
- a provisional Test 1/Test 2 correction manifest with timestamp and paired-GPS evidence;
- the corrected single-UE ASUS route with 10, 15, and 20 metre assignments;
- a geometry-only locked calibration/spatial-validation split; and
- a protocol manifest plus checksums for every generated table.

The source filename interpretation remains in the correction manifest for the
required sensitivity analysis. Generated tables do not constitute an ABC
posterior or a claim of successful RFsim calibration.
