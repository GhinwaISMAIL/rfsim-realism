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
