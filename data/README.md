# Data directories

- `raw/` stores checksum-verified external source archives.
- `curated/` stores deterministic tables generated from the source manifest.
- `calibration_runs/` stores imported immutable POWDER execution archives.

The contents of these directories are excluded from normal Git history. Source
records, selection manifests, fitted model metadata, and checksums are tracked.
