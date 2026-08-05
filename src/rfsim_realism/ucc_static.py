from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path, PurePosixPath

SOURCE_REPOSITORY = "https://github.com/uccmisl/5Gdataset"
SOURCE_REF = "v1.0.0"
SOURCE_COMMIT = "ba876f77e9330ebd227b2497cf4f90b37d624abc"
SOURCE_ARCHIVE = "5G-production-dataset.zip"
SOURCE_ARCHIVE_SHA256 = (
    "abc729d696b5e0ba34a9c6ed35a851f2868d2b5071be5efb8a49691c23bb1a9b"
)
TIMESTAMP_FORMAT = "%Y.%m.%d_%H.%M.%S"
RADIO_METRICS = ("RSRP", "RSRQ", "SNR", "CQI", "RSSI")
PRIMARY_RADIO_METRICS = ("RSRP", "RSRQ", "SNR")

DEFAULT_POLICY = {
    "window_seconds": 180,
    "min_timestamp_coverage": 0.80,
    "min_primary_radio_coverage": 0.75,
    "max_sample_gap_seconds": 3,
    "max_static_speed_p95_kph": 5.0,
    "static_speed_warning_kph": 1.0,
    "min_rssi_coverage": 0.75,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: Iterable[float | None], quantile: float) -> float | None:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _normalized_relative_path(value: str | Path) -> str | None:
    path = PurePosixPath(str(value).replace("\\", "/"))
    parts = path.parts
    if "__MACOSX" in parts or path.name.startswith("._"):
        return None
    if "5G-production-dataset" in parts:
        parts = parts[parts.index("5G-production-dataset") + 1:]
    if len(parts) < 3 or parts[1] != "Static" or not parts[-1].endswith(".csv"):
        return None
    return PurePosixPath(*parts).as_posix()


def _source_csvs(source: Path) -> tuple[list[tuple[str, bytes]], dict]:
    if source.is_file():
        archive_sha256 = _sha256_file(source)
        if source.name == SOURCE_ARCHIVE and archive_sha256 != SOURCE_ARCHIVE_SHA256:
            raise ValueError(
                f"{source} has SHA-256 {archive_sha256}; expected "
                f"{SOURCE_ARCHIVE_SHA256} for {SOURCE_REF}"
            )
        rows = []
        with zipfile.ZipFile(source) as archive:
            for member in sorted(archive.namelist()):
                relative = _normalized_relative_path(member)
                if relative is not None:
                    rows.append((relative, archive.read(member)))
        metadata = {
            "input_type": "zip",
            "archive_name": source.name,
            "archive_sha256": archive_sha256,
            "official_archive_verified": archive_sha256 == SOURCE_ARCHIVE_SHA256,
        }
        return rows, metadata

    if not source.is_dir():
        raise ValueError(f"dataset source does not exist: {source}")
    rows = []
    for path in sorted(source.rglob("*.csv")):
        relative = _normalized_relative_path(path.relative_to(source))
        if relative is not None:
            rows.append((relative, path.read_bytes()))
    return rows, {
        "input_type": "directory",
        "archive_name": None,
        "archive_sha256": None,
        "official_archive_verified": False,
    }


def _deduplicate(raw_rows: list[dict]) -> tuple[list[tuple[datetime, dict]], int, int]:
    by_second: dict[datetime, tuple[int, int, dict]] = {}
    invalid_timestamps = 0
    valid_rows = 0
    for index, row in enumerate(raw_rows):
        try:
            timestamp = datetime.strptime(row.get("Timestamp", ""), TIMESTAMP_FORMAT)
        except ValueError:
            invalid_timestamps += 1
            continue
        valid_rows += 1
        quality = sum(_number(row.get(metric)) is not None for metric in RADIO_METRICS)
        previous = by_second.get(timestamp)
        if previous is None or (quality, index) >= previous[:2]:
            by_second[timestamp] = (quality, index, row)
    observations = sorted(
        (timestamp, value[2]) for timestamp, value in by_second.items()
    )
    return observations, valid_rows - len(observations), invalid_timestamps


def _metric_stats(rows: list[dict], metric: str, window_seconds: int) -> dict:
    values = [_number(row.get(metric)) for row in rows]
    present = [value for value in values if value is not None]
    return {
        "observed": len(present),
        "coverage": len(present) / window_seconds,
        "unique_values": len(set(present)),
        "p10": _percentile(present, 0.10),
        "p50": _percentile(present, 0.50),
        "p90": _percentile(present, 0.90),
    }


def _window(
    observations: list[tuple[datetime, dict]],
    policy: dict,
) -> dict | None:
    duration = int(policy["window_seconds"])
    best: tuple[tuple, datetime, list[tuple[datetime, dict]], dict] | None = None
    right = 0
    for left, (start, _) in enumerate(observations):
        end = start + timedelta(seconds=duration - 1)
        right = max(right, left)
        while right < len(observations) and observations[right][0] <= end:
            right += 1
        sample = observations[left:right]
        if not sample:
            continue
        rows = [row for _, row in sample]
        timestamp_coverage = len(sample) / duration
        primary_coverage = sum(
            all(_number(row.get(metric)) is not None
                for metric in PRIMARY_RADIO_METRICS)
            for row in rows
        ) / duration
        deltas = [
            int((current[0] - previous[0]).total_seconds())
            for previous, current in pairwise(sample)
        ]
        max_gap = max(deltas, default=0)
        modes = sorted({row.get("NetworkMode", "") for row in rows})
        cells = sorted({row.get("CellID", "") for row in rows})
        eligible = (
            modes == ["5G"]
            and len(cells) == 1
            and timestamp_coverage >= policy["min_timestamp_coverage"]
            and primary_coverage >= policy["min_primary_radio_coverage"]
            and max_gap <= policy["max_sample_gap_seconds"]
        )
        score = (eligible, timestamp_coverage + primary_coverage,
                 timestamp_coverage, primary_coverage)
        details = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "observed_seconds": len(sample),
            "timestamp_coverage": timestamp_coverage,
            "primary_radio_coverage": primary_coverage,
            "max_sample_gap_seconds": max_gap,
            "network_modes": modes,
            "cell_ids": cells,
            "speed_p95_kph": _percentile(
                [_number(row.get("Speed")) for row in rows], 0.95),
            "state_counts": dict(sorted(Counter(
                row.get("State", "") for row in rows).items())),
            "metrics": {
                metric: _metric_stats(rows, metric, duration)
                for metric in RADIO_METRICS
            },
            "quality_eligible": eligible,
        }
        if best is None or score > best[0]:
            best = (score, start, sample, details)
    return best[3] if best is not None else None


def _classification(window: dict | None, policy: dict) -> tuple[str, list[str]]:
    if window is None or not window["quality_eligible"]:
        return "review", ["no_eligible_180s_window"]
    flags = []
    speed = window["speed_p95_kph"]
    if speed is not None and speed > policy["max_static_speed_p95_kph"]:
        return "quarantine_mobility", ["static_label_conflicts_with_speed"]
    if speed is not None and speed > policy["static_speed_warning_kph"]:
        flags.append("static_speed_warning")
    if window["metrics"]["RSSI"]["coverage"] < policy["min_rssi_coverage"]:
        flags.append("incomplete_rssi")
    frozen = all(
        window["metrics"][metric]["unique_values"] <= 1
        for metric in PRIMARY_RADIO_METRICS
    )
    if frozen:
        flags.append("steady_radio_values")
        return "steady_anchor", flags
    return "dynamic_static", flags


def _trace(relative_path: str, payload: bytes, policy: dict) -> dict:
    text = payload.decode("utf-8-sig")
    raw_rows = list(csv.DictReader(io.StringIO(text)))
    observations, duplicates, invalid_timestamps = _deduplicate(raw_rows)
    modes = Counter(row.get("NetworkMode", "") for _, row in observations)
    cells = [row.get("CellID", "") for _, row in observations]
    cell_changes = sum(a != b for a, b in pairwise(cells))
    window = _window(observations, policy)
    classification, flags = _classification(window, policy)
    if set(modes) != {"5G"}:
        flags.append("source_contains_non_5g")
    if cell_changes:
        flags.append("source_contains_cell_changes")
    if duplicates:
        flags.append("duplicate_timestamps")
    parts = PurePosixPath(relative_path).parts
    content = "/".join(parts[2:-1]) or None
    duration = (
        int((observations[-1][0] - observations[0][0]).total_seconds()) + 1
        if observations else 0
    )
    return {
        "trace_id": hashlib.sha256(relative_path.encode()).hexdigest()[:16],
        "source_path": relative_path,
        "source_sha256": _sha256_bytes(payload),
        "app": parts[0],
        "mobility": "Static",
        "content": content,
        "raw_rows": len(raw_rows),
        "unique_seconds": len(observations),
        "duplicate_rows": duplicates,
        "invalid_timestamps": invalid_timestamps,
        "wall_duration_seconds": duration,
        "network_mode_counts": dict(sorted(modes.items())),
        "cell_changes": cell_changes,
        "classification": classification,
        "quality_flags": sorted(set(flags)),
        "calibration_eligible": classification in {"dynamic_static", "steady_anchor"},
        "dynamic_replay_eligible": classification == "dynamic_static",
        "selected_window": window,
    }


def build_manifest(source: str | Path, policy: dict | None = None) -> dict:
    source = Path(source)
    effective_policy = dict(DEFAULT_POLICY)
    if policy:
        unknown = set(policy) - set(effective_policy)
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        effective_policy.update(policy)
    csvs, source_metadata = _source_csvs(source)
    if not csvs:
        raise ValueError(f"no static CSV traces found in {source}")
    traces = [_trace(path, payload, effective_policy) for path, payload in csvs]
    classifications = Counter(trace["classification"] for trace in traces)
    apps = Counter(trace["app"] for trace in traces)
    return {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "commit": SOURCE_COMMIT,
            "archive": SOURCE_ARCHIVE,
            "expected_archive_sha256": SOURCE_ARCHIVE_SHA256,
            **source_metadata,
        },
        "policy": effective_policy,
        "inventory": {
            "static_trace_files": len(traces),
            "raw_rows": sum(trace["raw_rows"] for trace in traces),
            "unique_seconds": sum(trace["unique_seconds"] for trace in traces),
            "duplicate_rows": sum(trace["duplicate_rows"] for trace in traces),
            "applications": dict(sorted(apps.items())),
            "classifications": dict(sorted(classifications.items())),
            "calibration_eligible": sum(
                trace["calibration_eligible"] for trace in traces),
            "dynamic_replay_eligible": sum(
                trace["dynamic_replay_eligible"] for trace in traces),
        },
        "traces": traces,
    }


def write_manifest(manifest: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination
