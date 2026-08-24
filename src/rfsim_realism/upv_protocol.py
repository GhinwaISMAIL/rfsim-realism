from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import yaml

RADIO_COLUMNS = {
    "rsrp_dbm": "RSRP (NR SpCell)",
    "rsrq_db": "RSRQ (NR SpCell)",
    "sinr_db": "SINR (NR SpCell)",
    "serving_pci": "Physical cell identity (NR SpCell)",
    "neighbor_rsrp_dbm": "RSRP (NR neighbor)",
    "neighbor_pci": "Physical cell identity (NR neighbor)",
    "longitude_deg": "Longitude",
    "latitude_deg": "Latitude",
}
NUMERIC_COLUMNS = tuple(RADIO_COLUMNS)
PRIMARY_METRICS = ("rsrp_dbm", "rsrq_db", "sinr_db")
DIRECTION_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
XLSX_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.6f", lineterminator="\n")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _normal_member_path(value: str) -> str:
    parts = PurePosixPath(value).parts
    if parts and parts[0].startswith("Remote Driving Dataset in UPV's 5G Private network"):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _simple_xlsx_rows(payload: bytes) -> list[dict[str, object]]:
    namespace = {"m": XLSX_NAMESPACE}
    with zipfile.ZipFile(io.BytesIO(payload)) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", namespace):
                shared.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{XLSX_NAMESPACE}}}t")
                    )
                )
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        matrix: list[list[object]] = []
        for row in sheet.findall(".//m:sheetData/m:row", namespace):
            cells: dict[int, object] = {}
            for cell in row.findall("m:c", namespace):
                index = _column_index(cell.attrib["r"])
                kind = cell.attrib.get("t")
                value_node = cell.find("m:v", namespace)
                value: object = "" if value_node is None else value_node.text or ""
                if kind == "s" and value != "":
                    value = shared[int(str(value))]
                elif kind == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iter(f"{{{XLSX_NAMESPACE}}}t")
                    )
                cells[index] = value
            width = max(cells, default=-1) + 1
            matrix.append([cells.get(index, "") for index in range(width)])
    if not matrix:
        return []
    headers = [str(value).strip() for value in matrix[0]]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        for row in matrix[1:]
    ]


def _excel_clock(value: object) -> tuple[str, float]:
    fraction = float(value) % 1.0
    seconds = round(fraction * 86400) % 86400
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}", float(
        hours * 3600 + minutes * 60 + seconds
    )


def _time_offset_seconds(observed: float, expected: float) -> float:
    return ((observed - expected + 43200.0) % 86400.0) - 43200.0


def _test_id(source_path: str) -> int:
    match = re.search(r"Test[_ ]?(\d+)", source_path, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot infer test ID from {source_path}")
    return int(match.group(1))


def _device(source_path: str) -> str:
    name = PurePosixPath(source_path).name.lower()
    if "asus" in name:
        return "ASUS"
    if "s25" in name:
        return "S25"
    raise ValueError(f"cannot infer device from {source_path}")


def _measurement_csvs(payloads: dict[str, bytes]) -> list[str]:
    return sorted(
        path
        for path in payloads
        if len(PurePosixPath(path).parts) == 2 and path.lower().endswith(".csv")
    )


def _load_radio_csv(payload: bytes) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(io.BytesIO(payload), sep=";", dtype=str, low_memory=False)
    frame = pd.DataFrame(index=raw.index)
    frame["time_of_day"] = raw["Time"]
    for output, source in RADIO_COLUMNS.items():
        if source not in raw:
            frame[output] = np.nan
            continue
        if output in NUMERIC_COLUMNS:
            frame[output] = pd.to_numeric(
                raw[source].str.replace(",", ".", regex=False), errors="coerce"
            )
        else:
            frame[output] = raw[source]
    frame["seconds_of_day"] = pd.to_timedelta(
        frame["time_of_day"], errors="coerce"
    ).dt.total_seconds()
    radio = frame.dropna(subset=[*PRIMARY_METRICS, "seconds_of_day"])
    route_required = [
        *PRIMARY_METRICS,
        "longitude_deg",
        "latitude_deg",
        "seconds_of_day",
    ]
    route = frame.dropna(subset=route_required).sort_values("seconds_of_day")
    duplicate_route_timestamps = int(route.duplicated("seconds_of_day").sum())
    route = route.drop_duplicates("seconds_of_day", keep="last").reset_index(drop=True)
    return route, {
        "complete_radio_triplets": len(radio),
        "complete_radio_gps_rows": len(route) + duplicate_route_timestamps,
        "duplicate_route_timestamps": duplicate_route_timestamps,
        "route_rows_after_timestamp_deduplication": len(route),
    }


def _local_steps(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = frame["latitude_deg"].to_numpy(float)
    longitude = frame["longitude_deg"].to_numpy(float)
    latitude_midpoint = np.deg2rad((latitude[1:] + latitude[:-1]) / 2.0)
    east = np.diff(longitude) * 111320.0 * np.cos(latitude_midpoint)
    north = np.diff(latitude) * 110540.0
    return np.r_[0.0, east], np.r_[0.0, north], np.r_[0.0, np.hypot(east, north)]


def build_route_table(
    frame: pd.DataFrame,
    *,
    source_path: str,
    corrected_test_id: int,
    bin_sizes_m: list[int],
    minimum_step_m_for_heading: float,
    direction_sectors: int,
) -> pd.DataFrame:
    if direction_sectors != len(DIRECTION_LABELS):
        raise ValueError("this protocol version requires eight direction sectors")
    route = frame.copy().reset_index(drop=True)
    east_step, north_step, step = _local_steps(route)
    route["sample_sequence"] = np.arange(len(route), dtype=int)
    route["source_path"] = source_path
    route["corrected_test_id"] = corrected_test_id
    route["device"] = _device(source_path)
    route["elapsed_seconds"] = route["seconds_of_day"] - route["seconds_of_day"].iloc[0]
    route["east_step_m"] = east_step
    route["north_step_m"] = north_step
    route["step_distance_m"] = step
    route["route_distance_m"] = np.cumsum(step)
    total_distance = float(route["route_distance_m"].iloc[-1])
    route["route_fraction"] = (
        route["route_distance_m"] / total_distance if total_distance else 0.0
    )
    heading = pd.Series(
        np.where(
            step >= minimum_step_m_for_heading,
            (np.degrees(np.arctan2(east_step, north_step)) + 360.0) % 360.0,
            np.nan,
        )
    ).ffill().bfill()
    route["heading_deg"] = heading
    sector_width = 360.0 / direction_sectors
    sector = np.floor((heading + sector_width / 2.0) / sector_width).astype(int)
    route["direction_sector"] = [DIRECTION_LABELS[value % direction_sectors] for value in sector]
    for size in bin_sizes_m:
        route[f"route_bin_{size}m"] = np.floor(route["route_distance_m"] / size).astype(int)
    identifier = source_path + "|"
    route["route_observation_id"] = [
        hashlib.sha256(f"{identifier}{value:.3f}".encode()).hexdigest()[:16]
        for value in route["seconds_of_day"]
    ]
    columns = [
        "route_observation_id",
        "source_path",
        "corrected_test_id",
        "device",
        "sample_sequence",
        "time_of_day",
        "seconds_of_day",
        "elapsed_seconds",
        "longitude_deg",
        "latitude_deg",
        "east_step_m",
        "north_step_m",
        "step_distance_m",
        "route_distance_m",
        "route_fraction",
        "heading_deg",
        "direction_sector",
        *[f"route_bin_{size}m" for size in bin_sizes_m],
        "rsrp_dbm",
        "rsrq_db",
        "sinr_db",
        "serving_pci",
        "neighbor_rsrp_dbm",
        "neighbor_pci",
    ]
    return route[columns]


def build_locked_split(route: pd.DataFrame, config: dict) -> pd.DataFrame:
    route_config = config["route"]
    split_config = config["locked_split"]
    primary_size = int(split_config["primary_bin_size_m"])
    rows: list[dict[str, object]] = []
    for size in [int(value) for value in route_config["bin_sizes_m"]]:
        bin_column = f"route_bin_{size}m"
        for bin_id, group in route.groupby(bin_column, sort=True):
            rows.append(
                {
                    "bin_size_m": size,
                    "route_bin_id": int(bin_id),
                    "route_start_m": float(group["route_distance_m"].min()),
                    "route_end_m": float(group["route_distance_m"].max()),
                    "route_center_m": float(
                        (group["route_distance_m"].min() + group["route_distance_m"].max())
                        / 2.0
                    ),
                    "sample_count": len(group),
                    "dwell_seconds": float(
                        group["seconds_of_day"].max() - group["seconds_of_day"].min()
                    ),
                    "eligible": bool(
                        len(group) >= int(split_config["minimum_samples"])
                        and group["seconds_of_day"].max()
                        - group["seconds_of_day"].min()
                        >= float(split_config["minimum_dwell_seconds"])
                    ),
                    "locked_role": "binning_sensitivity_only"
                    if size != primary_size
                    else "not_selected",
                    "target_route_fraction": np.nan,
                    "selection_basis": split_config["selection_basis"],
                }
            )
    result = pd.DataFrame(rows)
    primary = result[(result["bin_size_m"] == primary_size) & result["eligible"]]
    if primary.empty:
        raise ValueError("no primary spatial bins meet the locked geometry criteria")
    total_distance = float(route["route_distance_m"].max())
    selections = [
        ("calibration", float(split_config["calibration_fraction"])),
        *[
            (f"spatial_validation_{index}", float(fraction))
            for index, fraction in enumerate(
                split_config["spatial_validation_fractions"], start=1
            )
        ],
    ]
    used: set[int] = set()
    for role, fraction in selections:
        candidates = primary[~primary["route_bin_id"].isin(used)].copy()
        candidates["target_error_m"] = (
            candidates["route_center_m"] - fraction * total_distance
        ).abs()
        selected = candidates.sort_values(["target_error_m", "route_bin_id"]).iloc[0]
        mask = (
            (result["bin_size_m"] == primary_size)
            & (result["route_bin_id"] == int(selected["route_bin_id"]))
        )
        result.loc[mask, "locked_role"] = role
        result.loc[mask, "target_route_fraction"] = fraction
        used.add(int(selected["route_bin_id"]))
    if (result["locked_role"] == "calibration").sum() != 1:
        raise AssertionError("the locked split must contain exactly one calibration bin")
    expected_validation = len(split_config["spatial_validation_fractions"])
    if result["locked_role"].str.startswith("spatial_validation_").sum() != expected_validation:
        raise AssertionError("the locked split does not contain every validation bin")
    return result.sort_values(["bin_size_m", "route_bin_id"]).reset_index(drop=True)


def _gps_separation(left: pd.DataFrame, right: pd.DataFrame) -> tuple[int, float, float]:
    matched = pd.merge_asof(
        left.sort_values("seconds_of_day"),
        right.sort_values("seconds_of_day"),
        on="seconds_of_day",
        direction="nearest",
        tolerance=1.0,
        suffixes=("_left", "_right"),
    ).dropna(subset=["latitude_deg_right", "longitude_deg_right"])
    if matched.empty:
        return 0, math.nan, math.nan
    latitude_midpoint = np.deg2rad(
        (matched["latitude_deg_left"] + matched["latitude_deg_right"]) / 2.0
    )
    east = (
        (matched["longitude_deg_left"] - matched["longitude_deg_right"])
        * 111320.0
        * np.cos(latitude_midpoint)
    )
    north = (
        matched["latitude_deg_left"] - matched["latitude_deg_right"]
    ) * 110540.0
    distance = np.hypot(east, north)
    return len(matched), float(np.median(distance)), float(np.quantile(distance, 0.95))


def _description_index(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        if not str(row.get("Test #", "")).strip():
            continue
        test_id = int(float(str(row["Test #"])))
        clock, seconds = _excel_clock(row["Started At:"])
        result[test_id] = {
            "description": str(row.get("Test Description", "")),
            "comments": str(row.get("Comments", "")),
            "recorded_start_time": clock,
            "recorded_start_seconds": seconds,
            "ASUS_status": str(row.get("UE3 (FR1 -ASUS SnapDragon Bronze, IP: 10.45.21.19)", "")),
            "S25_status": str(row.get("UE2 (FR1 - S25 Nemo Silver, IP: 10.45.22.4)", "")),
        }
    return result


def _archive_inventory(
    archive_path: Path,
) -> tuple[pd.DataFrame, dict[str, bytes], str, str]:
    archive_sha256 = _digest_file(archive_path, "sha256")
    archive_md5 = _digest_file(archive_path, "md5")
    rows = [
        {
            "entry_kind": "source_archive",
            "source_path": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "crc32": "",
            "sha256": archive_sha256,
        }
    ]
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.infolist(), key=lambda value: value.filename):
            if member.is_dir():
                continue
            source_path = _normal_member_path(member.filename)
            payload = archive.read(member)
            if source_path in payloads:
                raise ValueError(f"duplicate normalized archive member: {source_path}")
            payloads[source_path] = payload
            rows.append(
                {
                    "entry_kind": "archive_member",
                    "source_path": source_path,
                    "size_bytes": len(payload),
                    "crc32": f"0x{member.CRC:08x}",
                    "sha256": _sha256_bytes(payload),
                }
            )
    return pd.DataFrame(rows), payloads, archive_sha256, archive_md5


def prepare_upv_protocol(
    archive_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    archive_path = Path(archive_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = yaml.safe_load(config_path.read_text())
    inventory, payloads, archive_sha256, archive_md5 = _archive_inventory(archive_path)
    source = config["source"]
    if archive_sha256 != source["expected_sha256"]:
        raise ValueError(f"archive SHA-256 {archive_sha256} does not match the pinned source")
    if archive_md5 != source["expected_md5"]:
        raise ValueError(f"archive MD5 {archive_md5} does not match the pinned source")
    if archive_path.stat().st_size != int(source["expected_size_bytes"]):
        raise ValueError("archive size does not match the pinned source")
    descriptions_payload = payloads.get("TestDescriptions.xlsx")
    if descriptions_payload is None:
        raise ValueError("TestDescriptions.xlsx is missing from the source archive")
    descriptions = _description_index(_simple_xlsx_rows(descriptions_payload))
    corrections = config["scenario_correction"]["corrected_test_ids"]
    trim_by_test = config["scenario_correction"]["trim_last_seconds_by_corrected_test"]
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, object]] = {}
    for source_path in _measurement_csvs(payloads):
        frame, frame_counts = _load_radio_csv(payloads[source_path])
        filename_test_id = _test_id(source_path)
        corrected_test_id = int(corrections.get(source_path, filename_test_id))
        trim_seconds = float(trim_by_test.get(str(corrected_test_id), 0))
        cutoff = float(frame["seconds_of_day"].max() - trim_seconds)
        retained = frame[frame["seconds_of_day"] <= cutoff].copy()
        frames[source_path] = frame
        metadata[source_path] = {
            "filename_test_id": filename_test_id,
            "corrected_test_id": corrected_test_id,
            "device": _device(source_path),
            "trim_last_seconds": trim_seconds,
            "retained": retained,
            "frame_counts": frame_counts,
        }
    scenario_rows: list[dict[str, object]] = []
    for source_path in sorted(frames):
        frame = frames[source_path]
        details = metadata[source_path]
        filename_test_id = int(details["filename_test_id"])
        corrected_test_id = int(details["corrected_test_id"])
        device = str(details["device"])
        counterpart = next(
            (
                path
                for path, candidate in metadata.items()
                if path != source_path
                and int(candidate["corrected_test_id"]) == corrected_test_id
                and candidate["device"] != device
            ),
            None,
        )
        gps_pairs, gps_median, gps_p95 = (
            _gps_separation(frame, frames[counterpart])
            if counterpart is not None
            else (0, math.nan, math.nan)
        )
        observed_start = float(frame["seconds_of_day"].min())
        original_description = descriptions[filename_test_id]
        corrected_description = descriptions[corrected_test_id]
        active_status = str(corrected_description[f"{device}_status"])
        scenario_rows.append(
            {
                "source_path": source_path,
                "source_sha256": _sha256_bytes(payloads[source_path]),
                "device": device,
                "filename_test_id": filename_test_id,
                "corrected_test_id": corrected_test_id,
                "sensitivity_test_id": filename_test_id,
                "correction_status": "provisional_timestamp_gps_supported"
                if filename_test_id != corrected_test_id
                else "filename_consistent",
                "corrected_test_description": corrected_description["description"],
                "corrected_test_comments": corrected_description["comments"],
                "device_status_in_corrected_test": active_status,
                "analysis_role": "primary_calibration_reference"
                if source_path == config["route"]["reference_source_path"]
                else (
                    "inactive_device_diagnostic_only"
                    if active_status.strip().lower() == "disabled"
                    else "external_transfer_candidate"
                ),
                "observed_radio_start_time": frame["time_of_day"].iloc[0],
                "observed_radio_end_time": frame["time_of_day"].iloc[-1],
                "recorded_original_test_start_time": original_description[
                    "recorded_start_time"
                ],
                "recorded_corrected_test_start_time": corrected_description[
                    "recorded_start_time"
                ],
                "original_start_offset_seconds": _time_offset_seconds(
                    observed_start, float(original_description["recorded_start_seconds"])
                ),
                "corrected_start_offset_seconds": _time_offset_seconds(
                    observed_start, float(corrected_description["recorded_start_seconds"])
                ),
                "gps_pair_source_path": counterpart or "",
                "gps_pairs_within_1s": gps_pairs,
                "gps_median_separation_m": gps_median,
                "gps_p95_separation_m": gps_p95,
                "complete_radio_triplets_raw": details["frame_counts"][
                    "complete_radio_triplets"
                ],
                "complete_radio_gps_rows_raw": details["frame_counts"][
                    "complete_radio_gps_rows"
                ],
                "duplicate_route_timestamps": details["frame_counts"][
                    "duplicate_route_timestamps"
                ],
                "trim_last_seconds": details["trim_last_seconds"],
                "route_rows_retained": len(details["retained"]),
            }
        )
    scenario_manifest = pd.DataFrame(scenario_rows)
    reference_path = config["route"]["reference_source_path"]
    if reference_path not in metadata:
        raise ValueError(f"route reference is missing: {reference_path}")
    reference = metadata[reference_path]
    route = build_route_table(
        reference["retained"],
        source_path=reference_path,
        corrected_test_id=int(reference["corrected_test_id"]),
        bin_sizes_m=[int(value) for value in config["route"]["bin_sizes_m"]],
        minimum_step_m_for_heading=float(
            config["route"]["minimum_step_m_for_heading"]
        ),
        direction_sectors=int(config["route"]["direction_sectors"]),
    )
    split = build_locked_split(route, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_inventory.csv": inventory,
        "scenario_correction_manifest.csv": scenario_manifest,
        "route_observations.csv": route,
        "locked_spatial_split.csv": split,
    }
    for name, frame in outputs.items():
        _write_csv(output_dir / name, frame)
    _write_parquet(output_dir / "route_observations.parquet", route)
    output_hashes = {
        name: _digest_file(output_dir / name, "sha256")
        for name in [*outputs, "route_observations.parquet"]
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "archive": archive_path.name,
            "archive_sha256": archive_sha256,
            "archive_md5": archive_md5,
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_members": len(inventory) - 1,
        },
        "protocol": {
            "config": config_path.name,
            "config_sha256": _digest_file(config_path, "sha256"),
            "scenario_correction_status": config["scenario_correction"]["status"],
            "primary_interpretation": config["scenario_correction"][
                "primary_interpretation"
            ],
            "sensitivity_interpretation": config["scenario_correction"][
                "sensitivity_interpretation"
            ],
            "selection_basis": config["locked_split"]["selection_basis"],
            "preliminary_inspection_disclosure": config["locked_split"][
                "preliminary_inspection_disclosure"
            ],
            "kpi_fields_used_for_split": [],
        },
        "summary": {
            "measurement_files": len(scenario_manifest),
            "complete_radio_triplets": int(
                scenario_manifest["complete_radio_triplets_raw"].sum()
            ),
            "complete_radio_gps_rows": int(
                scenario_manifest["complete_radio_gps_rows_raw"].sum()
            ),
            "reference_source_path": reference_path,
            "reference_observations": len(route),
            "reference_route_length_m": float(route["route_distance_m"].max()),
            "reference_duration_seconds": float(route["elapsed_seconds"].max()),
            "primary_bin_size_m": int(config["locked_split"]["primary_bin_size_m"]),
            "calibration_bins": int((split["locked_role"] == "calibration").sum()),
            "spatial_validation_bins": int(
                split["locked_role"].str.startswith("spatial_validation_").sum()
            ),
        },
        "outputs": output_hashes,
    }
    _write_json(output_dir / "protocol_manifest.json", manifest)
    checksums = {
        name: _digest_file(output_dir / name, "sha256")
        for name in sorted([*output_hashes, "protocol_manifest.json"])
    }
    _write_json(output_dir / "SHA256SUMS.json", checksums)
    return manifest
