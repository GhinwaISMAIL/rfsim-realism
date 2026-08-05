import csv
from datetime import datetime, timedelta
from pathlib import Path

from rfsim_realism import ucc_static

HEADER = [
    "Timestamp", "Longitude", "Latitude", "Speed", "Operatorname", "CellID",
    "NetworkMode", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate",
    "UL_bitrate", "State", "PINGAVG", "PINGMIN", "PINGMAX", "PINGSTDEV",
    "PINGLOSS", "CELLHEX", "NODEHEX", "LACHEX", "RAWCELLID", "NRxRSRP",
    "NRxRSRQ",
]


def write_trace(root: Path, name: str, *, speed: float = 0,
                dynamic: bool = True, network_mode: str = "5G",
                duplicate: bool = False) -> Path:
    path = root / "Download" / "Static" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 1, 12, 0, 0)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for second in range(180):
            timestamp = (start + timedelta(seconds=second)).strftime(
                ucc_static.TIMESTAMP_FORMAT)
            row = {
                "Timestamp": timestamp,
                "Longitude": -8.0,
                "Latitude": 51.0,
                "Speed": speed,
                "Operatorname": "B",
                "CellID": 11,
                "NetworkMode": network_mode,
                "RSRP": -100 + (second % 4 if dynamic else 0),
                "RSRQ": -15 + (second % 3 if dynamic else 0),
                "SNR": second % 8 if dynamic else 2,
                "CQI": 10 + (second % 3 if dynamic else 0),
                "RSSI": -90,
                "DL_bitrate": 1000,
                "UL_bitrate": 100,
                "State": "D",
                "NRxRSRP": -110,
                "NRxRSRQ": -18,
            }
            writer.writerow(row)
            if duplicate and second == 10:
                writer.writerow(row)
    return path


def only_trace(manifest: dict) -> dict:
    assert manifest["inventory"]["static_trace_files"] == 1
    return manifest["traces"][0]


def test_dynamic_trace_is_deduplicated_and_selected(tmp_path):
    write_trace(tmp_path, "dynamic.csv", duplicate=True)
    manifest = ucc_static.build_manifest(tmp_path)
    trace = only_trace(manifest)
    assert trace["classification"] == "dynamic_static"
    assert trace["calibration_eligible"] is True
    assert trace["dynamic_replay_eligible"] is True
    assert trace["duplicate_rows"] == 1
    assert trace["selected_window"]["observed_seconds"] == 180
    assert trace["selected_window"]["quality_eligible"] is True


def test_constant_trace_is_retained_as_steady_anchor(tmp_path):
    write_trace(tmp_path, "anchor.csv", dynamic=False)
    trace = only_trace(ucc_static.build_manifest(tmp_path))
    assert trace["classification"] == "steady_anchor"
    assert trace["calibration_eligible"] is True
    assert trace["dynamic_replay_eligible"] is False
    assert "steady_radio_values" in trace["quality_flags"]


def test_static_label_with_moving_speed_is_quarantined(tmp_path):
    write_trace(tmp_path, "moving.csv", speed=15)
    trace = only_trace(ucc_static.build_manifest(tmp_path))
    assert trace["classification"] == "quarantine_mobility"
    assert trace["calibration_eligible"] is False
    assert "static_label_conflicts_with_speed" in trace["quality_flags"]


def test_non_5g_trace_has_no_eligible_window(tmp_path):
    write_trace(tmp_path, "lte.csv", network_mode="LTE")
    trace = only_trace(ucc_static.build_manifest(tmp_path))
    assert trace["classification"] == "review"
    assert trace["selected_window"]["quality_eligible"] is False
    assert "source_contains_non_5g" in trace["quality_flags"]


def test_official_named_archive_requires_expected_checksum(tmp_path):
    archive = tmp_path / ucc_static.SOURCE_ARCHIVE
    archive.write_bytes(b"not the official archive")
    try:
        ucc_static.build_manifest(archive)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("checksum mismatch was not rejected")


def test_macos_archive_metadata_is_not_a_trace():
    value = "__MACOSX/5G-production-dataset/Download/Static/._trace.csv"
    assert ucc_static._normalized_relative_path(value) is None
