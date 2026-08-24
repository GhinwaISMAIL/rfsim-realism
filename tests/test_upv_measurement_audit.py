from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_measurement_audit import (
    _decision,
    _upv_inventory,
    validate_measurement_audit_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_measurement_audit_v1.yaml").read_text()
    )


def test_config_preserves_phase2_and_freezes_nonnegative_future_mmd() -> None:
    config = _config()

    validate_measurement_audit_config(config)

    assert config["frozen_inputs"]["phase2_estimator"] == "unbiased_mmd_squared"
    assert config["future_mmd_protocol"]["primary_estimator"] == (
        "biased_mmd_squared_v_statistic"
    )
    assert config["future_mmd_protocol"]["clipping_unbiased_estimates"] == "prohibited"
    assert config["reservation_policy"]["preparation_lead_time_minutes"] == 30


def _write_upv_archive(path: Path, pathloss_values: list[object]) -> None:
    frame = pd.DataFrame({
        "RSRP (NR SpCell)": [-85.0, -84.5],
        "RSRQ (NR SpCell)": [-10.5, -10.4],
        "SINR (NR SpCell)": [16.0, 17.0],
        "RSSI (NR SpCell)": [-56.0, -55.5],
        "Pathloss (NR SpCell)": pathloss_values,
        "Measurement type (NR SpCell)": ["SSB", "SSB"],
        "Beam index (NR SpCell)": [0, 0],
        "RX beam (SSB beam)": [None, None],
        "Rank indicator (NR)": [1, 1],
    })
    payload = io.StringIO()
    frame.to_csv(payload, sep=";", index=False)
    root = "Remote Driving Dataset in UPV's 5G Private network (n40)"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/Test_1/Test_1_ASUS.csv", payload.getvalue())


def test_upv_inventory_detects_empty_pathloss_and_ssb_labels(tmp_path: Path) -> None:
    archive = tmp_path / "upv.zip"
    _write_upv_archive(archive, [None, None])

    _, reference, summary = _upv_inventory(archive, _config())

    assert summary["serving_rsrp_rows"] == 2
    assert summary["serving_pathloss_rows"] == 0
    assert summary["measurement_type_values"] == ["SSB"]
    assert summary["all_observed_measurement_types_are_ssb"] is True
    assert summary["pathloss_reference_power_diagnostic_available"] is False
    assert reference.iloc[0]["status"] == "unavailable_pathloss_column_has_no_values"


def test_upv_inventory_computes_reference_power_when_available(tmp_path: Path) -> None:
    archive = tmp_path / "upv.zip"
    _write_upv_archive(archive, [105.0, 104.5])

    _, reference, summary = _upv_inventory(archive, _config())

    assert summary["pathloss_plus_rsrp_complete_rows"] == 2
    assert reference.iloc[0]["implied_reference_power_median_dbm"] == pytest.approx(20.0)


def test_decision_closes_reservation_and_absolute_rsrp_gate() -> None:
    phase2 = {"decision_code": "systematic_rsrp_offset_but_sinr_support"}
    phase2_gate = {
        "gate_before_next_reservation": {"reservation_should_be_requested_now": False}
    }
    summary = {
        "measurement_type_values": ["SSB"],
        "all_observed_measurement_types_are_ssb": True,
        "serving_rsrp_rows": 6969,
        "serving_pathloss_rows": 0,
        "pathloss_plus_rsrp_complete_rows": 0,
        "pathloss_reference_power_diagnostic_available": False,
    }

    result = _decision(_config(), phase2, phase2_gate, summary)

    assert result["decision_code"] == "insufficient_metadata_absolute_rsrp_not_identified"
    assert result["absolute_rsrp_calibration_authorized"] is False
    assert result["sinr_and_relative_rsrp_analysis_authorized"] is True
    assert result["reservation_should_be_requested_now"] is False


def test_repository_config_is_yaml_serializable() -> None:
    encoded = yaml.safe_dump(_config(), sort_keys=True)
    assert json.loads(json.dumps(yaml.safe_load(encoded)))["schema_version"] == 1


def test_cli_exposes_measurement_audit_command() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "rfsim_realism.cli", "audit-upv-measurement", "--help"],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--profile-source" in completed.stdout
