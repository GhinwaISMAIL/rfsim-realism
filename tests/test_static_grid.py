import hashlib
import json

import pandas as pd
import pytest

from rfsim_realism.static_grid import (
    build_plan,
    plan_document,
    plan_sha256,
    prepare_run_config,
    schedule_for,
    validate_archive,
)
from rfsim_realism.sweep import load_config


def config_path():
    return "configs/ucc_static_grid_v1.yaml"


def test_plan_is_deterministic_and_contains_24_joint_points():
    config = load_config(config_path())
    first = plan_document(config)
    second = plan_document(config)
    points = build_plan(config)

    assert first == second
    assert plan_sha256(first) == plan_sha256(second)
    assert len(points) == 24
    assert points[0].point_id == "ploss-p0_noise-m2-r1"
    assert points[-1].point_id == "ploss-m7_noise-p4-r2"
    assert points[0].controls == {
        "ploss": 0.0,
        "noise_power_dB": -2.0,
    }
    assert {point.repetition for point in points} == {1, 2}


def test_grid_rejects_positive_ploss():
    config = load_config(config_path())
    config["channel"]["grid"]["ploss"] = [0, 3]

    with pytest.raises(ValueError, match="path gain"):
        build_plan(config)


def test_schedule_keeps_both_controls_static_from_zero():
    point = build_plan(load_config(config_path()))[0]

    schedule = schedule_for(point)

    assert [event["at_s"] for event in schedule["events"]] == [0.0, 0.0]
    assert [event["parameter"] for event in schedule["events"]] == [
        "ploss",
        "noise_power_dB",
    ]
    assert [event["value"] for event in schedule["events"]] == [0.0, -2.0]


def test_prepare_run_config_freezes_the_grid_point(tmp_path):
    run = tmp_path / "run_grid"
    run.mkdir()
    target = run / "config.json"
    target.write_text(json.dumps({
        "run_name": "run_grid",
        "simulation_duration": 180,
        "rf_calibration": {"target_noise_power_db": -30},
    }))
    config = load_config(config_path())
    point = build_plan(config)[0]

    prepare_run_config(run, point, config)
    saved = json.loads(target.read_text())

    assert saved["rf_calibration"]["experiment_role"] == "ucc_static_grid"
    assert saved["rf_calibration"]["grid_point"] == {
        "point_id": point.point_id,
        "repetition": 1,
        "controls": {"ploss": 0.0, "noise_power_dB": -2.0},
    }
    assert saved["rf_calibration"]["ue_image_digest"].startswith("sha256:")
    assert "target_noise_power_db" not in saved["rf_calibration"]


def test_archive_gate_accepts_one_complete_joint_segment(tmp_path):
    config = load_config(config_path())
    point = build_plan(config)[0]
    archive = tmp_path / "mgen-grid"
    archive.mkdir()
    (archive / "metadata.json").write_text(json.dumps({
        "execution_id": "mgen-grid",
        "quality": {
            "channel_state_verified": True,
            "xapp": {"clean_shutdown": True, "errors": 0},
        },
    }))
    (archive / "config.json").write_text(json.dumps({
        "rf_calibration": {
            "grid_point": {"point_id": point.point_id},
        },
    }))
    pd.DataFrame([{
        "execution_id": "mgen-grid",
        "segment_id": "mgen-grid:ue1:dl:0",
        "ue": "ue1",
        "direction": "dl",
        "parameter": "joint",
        "control_count": 2,
        "requested_ploss": 0.0,
        "applied_ploss": 0.0,
        "requested_noise_power_dB": -2.0,
        "applied_noise_power_dB": -2.0,
        "duration_s": 180.0,
        "segment_start_utc": 1000.0,
        "segment_end_utc": 1180.0,
        "controlled": True,
        "verified": True,
        "channel_agreement": True,
        "training_eligible": True,
        "ploss_verified": True,
        "ploss_agreement": True,
        "noise_power_dB_verified": True,
        "noise_power_dB_agreement": True,
        "valid_clock_fraction": 1.0,
        "radio_samples": 180,
        "ue_radio_samples": 180,
        "ue_radio_clock_valid": True,
        "ue_radio_emit_lag_s_p95": 0.03,
        "radio_clock_lag_warning": False,
        "radio_clock_lag_s_segment_p95": 0.01,
        "sent_packets": 2,
        "received_packets": 2,
        "loss_rate": 0.0,
        "latency_ms_p95": 3.9,
        "received_mbps": 0.02,
        "ss_rsrp_dbm_segment_mean": -95.0,
        "ss_rsrq_db_segment_mean": -10.5,
        "ss_sinr_db_segment_mean": 4.2,
    }]).to_parquet(archive / "segment_training_table.parquet", index=False)
    pd.DataFrame({
        "ue": ["ue1", "ue1"],
        "direction": ["dl", "dl"],
        "sent_time_utc": [1010.0, 1011.0],
        "packet_clock_valid": [True, True],
        "received": [True, True],
        "latency_ms": [2.0, 4.0],
    }).to_parquet(archive / "packet_outcomes.parquet", index=False)
    checksums = {}
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        if path.name != "SHA256SUMS.json":
            checksums[str(path.relative_to(archive))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    (archive / "SHA256SUMS.json").write_text(json.dumps(checksums))

    result = validate_archive(archive, point, config)

    assert result["verified_files"] == 4
    assert result["controls"] == point.controls
    assert result["latency_ms_p95"] == pytest.approx(3.9)
    assert result["ss_sinr_db_segment_mean"] == pytest.approx(4.2)
