import hashlib
import json

import pandas as pd
import pytest

from rfsim_realism.sweep import (
    SweepPoint,
    build_plan,
    load_config,
    parse_ntpq,
    plan_document,
    plan_sha256,
    schedule_for,
    validate_archive,
)


def config_path():
    return "configs/awgn_calibration_v1.yaml"


def test_plan_is_deterministic_and_has_two_repetitions():
    config = load_config(config_path())
    first = plan_document(config)
    second = plan_document(config)
    points = build_plan(config)

    assert first == second
    assert plan_sha256(first) == plan_sha256(second)
    assert first["quality_requirements"]["max_ntp_offset_spread_ms"] == 5.0
    assert "ue_rsrp_dbm" in first["required_observations"]
    assert len(points) == 28
    assert points[0].point_id == "noise_power_dB-m30-r1"
    assert points[1].point_id == "noise_power_dB-m25-r1"
    assert points[-1].point_id == "ploss-m30-r2"
    assert {point.repetition for point in points} == {1, 2}


def test_positive_ploss_is_rejected():
    config = load_config(config_path())
    config["experiments"][1]["values"] = [0, 5]

    with pytest.raises(ValueError, match="path gain"):
        build_plan(config)


def test_schedule_has_baseline_treatment_and_return():
    point = SweepPoint(
        point_id="noise_power_dB-m25-r1",
        parameter="noise_power_dB",
        value=-25,
        baseline=-30,
        repetition=1,
        treatment_at_s=45,
        return_at_s=135,
        measurement_start_s=90,
        measurement_end_s=135,
    )

    schedule = schedule_for(point)

    assert [event["at_s"] for event in schedule["events"]] == [0, 45, 135]
    assert [event["value"] for event in schedule["events"]] == [-30, -25, -30]
    assert all(event["target"] == "ue1" for event in schedule["events"])


def test_ntpq_parser_reads_selected_peer():
    output = """
     remote           refid      st t when poll reach   delay   offset  jitter
==============================================================================
*155.98.33.74    44.190.5.123     3 u   37   64  377    0.182   +0.013   0.012
"""

    peer = parse_ntpq(output, "ghinwa@core.example")

    assert peer.server == "155.98.33.74"
    assert peer.reach == int("377", 8)
    assert peer.offset_ms == pytest.approx(0.013)
    assert peer.jitter_ms == pytest.approx(0.012)


def test_archive_validation_selects_the_treatment_segment(tmp_path):
    archive = tmp_path / "mgen-test"
    archive.mkdir()
    metadata = {
        "execution_id": "mgen-test",
        "quality": {
            "channel_state_verified": True,
            "xapp": {"clean_shutdown": True, "errors": 0},
        },
    }
    (archive / "metadata.json").write_text(json.dumps(metadata))
    frame = pd.DataFrame(
        [
            {
                    "segment_id": "baseline",
                    "ue": "ue1",
                "direction": "dl",
                "parameter": "noise_power_dB",
                "requested_value": -30,
                "duration_s": 45,
                "valid_clock_fraction": 1.0,
                "controlled": True,
                "verified": True,
                "channel_agreement": True,
                "training_eligible": True,
                "radio_samples": 40,
                "radio_clock_lag_warning": False,
                "radio_clock_lag_s_segment_p95": 0.01,
                "sent_packets": 10,
                "received_packets": 10,
                "loss_rate": 0.0,
                "latency_ms_p95": 2.0,
                    "segment_start_utc": 1000,
                    "segment_end_utc": 1045,
            },
            {
                    "segment_id": "treatment",
                    "ue": "ue1",
                "direction": "dl",
                "parameter": "noise_power_dB",
                "requested_value": -25,
                "duration_s": 90,
                "valid_clock_fraction": 1.0,
                "controlled": True,
                "verified": True,
                "channel_agreement": True,
                "training_eligible": True,
                "radio_samples": 80,
                "ue_radio_samples": 80,
                "ue_radio_clock_valid": True,
                "ue_radio_emit_lag_s_p95": 0.03,
                "ss_rsrp_dbm_segment_mean": -51.0,
                "ss_rsrq_db_segment_mean": -10.46,
                "ss_sinr_db_segment_mean": 38.1,
                "radio_clock_lag_warning": False,
                "radio_clock_lag_s_segment_p95": 0.02,
                "sent_packets": 20,
                "received_packets": 19,
                "loss_rate": 0.05,
                "latency_ms_p95": 3.0,
                    "segment_start_utc": 1045,
                    "segment_end_utc": 1135,
            },
        ]
    )
    frame.to_parquet(archive / "segment_training_table.parquet", index=False)
    logs = archive / "logs"
    logs.mkdir()
    pd.DataFrame({
        "utc_second": list(range(1090, 1135)),
        "emitted_epoch_us": [(second + 1) * 1_000_000 + 30_000
                              for second in range(1090, 1135)],
        "ue": ["ue1"] * 45,
        "cell": [1] * 45,
        "ue_index": [1] * 45,
        "ssb": [0] * 45,
        "samples": [15] * 45,
        "ss_rsrp_dbm": [-51.0] * 45,
        "ss_rsrq_db": [-10.46] * 45,
        "ss_sinr_db": [38.1] * 45,
    }).to_csv(logs / "ue_radio_by_second.csv", index=False)
    pd.DataFrame({
        "ue": ["ue1", "ue1"],
        "direction": ["dl", "dl"],
        "sent_time_utc": [1100.0, 1101.0],
        "packet_clock_valid": [True, True],
        "received": [True, True],
        "lost": [False, False],
        "latency_ms": [2.0, 4.0],
    }).to_parquet(archive / "packet_outcomes.parquet", index=False)
    checksums = {}
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        if path.name != "SHA256SUMS.json":
            checksums[str(path.relative_to(archive))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    (archive / "SHA256SUMS.json").write_text(json.dumps(checksums))
    point = SweepPoint("noise_power_dB-m25-r1", "noise_power_dB", -25, -30, 1, 45, 135, 90, 135)
    config = load_config(config_path())

    result = validate_archive(archive, point, config)

    assert result["segment_id"] == "treatment"
    assert result["verified_files"] == 4
    assert result["valid_clock_fraction"] == 1.0
    assert result["measurement_duration_s"] == pytest.approx(45.0)
    assert result["ss_rsrp_dbm_segment_mean"] == pytest.approx(-51.0)
    assert result["latency_ms_p95"] == pytest.approx(3.9)
