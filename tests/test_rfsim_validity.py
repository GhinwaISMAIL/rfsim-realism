from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from rfsim_realism.rfsim_validity import (
    NOISE_REFERENCE_RMS,
    corrected_noise_rms,
    legacy_equivalent_corrected_db,
    legacy_noise_rms,
    relative_power_db_from_rms,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "manifests" / "upv_rfsim_validity_audit_v1"


def test_corrected_power_db_round_trip() -> None:
    for noise_db in (3.0, 0.0, -5.0, -10.0, -20.0, -30.0, -60.0):
        rms = corrected_noise_rms(noise_db)
        assert relative_power_db_from_rms(rms) == pytest.approx(noise_db)
        assert math.pow(rms / NOISE_REFERENCE_RMS, 2.0) == pytest.approx(
            math.pow(10.0, noise_db / 10.0)
        )


def test_legacy_equivalence_mapping() -> None:
    for legacy_db in (-30.0, -20.0, -10.0, -7.0, -5.0, 0.0):
        corrected_db = legacy_equivalent_corrected_db(legacy_db)
        assert corrected_noise_rms(corrected_db) == pytest.approx(legacy_noise_rms(legacy_db))


def test_rms_validation_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        relative_power_db_from_rms(0.0)
    with pytest.raises(ValueError, match="positive"):
        relative_power_db_from_rms(1.0, reference_rms=0.0)


def test_erratum_mapping_is_computed_not_relabelled() -> None:
    erratum = json.loads((AUDIT / "prior_result_erratum.json").read_text())
    mapping = erratum["legacy_to_corrected_equivalence"]
    for row in mapping:
        assert row["corrected_equivalent_noise_power_db"] == pytest.approx(
            legacy_equivalent_corrected_db(row["legacy_command_noise_power_db"])
        )
    assert erratum["raw_data_policy"] == "preserve_commands_and_outputs_without_relabelling"


def test_reservation_opens_after_profile_is_available_remotely() -> None:
    gate = json.loads((AUDIT / "reservation_gate.json").read_text())
    assert gate["request_now"] is True
    assert gate["gate_state"] == "ready_to_request"
    assert gate["reservation_notice_minutes"] == 30
    assert gate["remote_profile_state"]["required_revision_available"] is True


def test_corrected_noise_validation_protocol_is_bounded() -> None:
    protocol = json.loads((AUDIT / "corrected_noise_validation_protocol.json").read_text())
    design = protocol["design"]
    assert (
        protocol["protocol_status"] == "frozen_remote_profile_available_hardware_execution_pending"
    )
    assert protocol["reservation"]["cell_nodes_gnbs"] == 1
    assert protocol["reservation"]["ues_per_cell"] == 1
    assert protocol["reservation"]["channel_family"] == "AWGN"
    assert design["noise_power_db_states"] == [-60.0, -40.0, -30.0, -25.0, -20.0]
    assert design["independent_executions_per_state"] == 3
    flattened = [
        state for repetition in design["state_order_by_repetition"] for state in repetition
    ]
    assert sorted(flattened) == sorted(design["noise_power_db_states"] * 3)
    assert "final UPV validation" in protocol["not_authorized"]


def test_audit_manifest_checksums() -> None:
    checksums = json.loads((AUDIT / "SHA256SUMS.json").read_text())
    for relative_path, expected in checksums.items():
        assert hashlib.sha256((AUDIT / relative_path).read_bytes()).hexdigest() == expected


def test_hardware_execution_freeze_matches_protocol() -> None:
    protocol = json.loads((AUDIT / "corrected_noise_validation_protocol.json").read_text())
    freeze = json.loads((AUDIT / "hardware_execution_freeze.json").read_text())
    plan = freeze["execution_plan"]
    expected = [
        state
        for repetition in protocol["design"]["state_order_by_repetition"]
        for state in repetition
    ]
    assert [row["noise_power_db"] for row in plan] == expected
    assert len({row["oai_rng_seed"] for row in plan}) == 15
    assert freeze["image_identity"]["corrected_image_id"].startswith("sha256:")
    assert freeze["authorization"]["execute_frozen_15_run_validation"] is True
    assert freeze["authorization"]["final_upv_test6_access"] is False


def test_audit_preserves_independent_results() -> None:
    audit = json.loads((AUDIT / "source_audit.json").read_text())
    assert audit["findings"]["noise_power_scaling"]["decision"] == "confirmed_in_pinned_source"
    assert audit["result_impact"]["phase3c15_fixed_noise_conclusion"] == "reopen"
    assert audit["result_impact"]["phase3d_measurement_only_analysis"] == "unchanged"
    assert audit["result_impact"]["cirdb_timing_diagnosis"] == "unchanged"
