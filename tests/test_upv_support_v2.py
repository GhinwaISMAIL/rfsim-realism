from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rfsim_realism.upv_support_v2 import (
    build_upv_support_v2_plan,
    validate_upv_support_v2_protocol,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_support_v2_protocol.yaml").read_text()
    )


def _config_v2_1() -> dict:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_support_v2_1_protocol.yaml").read_text()
    )


def test_v2_protocol_is_offline_unclipped_and_reservation_closed() -> None:
    config = _config()

    validate_upv_support_v2_protocol(config)

    assert config["execution_authorized"] is False
    assert config["preprocessing"]["distance_calculation_authorized"] is False
    assert config["kernel"]["primary_estimator"] == "biased_mmd_squared_v_statistic"
    assert config["kernel"]["posthoc_clipping"] == "prohibited"
    assert config["reservation"]["request_now"] is False


def test_v2_plan_freezes_safety_probe_without_authorizing_it() -> None:
    plan = build_upv_support_v2_plan(
        phase3a_decision=(
            REPOSITORY / "manifests/upv_measurement_audit_v1/phase3a_decision.json"
        ),
        phase3a_gate=(
            REPOSITORY / "manifests/upv_measurement_audit_v1/reservation_gate_v2.json"
        ),
        config_path=REPOSITORY / "configs/upv_support_v2_protocol.yaml",
    )

    assert plan["measurement_branch_status"] == "unresolved"
    assert plan["execution_authorized"] is False
    assert plan["abc_authorized"] is False
    assert plan["conditional_safety_probe"]["state_count"] == 6
    assert plan["conditional_safety_probe"]["minimum_executions"] == 18
    assert plan["conditional_safety_probe"]["preferred_executions"] == 30
    assert plan["reservation"]["reservation_should_be_requested_now"] is False


def test_v2_plan_rejects_an_open_reservation_gate(tmp_path: Path) -> None:
    decision = json.loads(
        (REPOSITORY / "manifests/upv_measurement_audit_v1/phase3a_decision.json")
        .read_text()
    )
    gate = json.loads(
        (REPOSITORY / "manifests/upv_measurement_audit_v1/reservation_gate_v2.json")
        .read_text()
    )
    gate["reservation_should_be_requested_now"] = True
    decision_path = tmp_path / "decision.json"
    gate_path = tmp_path / "gate.json"
    decision_path.write_text(json.dumps(decision))
    gate_path.write_text(json.dumps(gate))

    with pytest.raises(ValueError, match="reservation gate is open"):
        build_upv_support_v2_plan(
            phase3a_decision=decision_path,
            phase3a_gate=gate_path,
            config_path=REPOSITORY / "configs/upv_support_v2_protocol.yaml",
        )


def test_v2_1_clarifies_offset_relative_rsrp_and_positive_ploss() -> None:
    config = _config_v2_1()

    validate_upv_support_v2_protocol(config)

    branch = config["measurement_branches"]["no_offset_positive_ploss_valid"]
    assert set(branch["trigger"]["all_of"]) == {
        "nemo_oai_rsrp_definitions_comparable_without_offset",
        "positive_ploss_accepted_as_controlled_simulator_gain_only",
    }
    assert branch["physical_propagation_loss_interpretation"] == "prohibited"
    assert config["offset_contract"]["delta_definition"] == (
        "delta_OAI_to_NEMO = RSRP_NEMO - RSRP_OAI"
    )
    relative = config["relative_rsrp_diagnostic"]
    assert relative["status"] == "diagnostic_only"
    assert relative["absolute_location_information"] == "removed"


def test_v2_1_probe_gate_is_executable_but_fail_closed() -> None:
    config = _config_v2_1()
    gate = config["probe_quality_gate"]

    assert gate["authorization_status"] == "blocked_pending_branch_and_parser"
    assert gate["minimum_usable_telemetry_seconds"] == 120
    assert gate["attachment"]["required_fraction"] == 1.0
    assert gate["failure_limits"]["pbch_failure_events_maximum"] == 0
    assert gate["failure_limits"]["pusch_ul_failure_events_maximum"] == 0
    assert gate["failure_limits"]["parser_status"] == (
        "required_not_yet_implemented"
    )
    assert gate["rsrp_direction_test"]["test"] == (
        "exact_one_sided_execution_level_permutation"
    )


def test_v2_1_plan_carries_clarifications_and_keeps_reservation_closed() -> None:
    plan = build_upv_support_v2_plan(
        phase3a_decision=(
            REPOSITORY / "manifests/upv_measurement_audit_v1/phase3a_decision.json"
        ),
        phase3a_gate=(
            REPOSITORY / "manifests/upv_measurement_audit_v1/reservation_gate_v2.json"
        ),
        config_path=REPOSITORY / "configs/upv_support_v2_1_protocol.yaml",
    )

    assert plan["protocol_revision"] == "2.1"
    assert plan["probe_quality_gate"]["authorization_status"] == (
        "blocked_pending_branch_and_parser"
    )
    assert plan["external_request"]["status"] == "prepared_not_sent"
    assert plan["reservation"]["reservation_should_be_requested_now"] is False
