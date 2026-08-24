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
