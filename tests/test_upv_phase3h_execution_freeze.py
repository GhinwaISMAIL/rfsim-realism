from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3h_execution_v1.yaml"
FREEZE = ROOT / "manifests/upv_phase3h_execution_v1/hardware_freeze.json"
EXCLUDED_ATTEMPT = (
    ROOT / "manifests/upv_phase3h_attempt1_exclusion_v1/exclusion_decision.json"
)


def test_phase3h_execution_freeze_pins_protocol_profile_runner_and_plan() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["stage"] == "phase_3h_dynamic_staircase_execution"
    assert (
        config["frozen_research"]["protocol_config_sha256"]
        == freeze["protocol"]["config_sha256"]
    )
    assert (
        config["frozen_research"]["protocol_manifest_sha256"]
        == freeze["protocol"]["manifest_sha256"]
    )
    assert config["frozen_profile"]["revision"] == freeze["profile"]["revision"]
    assert config["frozen_profile"]["published"] is True
    assert config["frozen_profile"]["runner_sha256"] == freeze["profile"]["runner_sha256"]
    assert (
        config["frozen_profile"]["execution_plan_sha256"]
        == freeze["profile"]["execution_plan_sha256"]
    )
    assert len(config["frozen_oai"]["revision"]) == 40
    assert len(config["frozen_oai"]["patch_sha256"]) == 7


def test_phase3h_execution_authorizes_only_dynamic_staircase() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    authorization = freeze["execution_authorization"]
    assert authorization["dynamic_staircase_experiment"] is True
    assert authorization["short_trace_replay"] is False
    assert authorization["full_trace_replay"] is False
    assert authorization["final_test6_access"] is False
    assert authorization["abc"] is False
    assert (
        config["execution_authorization"]["dynamic_staircase_experiment"]
        == "conditional_on_all_runtime_preflight_checks"
    )


def test_phase3h_execution_limits_hardware_work_to_three_staircases() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["execution_scope"]["complete_staircase_executions"] == 3
    assert config["execution_scope"]["ue_recreations"] == 3
    assert config["execution_scope"]["total_segments"] == 27
    assert config["execution_scope"]["outer_state_observations"] == 21
    assert freeze["planned_execution_units"] == 3
    assert freeze["planned_ue_recreations"] == 3
    assert freeze["planned_segments"] == 27
    assert freeze["independent_statistical_unit"].startswith("complete_staircase")
    assert config["execution_scope"]["post_attachment_stabilization_seconds"] >= 5
    assert freeze["post_attachment_stabilization_seconds"] >= 5


def test_phase3h_execution_reservation_is_ready_with_lead_time() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    reservation = config["reservation"]
    assert reservation == freeze["reservation"]
    assert reservation["request_now"] is True
    assert reservation["user_notified"] is True
    assert reservation["preparation_lead_time_minutes"] >= 30
    assert reservation["cell_nodes_gnbs"] == 1
    assert reservation["ues_per_cell"] == 1
    assert reservation["channel_family"] == "AWGN"


def test_phase3h_execution_freeze_checksum() -> None:
    checksums = json.loads((FREEZE.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == checksums[FREEZE.name]


def test_phase3h_attempt1_is_immutable_and_excluded() -> None:
    decision = json.loads(EXCLUDED_ATTEMPT.read_text())
    checksums = json.loads((EXCLUDED_ATTEMPT.parent / "SHA256SUMS.json").read_text())
    assert decision["disposition"]["included_in_phase3h_inference"] is False
    assert decision["disposition"]["individual_states_reused"] is False
    assert decision["measurement_completion"]["continuous_attachment"] is True
    assert decision["measurement_completion"]["ping_successes"] == 81
    assert hashlib.sha256(EXCLUDED_ATTEMPT.read_bytes()).hexdigest() == checksums[
        EXCLUDED_ATTEMPT.name
    ]
    assert len(checksums) == 8
