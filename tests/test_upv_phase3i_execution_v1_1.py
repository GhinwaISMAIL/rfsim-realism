from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3i_execution_v1_1.yaml"
FREEZE = ROOT / "manifests/upv_phase3i_execution_v1_1/hardware_freeze.json"
EXCLUSION = (
    ROOT / "manifests/upv_phase3i_attempt1_exclusion_v1/exclusion_decision.json"
)
DIAGNOSIS = EXCLUSION.with_name("transport_diagnosis.json")


def test_attempt1_is_excluded_before_scientific_trace() -> None:
    decision = json.loads(EXCLUSION.read_text())
    assert decision["attempt"]["paired_trace_rows"] == 0
    assert decision["integrity"]["rollback_passed"] is True
    assert decision["disposition"]["included_in_phase3i_fidelity_analysis"] is False
    assert decision["disposition"]["scientific_protocol_or_threshold_changed"] is False


def test_replacement_retains_protocol_commands_and_timing_gate() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    protocol = config["immutable_scientific_protocol"]
    assert protocol["scientific_targets_or_gates_changed"] is False
    assert protocol["commands_sha256"] == freeze["scientific_protocol"]["commands_sha256"]
    assert config["transport"]["mode"] == "persistent_telnet_session"
    assert config["transport"]["readback_verification"] == "required"
    assert config["transport"]["maximum_command_completion_lateness_seconds"] == 0.5
    assert config["transport"]["timing_gate_changed"] is False


def test_only_one_replacement_attempt_is_authorized() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    authorization = freeze["execution_authorization"]
    assert authorization["one_replacement_representative_short_trace"] is True
    assert authorization["additional_repeat_after_replacement"] is False
    assert authorization["complete_test1_trace"] is False
    assert authorization["final_test6_access"] is False
    assert authorization["abc"] is False
    assert (
        config["execution_authorization"]["one_replacement_representative_short_trace"]
        == "conditional_on_all_runtime_preflight_checks"
    )


def test_exclusion_evidence_and_manifest_checksums() -> None:
    exclusion_checksums = json.loads((EXCLUSION.parent / "SHA256SUMS.json").read_text())
    for path in (EXCLUSION, DIAGNOSIS):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == exclusion_checksums[path.name]
    freeze_checksums = json.loads((FREEZE.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == freeze_checksums[FREEZE.name]


def test_replacement_reuses_current_awgn_reservation() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["reservation"] == freeze["reservation"]
    assert config["reservation"]["request_now"] is False
    assert config["reservation"]["reuse_current_if_healthy"] is True
    assert config["reservation"]["channel_family"] == "AWGN"
