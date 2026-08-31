from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3i_execution_v1_2.yaml"
FREEZE = ROOT / "manifests/upv_phase3i_execution_v1_2/hardware_freeze.json"


def test_python_compatibility_fix_did_not_change_protocol_or_gates() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["immutable_scientific_protocol"]["scientific_targets_or_gates_changed"] is False
    assert freeze["scientific_protocol"]["targets_or_gates_changed"] is False
    assert config["transport"]["maximum_command_completion_lateness_seconds"] == 0.5
    assert config["transport"]["timing_gate_changed"] is False


def test_import_failure_did_not_start_or_consume_replacement_attempt() -> None:
    freeze = json.loads(FREEZE.read_text())
    failure = freeze["pre_execution_failure"]
    assert failure["output_directory_created"] is False
    assert failure["ue_recreated"] is False
    assert failure["channel_controls_applied"] is False
    assert failure["consumes_authorized_replacement_attempt"] is False


def test_v1_2_authorizes_only_one_replacement_short_trace() -> None:
    freeze = json.loads(FREEZE.read_text())
    authorization = freeze["execution_authorization"]
    assert authorization["one_replacement_representative_short_trace"] is True
    assert authorization["additional_repeat_after_replacement"] is False
    assert authorization["complete_test1_trace"] is False
    assert authorization["final_test6_access"] is False
    assert authorization["abc"] is False


def test_v1_2_freeze_checksum() -> None:
    checksums = json.loads((FREEZE.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == checksums[FREEZE.name]
