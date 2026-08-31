from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3g_execution_v1.yaml"
FREEZE = ROOT / "manifests/upv_phase3g_execution_v1/hardware_freeze.json"


def test_phase3g_execution_freeze_pins_profile_runner_and_plan() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["stage"] == "phase_3g_bounded_gain_noise_response_execution"
    assert config["frozen_profile"]["revision"] == freeze["profile"]["revision"]
    assert config["frozen_profile"]["runner_sha256"] == freeze["profile"]["runner_sha256"]
    assert (
        config["frozen_profile"]["execution_plan_sha256"]
        == freeze["profile"]["execution_plan_sha256"]
    )
    assert len(config["frozen_oai"]["revision"]) == 40
    assert len(config["frozen_oai"]["patch_sha256"]) == 7


def test_phase3g_execution_authorizes_only_bounded_response() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    authorization = freeze["execution_authorization"]
    assert authorization["bounded_response_experiment"] is True
    assert authorization["direct_trace_replay"] is False
    assert authorization["final_test6_access"] is False
    assert authorization["abc"] is False
    assert (
        config["execution_authorization"]["bounded_response_experiment"]
        == "conditional_on_all_runtime_preflight_checks"
    )
    assert config["reservation"]["request_now"] is True
    assert config["reservation"]["preparation_lead_time_minutes"] >= 30


def test_phase3g_execution_freeze_checksum() -> None:
    checksums = json.loads((FREEZE.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == checksums[FREEZE.name]
