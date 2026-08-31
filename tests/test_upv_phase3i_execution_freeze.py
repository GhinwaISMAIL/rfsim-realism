from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3i_execution_v1.yaml"
FREEZE = ROOT / "manifests/upv_phase3i_execution_v1/hardware_freeze.json"
PROTOCOL_DIR = ROOT / "manifests/upv_phase3i_short_trace_v1"


def test_phase3i_execution_freeze_pins_protocol_profile_runner_and_commands() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert config["stage"] == "phase_3i_representative_short_trace_execution"
    assert config["frozen_research"]["repository_revision"] == freeze["research_revision"]
    assert config["frozen_profile"]["revision"] == freeze["profile"]["revision"]
    assert config["frozen_profile"]["published"] is True
    assert config["frozen_profile"]["runner_sha256"] == freeze["profile"]["runner_sha256"]
    assert config["frozen_profile"]["commands_sha256"] == freeze["profile"]["commands_sha256"]
    assert len(config["frozen_oai"]["revision"]) == 40


def test_phase3i_execution_authorizes_only_representative_short_trace() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    authorization = freeze["execution_authorization"]
    assert authorization["representative_short_trace"] is True
    assert authorization["complete_test1_trace"] is False
    assert authorization["final_test6_access"] is False
    assert authorization["abc"] is False
    assert (
        config["execution_authorization"]["representative_short_trace"]
        == "conditional_on_all_runtime_preflight_checks"
    )


def test_phase3i_execution_scope_is_one_unclipped_sixty_second_trace() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    scope = config["execution_scope"]
    assert scope["source_rows_inclusive"] == [154, 213]
    assert scope["target_rows"] == 60
    assert scope["command_interval_seconds"] == 1.0
    assert scope["clean_ue_recreations"] == 1
    assert scope["command_trace_clipped_fraction"] == 0.0
    assert freeze["planned_execution_units"] == 1
    assert freeze["planned_commands"] == 60


def test_phase3i_execution_protocol_hashes_match_immutable_files() -> None:
    freeze = json.loads(FREEZE.read_text())
    targets = {
        "config_sha256": ROOT / "configs/upv_phase3i_short_trace_v1.yaml",
        "manifest_sha256": PROTOCOL_DIR / "protocol.json",
        "commands_sha256": PROTOCOL_DIR / "short_trace_commands.csv",
        "selected_trace_sha256": PROTOCOL_DIR / "selected_target_trace.csv",
    }
    for key, path in targets.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze["protocol"][key]


def test_phase3i_execution_freeze_checksum() -> None:
    checksums = json.loads((FREEZE.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == checksums[FREEZE.name]


def test_phase3i_execution_reuses_healthy_reservation() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    reservation = config["reservation"]
    assert reservation == freeze["reservation"]
    assert reservation["request_now"] is False
    assert reservation["reuse_current_if_healthy"] is True
    assert reservation["preparation_lead_time_minutes"] >= 30
    assert reservation["channel_family"] == "AWGN"
