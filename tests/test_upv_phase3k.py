from __future__ import annotations

import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3k import (
    check_phase3k_test6_support,
    freeze_phase3k_model_release,
    validate_phase3k_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3k_test6_validation_v1.yaml"
PHASE3D_CONFIG = ROOT / "configs/upv_phase3d_radio_process_v1.yaml"
PHASE3J_CONFIG = ROOT / "configs/upv_phase3j_full_trace_v1.yaml"
PHASE3J_PROTOCOL = ROOT / "manifests/upv_phase3j_full_trace_v1"
PHASE3J_RESULT = ROOT / "manifests/upv_phase3j_full_trace_result_v1"
SUPPORT = PHASE3J_PROTOCOL / "translator_support_nodes.csv"


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _make_profile_repository(root: Path, config: dict) -> tuple[Path, Path]:
    profile = root / "profile"
    binary = profile / "bin"
    binary.mkdir(parents=True)
    runner = binary / "run-phase3j-full-trace.py"
    engine = binary / "run-phase3i-short-trace.py"
    runner.write_text("runner\n")
    engine.write_text("engine\n")
    subprocess.run(["git", "init", "-q"], cwd=profile, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=profile, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=profile, check=True)
    subprocess.run(["git", "add", "bin"], cwd=profile, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=profile, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=profile,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    frozen = config["frozen_inputs"]["profile"]
    frozen["expected_revision"] = revision
    frozen["full_trace_runner_sha256"] = _sha256(runner)
    frozen["replay_engine_sha256"] = _sha256(engine)
    return runner, engine


def _freeze(tmp_path: Path, config: dict, phase3d_config: Path = PHASE3D_CONFIG) -> Path:
    runner, engine = _make_profile_repository(tmp_path, config)
    config_path = tmp_path / "phase3k.yaml"
    _write_yaml(config_path, config)
    output = tmp_path / "release"
    freeze_phase3k_model_release(
        config_path=config_path,
        phase3d_config_path=phase3d_config,
        phase3j_config_path=PHASE3J_CONFIG,
        phase3j_protocol_dir=PHASE3J_PROTOCOL,
        phase3j_result_dir=PHASE3J_RESULT,
        profile_runner_path=runner,
        profile_engine_path=engine,
        pyproject_path=ROOT / "pyproject.toml",
        uv_lock_path=ROOT / "uv.lock",
        output_dir=output,
        require_clean_repositories=False,
    )
    return output


def test_phase3k_protocol_freezes_pre_access_rules() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3k_config(config)
    amendment = config["test6_support_gate"]["pre_access_runtime_gate_amendment"]
    assert amendment["legacy_absolute_minimum_paired_rows"] == 299
    assert amendment["reference_target_rows"] == 305
    assert amendment["minimum_paired_fraction"] == 299 / 305
    assert amendment["thresholds_selected_from_test6_kpi_distribution"] is False
    assert config["test6_source"]["use"] == "observed_target_trajectory_not_prediction_target"
    assert config["hardware_execution_authorized"] is False
    assert config["reservation"]["request_now"] is False


def test_phase3k_release_does_not_open_test6(tmp_path: Path) -> None:
    config = _read_yaml(CONFIG)
    output = _freeze(tmp_path, config)
    release = json.loads((output / "model_release.json").read_text())
    authorization = json.loads(
        (output / "offline_test6_access_authorization.json").read_text()
    )
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    assert release["phase3j_decision_code"] == "complete_test1_development_replay_passed"
    assert release["test6_payload_opened"] is False
    assert release["hardware_execution_authorized"] is False
    assert authorization["status_before_release_commit"] == "inactive_pending_commit"
    assert authorization["test6_payload_opened"] is False
    assert checksums == {
        name: _sha256(output / name)
        for name in (
            "frozen_file_inventory.csv",
            "model_release.json",
            "offline_test6_access_authorization.json",
        )
    }


def test_phase3k_support_check_uses_frozen_member_and_fractional_runtime_gate(
    tmp_path: Path,
) -> None:
    config = _read_yaml(CONFIG)
    support = pd.read_csv(SUPPORT)
    anchor = support.iloc[(support["observed_relative_rsrp_db"].abs()).argmin()]
    rows = 30
    radio = pd.DataFrame(
        {
            "Time": [f"00:00:{index:02d}" for index in range(rows)],
            "RSRP (NR SpCell)": [-90.0] * rows,
            "RSRQ (NR SpCell)": [-10.5] * rows,
            "SINR (NR SpCell)": [float(anchor["observed_sinr_db"])] * rows,
            "Physical cell identity (NR SpCell)": [41] * rows,
            "Longitude": [2.0 + index * 0.000002 for index in range(rows)],
            "Latitude": [39.0] * rows,
        }
    )
    csv_payload = radio.to_csv(index=False, sep=";").encode()
    archive_path = tmp_path / "upv.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Test_6/Test_6_ASUS.csv", csv_payload)

    phase3d = _read_yaml(PHASE3D_CONFIG)
    archive_sha256 = _sha256(archive_path)
    member_sha256 = hashlib.sha256(csv_payload).hexdigest()
    phase3d["source"]["expected_sha256"] = archive_sha256
    phase3d["final_evaluation"]["source_sha256"] = member_sha256
    phase3d_path = tmp_path / "phase3d.yaml"
    _write_yaml(phase3d_path, phase3d)
    config["frozen_inputs"]["phase3d_config"]["sha256"] = _sha256(phase3d_path)
    config["test6_source"]["archive_sha256"] = archive_sha256
    config["test6_source"]["member_sha256"] = member_sha256
    release = _freeze(tmp_path / "freeze", config, phase3d_path)
    config_path = tmp_path / "freeze" / "phase3k.yaml"
    output = tmp_path / "support-result"
    result = check_phase3k_test6_support(
        config_path=config_path,
        phase3d_config_path=phase3d_path,
        phase3j_config_path=PHASE3J_CONFIG,
        model_release_dir=release,
        translator_support_path=SUPPORT,
        archive_path=archive_path,
        output_dir=output,
        require_committed_release=False,
    )
    decision = json.loads((output / "test6_support_decision.json").read_text())
    audit = json.loads((output / "source_audit.json").read_text())
    assert result["decision_code"] == (
        "test6_target_supported_runner_freeze_and_reservation_required"
    )
    assert decision["target_rows"] == rows
    assert decision["clipped_rows"] == 0
    assert decision["runtime_gate_for_future_execution"]["minimum_paired_rows"] == math.ceil(
        rows * 299 / 305
    )
    assert decision["hardware_execution_authorized"] is False
    assert audit["member_payload_opened"] is True
    assert audit["other_archive_member_payloads_opened_by_this_command"] == 0
