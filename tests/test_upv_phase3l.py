from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3l import (
    analyze_phase3l_exploratory_replay,
    freeze_phase3l_exploratory_protocol,
    validate_phase3l_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3l_test6_exploratory_v1.yaml"
PHASE3K_CONFIG = ROOT / "configs/upv_phase3k_test6_validation_v1.yaml"
MODEL_RELEASE = ROOT / "manifests/upv_phase3k_model_release_v1"
SUPPORT_RESULT = ROOT / "manifests/upv_phase3k_test6_support_result_v1"
PHASE3J_CONFIG = ROOT / "configs/upv_phase3j_full_trace_v1.yaml"
AUTHORIZATION = ROOT / "manifests/upv_phase3l_execution_authorization_v1"


def _freeze(output: Path) -> dict[str, str]:
    return freeze_phase3l_exploratory_protocol(
        config_path=CONFIG,
        phase3k_config_path=PHASE3K_CONFIG,
        model_release_dir=MODEL_RELEASE,
        support_result_dir=SUPPORT_RESULT,
        phase3j_config_path=PHASE3J_CONFIG,
        output_dir=output,
    )


def test_phase3l_protocol_preserves_confirmatory_failure() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3l_config(config)
    assert config["official_confirmatory_result"]["support_gate_passed"] is False
    assert config["official_confirmatory_result"]["immutable"] is True
    assert config["exploratory_rationale"]["new_support_thresholds_defined"] is False
    assert config["exploratory_rationale"]["translator_changed"] is False
    assert config["radio_state_classification"]["primary_classification_claim"] == (
        "prohibited"
    )
    assert config["execution"]["exploratory_executions"] == 1
    assert config["reservation"]["request_now"] is False


def test_phase3l_freeze_records_raw_errors_intervals_and_sensitivity(tmp_path: Path) -> None:
    output = tmp_path / "phase3l"
    result = _freeze(output)
    diagnosis = json.loads((output / "clipping_diagnosis.json").read_text())
    intervals = pd.read_csv(output / "clipping_intervals.csv")
    sensitivity = pd.read_csv(output / "classification_threshold_sensitivity.csv")
    protocol = json.loads((output / "protocol.json").read_text())
    commands = pd.read_csv(output / "exploratory_test6_commands.csv")

    assert result["hardware_execution_authorized"] == "false"
    assert diagnosis["clipped_rows"] == 21
    assert diagnosis["clipped_intervals"] == 11
    assert diagnosis["longest_clipped_interval_rows"] == 7
    assert diagnosis["clipped_point_errors"][
        "relative_rsrp_mean_absolute_db"
    ] == pytest.approx(0.261338837222)
    assert diagnosis["clipped_point_errors"]["sinr_maximum_absolute_db"] == pytest.approx(
        1.6926155148
    )
    assert intervals.loc[intervals["duration_rows"].idxmax(), "start_command_index"] == 122
    changed = sensitivity[sensitivity["changed_classification_rows"] > 0]
    assert set(changed.loc[changed["metric"] == "sinr_db", "threshold_db"]) == {
        11,
        12,
        13,
        22,
        23,
    }
    assert len(commands) == 297
    assert _sha256(output / "exploratory_test6_commands.csv") == _sha256(
        SUPPORT_RESULT / "test6_commands.csv"
    )
    assert protocol["official_confirmatory_result"]["support_gate_passed"] is False
    assert protocol["hardware_execution_authorized"] is False


def _synthetic_campaign(root: Path, protocol: Path) -> Path:
    campaign = root / "campaign"
    campaign.mkdir()
    commands = pd.read_csv(protocol / "exploratory_test6_commands.csv")
    telemetry = commands.copy()
    telemetry["applied_gain_db"] = telemetry["commanded_gain_db"]
    telemetry["applied_noise_power_db"] = telemetry["commanded_noise_power_db"]
    telemetry["rsrp_db_per_re_unquantized"] = telemetry["target_relative_rsrp_db"] + 40.0
    telemetry["ss_sinr_db"] = telemetry["target_sinr_db"]
    telemetry["command_completion_lateness_seconds"] = 0.1
    telemetry["attached"] = True
    telemetry["channel_family"] = "AWGN"
    telemetry["channel_length"] = 1
    telemetry["nb_taps"] = 1
    telemetry["tap_energy_linear"] = 1.0
    telemetry.to_csv(campaign / "phase3l_test6_telemetry.csv", index=False)
    pd.DataFrame(
        {
            "anchor_type": ["anchor_start"] * 10 + ["anchor_end"] * 10,
            "rsrp_db_per_re_unquantized": [40.0] * 20,
            "ss_sinr_db": [15.0] * 20,
        }
    ).to_csv(campaign / "phase3l_anchor_telemetry.csv", index=False)
    state = {
        "stage": "phase_3l_posthoc_test6_exploratory_replay",
        "evaluation_status": "posthoc_exploratory_not_confirmatory_validation",
        "execution_number": 1,
        "execution_id": "phase3l-test6-exploratory-execution-1",
        "oai_rng_seed": 48001,
        "execution_completed": True,
        "error": None,
        "commands_sha256": _sha256(protocol / "exploratory_test6_commands.csv"),
        "research_protocol_sha256": _sha256(CONFIG),
        "test6_accessed": True,
        "translator_update_authorized": False,
        "gNB_untouched": True,
        "control_echo_abs_tolerance_db": 5e-6,
        "control_application_verification_source": "immediate_persistent_telnet_show",
        "channel_snapshot_purpose": "static_channel_identity_and_tap_invariants_only",
        "channel_snapshot_control_match_required": False,
        "research_revision": "test-research-revision",
        "oai_revision": "test-oai-revision",
        "profile_revision": "test-profile-revision",
        "runner_sha256": "test-runner-sha256",
        "compose_sha256": "test-compose-sha256",
        "channel_config_sha256": "test-channel-sha256",
        "ue_config_sha256": "test-ue-sha256",
        "debug_image_id": "test-image-id",
        "ping_success_fraction": 1.0,
        "critical_failure_count": 0,
        "ue_restart_count": 0,
        "gnb_restart_count_change": 0,
        "rollback": {"passed": True},
        "exploratory_replay": True,
        "frozen_v1_support_gate_passed": False,
    }
    (campaign / "execution_state.json").write_text(json.dumps(state))
    return campaign


def test_phase3l_analyzer_reports_exploratory_runtime_without_confirmatory_pass(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol"
    _freeze(protocol)
    campaign = _synthetic_campaign(tmp_path, protocol)
    output = tmp_path / "result"
    result = analyze_phase3l_exploratory_replay(
        campaign_dir=campaign,
        protocol_dir=protocol,
        config_path=CONFIG,
        output_dir=output,
    )
    decision = json.loads((output / "phase3l_exploratory_decision.json").read_text())
    metrics = pd.read_csv(output / "exploratory_execution_metrics.csv").iloc[0]
    assert result["decision"] == "exploratory_test6_replay_completed_runtime_valid"
    assert decision["official_phase3k_support_status"] == (
        "unsupported_under_version_1_protocol"
    )
    assert decision["confirmatory_support_pass_claimed"] is False
    assert decision["exploratory_execution"]["reference_limits_are_acceptance_gates"] is False
    assert metrics["paired_rows"] == 297
    assert metrics["clipped_rows"] == 21
    assert metrics["runtime_gate_passed"]
    assert np.isclose(metrics["total_relative_rsrp_mae_db"], 0.0)


def test_phase3l_hardware_authorization_is_exploratory_and_checksums_match() -> None:
    authorization = json.loads((AUTHORIZATION / "hardware_authorization.json").read_text())
    checksums = json.loads((AUTHORIZATION / "SHA256SUMS.json").read_text())
    assert checksums == {
        "hardware_authorization.json": _sha256(
            AUTHORIZATION / "hardware_authorization.json"
        )
    }
    assert authorization["official_confirmatory_result"][
        "phase3k_support_gate_passed"
    ] is False
    assert authorization["execution_authorization"]["authorized_execution_numbers"] == [1]
    assert authorization["execution_authorization"]["oai_rng_seeds"] == [48001]
    assert authorization["execution_authorization"]["confirmatory_validation"] is False
    assert authorization["evaluation_contract"][
        "phase3j_fidelity_limits_are_acceptance_gates"
    ] is False
    assert authorization["reservation"]["request_now"] is True
