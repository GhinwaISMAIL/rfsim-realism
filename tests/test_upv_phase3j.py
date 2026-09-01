from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3j import (
    analyze_phase3j_full_trace,
    freeze_phase3j_full_trace,
    validate_phase3j_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3j_full_trace_v1.yaml"
PHASE3I_DECISION = (
    ROOT
    / "manifests/upv_phase3i_short_trace_result_v1"
    / "phase3i_short_trace_decision.json"
)
PHASE3G = ROOT / "manifests/upv_phase3g_response_v1/execution_medians.csv"
PHASE3H = (
    ROOT
    / "manifests/upv_phase3h_translation_validation_v1_1"
    / "state_translation_validation.csv"
)
TRACE = ROOT / "data/model_runs/upv_phase3g_direct_trace_v1/direct_test1_target_trace.csv"
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"


def _freeze(output: Path) -> dict[str, str]:
    return freeze_phase3j_full_trace(
        config_path=CONFIG,
        phase3i_decision_path=PHASE3I_DECISION,
        phase3g_execution_medians_path=PHASE3G,
        phase3h_state_validation_path=PHASE3H,
        direct_trace_path=TRACE,
        pyproject_path=PYPROJECT,
        uv_lock_path=UV_LOCK,
        output_dir=output,
    )


def test_phase3j_protocol_freezes_development_and_test6_rules() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3j_config(config)
    assert config["evaluation_status"] == "development_not_independent_final_validation"
    assert config["target_trace"]["input_trace_is_development_data"] is True
    assert config["test6_access_authorized"] is False
    assert config["test6_support_gate"] == {
        "freeze_before_test6_access": True,
        "maximum_clipped_fraction": 0.05,
        "maximum_clipping_distance_scaled": 1.0,
        "excessive_out_of_hull_status": "unsupported_not_emulator_failure",
        "hardware_execution_if_unsupported": "prohibited",
        "primary_temporal_metrics": "include_clipped_rows_against_original_target",
        "supported_only_temporal_metrics": (
            "exclude_clipped_rows_and_adjacent_crossing_pairs"
        ),
        "minimum_paired_rows": 299,
        "missing_telemetry_interpolation": "prohibited",
        "temporal_gap_rule": "split_at_missing_rows_and_never_bridge",
        "primary_kpi_alignment_seconds": 0,
        "channel_verification_alignment_seconds": 1,
        "post_hoc_lag_selection": "prohibited",
    }
    assert config["runtime_gates"]["missing_row_interpolation"] == "prohibited"
    assert config["clipping_evaluation"]["bridge_across_clipped_rows"] == "prohibited"
    assert config["metric_definitions"]["primary_metrics_target"] == (
        "original_measured_target_including_clipped_rows"
    )
    assert config["repeatability_gates"]["aggregate_statistic"] == (
        "root_mean_square_over_command_indices_present_in_all_executions"
    )
    assert (
        config["model_update_policy"]["translator_update_from_phase3j_residuals"]
        == "prohibited"
    )
    assert config["reservation"]["request_now"] is False


def test_phase3j_freeze_builds_bounded_complete_trace(tmp_path: Path) -> None:
    output = tmp_path / "phase3j"
    result = _freeze(output)
    commands = pd.read_csv(output / "full_trace_commands.csv")
    support = json.loads((output / "support_report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())

    assert result["command_rows"] == "305"
    assert result["clipped_rows"] == "8"
    assert result["development_support_gate_passed"] == "true"
    assert len(commands) == 305
    assert commands["command_index"].tolist() == list(range(305))
    assert commands.loc[commands["clipped"], "trace_row_index"].tolist() == [
        72,
        101,
        102,
        107,
        124,
        125,
        139,
        141,
    ]
    assert commands["commanded_gain_db"].between(-18.0, 0.0).all()
    assert commands["commanded_noise_power_db"].between(-35.0, -17.0).all()
    inside = ~commands["clipped"]
    assert np.allclose(
        commands.loc[inside, "target_relative_rsrp_db"],
        commands.loc[inside, "projected_relative_rsrp_db"],
    )
    assert np.allclose(
        commands.loc[inside, "target_sinr_db"],
        commands.loc[inside, "projected_sinr_db"],
    )
    assert support["target_rows"] == 305
    assert support["inside_rows"] == 297
    assert support["clipped_rows"] == 8
    assert support["clipped_fraction"] == pytest.approx(8 / 305)
    assert support["maximum_clipping_distance_scaled"] == pytest.approx(
        0.672419638881
    )
    assert support["development_support_gate_passed"] is True
    assert protocol["evaluation_status"] == "development_not_independent_final_validation"
    assert protocol["execution_authorized"] is False
    assert protocol["test6_access_authorized"] is False
    assert protocol["test6_interpretation"]["predicts_test6_without_observing_it"] is False


def test_phase3j_freeze_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _freeze(first)
    _freeze(second)
    for name in (
        "translator_support_nodes.csv",
        "complete_test1_target_trace.csv",
        "full_trace_commands.csv",
        "clipped_targets.csv",
        "support_report.json",
        "test6_support_rules.json",
        "protocol.json",
        "analysis_manifest.json",
        "SHA256SUMS.json",
    ):
        assert _sha256(first / name) == _sha256(second / name)


def test_phase3j_rejects_tampered_input(tmp_path: Path) -> None:
    tampered = tmp_path / TRACE.name
    shutil.copyfile(TRACE, tampered)
    tampered.write_text(tampered.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        freeze_phase3j_full_trace(
            config_path=CONFIG,
            phase3i_decision_path=PHASE3I_DECISION,
            phase3g_execution_medians_path=PHASE3G,
            phase3h_state_validation_path=PHASE3H,
            direct_trace_path=tampered,
            pyproject_path=PYPROJECT,
            uv_lock_path=UV_LOCK,
            output_dir=tmp_path / "output",
        )


def _synthetic_campaign(
    root: Path,
    execution_number: int,
    *,
    sinr_offset_db: float = 0.0,
    applied_gain_offset_db: float = 0.0,
) -> Path:
    campaign = root / f"campaign-{execution_number}"
    campaign.mkdir()
    commands = pd.read_csv(
        ROOT / "manifests/upv_phase3j_full_trace_v1/full_trace_commands.csv"
    )
    telemetry = commands[
        [
            "command_index",
            "trace_row_index",
            "trace_time_bin",
            "trace_t_s",
            "target_relative_rsrp_db",
            "target_sinr_db",
            "projected_relative_rsrp_db",
            "projected_sinr_db",
            "commanded_gain_db",
            "commanded_noise_power_db",
            "clipped",
        ]
    ].copy()
    telemetry["applied_gain_db"] = (
        telemetry["commanded_gain_db"] + applied_gain_offset_db
    )
    telemetry["applied_noise_power_db"] = telemetry["commanded_noise_power_db"]
    telemetry["rsrp_db_per_re_unquantized"] = (
        telemetry["target_relative_rsrp_db"] + 40.0
    )
    telemetry["ss_sinr_db"] = telemetry["target_sinr_db"] + sinr_offset_db
    telemetry["command_completion_lateness_seconds"] = 0.1
    telemetry["attached"] = True
    telemetry["channel_family"] = "AWGN"
    telemetry["channel_length"] = 1
    telemetry["nb_taps"] = 1
    telemetry["tap_energy_linear"] = 1.0
    telemetry.to_csv(campaign / "phase3j_full_trace_telemetry.csv", index=False)
    anchors = pd.DataFrame(
        {
            "anchor_type": ["anchor_start"] * 10 + ["anchor_end"] * 10,
            "rsrp_db_per_re_unquantized": [40.0] * 20,
            "ss_sinr_db": [15.0] * 20,
        }
    )
    anchors.to_csv(campaign / "phase3j_anchor_telemetry.csv", index=False)
    config_sha256 = _sha256(CONFIG)
    commands_sha256 = _sha256(
        ROOT / "manifests/upv_phase3j_full_trace_v1/full_trace_commands.csv"
    )
    state = {
        "stage": "phase_3j_complete_test1_development_fidelity_and_repeatability",
        "evaluation_status": "development_not_independent_final_validation",
        "execution_number": execution_number,
        "execution_id": f"phase3j-test1-execution-{execution_number}",
        "oai_rng_seed": 47000 + execution_number,
        "execution_completed": True,
        "error": None,
        "commands_sha256": commands_sha256,
        "research_protocol_sha256": config_sha256,
        "test6_accessed": False,
        "translator_update_authorized": False,
        "gNB_untouched": True,
        "control_echo_abs_tolerance_db": 5e-6,
        "control_application_verification_source": "immediate_persistent_telnet_show",
        "channel_snapshot_purpose": "static_channel_identity_and_tap_invariants_only",
        "channel_snapshot_control_match_required": False,
        "research_revision": "research-revision",
        "oai_revision": "oai-revision",
        "profile_revision": "profile-revision",
        "runner_sha256": "runner-sha256",
        "compose_sha256": "compose-sha256",
        "channel_config_sha256": "channel-config-sha256",
        "ue_config_sha256": "ue-config-sha256",
        "debug_image_id": "debug-image-id",
        "ping_success_fraction": 1.0,
        "critical_failure_count": 0,
        "ue_restart_count": 0,
        "gnb_restart_count_change": 0,
        "rollback": {"passed": True},
    }
    (campaign / "execution_state.json").write_text(json.dumps(state))
    return campaign


def test_phase3j_analyzer_passes_exact_three_execution_replay(tmp_path: Path) -> None:
    campaigns = [
        _synthetic_campaign(
            tmp_path,
            number,
            applied_gain_offset_db=1.572e-6 if number == 1 else 0.0,
        )
        for number in (1, 2, 3)
    ]
    output = tmp_path / "analysis"
    result = analyze_phase3j_full_trace(
        campaign_dirs=campaigns,
        protocol_dir=ROOT / "manifests/upv_phase3j_full_trace_v1",
        config_path=CONFIG,
        output_dir=output,
    )
    decision = json.loads((output / "phase3j_full_trace_decision.json").read_text())
    metrics = pd.read_csv(output / "per_execution_metrics.csv")
    repeatability = pd.read_csv(output / "repeatability_by_command.csv")
    assert result["decision"] == "complete_test1_development_replay_passed"
    assert result["model_release_freeze_authorized"] == "true"
    assert result["test6_access_authorized"] == "false"
    assert len(metrics) == 3
    assert metrics["runtime_gate_passed"].all()
    assert metrics["fidelity_gate_passed"].all()
    assert np.allclose(metrics["total_relative_rsrp_mae_db"], 0.0)
    assert np.allclose(metrics["total_sinr_mae_db"], 0.0)
    assert len(repeatability) == 305
    assert np.allclose(
        repeatability["relative_rsrp_sample_standard_deviation_db"], 0.0
    )
    assert np.allclose(repeatability["sinr_sample_standard_deviation_db"], 0.0)
    assert decision["evaluation_status"] == "development_not_independent_final_validation"
    assert decision["model_release_freeze_authorized"] is True
    assert decision["translator_update_from_residuals_authorized"] is False
    assert decision["test6_accessed"] is False


def test_phase3j_analyzer_rejects_reused_execution_directory(tmp_path: Path) -> None:
    campaign = _synthetic_campaign(tmp_path, 1)
    with pytest.raises(ValueError, match="must be distinct"):
        analyze_phase3j_full_trace(
            campaign_dirs=[campaign, campaign, campaign],
            protocol_dir=ROOT / "manifests/upv_phase3j_full_trace_v1",
            config_path=CONFIG,
            output_dir=tmp_path / "analysis",
        )


def test_phase3j_analyzer_rejects_control_error_above_frozen_tolerance(
    tmp_path: Path,
) -> None:
    campaigns = [
        _synthetic_campaign(
            tmp_path,
            number,
            applied_gain_offset_db=1e-5 if number == 1 else 0.0,
        )
        for number in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="applied different controls"):
        analyze_phase3j_full_trace(
            campaign_dirs=campaigns,
            protocol_dir=ROOT / "manifests/upv_phase3j_full_trace_v1",
            config_path=CONFIG,
            output_dir=tmp_path / "analysis",
        )


def test_phase3j_hardware_authorization_matches_frozen_packages() -> None:
    authorization_root = ROOT / "manifests/upv_phase3j_execution_authorization_v1"
    authorization = json.loads(
        (authorization_root / "hardware_authorization.json").read_text()
    )
    checksums = json.loads((authorization_root / "SHA256SUMS.json").read_text())
    assert checksums["hardware_authorization.json"] == _sha256(
        authorization_root / "hardware_authorization.json"
    )
    assert authorization["research_package"]["repository_revision"] == (
        "16117beead9da5a72a862e820a25ea7fee810345"
    )
    assert authorization["research_package"]["protocol_config_sha256"] == _sha256(
        CONFIG
    )
    assert authorization["research_package"]["protocol_manifest_sha256"] == _sha256(
        ROOT / "manifests/upv_phase3j_full_trace_v1/protocol.json"
    )
    assert authorization["research_package"]["commands_sha256"] == _sha256(
        ROOT / "manifests/upv_phase3j_full_trace_v1/full_trace_commands.csv"
    )
    assert authorization["execution_authorization"]["authorized_execution_numbers"] == [
        1,
        2,
        3,
    ]
    assert authorization["rollback"]["mandatory_after_each_execution"] is True
    assert authorization["reservation"]["request_now"] is True
    assert authorization["claim_limits"]["test6_access"] == "prohibited"
