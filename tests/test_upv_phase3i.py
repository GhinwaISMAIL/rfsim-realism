from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3i import (
    _support_bank,
    analyze_phase3i_short_trace,
    freeze_phase3i_short_trace,
    validate_phase3i_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3i_short_trace_v1.yaml"
DECISION = (
    ROOT
    / "manifests/upv_phase3h_translation_validation_v1_1"
    / "phase3h_staircase_decision.json"
)
PHASE3G = ROOT / "manifests/upv_phase3g_response_v1/execution_medians.csv"
PHASE3H = (
    ROOT
    / "manifests/upv_phase3h_translation_validation_v1_1"
    / "state_translation_validation.csv"
)
TRACE = ROOT / "data/model_runs/upv_phase3g_direct_trace_v1/direct_test1_target_trace.csv"


def _freeze(output: Path) -> dict[str, str]:
    return freeze_phase3i_short_trace(
        config_path=CONFIG,
        phase3h_decision_path=DECISION,
        phase3g_execution_medians_path=PHASE3G,
        phase3h_state_validation_path=PHASE3H,
        direct_trace_path=TRACE,
        output_dir=output,
    )


def test_phase3i_support_and_protocol_are_bounded() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3i_config(config)
    support = _support_bank(pd.read_csv(PHASE3G), pd.read_csv(PHASE3H))
    assert len(support) == 20
    assert support["commanded_gain_db"].between(-18.0, 0.0).all()
    assert support["commanded_noise_power_db"].between(-35.0, -17.0).all()
    assert config["translator"]["extrapolation"] == "prohibited"
    assert config["reservation"]["request_now"] is False


def test_phase3i_freeze_selects_representative_supported_window(tmp_path: Path) -> None:
    output = tmp_path / "protocol"
    result = _freeze(output)
    commands = pd.read_csv(output / "short_trace_commands.csv")
    protocol = json.loads((output / "protocol.json").read_text())
    assert result["selected_start_row"] == "154"
    assert result["selected_end_row_inclusive"] == "213"
    assert len(commands) == 60
    assert not commands["clipped"].astype(bool).any()
    assert commands["commanded_gain_db"].between(-18.0, 0.0).all()
    assert commands["commanded_noise_power_db"].between(-35.0, -17.0).all()
    assert protocol["selection"]["inside_fraction"] == 1.0
    assert protocol["execution_authorized"] is False
    assert protocol["full_trace_replay_authorized"] is False


def test_phase3i_analyzer_passes_exact_synthetic_replay(tmp_path: Path) -> None:
    protocol_root = tmp_path / "protocol"
    _freeze(protocol_root)
    commands = pd.read_csv(protocol_root / "short_trace_commands.csv")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    telemetry = commands[
        [
            "command_index",
            "target_relative_rsrp_db",
            "target_sinr_db",
            "projected_relative_rsrp_db",
            "projected_sinr_db",
            "commanded_gain_db",
            "commanded_noise_power_db",
        ]
    ].copy()
    telemetry["applied_gain_db"] = telemetry["commanded_gain_db"]
    telemetry["applied_noise_power_db"] = telemetry["commanded_noise_power_db"]
    telemetry["rsrp_db_per_re_unquantized"] = (
        telemetry["projected_relative_rsrp_db"] + 40.0
    )
    telemetry["ss_sinr_db"] = telemetry["projected_sinr_db"]
    telemetry["command_completion_lateness_seconds"] = 0.1
    telemetry["attached"] = True
    telemetry.to_csv(campaign / "phase3i_short_trace_telemetry.csv", index=False)
    anchors = pd.DataFrame(
        {
            "anchor_type": ["anchor_start"] * 10 + ["anchor_end"] * 10,
            "rsrp_db_per_re_unquantized": [40.0] * 20,
            "ss_sinr_db": [15.0] * 20,
        }
    )
    anchors.to_csv(campaign / "phase3i_anchor_telemetry.csv", index=False)
    state = {
        "execution_completed": True,
        "error": None,
        "final_test6_accessed": False,
        "full_trace_replay_authorized": False,
        "commands_sha256": _sha256(protocol_root / "short_trace_commands.csv"),
        "research_protocol_sha256": _sha256(CONFIG),
        "gNB_untouched": True,
        "ping_success_fraction": 1.0,
        "critical_failure_count": 0,
        "ue_restart_count": 0,
        "gnb_restart_count_change": 0,
        "rollback": {"passed": True},
    }
    (campaign / "execution_state.json").write_text(json.dumps(state))
    result = analyze_phase3i_short_trace(
        campaign_dir=campaign,
        protocol_dir=protocol_root,
        config_path=CONFIG,
        output_dir=tmp_path / "analysis",
    )
    assert result["decision"] == "representative_short_trace_replay_passed"
    assert result["paired_rows"] == "60"
    assert result["full_trace_protocol_freeze_authorized"] == "true"
    assert result["full_trace_replay_currently_authorized"] == "false"
