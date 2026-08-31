from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3i import (
    REJECTED_RUNNER_SHA256,
    _support_bank,
    analyze_phase3i_short_trace,
    freeze_phase3i_short_trace,
    recover_phase3i_short_trace,
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


def test_phase3i_recovery_uses_next_channel_second_without_hardware_rerun(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "rejected"
    campaign.mkdir()
    start = 1_700_000_000
    boot_epoch = start - 100.0
    events = []
    ue_lines = ["== Starting NR UE soft modem"]
    channel_lines = []

    def channel_line(second: int, gain: float, noise: float) -> str:
        monotonic = second - boot_epoch + 0.001
        return (
            f"{monotonic:.6f} [HW] I RFSIM_CHANNEL_DEBUG_V1 "
            f"utc_second={second} emitted_epoch_us={second * 1_000_000 + 1000} "
            "model=rfsimu_channel_enB0 channel_snapshot_id=static-0 "
            f"channel_snapshot_timestamp_ns={second * 1_000_000_000} "
            "tap_energy_linear=1 tap_fingerprint_fnv1a64=abc channel_length=1 "
            "nb_taps=1 nb_tx=1 nb_rx=1 oai_rng_seed=46001 "
            f"applied_gain_db={gain:.9f} noise_power_db={noise:.9f}"
        )

    def ue_line(second: int) -> str:
        monotonic = second - boot_epoch + 1.05
        return (
            f"{monotonic:.6f} [PHY] I UE_RADIO_DEBUG_V1 utc_second={second} "
            f"emitted_epoch_us={(second + 1) * 1_000_000 + 50_000} "
            "rsrp_digital_power_linear=10000 rsrp_db_per_re_unquantized=40 "
            "ss_rsrp_dbm_integer=-107 ss_sinr_db=15"
        )

    for second in range(start - 10, start + 1):
        channel_lines.append(channel_line(second, -10.0, -25.0))
        if second < start:
            ue_lines.append(ue_line(second))
    for index in range(60):
        second = start + index
        gain = -8.0 - index / 100.0
        noise = -20.0 - index / 100.0
        events.append(
            {
                "command_index": index,
                "trace_row_index": 154 + index,
                "trace_time_bin": 154 + index,
                "trace_t_s": 154.0 + index,
                "target_relative_rsrp_db": 0.0,
                "target_sinr_db": 15.0,
                "projected_relative_rsrp_db": 0.0,
                "projected_sinr_db": 15.0,
                "commanded_gain_db": gain,
                "commanded_noise_power_db": noise,
                "clipped": False,
                "scheduled_epoch": float(second),
                "command_complete_epoch": second + 0.1,
                "command_completion_lateness_seconds": 0.1,
                "sample_utc_second": second,
            }
        )
        ue_lines.append(ue_line(second))
        channel_lines.append(channel_line(second + 1, gain, noise))
    for second in range(start + 70, start + 80):
        channel_lines.append(channel_line(second, -10.0, -25.0))
        ue_lines.append(ue_line(second))
    (campaign / "phase3i-ue.log").write_text(
        "\n".join(ue_lines + channel_lines) + "\n"
    )
    (campaign / "phase3i-gnb.log").write_text("")
    (campaign / "phase3i-command-events.json").write_text(json.dumps(events))
    windows = [
        {
            "usable_start_epoch": start - 10.0,
            "usable_end_epoch": float(start),
            "attachment_checks": [{"attached": True}],
        },
        {
            "usable_start_epoch": start + 70.0,
            "usable_end_epoch": start + 80.0,
            "attachment_checks": [{"attached": True}],
        },
    ]
    (campaign / "phase3i-anchor-windows.json").write_text(json.dumps(windows))
    (campaign / "phase3i-ping-checks.json").write_text(
        json.dumps([{"passed": True}])
    )
    state = {
        "execution_completed": False,
        "error": "applied gain mismatch at command 0",
        "runner_sha256": REJECTED_RUNNER_SHA256,
        "research_protocol_sha256": _sha256(CONFIG),
        "rollback": {
            "passed": True,
            "gnb_restart_count_before": 0,
            "gnb_restart_count_after": 0,
        },
        "commands_sha256": "frozen-commands",
        "gNB_untouched": True,
        "full_trace_replay_authorized": False,
        "final_test6_accessed": False,
    }
    (campaign / "execution_state.json").write_text(json.dumps(state))
    output = tmp_path / "recovered"
    result = recover_phase3i_short_trace(
        campaign_dir=campaign,
        config_path=CONFIG,
        output_dir=output,
    )
    telemetry = pd.read_csv(output / "phase3i_short_trace_telemetry.csv")
    recovered_state = json.loads((output / "execution_state.json").read_text())
    assert result["paired_rows"] == "60"
    assert result["hardware_rerun_used"] == "false"
    assert (
        telemetry["channel_verification_utc_second"]
        == telemetry["sample_utc_second"] + 1
    ).all()
    assert recovered_state["execution_completed"] is True
    assert recovered_state["primary_kpi_alignment_seconds"] == 0
