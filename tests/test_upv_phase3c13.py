from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3c13 import (
    evaluate_static_tdlb_pilot,
    validate_phase3c13_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/upv_phase3c13_static_tdlb_pilot_v1.yaml"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _inputs(tmp_path: Path, *, duplicate_fingerprint: bool = False) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    sequence = [0.0, -2.0, -4.0, -2.0, 0.0]
    for replay_number, seed in enumerate(range(31001, 31006), start=1):
        replay_id = f"tdlb-{replay_number}"
        fingerprint = "0000000000000001" if duplicate_fingerprint else f"{replay_number:016x}"
        run_shift = 0.2 * (replay_number - 1)
        for second in range(50):
            gain = sequence[second // 10]
            float_rsrp = -40.0 + run_shift + gain
            rows.append(
                {
                    "replay_id": replay_id,
                    "oai_rng_seed": seed,
                    "t_s": float(second),
                    "commanded_gain_db": gain,
                    "applied_gain_db": gain,
                    "channel_family": "TDL_B",
                    "channel_model_name": "rfsimu_channel_enB0",
                    "channel_snapshot_id": "static-0",
                    "channel_snapshot_timestamp_ns": seed,
                    "tap_energy_linear": 1.0 + replay_number / 10,
                    "tap_fingerprint_fnv1a64": fingerprint,
                    "channel_length": 7,
                    "nb_taps": 23,
                    "nb_tx": 1,
                    "nb_rx": 1,
                    "noise_power_db": -30.0,
                    "rsrp_digital_power_linear": 10 ** (float_rsrp / 10),
                    "rsrp_db_per_re_unquantized": float_rsrp,
                    "ss_rsrp_dbm_integer": round(float_rsrp),
                    "ss_sinr_db": 20.0 + run_shift + gain,
                    "attached": True,
                }
            )
        summaries.append(
            {
                "replay_id": replay_id,
                "oai_rng_seed": seed,
                "tap_fingerprint_fnv1a64": fingerprint,
                "continuous_attachment": True,
                "operational_runtime_pass": True,
                "critical_failure_count": 0,
                "ue_restart_count": 0,
                "gnb_health": "healthy",
            }
        )
    telemetry = tmp_path / "telemetry.csv"
    pd.DataFrame(rows).to_csv(telemetry, index=False)
    state = tmp_path / "state.json"
    config = _config()
    state.write_text(
        json.dumps(
            {
                "stage": "phase_3c13_static_tdlb_scalar_pilot_execution",
                "execution_completed": True,
                "error": None,
                "oai_revision": config["frozen_inputs"]["oai_revision"],
                "profile_revision": config["frozen_inputs"]["profile_revision"],
                "runner_sha256": config["frozen_inputs"]["profile_runner_sha256"],
                "channel_family": "TDL_B",
                "tdl_rms_delay_spread_ns": 30,
                "rng_seeds": list(range(31001, 31006)),
                "distinct_fingerprint_count": 1 if duplicate_fingerprint else 5,
                "gNB_untouched": True,
                "replays": summaries,
                "rollback": {"passed": True},
            }
        )
    )
    return telemetry, state


def test_phase3c13_config_is_scoped_and_fail_closed() -> None:
    config = _config()
    validate_phase3c13_config(config)
    assert config["execution_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["reservation"]["reservation_should_be_requested_now"] is True
    assert config["reservation"]["gate_state"] == "open_for_build_and_smoke_only"
    assert config["reservation"]["pilot_execution_authorized"] is False
    assert config["design"]["control"].startswith("phase3c1_static_awgn")
    assert config["claim_limits"]["fully_time_varying_tdl_b"] == "prohibited"


def test_phase3c13_accepts_seeded_static_tdlb_transfer(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path)
    result = evaluate_static_tdlb_pilot(
        telemetry_path=telemetry,
        execution_state_path=state,
        config_path=CONFIG,
    )
    assert result["pilot_gate_pass"] is True
    assert result["decision_code"] == "static_tdlb_scalar_candidate_accepted"
    assert result["cross_execution"]["baseline_rsrp_range_db"] == pytest.approx(0.8)
    assert all(row["replay_pass"] for row in result["replay_results"])
    assert result["abc_authorized"] is False


def test_phase3c13_rejects_duplicate_realization_fingerprints(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path, duplicate_fingerprint=True)
    result = evaluate_static_tdlb_pilot(
        telemetry_path=telemetry,
        execution_state_path=state,
        config_path=CONFIG,
    )
    assert result["pilot_gate_pass"] is False
    assert result["cross_execution"]["gate_results"]["five_distinct_fingerprints"] is False
    assert result["decision_code"] == "static_tdlb_realization_gate_failed"


def test_phase3c13_rejects_between_execution_rsrp_instability(
    tmp_path: Path,
) -> None:
    telemetry, state = _inputs(tmp_path)
    frame = pd.read_csv(telemetry, dtype={"tap_fingerprint_fnv1a64": str})
    frame.loc[frame["replay_id"].eq("tdlb-5"), "rsrp_db_per_re_unquantized"] += 4.0
    frame.to_csv(telemetry, index=False)
    result = evaluate_static_tdlb_pilot(
        telemetry_path=telemetry,
        execution_state_path=state,
        config_path=CONFIG,
    )
    assert result["pilot_gate_pass"] is False
    assert result["cross_execution"]["gate_results"]["baseline_rsrp_range"] is False
    assert result["decision_code"] == "static_tdlb_execution_variance_too_large"


def test_phase3c13_validates_corrected_runtime_identity(tmp_path: Path) -> None:
    telemetry, state_path = _inputs(tmp_path)
    state = json.loads(state_path.read_text())
    state.update(
        {
            "profile_revision": "ca4e78b0f2fe0630c3a77c3f2e7506fa38f8206b",
            "runner_sha256": "d1fb090550299d9f3b9e4a51593af636422975c0c03fd08bf6955c6a6d498f8e",
            "debug_image_id": (
                "sha256:d6b87ce2e446f8750727121a2463dfb06eee747b599492bdb78afc36c9dcc664"
            ),
        }
    )
    state_path.write_text(json.dumps(state))
    amendment = tmp_path / "identity_amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "phase_3c13_fixed_tap_scalar_control_correction",
                "decision": "corrected_five_execution_pilot_authorized",
                "stopping_rules_unchanged": True,
                "claim_limits_unchanged": True,
                "rerun_identity": {
                    "profile_revision": "ca4e78b0f2fe0630c3a77c3f2e7506fa38f8206b",
                    "runner_sha256": (
                        "d1fb090550299d9f3b9e4a51593af636422975c0c03fd08bf6955c6a6d498f8e"
                    ),
                    "debug_image_id": (
                        "sha256:d6b87ce2e446f8750727121a2463dfb06eee747b599492bdb78afc36c9dcc664"
                    ),
                },
            }
        )
    )

    result = evaluate_static_tdlb_pilot(
        telemetry_path=telemetry,
        execution_state_path=state_path,
        config_path=CONFIG,
        identity_amendment_path=amendment,
    )

    assert result["pilot_gate_pass"] is True
    assert result["state_gate_results"]["profile_revision"] is True
    assert result["state_gate_results"]["runner_sha256"] is True
    assert result["state_gate_results"]["debug_image_id"] is True
    assert result["input_sha256"]["identity_amendment"]
