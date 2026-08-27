from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3c14 import (
    evaluate_awgn_execution_control,
    validate_phase3c14_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/upv_phase3c14_awgn_control_v1.yaml"
TDLB_EVALUATION = (
    REPOSITORY
    / "data/model_runs/upv_phase3c13_static_tdlb_v1/static_tdlb_pilot_evaluation_amended.json"
)
TDLB_RESULT = REPOSITORY / "manifests/upv_phase3c13_static_tdlb_pilot_v1/execution_result.json"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _inputs(
    tmp_path: Path,
    *,
    duplicate_identity: bool = True,
) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    sequence = [0.0, -2.0, -4.0, -2.0, 0.0]
    for replay_number, seed in enumerate(range(32001, 32006), start=1):
        replay_id = f"awgn-{replay_number}"
        fingerprint = "0000000000000001" if duplicate_identity else f"{replay_number:016x}"
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
                    "channel_family": "AWGN",
                    "channel_model_name": "rfsimu_channel_enB0",
                    "channel_snapshot_id": "static-0",
                    "channel_snapshot_timestamp_ns": seed,
                    "tap_energy_linear": 1.0,
                    "tap_fingerprint_fnv1a64": fingerprint,
                    "channel_length": 1,
                    "nb_taps": 1,
                    "nb_tx": 1,
                    "nb_rx": 1,
                    "noise_power_db": -30.0,
                    "rsrp_digital_power_linear": 10 ** (float_rsrp / 10),
                    "rsrp_db_per_re_unquantized": float_rsrp,
                    "ss_rsrp_dbm_integer": round(float_rsrp),
                    "ss_sinr_db": 20.0 + run_shift / 2 + gain,
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
    config = _config()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "stage": "phase_3c14_awgn_execution_control",
                "execution_completed": True,
                "error": None,
                "oai_revision": config["frozen_inputs"]["oai_revision"],
                "profile_revision": config["frozen_inputs"]["profile_revision"],
                "runner_sha256": config["frozen_inputs"]["profile_runner_sha256"],
                "channel_family": "AWGN",
                "rng_seeds": list(range(32001, 32006)),
                "debug_image_id": f"sha256:{'a' * 64}",
                "debug_image_revision_label": config["frozen_inputs"]["oai_revision"],
                "gNB_untouched": True,
                "replays": summaries,
                "rollback": {"passed": True},
            }
        )
    )
    return telemetry, state


def _evaluate(telemetry: Path, state: Path) -> dict[str, object]:
    return evaluate_awgn_execution_control(
        telemetry_path=telemetry,
        execution_state_path=state,
        config_path=CONFIG,
        tdlb_evaluation_path=TDLB_EVALUATION,
        tdlb_result_path=TDLB_RESULT,
    )


def test_phase3c14_config_is_scoped_and_fail_closed() -> None:
    config = _config()
    validate_phase3c14_config(config)
    assert config["execution_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["reservation"]["reservation_should_be_requested_now"] is False
    assert config["reservation"]["preparation_lead_time_minutes"] == 30
    assert config["control"]["channel_family"] == "AWGN"
    assert config["design"]["static_TDL_B_status"] == "sensitivity_analysis_only"


def test_phase3c14_accepts_stable_five_execution_awgn_control(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path)
    result = _evaluate(telemetry, state)

    assert result["control_gate_pass"] is True
    assert result["decision_code"] == "awgn_execution_control_accepted"
    assert result["cross_execution"]["baseline_rsrp_range_db"] == pytest.approx(0.8)
    assert result["cross_execution"]["baseline_sinr_range_db"] == pytest.approx(0.4)
    assert result["static_tdlb_comparison"]["awgn_to_tdlb_baseline_rsrp_range_ratio"] < 1
    assert all(row["replay_pass"] for row in result["replay_results"])
    assert result["abc_authorized"] is False


def test_phase3c14_rejects_nonidentity_fingerprints(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path, duplicate_identity=False)
    result = _evaluate(telemetry, state)

    assert result["control_gate_pass"] is False
    assert (
        result["cross_execution"]["identity_gate_results"]["one_common_awgn_fingerprint"] is False
    )
    assert result["decision_code"] == "awgn_identity_gate_failed"


def test_phase3c14_rejects_between_execution_rsrp_instability(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path)
    frame = pd.read_csv(telemetry, dtype={"tap_fingerprint_fnv1a64": str})
    frame.loc[frame["replay_id"].eq("awgn-5"), "rsrp_db_per_re_unquantized"] += 4.0
    frame.to_csv(telemetry, index=False)
    result = _evaluate(telemetry, state)

    assert result["control_gate_pass"] is False
    assert result["cross_execution"]["stability_gate_results"]["baseline_rsrp_range"] is False
    assert result["decision_code"] == "awgn_execution_variance_too_large"


def test_phase3c14_rejects_wrong_image_revision_label(tmp_path: Path) -> None:
    telemetry, state_path = _inputs(tmp_path)
    state = json.loads(state_path.read_text())
    state["debug_image_revision_label"] = "wrong"
    state_path.write_text(json.dumps(state))
    result = _evaluate(telemetry, state_path)

    assert result["control_gate_pass"] is False
    assert result["state_gate_results"]["image_revision_label"] is False
    assert result["decision_code"] == "awgn_identity_gate_failed"


def test_phase3c14_rejects_unfrozen_tdlb_evidence(tmp_path: Path) -> None:
    telemetry, state = _inputs(tmp_path)
    altered = tmp_path / "tdlb.json"
    altered.write_text(TDLB_EVALUATION.read_text() + "\n")

    with pytest.raises(ValueError, match="evaluation checksum"):
        evaluate_awgn_execution_control(
            telemetry_path=telemetry,
            execution_state_path=state,
            config_path=CONFIG,
            tdlb_evaluation_path=altered,
            tdlb_result_path=TDLB_RESULT,
        )
