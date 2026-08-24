from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3c import (
    deterministic_envelope,
    evaluate_deterministic_replay,
    validate_phase3c_config,
    write_deterministic_replay_evaluation,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_phase3c0_protocol_v1.yaml").read_text()
    )


def test_phase3c_config_is_offline_fail_closed_and_scoped() -> None:
    config = _config()

    validate_phase3c_config(config)

    assert config["execution_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["reservation"]["request_now"] is False
    assert config["reservation"]["preparation_lead_time_minutes"] == 30
    assert config["external_generator_status"]["state"].startswith("provisional_")
    assert config["model_scope"]["prohibited_conclusion"].startswith("all_RFsim")


def test_phase3c_rejects_universal_rfsim_claim() -> None:
    config = _config()
    config["model_scope"]["prohibited_conclusion"] = "none"

    with pytest.raises(ValueError, match="generalize"):
        validate_phase3c_config(config)


def test_deterministic_envelope_freezes_times_and_linear_semantics() -> None:
    frame = deterministic_envelope(_config())

    assert frame["commanded_gain_db"].tolist() == [0.0, -2.0, -4.0, -2.0, 0.0]
    assert frame["start_s"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert frame["analysis_start_s"].tolist() == [3.0, 13.0, 23.0, 33.0, 43.0]
    assert frame["end_s"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert frame.iloc[1]["expected_amplitude_multiplier"] == pytest.approx(
        10 ** (-2 / 20)
    )
    assert frame.iloc[2]["expected_power_ratio"] == pytest.approx(10 ** (-4 / 10))


def _write_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan"
    plan.mkdir()
    deterministic_envelope(_config()).to_csv(
        plan / "deterministic_scalar_envelope.csv", index=False
    )
    (plan / "deterministic_replay_acceptance.json").write_text(
        json.dumps(_config()["deterministic_replay_acceptance"])
    )
    return plan


def _telemetry(*, constant_rsrp: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sequence = [0.0, -2.0, -4.0, -2.0, 0.0]
    for replay_id in ["local-1", "local-2"]:
        for second in range(50):
            gain = sequence[second // 10]
            float_rsrp = -40.0 if constant_rsrp else -40.0 + gain
            rows.append({
                "replay_id": replay_id,
                "t_s": float(second),
                "commanded_gain_db": gain,
                "applied_gain_db": gain,
                "channel_snapshot_id": "static-snapshot-0",
                "channel_snapshot_timestamp_ns": 0,
                "tap_energy_linear": 1.0,
                "noise_power_db": -30.0,
                "rsrp_digital_power_linear": 10 ** (float_rsrp / 10),
                "rsrp_db_per_re_unquantized": float_rsrp,
                "ss_rsrp_dbm_integer": int(np.rint(float_rsrp)),
                "ss_sinr_db": 30.0 + gain,
                "attached": True,
            })
    return pd.DataFrame(rows)


def test_deterministic_replay_evaluator_accepts_exact_transfer(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    telemetry = tmp_path / "telemetry.csv"
    _telemetry().to_csv(telemetry, index=False)

    result = evaluate_deterministic_replay(
        telemetry_path=telemetry,
        plan_dir=plan,
    )

    assert result["replays_evaluated"] == 2
    assert result["deterministic_replay_gate_pass"] is True
    assert result["reservation_should_be_requested_now"] is False
    for replay in result["replay_results"]:
        assert replay["float_rsrp_transfer_slope"] == pytest.approx(1.0)
        assert replay["replay_pass"] is True


def test_deterministic_replay_writer_serializes_gate_results(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    telemetry = tmp_path / "telemetry.csv"
    output = tmp_path / "evaluation.json"
    _telemetry().to_csv(telemetry, index=False)

    write_deterministic_replay_evaluation(
        telemetry_path=telemetry,
        plan_dir=plan,
        output_path=output,
    )

    result = json.loads(output.read_text())
    assert result["deterministic_replay_gate_pass"] is True
    assert all(
        isinstance(value, bool)
        for replay in result["replay_results"]
        for value in replay["gate_results"].values()
    )


def test_deterministic_replay_evaluator_rejects_constant_rsrp(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    telemetry = tmp_path / "constant.csv"
    _telemetry(constant_rsrp=True).to_csv(telemetry, index=False)

    result = evaluate_deterministic_replay(
        telemetry_path=telemetry,
        plan_dir=plan,
    )

    assert result["deterministic_replay_gate_pass"] is False
    assert all(not replay["gate_results"]["float_slope"] for replay in result[
        "replay_results"
    ])
