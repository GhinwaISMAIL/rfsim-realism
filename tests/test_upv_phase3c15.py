from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3c15 import (
    validate_phase3c15_config,
    write_phase3c15_support_analysis,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/upv_phase3c15_offline_support_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    roles = [
        "calibration",
        "spatial_validation_1",
        "spatial_validation_2",
        "spatial_validation_3",
        "spatial_validation_4",
    ]
    split_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    pattern = np.asarray([0, 1, 2, 1, 0, -1, -2, -1] * 2, dtype=float)
    for bin_id, role in enumerate(roles, start=1):
        split_rows.append({"bin_size_m": 15, "route_bin_id": bin_id, "locked_role": role})
        for second, value in enumerate(pattern):
            route_rows.append(
                {
                    "source_path": "Test_2/Test_2_ASUS.csv",
                    "seconds_of_day": 1000.0 + bin_id * 100 + second,
                    "route_bin_15m": bin_id,
                    "rsrp_dbm": -85.0 + value + 0.1 * bin_id,
                    "sinr_db": 16.0 + 1.5 * value + 0.2 * bin_id,
                }
            )
    route = tmp_path / "route.parquet"
    pd.DataFrame(route_rows).to_parquet(route, index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame(split_rows).to_csv(split, index=False)

    telemetry_rows: list[dict[str, object]] = []
    sequence = [0.0, -2.0, -4.0, -2.0, 0.0]
    for replay_number in range(1, 6):
        for second in range(50):
            gain = sequence[second // 10]
            telemetry_rows.append(
                {
                    "replay_id": f"awgn-{replay_number}",
                    "t_s": second + 0.5,
                    "commanded_gain_db": gain,
                    "applied_gain_db": gain,
                    "channel_family": "AWGN",
                    "noise_power_db": -30.0,
                    "rsrp_db_per_re_unquantized": 50.0 + gain + replay_number * 1e-4,
                    "ss_sinr_db": 45.0 + 0.2 * gain + (second % 3) * 0.1,
                    "attached": True,
                }
            )
    telemetry = tmp_path / "telemetry.csv"
    pd.DataFrame(telemetry_rows).to_csv(telemetry, index=False)

    phase3c14_evaluation = tmp_path / "phase3c14_evaluation.json"
    _json(
        phase3c14_evaluation,
        {"decision_code": "awgn_execution_control_accepted", "control_gate_pass": True},
    )
    phase3c14_result = tmp_path / "phase3c14_result.json"
    _json(phase3c14_result, {"decision": "awgn_execution_control_accepted"})
    phase3b_decision = tmp_path / "phase3b_decision.json"
    _json(
        phase3b_decision,
        {
            "decision_code": "relative_rsrp_shape_mismatch",
            "abc_performed": False,
            "metric_support": {
                "SINR": {
                    "best_candidate_id": "ploss=0|noise=-10",
                    "primary_calibration_supported": True,
                    "locked_validation_regions_supported": 2,
                    "locked_validation_regions_evaluated": 4,
                }
            },
        },
    )
    diagnostics = tmp_path / "diagnostics.csv"
    diagnostics.write_text("metric,candidate_id\nSINR,ploss=0|noise=-10\n")
    validation = tmp_path / "validation.csv"
    validation.write_text("metric,supported\nSINR,True\n")
    return {
        "route_observations": route,
        "locked_spatial_split": split,
        "phase3c14_telemetry": telemetry,
        "phase3c14_evaluation": phase3c14_evaluation,
        "phase3c14_result": phase3c14_result,
        "phase3b_decision": phase3b_decision,
        "phase3b_distribution_diagnostics": diagnostics,
        "phase3b_locked_validation_support": validation,
    }


def _config_for_inputs(tmp_path: Path, inputs: dict[str, Path]) -> Path:
    config = yaml.safe_load(CONFIG.read_text())
    config["conditional_reference"]["repetitions"] = 100
    checksum_keys = {
        "route_observations": "route_observations_sha256",
        "locked_spatial_split": "locked_spatial_split_sha256",
        "phase3c14_telemetry": "phase3c14_telemetry_sha256",
        "phase3c14_evaluation": "phase3c14_evaluation_sha256",
        "phase3c14_result": "phase3c14_result_sha256",
        "phase3b_decision": "phase3b_decision_sha256",
        "phase3b_distribution_diagnostics": "phase3b_distribution_diagnostics_sha256",
        "phase3b_locked_validation_support": "phase3b_locked_validation_support_sha256",
    }
    for name, key in checksum_keys.items():
        config["frozen_inputs"][key] = _sha256(inputs[name])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_phase3c15_config_is_offline_and_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_phase3c15_config(config)
    assert config["execution_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["reservation"]["request_now"] is False
    assert config["reservation"]["preparation_lead_time_minutes"] == 30
    assert config["preprocessing"]["fitted_from"] == "UPV_calibration_trace_only"
    assert (
        config["support_rules"][
            "control_envelope_failure_does_not_prove_AWGN_model_class_impossibility"
        ]
        is True
    )


def test_phase3c15_rejects_holdout_leakage() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["preprocessing"]["holdouts_excluded_from_all_fit_and_threshold_steps"] = False
    with pytest.raises(ValueError, match=r"Holdouts|holdouts"):
        validate_phase3c15_config(config)


def test_phase3c15_rejects_gain_only_when_sinr_is_disjoint(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = _config_for_inputs(tmp_path, inputs)
    output = tmp_path / "output"
    result = write_phase3c15_support_analysis(
        **inputs,
        config_path=config,
        output_dir=output,
    )
    assert result["decision_code"] == "gain_only_rejected_noise_control_required"
    assert result["abc_authorized"] is False
    assert result["reservation_requested"] is False
    decision = json.loads((output / "phase3c15_decision.json").read_text())
    assert decision["noise_control_dimension_required"] is True
    assert decision["control_envelope_failure_proves_AWGN_impossibility"] is False
    support = pd.read_csv(output / "support_results.csv")
    assert support.loc[
        support["trace_role"].eq("calibration"), "sinr_distribution_pass"
    ].item() in (
        False,
        np.bool_(False),
    )
    assert set(support.loc[support["split_role"].eq("holdout"), "trace_role"]) == {
        "spatial_validation_1",
        "spatial_validation_2",
        "spatial_validation_3",
        "spatial_validation_4",
    }
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        assert _sha256(output / name) == expected


def test_phase3c15_refuses_checksum_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = _config_for_inputs(tmp_path, inputs)
    inputs["phase3c14_telemetry"].write_text("changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_phase3c15_support_analysis(
            **inputs,
            config_path=config,
            output_dir=tmp_path / "output",
        )
