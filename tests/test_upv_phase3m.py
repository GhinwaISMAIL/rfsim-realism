from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rfsim_realism.upv_phase3d import _read_yaml
from rfsim_realism.upv_phase3m import (
    _cross_validate,
    _fit_model,
    freeze_phase3m_sinr_dynamics,
    validate_phase3m_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3m_sinr_dynamics_v2_development_v1.yaml"
PHASE3J_RESULT = ROOT / "manifests/upv_phase3j_full_trace_result_v1"
PHASE3L_RESULT = ROOT / "manifests/upv_phase3l_test6_exploratory_result_v1"
PHASE3G_MEDIANS = ROOT / "manifests/upv_phase3g_response_v1/execution_medians.csv"
PHASE3G_CAMPAIGN = (
    ROOT
    / "data/model_runs/upv_phase3g_bounded_response_v1"
    / "phase3g-bounded-response-20260831T132110Z"
)
PHASE3J_ANALYZER = ROOT / "src/rfsim_realism/upv_phase3j.py"


def test_phase3m_config_keeps_version1_and_hardware_frozen() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3m_config(config)
    assert config["version1_artifacts_mutable"] is False
    assert config["hardware_execution_authorized"] is False
    assert config["inverse_command_generation_authorized"] is False
    assert config["development_data"]["clipped_row_forward_input"] == ("projected_feasible_sinr")
    assert config["models"]["observed_previous_output_as_predictor"] == "prohibited"


def _synthetic_executions() -> list[dict[str, object]]:
    executions: list[dict[str, object]] = []
    alpha, a, b = 0.4, 2.0, 0.8
    for execution_number in range(1, 5):
        time = np.arange(80)
        projected = 15.0 + 4.0 * np.sin((time + execution_number * 3) / 5.0)
        observed = np.empty_like(projected)
        observed[0] = a + b * projected[0]
        for index in range(1, len(projected)):
            observed[index] = alpha * observed[index - 1] + (1 - alpha) * (a + b * projected[index])
        frame = pd.DataFrame(
            {
                "command_index": time,
                "projected_sinr_db": projected,
                "target_sinr_db": projected,
                "ss_sinr_db": observed,
                "clipped": False,
            }
        )
        executions.append(
            {
                "execution_key": f"synthetic_{execution_number}",
                "source_phase": "synthetic",
                "trajectory": f"trajectory_{execution_number}",
                "execution_number": execution_number,
                "frame": frame,
            }
        )
    return executions


def test_phase3m_combined_model_recovers_recursive_dynamics() -> None:
    config = _read_yaml(CONFIG)
    executions = _synthetic_executions()
    combined = _fit_model("combined", executions, config)
    static = _fit_model("static", executions, config)
    assert combined["alpha"] == 0.4
    assert np.isclose(combined["a"], 2.0, atol=1e-9)
    assert np.isclose(combined["b"], 0.8, atol=1e-9)
    assert combined["training_mse"] < static["training_mse"]
    metrics, parameters, predictions = _cross_validate(executions, config)
    mean_mae = metrics.groupby("model")["mae_db"].mean()
    assert mean_mae["combined"] < mean_mae["static"]
    assert set(parameters["model"]) == {"static", "memory_only", "combined"}
    assert predictions["command_index"].min() == 1


def test_phase3m_freeze_records_non_destructive_erratum(tmp_path: Path) -> None:
    output = tmp_path / "phase3m"
    result = freeze_phase3m_sinr_dynamics(
        config_path=CONFIG,
        phase3j_result_dir=PHASE3J_RESULT,
        phase3l_result_dir=PHASE3L_RESULT,
        phase3g_execution_medians_path=PHASE3G_MEDIANS,
        phase3g_campaign_dir=PHASE3G_CAMPAIGN,
        phase3j_analyzer_path=PHASE3J_ANALYZER,
        output_dir=output,
    )
    erratum = json.loads((output / "version1_sign_convention_erratum.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    inventory = pd.read_csv(output / "phase3g_telemetry_inventory.csv")
    assert result["version1_outputs_changed"] == "false"
    assert result["hardware_execution_authorized"] == "false"
    assert erratum["implemented_and_canonical_convention"] == (
        "projected_feasible_target_minus_original_target"
    )
    assert erratum["absolute_metrics_affected"] is False
    assert protocol["inverse_command_generation_authorized"] is False
    assert len(inventory) == 45
