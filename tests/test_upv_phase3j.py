from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfsim_realism.upv_phase3d import _read_yaml, _sha256
from rfsim_realism.upv_phase3j import (
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
    }
    assert config["runtime_gates"]["missing_row_interpolation"] == "prohibited"
    assert config["clipping_evaluation"]["bridge_across_clipped_rows"] == "prohibited"
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
