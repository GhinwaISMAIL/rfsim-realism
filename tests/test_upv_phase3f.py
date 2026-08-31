from __future__ import annotations

import json
from pathlib import Path

import yaml

from rfsim_realism.upv_phase3f import phase3f_decision, validate_phase3f_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3f_exchangeability_v1.yaml"
RESULT = ROOT / "manifests/upv_phase3f_exchangeability_v1/analysis_result.json"


def test_phase3f_config_is_diagnostic_only() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_phase3f_config(config)
    assert config["development"]["final_test6_access"] is False
    assert config["diagnostics"]["holdout_fitted_oracle"]["selectable_process"] is False
    assert config["reservation"]["request_now"] is False
    assert all(value == "prohibited" for value in config["claim_limits"].values())


def test_phase3f_classifies_cross_session_shift_first() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    result = phase3f_decision(
        exchangeable_folds=2,
        oracle_supported_folds=5,
        config=config,
    )
    assert result["decision_code"] == "cross_session_exchangeability_not_established"
    assert result["process_selection_authorized"] is False
    assert result["powder_reservation_authorized"] is False


def test_phase3f_classifies_unattainable_gate_after_exchangeability() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    result = phase3f_decision(
        exchangeable_folds=4,
        oracle_supported_folds=2,
        config=config,
    )
    assert result["decision_code"] == "current_joint_temporal_gate_not_attainable"


def test_phase3f_retains_model_failure_only_after_both_diagnostics_pass() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    result = phase3f_decision(
        exchangeable_folds=4,
        oracle_supported_folds=4,
        config=config,
    )
    assert result["decision_code"] == "training_process_model_still_inadequate"


def test_phase3f_recorded_result_selects_no_global_process() -> None:
    result = json.loads(RESULT.read_text())
    assert result["decision_code"] == "cross_session_exchangeability_not_established"
    assert result["exchangeable_folds"] == 3
    assert result["process_selection_authorized"] is False
    assert result["final_evaluation"]["payload_opened"] is False
    assert result["reservation"]["request_now"] is False
