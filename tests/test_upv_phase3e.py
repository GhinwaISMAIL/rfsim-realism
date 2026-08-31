from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import rfsim_realism.upv_phase3e as phase3e
from rfsim_realism.upv_phase3e import (
    _decision,
    _fit_switching_var1,
    _fit_var1,
    _sample_switching_var1,
    _sample_var1,
    validate_phase3e_config,
    write_phase3e_protocol_freeze,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3e_radio_process_v1.yaml"
PHASE3D_CONFIG = ROOT / "configs/upv_phase3d_radio_process_v1.yaml"
RESULT = ROOT / "manifests/upv_phase3e_analysis_v1/analysis_result.json"
CANDIDATES = [
    "paired_block_10",
    "paired_block_20",
    "paired_block_40",
    "paired_block_60",
    "empirical_innovation_var1",
    "student_t_2state_switching_var1",
]


def test_phase3e_config_is_offline_and_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_phase3e_config(config)
    assert config["development"]["final_test6_access"] is False
    assert config["reservation"]["request_now"] is False
    assert config["decision_rules"]["complexity_order"] == CANDIDATES
    assert all(value == "prohibited" for value in config["claim_limits"].values())


def test_phase3e_empirical_innovation_var_is_stable_and_reproducible() -> None:
    rng = np.random.default_rng(9)
    sequences = []
    matrix = np.asarray([[0.8, 0.1], [-0.05, 0.7]])
    for _ in range(4):
        values = np.zeros((120, 2), dtype=float)
        values[0] = rng.normal(size=2)
        for index in range(1, len(values)):
            values[index] = matrix @ values[index - 1] + rng.normal(scale=0.2, size=2)
        sequences.append(values)
    config = yaml.safe_load(CONFIG.read_text())
    model = _fit_var1(sequences, config)
    assert model["eligible"] is True
    assert model["spectral_radius"] <= 0.995
    first = _sample_var1(model, [80, 40], np.random.default_rng(100))
    second = _sample_var1(model, [80, 40], np.random.default_rng(100))
    assert [len(value) for value in first] == [80, 40]
    assert np.array_equal(first[0], second[0])
    assert np.isfinite(np.vstack(first)).all()


def test_phase3e_switching_var_generates_finite_paired_sequences() -> None:
    rng = np.random.default_rng(11)
    sequences = [np.cumsum(rng.normal(size=(80, 2)), axis=0) / 10 for _ in range(4)]
    config = yaml.safe_load(CONFIG.read_text())
    phase3d_config = yaml.safe_load(PHASE3D_CONFIG.read_text())
    model = _fit_switching_var1(sequences, phase3d_config, config, seed=17)
    generated = _sample_switching_var1(model, [50, 30], np.random.default_rng(3))
    assert model["eligible"] is True
    assert [len(value) for value in generated] == [50, 30]
    assert np.isfinite(np.vstack(generated)).all()


def _decision_frame(passing_candidate: str | None) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for candidate in CANDIDATES:
            passes = candidate == passing_candidate and fold < 4
            rows.append(
                {
                    "fold_id": f"fold-{fold}",
                    "candidate_id": candidate,
                    "model_eligible": True,
                    "joint_reference_p90": 1.0,
                    "temporal_reference_p90": 1.0,
                    "joint_mmd_squared_mean": 0.5 if passes else 1.5,
                    "temporal_error_mean": 0.5 if passes else 1.5,
                }
            )
    return pd.DataFrame(rows)


def test_phase3e_decision_selects_only_four_fold_joint_temporal_support() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    result = _decision(_decision_frame("paired_block_40"), config)
    assert result["decision_code"] == "development_radio_process_supported"
    assert result["selected_process"] == "paired_block_40"
    selected = next(
        value for value in result["candidate_support"] if value["candidate_id"] == "paired_block_40"
    )
    assert selected["joint_and_temporal_supported_folds"] == 4
    assert result["powder_reservation_authorized"] is False


def test_phase3e_decision_rejects_all_unsupported_candidates() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    result = _decision(_decision_frame(None), config)
    assert result["decision_code"] == "temporal_process_revision_still_unsupported"
    assert result["selected_process"] is None


def test_phase3e_protocol_freeze_records_clean_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        phase3e,
        "_git_revision",
        lambda: {"revision": "frozen", "tracked_worktree_dirty": False},
    )
    output = tmp_path / "protocol"
    paths = write_phase3e_protocol_freeze(
        config_path=CONFIG,
        phase3d_config_path=PHASE3D_CONFIG,
        output_dir=output,
    )
    freeze = json.loads(Path(paths["protocol_freeze"]).read_text())
    assert freeze["repository"]["revision"] == "frozen"
    assert freeze["final_test6_access"] is False
    assert freeze["candidate_ids"] == CANDIDATES
    with pytest.raises(FileExistsError):
        write_phase3e_protocol_freeze(
            config_path=CONFIG,
            phase3d_config_path=PHASE3D_CONFIG,
            output_dir=output,
        )


def test_phase3e_recorded_result_keeps_hardware_and_final_gates_closed() -> None:
    result = json.loads(RESULT.read_text())
    assert result["decision_code"] == "temporal_process_revision_still_unsupported"
    assert result["selected_process"] is None
    assert result["diagnosis"]["model_selection_authorized"] is False
    assert result["final_evaluation"]["payload_opened"] is False
    assert result["reservation"]["request_now"] is False
    assert not any(result["authorizations"].values())
    checksums = json.loads((RESULT.parent / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == checksums[RESULT.name]
