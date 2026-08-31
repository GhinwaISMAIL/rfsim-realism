from __future__ import annotations

from pathlib import Path

from rfsim_realism.upv_phase3d import _read_yaml
from rfsim_realism.upv_phase3h import _build_plan, validate_phase3h_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "upv_phase3h_dynamic_staircase_v1.yaml"


def test_phase3h_config_freezes_three_dynamic_sequence_units() -> None:
    config = _read_yaml(CONFIG)
    validate_phase3h_config(config)
    plan = _build_plan(config)
    assert len(plan) == 27
    assert plan["sequence_index"].nunique() == 3
    assert (plan["segment_type"] == "validation").sum() == 21
    assert (plan["segment_type"] == "anchor_start").sum() == 3
    assert (plan["segment_type"] == "anchor_end").sum() == 3
    assert plan.loc[plan["segment_type"] == "validation", "state_id"].value_counts().eq(3).all()


def test_phase3h_plan_counterbalances_positions_and_uses_unique_seeds() -> None:
    config = _read_yaml(CONFIG)
    plan = _build_plan(config)
    validation = plan.loc[plan["segment_type"] == "validation"]
    assert validation.groupby("state_id")["position"].nunique().eq(3).all()
    assert plan.groupby("sequence_index")["oai_rng_seed"].first().nunique() == 3
    assert (
        config["statistical_design"]["individual_radio_samples_are_independent_repetitions"]
        is False
    )
    assert config["reservation"]["request_now"] is False
