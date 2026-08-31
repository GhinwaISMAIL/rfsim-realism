from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3g import (
    _execution_plan,
    derive_provisional_controls,
    validate_phase3g_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/upv_phase3g_direct_trace_v1.yaml"


def test_phase3g_config_is_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_phase3g_config(config)
    assert config["target_trace"]["final_test6_access"] is False
    assert config["provisional_mapping"]["replay_authorized"] is False
    assert config["reservation"]["request_now"] is False
    assert config["reservation"]["preparation_lead_time_minutes"] >= 30
    assert config["hardware_design"]["channel_family"] == "AWGN"


def test_provisional_mapping_anchors_strongest_rsrp_at_zero_gain() -> None:
    frame = pd.DataFrame(
        {
            "relative_rsrp_db": [-5.0, 0.0, 3.0],
            "sinr_db": [10.0, 15.0, 20.0],
        }
    )
    mapped = derive_provisional_controls(
        frame,
        gain_response_slope=1.0,
        noise_intercept=1.0,
        noise_slope=-1.0,
        gain_to_sinr_coefficient=1.0,
    )
    assert mapped["provisional_gain_db"].tolist() == [-8.0, -3.0, 0.0]
    assert mapped.loc[2, "provisional_gain_db"] == 0.0
    assert np.isfinite(mapped["provisional_noise_power_db"]).all()


@pytest.mark.parametrize(
    ("gain_slope", "noise_slope"),
    [(0.0, -1.0), (-1.0, -1.0), (1.0, 0.0), (1.0, 1.0)],
)
def test_provisional_mapping_rejects_invalid_response_directions(
    gain_slope: float, noise_slope: float
) -> None:
    frame = pd.DataFrame({"relative_rsrp_db": [0.0], "sinr_db": [10.0]})
    with pytest.raises(ValueError, match="response directions"):
        derive_provisional_controls(
            frame,
            gain_response_slope=gain_slope,
            noise_intercept=0.0,
            noise_slope=noise_slope,
        )


def test_response_execution_plan_has_frozen_counts_and_unique_seeds() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    plan = _execution_plan(config)
    counts = Counter(row["stage"] for row in plan)
    assert len(plan) == 45
    assert counts == {
        "gain_safety": 4,
        "noise_safety": 2,
        "factorial": 27,
        "boundary": 12,
    }
    assert len({row["oai_rng_seed"] for row in plan}) == 45
    factorial = Counter(
        (row["gain_db"], row["noise_power_db"]) for row in plan if row["stage"] == "factorial"
    )
    assert len(factorial) == 9
    assert set(factorial.values()) == {3}
    boundary = Counter(
        (row["gain_db"], row["noise_power_db"]) for row in plan if row["stage"] == "boundary"
    )
    assert len(boundary) == 4
    assert set(boundary.values()) == {3}
