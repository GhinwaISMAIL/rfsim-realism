from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_phase3b import (
    _robust_scale,
    transfer_locked_regions,
    validate_phase3b_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_phase3b_config_freezes_nonabsolute_claims_and_closed_gate() -> None:
    config = yaml.safe_load(
        (REPOSITORY / "configs/upv_phase3b_support_v1.yaml").read_text()
    )

    validate_phase3b_config(config)

    assert config["selected_measurement_branch"] == "insufficient_metadata"
    assert config["kernel"]["estimator"] == "biased_mmd_squared_v_statistic"
    assert config["claim_limits"]["absolute_rsrp_calibration"] == "prohibited"
    assert config["claim_limits"]["absolute_noise_power_calibration"] == "prohibited"
    assert config["reservation"]["request_now"] is False
    assert config["balanced_reference"]["comparison_rows_per_distribution"] == 10


def _route(longitudes: list[float], directions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "longitude_deg": longitudes,
        "latitude_deg": [39.0] * len(longitudes),
        "direction_sector": directions,
    })


def test_region_transfer_requires_direction_and_distance() -> None:
    reference = _route([0.0, 0.0001], ["E", "W"])
    reference["locked_role"] = ["calibration", "spatial_validation_1"]
    target = _route([0.000001, 0.000099, 0.000001], ["E", "W", "W"])

    transferred, diagnostics = transfer_locked_regions(
        target,
        reference,
        roles=["calibration", "spatial_validation_1"],
        maximum_distance_m=2.0,
        minimum_rows=1,
        minimum_fraction=0.5,
    )

    assert transferred.iloc[0]["locked_role"] == "calibration"
    assert transferred.iloc[1]["locked_role"] == "spatial_validation_1"
    assert transferred.iloc[2]["locked_role"] == "outside_transfer_distance"
    assert diagnostics.set_index("locked_role").loc["calibration", "unit_valid"]


def test_invalid_transfer_unit_is_removed_fail_closed() -> None:
    reference = _route([0.0, 0.00001], ["E", "E"])
    reference["locked_role"] = ["calibration", "calibration"]
    target = _route([0.1], ["E"])

    transferred, diagnostics = transfer_locked_regions(
        target,
        reference,
        roles=["calibration"],
        maximum_distance_m=1.0,
        minimum_rows=1,
        minimum_fraction=1.0,
    )

    assert transferred.iloc[0]["locked_role"] == "invalid_transfer"
    assert not bool(diagnostics.iloc[0]["unit_valid"])


def test_phase3b_rejects_absolute_noise_claim() -> None:
    config = yaml.safe_load(
        (REPOSITORY / "configs/upv_phase3b_support_v1.yaml").read_text()
    )
    config["claim_limits"]["absolute_noise_power_calibration"] = "allowed"

    with pytest.raises(ValueError, match="prohibitions"):
        validate_phase3b_config(config)


def test_robust_scale_uses_population_standard_deviation_for_sparse_variation() -> None:
    center, scale = _robust_scale(
        pd.Series([0.0] * 16 + [-2.0, 2.0]).to_numpy(), 1.4826
    )

    assert center == 0.0
    assert scale > 0.0
