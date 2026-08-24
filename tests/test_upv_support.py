from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rfsim_realism.upv_support import (
    _sensitivity_tables,
    _support_tables,
    initial_positive_sequence_ess,
    validate_upv_support_config,
    validation_separation,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict:
    value = yaml.safe_load((REPOSITORY / "configs/upv_support_v1.yaml").read_text())
    value["bootstrap"]["repetitions"] = 20
    value["sensitivity"]["bootstrap_repetitions"] = 50
    return value


def test_frozen_config_declares_phase1_revision_and_prohibits_abc() -> None:
    config = _config()

    validate_upv_support_config(config)

    assert config["phase1_protocol"]["revision"] == (
        "242181ad8d2f0f3d6ef1e88c18fbabab6ff3b0d1"
    )
    assert config["decision_rules"]["abc_is_prohibited_for_existing_bank"] is True
    assert [item["name"] for item in config["features"]] == ["RSRP", "SINR"]


def test_initial_positive_sequence_ess_is_bounded_by_sample_count() -> None:
    independent = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 1.2, -1.2])
    correlated = np.repeat(np.arange(6, dtype=float), 3)

    independent_ess, _, _ = initial_positive_sequence_ess(independent)
    correlated_ess, _, _ = initial_positive_sequence_ess(correlated)

    assert 1 <= independent_ess <= len(independent)
    assert 1 <= correlated_ess < len(correlated)


def test_validation_separation_reports_route_and_euclidean_distance() -> None:
    route = pd.DataFrame({
        "route_bin_15m": [0, 0, 1, 1],
        "route_distance_m": [0.0, 5.0, 100.0, 105.0],
        "longitude_deg": [0.0, 0.00001, 0.00002, 0.00003],
        "latitude_deg": [39.0, 39.0, 39.0, 39.0],
    })
    split = pd.DataFrame([
        {
            "bin_size_m": 15,
            "route_bin_id": 0,
            "route_center_m": 2.5,
            "locked_role": "calibration",
        },
        {
            "bin_size_m": 15,
            "route_bin_id": 1,
            "route_center_m": 102.5,
            "locked_role": "spatial_validation_1",
        },
    ])

    result = validation_separation(
        route,
        split,
        primary_bin_size_m=15,
        calibration_role="calibration",
        validation_prefix="spatial_validation_",
        adjacency_threshold_m=5.0,
    ).iloc[0]

    assert result["minimum_along_route_gap_m"] == pytest.approx(95.0)
    assert result["minimum_euclidean_separation_m"] < 5.0
    assert bool(result["geographically_adjacent_under_threshold"])


def _frame(rsrp: float, sinr: float, count: int = 24) -> pd.DataFrame:
    offsets = np.linspace(-0.5, 0.5, count)
    return pd.DataFrame({
        "aggregate_index": np.arange(count),
        "time_seconds": np.arange(count, dtype=float),
        "ss_rsrp_dbm": rsrp + offsets,
        "ss_sinr_db": sinr + offsets[::-1] / 2,
    })


def test_support_ranking_is_execution_level_and_repeatability_anchored() -> None:
    config = _config()
    states = [(-20.0, -30.0), (-10.0, -30.0), (0.0, -20.0), (0.0, -10.0)]
    selected_rows = []
    simulation = {}
    for state_index, (ploss, noise) in enumerate(states):
        for repetition in (1, 2):
            execution_id = f"execution-{state_index}-{repetition}"
            selected_rows.append({
                "point_id": f"point-{state_index}-{repetition}",
                "execution_id": execution_id,
                "repetition": repetition,
                "ploss": ploss,
                "noise_power_dB": noise,
            })
            simulation[execution_id] = _frame(
                -100.0 + state_index * 3 + repetition * 0.02,
                8.0 + state_index * 2 + repetition * 0.02,
            )
    selected = pd.DataFrame(selected_rows)
    upv = _frame(-94.0, 12.0, count=16)
    balanced = pd.concat([
        upv,
        *[frame.iloc[np.linspace(0, len(frame) - 1, 16).round().astype(int)]
          for frame in simulation.values()],
    ])
    matrix = balanced[["ss_rsrp_dbm", "ss_sinr_db"]].to_numpy(float)
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0) * 1.4826

    executions, candidates, repeatability, threshold = _support_tables(
        upv,
        simulation,
        selected,
        balanced_count=16,
        feature_columns=["ss_rsrp_dbm", "ss_sinr_db"],
        features=config["features"],
        center=center,
        scale=scale,
        bandwidth=1.0,
        config=config,
    )

    assert len(executions) == 8
    assert len(candidates) == 4
    assert len(repeatability) == 4
    assert threshold == pytest.approx(np.quantile(repeatability["mmd_squared"], 0.9))
    assert candidates.iloc[0]["execution_count"] == 2


def test_sensitivity_uses_only_supported_axis_sweeps() -> None:
    config = _config()
    states = [
        (-20.0, -30.0),
        (-15.0, -30.0),
        (-10.0, -30.0),
        (-5.0, -30.0),
        (0.0, -20.0),
        (0.0, -10.0),
        (0.0, -5.0),
        (-5.0, -7.0),
    ]
    selected_rows = []
    simulation = {}
    for state_index, (ploss, noise) in enumerate(states):
        for repetition in (1, 2):
            execution_id = f"execution-{state_index}-{repetition}"
            selected_rows.append({
                "point_id": f"point-{state_index}-{repetition}",
                "execution_id": execution_id,
                "repetition": repetition,
                "ploss": ploss,
                "noise_power_dB": noise,
            })
            shift = 0.05 * repetition
            simulation[execution_id] = _frame(
                -95.0 + ploss + 0.1 * noise + shift,
                30.0 + 0.5 * ploss - 1.5 * noise + shift,
            )

    global_fit, local, matrix, summary = _sensitivity_tables(
        simulation, pd.DataFrame(selected_rows), config["features"], config
    )

    assert len(global_fit) == 4
    assert set(global_fit["execution_count"]) == {6, 8}
    assert len(local) == 10
    assert matrix.shape == (2, 3)
    assert summary["full_interaction_surface_identified"] is False
    assert summary["condition_number"] > 0
