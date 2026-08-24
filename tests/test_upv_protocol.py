from __future__ import annotations

import numpy as np
import pandas as pd

from rfsim_realism.upv_protocol import build_locked_split, build_route_table


def _radio_route() -> pd.DataFrame:
    samples = 251
    longitude = np.linspace(-0.001, 0.001, samples)
    return pd.DataFrame(
        {
            "time_of_day": [f"12:00:{index / 2:06.3f}" for index in range(samples)],
            "seconds_of_day": 43200.0 + np.arange(samples) / 2.0,
            "longitude_deg": longitude,
            "latitude_deg": np.full(samples, 39.0),
            "rsrp_dbm": np.linspace(-95.0, -80.0, samples),
            "rsrq_db": np.full(samples, -10.5),
            "sinr_db": np.linspace(8.0, 24.0, samples),
            "serving_pci": np.full(samples, 41),
            "neighbor_rsrp_dbm": np.full(samples, -104.0),
            "neighbor_pci": np.full(samples, 61),
        }
    )


def _config() -> dict:
    return {
        "route": {"bin_sizes_m": [10, 15, 20]},
        "locked_split": {
            "primary_bin_size_m": 15,
            "minimum_samples": 10,
            "minimum_dwell_seconds": 4,
            "calibration_fraction": 0.50,
            "spatial_validation_fractions": [0.20, 0.35, 0.65, 0.80],
            "selection_basis": "geometry_sample_count_and_dwell_only",
        },
    }


def test_route_table_has_distance_direction_and_all_bin_sizes() -> None:
    route = build_route_table(
        _radio_route(),
        source_path="Test_2/Test_2_ASUS.csv",
        corrected_test_id=1,
        bin_sizes_m=[10, 15, 20],
        minimum_step_m_for_heading=0.1,
        direction_sectors=8,
    )

    assert route["route_distance_m"].is_monotonic_increasing
    assert route["route_distance_m"].iloc[-1] > 150
    assert set(route["direction_sector"]) == {"E"}
    assert {"route_bin_10m", "route_bin_15m", "route_bin_20m"} <= set(route)
    assert route["route_observation_id"].is_unique


def test_locked_split_is_geometry_only_and_kpi_invariant() -> None:
    route = build_route_table(
        _radio_route(),
        source_path="Test_2/Test_2_ASUS.csv",
        corrected_test_id=1,
        bin_sizes_m=[10, 15, 20],
        minimum_step_m_for_heading=0.1,
        direction_sectors=8,
    )
    baseline = build_locked_split(route, _config())
    changed = route.copy()
    changed[["rsrp_dbm", "rsrq_db", "sinr_db"]] = np.random.default_rng(42).normal(
        size=(len(changed), 3)
    )
    repeated = build_locked_split(changed, _config())

    pd.testing.assert_frame_equal(baseline, repeated)
    assert (baseline["locked_role"] == "calibration").sum() == 1
    assert baseline["locked_role"].str.startswith("spatial_validation_").sum() == 4
    assert not {"rsrp_dbm", "rsrq_db", "sinr_db"} & set(baseline)
