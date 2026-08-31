from __future__ import annotations

import numpy as np
import pandas as pd

from rfsim_realism.upv_phase3g_diagnosis import (
    _convex_hull,
    _expanded_development_diagnostics,
    _inside_convex_hull,
    _solve_trace_controls,
)


def test_convex_hull_membership_handles_edges() -> None:
    hull = _convex_hull(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    )
    inside = _inside_convex_hull(
        np.array([[0.5, 0.5], [1.0, 0.5], [1.1, 0.5], [-0.1, -0.1]]), hull
    )
    assert inside.tolist() == [True, True, False, False]


def test_trace_inverse_recovers_bilinear_controls() -> None:
    coefficients = np.array(
        [
            [41.0, 15.0],
            [0.97, 1.04],
            [0.035, -1.0],
            [-0.0075, 0.0095],
        ]
    )
    expected_gain = np.array([-12.0, -8.0, -4.0])
    expected_noise = np.array([-28.0, -25.0, -20.0])
    centered_gain = expected_gain + 10.0
    centered_noise = expected_noise + 25.0
    design = np.column_stack(
        (
            np.ones(3),
            centered_gain,
            centered_noise,
            centered_gain * centered_noise,
        )
    )
    targets = design @ coefficients
    trace = pd.DataFrame(
        {
            "relative_rsrp_db": targets[:, 0] - coefficients[0, 0],
            "sinr_db": targets[:, 1],
            "provisional_gain_db": expected_gain + 0.2,
            "provisional_noise_power_db": expected_noise - 0.3,
        }
    )
    gain, noise, residual = _solve_trace_controls(
        trace, coefficients, gain_center=-10.0, noise_center=-25.0
    )
    np.testing.assert_allclose(gain, expected_gain, atol=1e-10)
    np.testing.assert_allclose(noise, expected_noise, atol=1e-10)
    assert residual < 1e-10


def test_expanded_development_diagnostics_recovers_stable_mapping() -> None:
    rows = []
    for gain in (-12.0, -10.0, -8.0):
        for noise in (-28.0, -25.0, -22.0):
            centered_gain = gain + 10.0
            centered_noise = noise + 25.0
            for repetition in range(3):
                rows.append(
                    {
                        "commanded_gain_db": gain,
                        "commanded_noise_power_db": noise,
                        "rsrp_db_per_re_unquantized": 41.0 + centered_gain,
                        "ss_sinr_db": 15.0 + centered_gain - centered_noise + repetition * 0.01,
                    }
                )
    result = _expanded_development_diagnostics(
        pd.DataFrame(rows), gain_center=-10.0, noise_center=-25.0
    )
    assert result["state_count"] == 9
    assert result["execution_count"] == 27
    assert result["responses"]["relative_RSRP"]["state_loso_max_absolute_error_db"] < 1e-10
    assert result["responses"]["SINR"]["state_loso_max_absolute_error_db"] < 0.02
