from __future__ import annotations

import numpy as np
import pandas as pd

from rfsim_realism.upv_phase3g_response import (
    _condition_number,
    _design_matrix,
    _fit_response,
)


def test_phase3g_response_recovers_crossed_coefficients() -> None:
    rows = []
    for gain in (-12.0, -10.0, -8.0):
        for noise in (-28.0, -25.0, -22.0):
            gain_centered = gain + 10.0
            noise_centered = noise + 25.0
            for repetition in range(3):
                rows.append(
                    {
                        "commanded_gain_db": gain,
                        "commanded_noise_power_db": noise,
                        "rsrp_db_per_re_unquantized": (
                            40.0
                            + gain_centered
                            + 0.05 * noise_centered
                            + 0.01 * gain_centered * noise_centered
                            + repetition * 0.001
                        ),
                        "ss_sinr_db": (
                            15.0
                            + gain_centered
                            - noise_centered
                            - 0.02 * gain_centered * noise_centered
                            + repetition * 0.001
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    coefficients, predictions = _fit_response(frame, -10.0, -25.0)
    assert predictions.shape == (27, 2)
    np.testing.assert_allclose(coefficients[1], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(coefficients[2], [0.05, -1.0], atol=1e-12)
    np.testing.assert_allclose(coefficients[3], [0.01, -0.02], atol=1e-12)
    assert _condition_number(coefficients) < 3.0


def test_phase3g_response_design_is_centered_and_contains_interaction() -> None:
    frame = pd.DataFrame(
        {
            "commanded_gain_db": [-10.0, -8.0],
            "commanded_noise_power_db": [-25.0, -22.0],
        }
    )
    matrix = _design_matrix(frame, -10.0, -25.0)
    np.testing.assert_allclose(matrix, [[1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 6.0]])
