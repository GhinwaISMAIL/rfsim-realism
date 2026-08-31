from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .upv_phase3d import _git_revision, _read_json, _sha256, _write_csv, _write_json
from .upv_phase3g_response import RESPONSE_COLUMNS, _design_matrix, _fit_response


def _quadratic_design_matrix(
    frame: pd.DataFrame, gain_center: float, noise_center: float
) -> np.ndarray:
    gain = frame["commanded_gain_db"].to_numpy(float) - gain_center
    noise = frame["commanded_noise_power_db"].to_numpy(float) - noise_center
    return np.column_stack((np.ones(len(frame)), gain, noise, gain * noise, gain**2, noise**2))


def _fit_with_matrix(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    gain_center: float,
    noise_center: float,
    quadratic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _quadratic_design_matrix if quadratic else _design_matrix
    train_design = matrix(train, gain_center, noise_center)
    predict_design = matrix(predict, gain_center, noise_center)
    coefficients, _, _, _ = np.linalg.lstsq(
        train_design, train[list(RESPONSE_COLUMNS)].to_numpy(float), rcond=None
    )
    return predict_design @ coefficients, coefficients


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _inside_convex_hull(
    points: np.ndarray, hull: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(hull) < 3:
        return np.zeros(len(values), dtype=bool)
    crosses = np.column_stack(
        [
            (end[0] - start[0]) * (values[:, 1] - start[1])
            - (end[1] - start[1]) * (values[:, 0] - start[0])
            for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True)
        ]
    )
    return (crosses >= -tolerance).all(axis=1) | (crosses <= tolerance).all(axis=1)


def _solve_trace_controls(
    trace: pd.DataFrame,
    coefficients: np.ndarray,
    *,
    gain_center: float,
    noise_center: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    targets = np.column_stack(
        (
            coefficients[0, 0] + trace["relative_rsrp_db"].to_numpy(float),
            trace["sinr_db"].to_numpy(float),
        )
    )
    centered = np.column_stack(
        (
            trace["provisional_gain_db"].to_numpy(float) - gain_center,
            trace["provisional_noise_power_db"].to_numpy(float) - noise_center,
        )
    )
    maximum_step = float("inf")
    for _ in range(50):
        frame = pd.DataFrame(
            {
                "commanded_gain_db": centered[:, 0] + gain_center,
                "commanded_noise_power_db": centered[:, 1] + noise_center,
            }
        )
        error = targets - _design_matrix(frame, gain_center, noise_center) @ coefficients
        jacobians = np.empty((len(frame), 2, 2), dtype=float)
        jacobians[:, 0, 0] = coefficients[1, 0] + coefficients[3, 0] * centered[:, 1]
        jacobians[:, 0, 1] = coefficients[2, 0] + coefficients[3, 0] * centered[:, 0]
        jacobians[:, 1, 0] = coefficients[1, 1] + coefficients[3, 1] * centered[:, 1]
        jacobians[:, 1, 1] = coefficients[2, 1] + coefficients[3, 1] * centered[:, 0]
        step = np.linalg.solve(jacobians, error[..., None]).squeeze(-1)
        centered += step
        maximum_step = float(np.abs(step).max())
        if maximum_step < 1e-10:
            break
    if maximum_step >= 1e-8:
        raise ValueError("the empirical gain/noise inverse did not converge")
    commands = centered + np.array([gain_center, noise_center])
    final_frame = pd.DataFrame(
        {
            "commanded_gain_db": commands[:, 0],
            "commanded_noise_power_db": commands[:, 1],
        }
    )
    residual = np.abs(
        targets - _design_matrix(final_frame, gain_center, noise_center) @ coefficients
    ).max()
    return commands[:, 0], commands[:, 1], float(residual)


def _prediction_leverage(
    fit_states: pd.DataFrame,
    predict: pd.DataFrame,
    gain_center: float,
    noise_center: float,
) -> np.ndarray:
    fit_design = _design_matrix(fit_states, gain_center, noise_center)
    predict_design = _design_matrix(predict, gain_center, noise_center)
    covariance = np.linalg.pinv(fit_design.T @ fit_design)
    return np.einsum("ij,jk,ik->i", predict_design, covariance, predict_design)


def _bootstrap_boundary_attribution(
    factorial: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    gain_center: float,
    noise_center: float,
    repetitions: int,
    seed: int,
    rsrp_limit: float,
    sinr_limit: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    factorial_values = np.stack(
        [
            group.sort_values("execution_index")[list(RESPONSE_COLUMNS)].to_numpy(float)
            for _, group in factorial.groupby(
                ["commanded_gain_db", "commanded_noise_power_db"], sort=True
            )
        ]
    )
    boundary_values = np.stack(
        [
            group.sort_values("execution_index")[list(RESPONSE_COLUMNS)].to_numpy(float)
            for _, group in boundary.groupby(
                ["commanded_gain_db", "commanded_noise_power_db"], sort=True
            )
        ]
    )
    factorial_states = (
        factorial.groupby(["commanded_gain_db", "commanded_noise_power_db"], sort=True)
        .first()
        .reset_index()
    )
    boundary_states = (
        boundary.groupby(["commanded_gain_db", "commanded_noise_power_db"], sort=True)
        .first()
        .reset_index()
    )
    factorial_design = _design_matrix(factorial_states, gain_center, noise_center)
    boundary_design = _design_matrix(boundary_states, gain_center, noise_center)
    projection = np.linalg.pinv(factorial_design)
    point_coefficients, _ = _fit_response(factorial, gain_center, noise_center)
    point_boundary_means = boundary_values.mean(axis=1)
    point_predictions = boundary_design @ point_coefficients

    rng = np.random.default_rng(seed)
    factorial_choices = rng.integers(
        0, factorial_values.shape[1], size=(repetitions, len(factorial_values), 3)
    )
    factorial_samples = factorial_values[
        np.arange(len(factorial_values))[None, :, None], factorial_choices
    ].mean(axis=2)
    bootstrap_coefficients = np.einsum("ij,bjk->bik", projection, factorial_samples)
    boundary_choices = rng.integers(
        0, boundary_values.shape[1], size=(repetitions, len(boundary_values), 3)
    )
    boundary_samples = boundary_values[
        np.arange(len(boundary_values))[None, :, None], boundary_choices
    ].mean(axis=2)
    bootstrap_predictions = np.einsum(
        "ij,bjk->bik", boundary_design, bootstrap_coefficients
    )
    full_errors = np.abs(boundary_samples - bootstrap_predictions)
    model_only_errors = np.abs(point_boundary_means[None, :, :] - bootstrap_predictions)
    observation_only_errors = np.abs(boundary_samples - point_predictions[None, :, :])
    limits = (rsrp_limit, sinr_limit)
    response_names = ("relative_RSRP", "SINR")
    rows: list[dict[str, Any]] = []
    maximum_summary: dict[str, Any] = {}
    for response_index, response_name in enumerate(response_names):
        argmax = full_errors[:, :, response_index].argmax(axis=1)
        limit = limits[response_index]
        maximum = full_errors[:, :, response_index].max(axis=1)
        maximum_summary[response_name] = {
            "ci_low": float(np.quantile(maximum, 0.025)),
            "median": float(np.quantile(maximum, 0.5)),
            "ci_high": float(np.quantile(maximum, 0.975)),
            "limit_db": limit,
        }
        for state_index, state in boundary_states.iterrows():
            full = full_errors[:, state_index, response_index]
            model_only = model_only_errors[:, state_index, response_index]
            observation_only = observation_only_errors[:, state_index, response_index]
            model_high = float(np.quantile(model_only, 0.975))
            observation_high = float(np.quantile(observation_only, 0.975))
            rows.append(
                {
                    "response": response_name,
                    "commanded_gain_db": float(state["commanded_gain_db"]),
                    "commanded_noise_power_db": float(state["commanded_noise_power_db"]),
                    "point_absolute_error_db": float(
                        abs(
                            point_boundary_means[state_index, response_index]
                            - point_predictions[state_index, response_index]
                        )
                    ),
                    "full_bootstrap_ci_low_db": float(np.quantile(full, 0.025)),
                    "full_bootstrap_median_db": float(np.quantile(full, 0.5)),
                    "full_bootstrap_ci_high_db": float(np.quantile(full, 0.975)),
                    "model_only_ci_high_db": model_high,
                    "boundary_observation_only_ci_high_db": observation_high,
                    "maximum_error_attribution_fraction": float(np.mean(argmax == state_index)),
                    "threshold_exceedance_fraction": float(np.mean(full > limit)),
                    "dominant_uncertainty_component": (
                        "central_model_extrapolation"
                        if model_high > observation_high
                        else "boundary_execution_variability"
                    ),
                }
            )
    return pd.DataFrame(rows), maximum_summary


def _curvature_diagnostics(
    factorial: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    gain_center: float,
    noise_center: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_keys = list(
        factorial.groupby(["commanded_gain_db", "commanded_noise_power_db"], sort=True).groups
    )
    boundary_states = (
        boundary.groupby(["commanded_gain_db", "commanded_noise_power_db"], as_index=False)[
            list(RESPONSE_COLUMNS)
        ]
        .mean()
        .sort_values(["commanded_gain_db", "commanded_noise_power_db"])
    )
    factorial_state_count = len(state_keys)
    for quadratic in (False, True):
        model_name = "quadratic" if quadratic else "bilinear"
        fitted, coefficients = _fit_with_matrix(
            factorial,
            factorial,
            gain_center=gain_center,
            noise_center=noise_center,
            quadratic=quadratic,
        )
        boundary_prediction, _ = _fit_with_matrix(
            factorial,
            boundary_states,
            gain_center=gain_center,
            noise_center=noise_center,
            quadratic=quadratic,
        )
        fold_errors: list[np.ndarray] = []
        for gain, noise in state_keys:
            selected = (factorial["commanded_gain_db"] == gain) & (
                factorial["commanded_noise_power_db"] == noise
            )
            prediction, _ = _fit_with_matrix(
                factorial.loc[~selected],
                factorial.loc[selected],
                gain_center=gain_center,
                noise_center=noise_center,
                quadratic=quadratic,
            )
            fold_errors.append(
                factorial.loc[selected, list(RESPONSE_COLUMNS)].to_numpy(float) - prediction
            )
        cross_validation_errors = np.vstack(fold_errors)
        for response_index, response_name in enumerate(("relative_RSRP", "SINR")):
            response_column = RESPONSE_COLUMNS[response_index]
            residual = factorial[response_column].to_numpy(float) - fitted[:, response_index]
            pure_error = sum(
                float(((group[response_column] - group[response_column].mean()) ** 2).sum())
                for _, group in factorial.groupby(
                    ["commanded_gain_db", "commanded_noise_power_db"], sort=True
                )
            )
            total_error = float((residual**2).sum())
            lack_of_fit = max(0.0, total_error - pure_error)
            pure_error_df = len(factorial) - factorial_state_count
            lack_of_fit_df = factorial_state_count - len(coefficients)
            lack_of_fit_f = (
                (lack_of_fit / lack_of_fit_df) / (pure_error / pure_error_df)
                if lack_of_fit_df > 0 and pure_error > 0
                else float("nan")
            )
            boundary_error = (
                boundary_states[response_column].to_numpy(float)
                - boundary_prediction[:, response_index]
            )
            cv_error = cross_validation_errors[:, response_index]
            rows.append(
                {
                    "model": model_name,
                    "response": response_name,
                    "parameter_count": len(coefficients),
                    "central_execution_rmse_db": float(np.sqrt(np.mean(residual**2))),
                    "central_state_loso_rmse_db": float(np.sqrt(np.mean(cv_error**2))),
                    "central_state_loso_max_absolute_error_db": float(np.abs(cv_error).max()),
                    "lack_of_fit_f_ratio": lack_of_fit_f,
                    "boundary_state_rmse_db": float(np.sqrt(np.mean(boundary_error**2))),
                    "boundary_state_max_absolute_error_db": float(np.abs(boundary_error).max()),
                }
            )
    return pd.DataFrame(rows)


def _expanded_development_diagnostics(
    development: pd.DataFrame,
    *,
    gain_center: float,
    noise_center: float,
) -> dict[str, Any]:
    fitted, coefficients = _fit_with_matrix(
        development,
        development,
        gain_center=gain_center,
        noise_center=noise_center,
        quadratic=False,
    )
    fold_errors: list[np.ndarray] = []
    for gain, noise in development.groupby(
        ["commanded_gain_db", "commanded_noise_power_db"], sort=True
    ).groups:
        selected = (development["commanded_gain_db"] == gain) & (
            development["commanded_noise_power_db"] == noise
        )
        prediction, _ = _fit_with_matrix(
            development.loc[~selected],
            development.loc[selected],
            gain_center=gain_center,
            noise_center=noise_center,
            quadratic=False,
        )
        fold_errors.append(
            development.loc[selected, list(RESPONSE_COLUMNS)].to_numpy(float) - prediction
        )
    errors = development[list(RESPONSE_COLUMNS)].to_numpy(float) - fitted
    cross_validation_errors = np.vstack(fold_errors)
    jacobian = np.array(
        [
            [coefficients[1, 0], coefficients[2, 0]],
            [coefficients[1, 1], coefficients[2, 1]],
        ]
    )
    result: dict[str, Any] = {
        "state_count": int(
            development.groupby(["commanded_gain_db", "commanded_noise_power_db"]).ngroups
        ),
        "execution_count": len(development),
        "condition_number_at_center": float(np.linalg.cond(jacobian)),
        "coefficients": {},
        "responses": {},
    }
    term_names = ("intercept", "gain", "noise", "gain_noise_interaction")
    response_names = ("relative_RSRP", "SINR")
    for response_index, response_name in enumerate(response_names):
        result["coefficients"][response_name] = {
            term: float(coefficients[term_index, response_index])
            for term_index, term in enumerate(term_names)
        }
        result["responses"][response_name] = {
            "fit_rmse_db": float(np.sqrt(np.mean(errors[:, response_index] ** 2))),
            "state_loso_rmse_db": float(
                np.sqrt(np.mean(cross_validation_errors[:, response_index] ** 2))
            ),
            "state_loso_max_absolute_error_db": float(
                np.abs(cross_validation_errors[:, response_index]).max()
            ),
        }
    return result


def diagnose_phase3g_boundary(
    *,
    response_dir: str | Path,
    direct_trace_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    response = Path(response_dir).resolve()
    trace_file = Path(direct_trace_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3G diagnosis output already exists: {output}")
    decision_file = response / "phase3g_response_decision.json"
    medians_file = response / "execution_medians.csv"
    decision = _read_json(decision_file)
    if decision.get("direct_trace_replay_authorized") is not False:
        raise ValueError("the frozen response result unexpectedly authorized replay")
    if decision.get("final_test6_accessed") is not False:
        raise ValueError("the frozen response result accessed Test 6")
    medians = pd.read_csv(medians_file)
    trace = pd.read_csv(trace_file)
    required_trace = {
        "relative_rsrp_db",
        "sinr_db",
        "provisional_gain_db",
        "provisional_noise_power_db",
    }
    if not required_trace.issubset(trace.columns):
        raise ValueError("the direct Test 1 trace is missing required columns")
    if set(trace["session_id"]) != {"corrected_test_1_ASUS"}:
        raise ValueError("only the designated Test 1 ASUS development trace is permitted")
    factorial = medians.loc[medians["stage"] == "factorial"].copy()
    boundary = medians.loc[medians["stage"] == "boundary"].copy()
    if len(factorial) != 27 or len(boundary) != 12:
        raise ValueError("the frozen Phase 3G execution counts are inconsistent")

    gain_center = float(decision["model"]["gain_center_db"])
    noise_center = float(decision["model"]["noise_center_db"])
    limits = {"relative_RSRP": 1.0, "SINR": 2.0}
    attribution, maximum_summary = _bootstrap_boundary_attribution(
        factorial,
        boundary,
        gain_center=gain_center,
        noise_center=noise_center,
        repetitions=20000,
        seed=20260831,
        rsrp_limit=limits["relative_RSRP"],
        sinr_limit=limits["SINR"],
    )
    curvature = _curvature_diagnostics(
        factorial, boundary, gain_center=gain_center, noise_center=noise_center
    )
    development = pd.concat((factorial, boundary), ignore_index=True)
    expanded_development = _expanded_development_diagnostics(
        development, gain_center=gain_center, noise_center=noise_center
    )

    factorial_states = (
        factorial.groupby(["commanded_gain_db", "commanded_noise_power_db"], sort=True)
        .first()
        .reset_index()
    )
    boundary_states = (
        boundary.groupby(["commanded_gain_db", "commanded_noise_power_db"], sort=True)
        .first()
        .reset_index()
    )
    all_states = pd.concat((factorial_states, boundary_states), ignore_index=True)
    coefficients, _ = _fit_response(factorial, gain_center, noise_center)
    empirical_gain, empirical_noise, inverse_residual = _solve_trace_controls(
        trace, coefficients, gain_center=gain_center, noise_center=noise_center
    )
    trace_support = trace.copy()
    trace_support["empirical_gain_db"] = empirical_gain
    trace_support["empirical_noise_power_db"] = empirical_noise
    trace_support["inside_central_factorial_rectangle"] = (
        (empirical_gain >= factorial_states["commanded_gain_db"].min())
        & (empirical_gain <= factorial_states["commanded_gain_db"].max())
        & (empirical_noise >= factorial_states["commanded_noise_power_db"].min())
        & (empirical_noise <= factorial_states["commanded_noise_power_db"].max())
    )
    tested_hull = _convex_hull(
        all_states[["commanded_gain_db", "commanded_noise_power_db"]].to_numpy(float)
    )
    trace_points = np.column_stack((empirical_gain, empirical_noise))
    trace_support["inside_current_tested_hull"] = _inside_convex_hull(trace_points, tested_hull)
    operational_bounds = {
        "gain_min_db": -18.0,
        "gain_max_db": 0.0,
        "noise_min_db": -35.0,
        "noise_max_db": -17.0,
    }
    trace_support["inside_tested_operational_bounds"] = (
        (empirical_gain >= operational_bounds["gain_min_db"])
        & (empirical_gain <= operational_bounds["gain_max_db"])
        & (empirical_noise >= operational_bounds["noise_min_db"])
        & (empirical_noise <= operational_bounds["noise_max_db"])
    )
    command_frame = pd.DataFrame(
        {
            "commanded_gain_db": empirical_gain,
            "commanded_noise_power_db": empirical_noise,
        }
    )
    trace_support["central_model_prediction_leverage"] = _prediction_leverage(
        factorial_states, command_frame, gain_center, noise_center
    )
    trace_support["expanded_13_state_prediction_leverage"] = _prediction_leverage(
        all_states, command_frame, gain_center, noise_center
    )

    operational_trace = trace_points[
        trace_support["inside_tested_operational_bounds"].to_numpy(bool)
    ]
    target_hull = _convex_hull(operational_trace)
    rounded_target_hull = np.round(target_hull)
    union_hull = _convex_hull(
        np.vstack(
            (
                all_states[["commanded_gain_db", "commanded_noise_power_db"]].to_numpy(float),
                rounded_target_hull,
            )
        )
    )
    existing_pairs = set(
        map(
            tuple,
            all_states[["commanded_gain_db", "commanded_noise_power_db"]].to_numpy(float),
        )
    )
    candidate_points = np.asarray(
        [point for point in union_hull if tuple(point) not in existing_pairs], dtype=float
    ).reshape(-1, 2)
    candidate_frame = pd.DataFrame(
        candidate_points, columns=["commanded_gain_db", "commanded_noise_power_db"]
    )
    if not candidate_frame.empty:
        candidate_frame["expanded_13_state_prediction_leverage"] = _prediction_leverage(
            all_states, candidate_frame, gain_center, noise_center
        )
        distances = np.sqrt(
            ((empirical_gain[:, None] - candidate_points[None, :, 0]) / 2.0) ** 2
            + ((empirical_noise[:, None] - candidate_points[None, :, 1]) / 3.0) ** 2
        )
        candidate_frame["target_rows_within_scaled_radius_1"] = (distances <= 1.0).sum(axis=0)
    candidate_frame["status"] = "preliminary_not_frozen_not_authorized"
    candidate_frame["recommended_execution_repetitions"] = 5

    central_count = int(trace_support["inside_central_factorial_rectangle"].sum())
    tested_hull_count = int(trace_support["inside_current_tested_hull"].sum())
    operational_count = int(trace_support["inside_tested_operational_bounds"].sum())
    dominant_sinr = attribution.loc[
        attribution["response"] == "SINR"
    ].sort_values("maximum_error_attribution_fraction", ascending=False)
    bilinear_sinr = curvature.loc[
        (curvature["model"] == "bilinear") & (curvature["response"] == "SINR")
    ].iloc[0]
    quadratic_sinr = curvature.loc[
        (curvature["model"] == "quadratic") & (curvature["response"] == "SINR")
    ].iloc[0]
    diagnosis = {
        "schema_version": 1,
        "stage": "phase_3g_boundary_and_trace_support_diagnosis",
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "phase3g_response_decision": _sha256(decision_file),
            "phase3g_execution_medians": _sha256(medians_file),
            "direct_test1_target_trace": _sha256(trace_file),
        },
        "bootstrap": {
            "repetitions": 20000,
            "seed": 20260831,
            "maximum_error": maximum_summary,
            "dominant_sinr_boundary_state": {
                "commanded_gain_db": float(dominant_sinr.iloc[0]["commanded_gain_db"]),
                "commanded_noise_power_db": float(
                    dominant_sinr.iloc[0]["commanded_noise_power_db"]
                ),
                "maximum_error_attribution_fraction": float(
                    dominant_sinr.iloc[0]["maximum_error_attribution_fraction"]
                ),
                "dominant_uncertainty_component": dominant_sinr.iloc[0][
                    "dominant_uncertainty_component"
                ],
            },
        },
        "curvature": {
            "quadratic_revision_supported": False,
            "reason": "quadratic_worsens_sinr_loso_and_boundary_prediction",
            "bilinear_sinr_loso_rmse_db": float(
                bilinear_sinr["central_state_loso_rmse_db"]
            ),
            "quadratic_sinr_loso_rmse_db": float(
                quadratic_sinr["central_state_loso_rmse_db"]
            ),
            "bilinear_sinr_boundary_max_error_db": float(
                bilinear_sinr["boundary_state_max_absolute_error_db"]
            ),
            "quadratic_sinr_boundary_max_error_db": float(
                quadratic_sinr["boundary_state_max_absolute_error_db"]
            ),
        },
        "expanded_13_state_development_fit": expanded_development,
        "trace_support": {
            "target_rows": len(trace_support),
            "inverse_maximum_residual_db": inverse_residual,
            "central_factorial_rectangle_rows": central_count,
            "central_factorial_rectangle_fraction": central_count / len(trace_support),
            "current_tested_hull_rows": tested_hull_count,
            "current_tested_hull_fraction": tested_hull_count / len(trace_support),
            "tested_operational_bounds_rows": operational_count,
            "tested_operational_bounds_fraction": operational_count / len(trace_support),
            "outside_tested_operational_bounds_rows": len(trace_support) - operational_count,
            "central_model_leverage_quantiles": {
                "median": float(trace_support["central_model_prediction_leverage"].median()),
                "q95": float(
                    trace_support["central_model_prediction_leverage"].quantile(0.95)
                ),
                "maximum": float(trace_support["central_model_prediction_leverage"].max()),
            },
            "expanded_13_state_leverage_quantiles": {
                "median": float(
                    trace_support["expanded_13_state_prediction_leverage"].median()
                ),
                "q95": float(
                    trace_support["expanded_13_state_prediction_leverage"].quantile(0.95)
                ),
                "maximum": float(
                    trace_support["expanded_13_state_prediction_leverage"].max()
                ),
            },
        },
        "diagnosis_code": "targeted_outer_envelope_validation_extension_required",
        "reason": (
            "the failed SINR uncertainty gate is dominated by central-model extrapolation, "
            "while the central rectangle covers too little of the designated trace"
        ),
        "next_action": (
            "freeze_a_new_protocol_that_uses_all_13_existing_states_for_development_and_"
            "collects_new_trace_informed_held_out_validation_states"
        ),
        "preliminary_validation_state_count": len(candidate_frame),
        "preliminary_repetitions_per_state": 5,
        "reservation_request_now": False,
        "direct_trace_replay_authorized": False,
        "final_test6_accessed": False,
        "abc_authorized": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "boundary_bootstrap_attribution.csv", attribution)
    _write_csv(output / "curvature_diagnostics.csv", curvature)
    _write_csv(output / "direct_test1_trace_support.csv", trace_support)
    _write_csv(output / "preliminary_validation_states.csv", candidate_frame)
    _write_csv(
        output / "current_tested_hull.csv",
        pd.DataFrame(tested_hull, columns=["commanded_gain_db", "commanded_noise_power_db"]),
    )
    _write_csv(
        output / "operational_target_hull.csv",
        pd.DataFrame(target_hull, columns=["empirical_gain_db", "empirical_noise_power_db"]),
    )
    _write_json(output / "phase3g_boundary_diagnosis.json", diagnosis)
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3g_boundary_and_trace_support_diagnosis_manifest",
            "development_trace": "corrected_test_1_ASUS",
            "final_test6_accessed": False,
            "direct_trace_replay_authorized": False,
            "reservation_requested": False,
            "abc_authorized": False,
        },
    )
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "diagnosis": diagnosis["diagnosis_code"],
        "central_trace_support_fraction": f"{central_count / len(trace_support):.6f}",
        "current_tested_hull_fraction": f"{tested_hull_count / len(trace_support):.6f}",
        "preliminary_validation_states": str(len(candidate_frame)),
        "reservation_request_now": "false",
        "direct_trace_replay_authorized": "false",
        "final_test6_accessed": "false",
    }
