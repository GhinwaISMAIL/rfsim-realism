from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .upv_phase3d import (
    _git_revision,
    _read_json,
    _read_yaml,
    _sha256,
    _write_csv,
    _write_json,
)


def validate_phase3m_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3M schema_version must be 1")
    if config.get("stage") != "phase_3m_version2_sinr_dynamics_offline_development":
        raise ValueError("unexpected Phase 3M stage")
    if config.get("evaluation_status") != "posthoc_offline_development_not_final_validation":
        raise ValueError("Phase 3M must remain explicitly post hoc development")
    for flag in (
        "hardware_execution_authorized",
        "inverse_command_generation_authorized",
        "version1_artifacts_mutable",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false")
    erratum = config["version1_erratum"]
    if erratum.get("frozen_outputs_recomputed") is not False:
        raise ValueError("the Version 1 erratum may not recompute frozen outputs")
    if erratum.get("thresholds_changed") is not False:
        raise ValueError("the Version 1 erratum may not change thresholds")
    if erratum.get("implemented_column_convention") != (
        "projected_feasible_target_minus_original_target"
    ):
        raise ValueError("the implemented clipping-error convention changed")
    development = config["development_data"]
    if development.get("random_row_splitting") != "prohibited":
        raise ValueError("random row splitting is prohibited")
    if development.get("clipped_row_forward_input") != "projected_feasible_sinr":
        raise ValueError("forward dynamics must use projected feasible SINR")
    if development.get("final_version2_validation_data") != "not_yet_selected":
        raise ValueError("final Version 2 validation data must remain unselected")
    models = config["models"]
    if models.get("prediction_mode") != "recursive_open_loop":
        raise ValueError("Phase 3M must evaluate recursive open-loop predictions")
    if models.get("observed_previous_output_as_predictor") != "prohibited":
        raise ValueError("held-out observations may not drive recursive predictions")
    alpha = models["alpha"]
    if not (0.0 <= float(alpha["minimum"]) < float(alpha["maximum"]) < 1.0):
        raise ValueError("invalid alpha bounds")
    if float(alpha["grid_step"]) <= 0:
        raise ValueError("alpha grid step must be positive")
    if config["cross_validation"].get("method") != "leave_one_complete_execution_out":
        raise ValueError("Phase 3M requires complete-execution cross-validation")
    if int(config["cross_validation"].get("folds", 0)) != 4:
        raise ValueError("Phase 3M requires four complete-execution folds")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("Phase 3M does not authorize a reservation")


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise ValueError(f"frozen Phase 3M input mismatch: {path}")


def _verify_bundle(directory: Path, expected_manifest_sha256: str) -> None:
    manifest = directory / "SHA256SUMS.json"
    _verify_file(manifest, expected_manifest_sha256)
    for name, digest in _read_json(manifest).items():
        _verify_file(directory / name, str(digest))


def _telemetry_inventory(campaign: Path) -> pd.DataFrame:
    files = sorted(campaign.glob("*-telemetry.csv"))
    records: list[dict[str, Any]] = []
    for path in files:
        frame = pd.read_csv(path)
        records.append(
            {
                "file_name": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "execution_id": str(frame["execution_id"].iloc[0]),
                "stage": str(frame["stage"].iloc[0]),
            }
        )
    return pd.DataFrame(records)


def _verify_protocol_checksums(directory: Path) -> None:
    manifest = directory / "SHA256SUMS.json"
    checksums = _read_json(manifest)
    for name, digest in checksums.items():
        _verify_file(directory / name, str(digest))


def freeze_phase3m_sinr_dynamics(
    *,
    config_path: str | Path,
    phase3j_result_dir: str | Path,
    phase3l_result_dir: str | Path,
    phase3g_execution_medians_path: str | Path,
    phase3g_campaign_dir: str | Path,
    phase3j_analyzer_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3j = Path(phase3j_result_dir).resolve()
    phase3l = Path(phase3l_result_dir).resolve()
    medians = Path(phase3g_execution_medians_path).resolve()
    campaign = Path(phase3g_campaign_dir).resolve()
    analyzer = Path(phase3j_analyzer_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3M protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3m_config(config)
    frozen = config["frozen_inputs"]
    _verify_bundle(phase3j, frozen["phase3j_result_checksums"]["sha256"])
    _verify_bundle(phase3l, frozen["phase3l_result_checksums"]["sha256"])
    _verify_file(
        phase3j / "paired_full_trace_fidelity.csv",
        frozen["phase3j_paired_fidelity"]["sha256"],
    )
    _verify_file(
        phase3l / "paired_test6_fidelity.csv",
        frozen["phase3l_paired_fidelity"]["sha256"],
    )
    _verify_file(medians, frozen["phase3g_execution_medians"]["sha256"])
    _verify_file(analyzer, frozen["phase3j_analyzer"]["sha256"])
    inventory = _telemetry_inventory(campaign)
    expected_files = int(frozen["phase3g_campaign"]["telemetry_files"])
    if len(inventory) != expected_files or inventory["execution_id"].nunique() != expected_files:
        raise ValueError("the frozen Phase 3G telemetry inventory changed")
    analyzer_text = analyzer.read_text()
    implemented_expressions = (
        'paired["projected_relative_rsrp_db"] - paired["target_relative_rsrp_db"]',
        'paired["projected_sinr_db"] - paired["target_sinr_db"]',
    )
    if not all(expression in analyzer_text for expression in implemented_expressions):
        raise ValueError("the Version 1 implemented clipping convention was not found")
    erratum = {
        "schema_version": 1,
        "status": config["version1_erratum"]["status"],
        "scope": "documentation_sign_only",
        "version1_config_sha256": _sha256(
            config_file.parent / "upv_phase3l_test6_exploratory_v1.yaml"
        ),
        "version1_analyzer_sha256": _sha256(analyzer),
        "declared_config_convention": config["version1_erratum"]["declared_config_convention"],
        "implemented_and_canonical_convention": config["version1_erratum"][
            "implemented_column_convention"
        ],
        "canonical_equations": {
            "clipping_error": "projected_feasible_target_minus_original_target",
            "dynamic_error": "oai_output_minus_projected_feasible_target",
            "total_error": "oai_output_minus_original_target",
            "identity": "total_error_equals_clipping_error_plus_dynamic_error",
        },
        "absolute_metrics_affected": False,
        "frozen_outputs_recomputed": False,
        "thresholds_changed": False,
        "version1_scientific_status_changed": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "phase3g_telemetry_inventory.csv", inventory)
    _write_json(output / "version1_sign_convention_erratum.json", erratum)
    protocol = {
        "schema_version": 1,
        "stage": config["stage"],
        "protocol_revision": config["protocol_revision"],
        "evaluation_status": config["evaluation_status"],
        "analysis_repository_revision": _git_revision(),
        "config_sha256": _sha256(config_file),
        "frozen_inputs": frozen,
        "phase3g_telemetry_inventory_sha256": _sha256(output / "phase3g_telemetry_inventory.csv"),
        "version1_erratum_sha256": _sha256(output / "version1_sign_convention_erratum.json"),
        "development_data": config["development_data"],
        "models": config["models"],
        "cross_validation": config["cross_validation"],
        "candidate_gates_against_static": config["candidate_gates_against_static"],
        "static_variability_reference": config["static_variability_reference"],
        "decision_rules": config["decision_rules"],
        "claim_limits": config["claim_limits"],
        "hardware_execution_authorized": False,
        "inverse_command_generation_authorized": False,
        "reservation_requested": False,
    }
    _write_json(output / "protocol.json", protocol)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "telemetry_files": str(len(inventory)),
        "version1_outputs_changed": "false",
        "hardware_execution_authorized": "false",
    }


def _execution_frames(phase3j: pd.DataFrame, phase3l: pd.DataFrame) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    required = {
        "execution_number",
        "command_index",
        "projected_sinr_db",
        "target_sinr_db",
        "ss_sinr_db",
        "clipped",
    }
    for source, frame, trajectory in (
        ("phase3j", phase3j, "test1"),
        ("phase3l", phase3l, "test6"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source} paired fidelity is missing columns: {sorted(missing)}")
        for number, group in frame.groupby("execution_number", sort=True):
            ordered = group.sort_values("command_index").reset_index(drop=True).copy()
            expected = np.arange(len(ordered))
            if not np.array_equal(ordered["command_index"].to_numpy(dtype=int), expected):
                raise ValueError(f"{source} execution {number} is not a complete ordered trace")
            if len(ordered) < 20 or ordered[list(required - {"clipped"})].isna().any().any():
                raise ValueError(f"{source} execution {number} is incomplete")
            executions.append(
                {
                    "execution_key": f"{source}_{trajectory}_execution_{int(number)}",
                    "source_phase": source,
                    "trajectory": trajectory,
                    "execution_number": int(number),
                    "frame": ordered,
                }
            )
    if len(executions) != 4:
        raise ValueError("Phase 3M requires exactly four complete executions")
    return executions


def _alpha_grid(config: dict[str, Any]) -> np.ndarray:
    spec = config["models"]["alpha"]
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["grid_step"])
    count = round((maximum - minimum) / step)
    return np.linspace(minimum, maximum, count + 1)


def _smoothed_input(projected: np.ndarray, alpha: float) -> np.ndarray:
    smooth = np.empty_like(projected, dtype=float)
    smooth[0] = projected[0]
    for index in range(1, len(projected)):
        smooth[index] = alpha * smooth[index - 1] + (1.0 - alpha) * projected[index]
    return smooth


def _bounded_affine_fit(
    executions: list[dict[str, Any]], alpha: float, config: dict[str, Any]
) -> tuple[float, float]:
    predictors: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for execution in executions:
        frame = execution["frame"]
        smooth = _smoothed_input(frame["projected_sinr_db"].to_numpy(dtype=float), alpha)[1:]
        observed = frame["ss_sinr_db"].to_numpy(dtype=float)[1:]
        predictors.append(smooth)
        outputs.append(observed)
        weights.append(np.full(len(smooth), 1.0 / len(smooth)))
    x = np.concatenate(predictors)
    y = np.concatenate(outputs)
    weight = np.concatenate(weights)
    design = np.column_stack([np.ones(len(x)), x])
    root_weight = np.sqrt(weight)
    a, b = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)[0]
    a_bounds = config["models"]["a_db"]
    b_bounds = config["models"]["b"]
    b = float(np.clip(b, float(b_bounds["minimum"]), float(b_bounds["maximum"])))
    a = float(np.average(y - b * x, weights=weight))
    a = float(np.clip(a, float(a_bounds["minimum"]), float(a_bounds["maximum"])))
    if a in (float(a_bounds["minimum"]), float(a_bounds["maximum"])):
        denominator = float(np.sum(weight * x * x))
        if denominator > 0:
            b = float(np.sum(weight * x * (y - a)) / denominator)
            b = float(np.clip(b, float(b_bounds["minimum"]), float(b_bounds["maximum"])))
    return a, b


def _predict(frame: pd.DataFrame, *, alpha: float, a: float, b: float) -> np.ndarray:
    projected = frame["projected_sinr_db"].to_numpy(dtype=float)
    return a + b * _smoothed_input(projected, alpha)


def _balanced_mse(executions: list[dict[str, Any]], *, alpha: float, a: float, b: float) -> float:
    losses = []
    for execution in executions:
        frame = execution["frame"]
        residual = (
            frame["ss_sinr_db"].to_numpy(dtype=float)[1:]
            - _predict(frame, alpha=alpha, a=a, b=b)[1:]
        )
        losses.append(float(np.mean(np.square(residual))))
    return float(np.mean(losses))


def _fit_model(
    model: str, executions: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, float]:
    if model == "static":
        a, b = _bounded_affine_fit(executions, 0.0, config)
        return {
            "alpha": 0.0,
            "a": a,
            "b": b,
            "training_mse": _balanced_mse(executions, alpha=0.0, a=a, b=b),
        }
    best: dict[str, float] | None = None
    for alpha in _alpha_grid(config):
        if model == "memory_only":
            a, b = 0.0, 1.0
        elif model == "combined":
            a, b = _bounded_affine_fit(executions, float(alpha), config)
        else:
            raise ValueError(f"unknown Phase 3M model: {model}")
        loss = _balanced_mse(executions, alpha=float(alpha), a=a, b=b)
        candidate = {"alpha": float(alpha), "a": a, "b": b, "training_mse": loss}
        if best is None or (loss, alpha) < (best["training_mse"], best["alpha"]):
            best = candidate
    if best is None:
        raise ValueError("the alpha grid is empty")
    return best


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _fold_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    absolute = np.abs(residual)
    return {
        "mae_db": float(np.mean(absolute)),
        "rmse_db": float(np.sqrt(np.mean(np.square(residual)))),
        "p95_absolute_error_db": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error_db": float(np.max(absolute)),
        "signed_bias_db": float(np.mean(residual)),
        "residual_lag1_correlation": _correlation(residual[:-1], residual[1:]),
        "prediction_correlation": _correlation(observed, predicted),
    }


def _cross_validate(
    executions: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    models = ("static", "memory_only", "combined")
    for fold, held_out in enumerate(executions, start=1):
        training = [item for item in executions if item is not held_out]
        for model in models:
            parameters = _fit_model(model, training, config)
            frame = held_out["frame"]
            predicted = _predict(frame, **{key: parameters[key] for key in ("alpha", "a", "b")})
            observed = frame["ss_sinr_db"].to_numpy(dtype=float)
            score = _fold_metrics(observed[1:], predicted[1:])
            metric_rows.append(
                {
                    "fold": fold,
                    "held_out_execution": held_out["execution_key"],
                    "held_out_trajectory": held_out["trajectory"],
                    "model": model,
                    "scored_rows": len(frame) - 1,
                    **score,
                }
            )
            parameter_rows.append(
                {
                    "fold": fold,
                    "held_out_execution": held_out["execution_key"],
                    "model": model,
                    "training_executions": ",".join(
                        str(item["execution_key"]) for item in training
                    ),
                    **parameters,
                }
            )
            for row_index in range(1, len(frame)):
                prediction_rows.append(
                    {
                        "fold": fold,
                        "held_out_execution": held_out["execution_key"],
                        "held_out_trajectory": held_out["trajectory"],
                        "model": model,
                        "command_index": int(frame.loc[row_index, "command_index"]),
                        "clipped": bool(frame.loc[row_index, "clipped"]),
                        "original_target_sinr_db": float(frame.loc[row_index, "target_sinr_db"]),
                        "projected_feasible_sinr_db": float(
                            frame.loc[row_index, "projected_sinr_db"]
                        ),
                        "observed_sinr_db": float(observed[row_index]),
                        "predicted_sinr_db": float(predicted[row_index]),
                        "residual_observed_minus_predicted_db": float(
                            observed[row_index] - predicted[row_index]
                        ),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows), pd.DataFrame(prediction_rows)


def _model_summary(metrics: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=False):
        params = parameters[parameters["model"] == model]
        records.append(
            {
                "model": model,
                "folds": len(group),
                "mean_mae_db": float(group["mae_db"].mean()),
                "mean_rmse_db": float(group["rmse_db"].mean()),
                "mean_p95_absolute_error_db": float(group["p95_absolute_error_db"].mean()),
                "maximum_absolute_error_db": float(group["maximum_absolute_error_db"].max()),
                "mean_signed_bias_db": float(group["signed_bias_db"].mean()),
                "mean_absolute_residual_lag1_correlation": float(
                    group["residual_lag1_correlation"].abs().mean()
                ),
                "mean_prediction_correlation": float(group["prediction_correlation"].mean()),
                "alpha_mean": float(params["alpha"].mean()),
                "alpha_min": float(params["alpha"].min()),
                "alpha_max": float(params["alpha"].max()),
                "a_mean_db": float(params["a"].mean()),
                "b_mean": float(params["b"].mean()),
                "b_min": float(params["b"].min()),
                "b_max": float(params["b"].max()),
            }
        )
    return pd.DataFrame(records)


def _candidate_gates(
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gate = config["candidate_gates_against_static"]
    lookup = summary.set_index("model")
    static = lookup.loc["static"]
    static_folds = metrics[metrics["model"] == "static"].set_index("fold")
    alpha_spec = config["models"]["alpha"]
    a_spec = config["models"]["a_db"]
    b_spec = config["models"]["b"]
    tolerance = float(gate["parameter_boundary_tolerance"])
    records: list[dict[str, Any]] = []
    for candidate in config["cross_validation"]["model_selection_order"]:
        row = lookup.loc[candidate]
        candidate_folds = metrics[metrics["model"] == candidate].set_index("fold")
        candidate_params = parameters[parameters["model"] == candidate]
        mae_improvement = float(static["mean_mae_db"] - row["mean_mae_db"])
        mae_relative = mae_improvement / float(static["mean_mae_db"])
        p95_improvement = float(
            static["mean_p95_absolute_error_db"] - row["mean_p95_absolute_error_db"]
        )
        p95_relative = p95_improvement / float(static["mean_p95_absolute_error_db"])
        lag_reduction = float(
            static["mean_absolute_residual_lag1_correlation"]
            - row["mean_absolute_residual_lag1_correlation"]
        )
        maximum_fold_degradation = float((candidate_folds["mae_db"] - static_folds["mae_db"]).max())
        alpha_range = float(candidate_params["alpha"].max() - candidate_params["alpha"].min())
        b_range = float(candidate_params["b"].max() - candidate_params["b"].min())
        alpha_boundary = bool(
            (candidate_params["alpha"] <= float(alpha_spec["minimum"]) + tolerance).any()
            or (candidate_params["alpha"] >= float(alpha_spec["maximum"]) - tolerance).any()
        )
        affine_boundary = False
        if candidate == "combined":
            affine_boundary = bool(
                (candidate_params["a"] <= float(a_spec["minimum"]) + tolerance).any()
                or (candidate_params["a"] >= float(a_spec["maximum"]) - tolerance).any()
                or (candidate_params["b"] <= float(b_spec["minimum"]) + tolerance).any()
                or (candidate_params["b"] >= float(b_spec["maximum"]) - tolerance).any()
            )
        checks = {
            "mae_absolute_gate": mae_improvement
            >= float(gate["minimum_absolute_mae_improvement_db"]),
            "mae_relative_gate": mae_relative
            >= float(gate["minimum_relative_mae_improvement_fraction"]),
            "p95_absolute_gate": p95_improvement
            >= float(gate["minimum_absolute_p95_improvement_db"]),
            "p95_relative_gate": p95_relative
            >= float(gate["minimum_relative_p95_improvement_fraction"]),
            "residual_lag_gate": lag_reduction
            >= float(gate["minimum_mean_absolute_residual_lag1_reduction"]),
            "fold_noninferiority_gate": maximum_fold_degradation
            <= float(gate["maximum_single_fold_mae_degradation_db"]),
            "alpha_stability_gate": alpha_range <= float(gate["maximum_alpha_range_across_folds"]),
            "b_stability_gate": b_range <= float(gate["maximum_b_range_across_folds"]),
            "parameter_interior_gate": not alpha_boundary and not affine_boundary,
        }
        records.append(
            {
                "candidate": candidate,
                "mae_improvement_db": mae_improvement,
                "mae_improvement_fraction": mae_relative,
                "p95_improvement_db": p95_improvement,
                "p95_improvement_fraction": p95_relative,
                "mean_absolute_residual_lag1_reduction": lag_reduction,
                "maximum_single_fold_mae_degradation_db": maximum_fold_degradation,
                "alpha_range_across_folds": alpha_range,
                "b_range_across_folds": b_range,
                "alpha_boundary_hit": alpha_boundary,
                "affine_boundary_hit": affine_boundary,
                **checks,
                "all_candidate_gates_passed": all(checks.values()),
            }
        )
    return pd.DataFrame(records)


def _static_variability(
    campaign: Path, inventory: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    included = set(config["static_variability_reference"]["included_stages"])
    rows: list[dict[str, Any]] = []
    for record in inventory.itertuples(index=False):
        path = campaign / record.file_name
        _verify_file(path, str(record.sha256))
        frame = pd.read_csv(path)
        if str(frame["stage"].iloc[0]) not in included:
            continue
        sinr = frame["ss_sinr_db"].to_numpy(dtype=float)
        rows.append(
            {
                "execution_id": str(frame["execution_id"].iloc[0]),
                "stage": str(frame["stage"].iloc[0]),
                "commanded_gain_db": float(frame["commanded_gain_db"].iloc[0]),
                "commanded_noise_power_db": float(frame["commanded_noise_power_db"].iloc[0]),
                "samples": len(sinr),
                "sinr_mean_db": float(np.mean(sinr)),
                "sinr_sd_db": float(np.std(sinr, ddof=1)),
                "sinr_mean_absolute_deviation_db": float(np.mean(np.abs(sinr - np.mean(sinr)))),
                "sinr_range_db": float(np.max(sinr) - np.min(sinr)),
            }
        )
    by_execution = pd.DataFrame(rows)
    state = (
        by_execution.groupby(["commanded_gain_db", "commanded_noise_power_db"])["sinr_mean_db"]
        .agg(["count", "std"])
        .reset_index()
    )
    repeated = state[state["count"] >= 3].dropna(subset=["std"])
    degrees = repeated["count"] - 1
    pooled_sd = math.sqrt(
        float(np.sum(degrees * np.square(repeated["std"]))) / float(degrees.sum())
    )
    summary = {
        "schema_version": 1,
        "included_stages": sorted(included),
        "executions": len(by_execution),
        "samples": int(by_execution["samples"].sum()),
        "median_within_execution_sinr_sd_db": float(by_execution["sinr_sd_db"].median()),
        "mean_within_execution_sinr_sd_db": float(by_execution["sinr_sd_db"].mean()),
        "p95_within_execution_sinr_sd_db": float(by_execution["sinr_sd_db"].quantile(0.95)),
        "median_within_execution_mean_absolute_deviation_db": float(
            by_execution["sinr_mean_absolute_deviation_db"].median()
        ),
        "repeated_control_states": len(repeated),
        "pooled_between_execution_mean_sinr_sd_db": pooled_sd,
        "interpretation": config["static_variability_reference"]["interpretation"],
    }
    return by_execution, summary


def analyze_phase3m_sinr_dynamics(
    *,
    config_path: str | Path,
    protocol_dir: str | Path,
    phase3j_result_dir: str | Path,
    phase3l_result_dir: str | Path,
    phase3g_campaign_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    protocol = Path(protocol_dir).resolve()
    phase3j = Path(phase3j_result_dir).resolve()
    phase3l = Path(phase3l_result_dir).resolve()
    campaign = Path(phase3g_campaign_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3M result output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3m_config(config)
    _verify_protocol_checksums(protocol)
    protocol_document = _read_json(protocol / "protocol.json")
    if protocol_document.get("config_sha256") != _sha256(config_file):
        raise ValueError("the Phase 3M configuration changed after protocol freeze")
    frozen = config["frozen_inputs"]
    _verify_bundle(phase3j, frozen["phase3j_result_checksums"]["sha256"])
    _verify_bundle(phase3l, frozen["phase3l_result_checksums"]["sha256"])
    inventory = pd.read_csv(protocol / "phase3g_telemetry_inventory.csv")
    if len(inventory) != int(frozen["phase3g_campaign"]["telemetry_files"]):
        raise ValueError("the Phase 3G telemetry inventory changed")
    executions = _execution_frames(
        pd.read_csv(phase3j / "paired_full_trace_fidelity.csv"),
        pd.read_csv(phase3l / "paired_test6_fidelity.csv"),
    )
    metrics, parameters, predictions = _cross_validate(executions, config)
    summary = _model_summary(metrics, parameters)
    gates = _candidate_gates(metrics, parameters, summary, config)
    selected: str | None = None
    for candidate in config["cross_validation"]["model_selection_order"]:
        row = gates[gates["candidate"] == candidate].iloc[0]
        if bool(row["all_candidate_gates_passed"]):
            selected = str(candidate)
            break
    variability, variability_summary = _static_variability(campaign, inventory, config)
    supported = selected is not None
    rule = config["decision_rules"][
        "candidate_supported" if supported else "no_candidate_supported"
    ]
    decision = {
        "schema_version": 1,
        "stage": "phase_3m_version2_sinr_dynamics_offline_result",
        "evaluation_status": config["evaluation_status"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "config": _sha256(config_file),
            "protocol": _sha256(protocol / "protocol.json"),
            "phase3j_paired_fidelity": _sha256(phase3j / "paired_full_trace_fidelity.csv"),
            "phase3l_paired_fidelity": _sha256(phase3l / "paired_test6_fidelity.csv"),
            "phase3g_telemetry_inventory": _sha256(protocol / "phase3g_telemetry_inventory.csv"),
        },
        "complete_execution_folds": len(executions),
        "selected_candidate": selected,
        "candidate_supported": supported,
        "decision_code": rule["code"],
        "next_action": rule["next_action"],
        "version1_status_changed": False,
        "test6_role": config["development_data"]["test6_role"],
        "hardware_execution_authorized": False,
        "inverse_command_generation_authorized": False,
        "final_version2_validation_performed": False,
        "claim_limits": config["claim_limits"],
    }
    output.mkdir(parents=True)
    _write_csv(output / "cross_validation_fold_metrics.csv", metrics)
    _write_csv(output / "cross_validation_parameters.csv", parameters)
    _write_csv(output / "cross_validation_predictions.csv", predictions)
    _write_csv(output / "model_summary.csv", summary)
    _write_csv(output / "candidate_gate_evaluation.csv", gates)
    _write_csv(output / "static_variability_by_execution.csv", variability)
    _write_json(output / "static_variability_summary.json", variability_summary)
    _write_json(output / "phase3m_decision.json", decision)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": str(rule["code"]),
        "selected_candidate": selected or "none",
        "hardware_execution_authorized": "false",
    }
