from __future__ import annotations

import platform
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import yaml

from .upv_phase3d import (
    _git_revision,
    _read_json,
    _read_yaml,
    _sha256,
    _write_csv,
    _write_json,
)
from .upv_phase3i import COMMAND_COLUMNS, _barycentric, _scaled_points, _support_bank


def validate_phase3j_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "phase_3j_complete_test1_development_fidelity_and_repeatability":
        raise ValueError("unexpected Phase 3J stage")
    if config.get("evaluation_status") != "development_not_independent_final_validation":
        raise ValueError("Phase 3J must remain development evaluation")
    for flag in ("execution_authorized", "test6_access_authorized", "abc_authorized"):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false during protocol preparation")
    target = config["target_trace"]
    if target.get("session_id") != "corrected_test_1_ASUS" or int(target["rows"]) != 305:
        raise ValueError("Phase 3J must use the complete designated Test 1 trace")
    if target.get("relative_rsrp_definition") != "subtract_complete_session_median":
        raise ValueError("the complete-session median definition must remain frozen")
    if target.get("input_trace_is_development_data") is not True:
        raise ValueError("Test 1 must be labelled as development data")
    translator = config["translator"]
    if translator.get("method") != "bounded_piecewise_affine_interpolation":
        raise ValueError("Phase 3J requires bounded piecewise-affine interpolation")
    if translator.get("extrapolation") != "prohibited":
        raise ValueError("translator extrapolation is prohibited")
    if translator.get("boundary_tie_break") != "lexicographically_smallest_vertex_pair":
        raise ValueError("the boundary projection tie-break changed")
    execution = config["execution"]
    if int(execution.get("repetitions", 0)) != 3:
        raise ValueError("Phase 3J requires three complete executions")
    if len(set(execution.get("oai_rng_seeds", []))) != 3:
        raise ValueError("Phase 3J requires three distinct frozen OAI seeds")
    if execution.get("commands_may_adapt_during_execution") is not False:
        raise ValueError("commands may not adapt during execution")
    if execution.get("primary_kpi_alignment_seconds") != 0:
        raise ValueError("primary KPI alignment must remain zero lag")
    if execution.get("channel_verification_alignment_seconds") != 1:
        raise ValueError("channel verification must use the following UTC second")
    test6 = config["test6_support_gate"]
    if test6.get("freeze_before_test6_access") is not True:
        raise ValueError("Test 6 support rules must be frozen before access")
    if test6.get("excessive_out_of_hull_status") != "unsupported_not_emulator_failure":
        raise ValueError("out-of-support Test 6 trajectories must not be called failures")
    if float(test6.get("maximum_clipped_fraction", -1)) <= 0:
        raise ValueError("the Test 6 maximum clipped fraction must be positive")
    if float(test6.get("maximum_clipping_distance_scaled", -1)) <= 0:
        raise ValueError("the Test 6 maximum clipping distance must be positive")
    if test6.get("hardware_execution_if_unsupported") != "prohibited":
        raise ValueError("unsupported Test 6 trajectories may not be executed")
    if test6.get("missing_telemetry_interpolation") != "prohibited":
        raise ValueError("Test 6 missing telemetry may not be interpolated")
    if test6.get("temporal_gap_rule") != "split_at_missing_rows_and_never_bridge":
        raise ValueError("Test 6 temporal metrics must split at missing telemetry")
    if test6.get("primary_kpi_alignment_seconds") != 0:
        raise ValueError("Test 6 primary KPI alignment must remain zero lag")
    if test6.get("channel_verification_alignment_seconds") != 1:
        raise ValueError("Test 6 channel verification must use the following second")
    if test6.get("post_hoc_lag_selection") != "prohibited":
        raise ValueError("Test 6 may not use post-hoc lag selection")
    clipping = config["clipping_evaluation"]
    if clipping.get("bridge_across_clipped_rows") != "prohibited":
        raise ValueError("temporal metrics may not bridge clipped rows")
    runtime = config["runtime_gates"]
    if runtime.get("missing_row_interpolation") != "prohibited":
        raise ValueError("missing telemetry may not be interpolated")
    if runtime.get("temporal_gap_rule") != "split_at_missing_rows_and_never_bridge":
        raise ValueError("temporal metrics must split at missing telemetry")
    if int(test6.get("minimum_paired_rows", 0)) != int(
        runtime["minimum_paired_rows_per_execution"]
    ):
        raise ValueError("Test 6 and development missing-telemetry gates must agree")
    metrics = config["metric_definitions"]
    if metrics.get("primary_metrics_target") != (
        "original_measured_target_including_clipped_rows"
    ):
        raise ValueError("primary fidelity metrics must retain the measured target")
    if metrics.get("missing_rows") != (
        "omit_and_split_without_interpolation_or_bridging"
    ):
        raise ValueError("metric missing-row handling changed")
    repeatability = config["repeatability_gates"]
    if repeatability.get("per_command_statistic") != (
        "sample_standard_deviation_across_three_executions_ddof_1"
    ):
        raise ValueError("the per-command repeatability statistic changed")
    if repeatability.get("aggregate_statistic") != (
        "root_mean_square_over_command_indices_present_in_all_executions"
    ):
        raise ValueError("the repeatability aggregation changed")
    if (
        config["model_update_policy"].get("translator_update_from_phase3j_residuals")
        != "prohibited"
    ):
        raise ValueError("Phase 3J residual-driven translator updates are prohibited")
    if (
        config["model_update_policy"].get(
            "package_presented_to_test6_must_equal_committed_phase3j_package"
        )
        is not True
    ):
        raise ValueError("Test 6 must receive the committed Phase 3J package unchanged")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("reservation must remain closed during the offline freeze")


def _boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    counts = Counter(
        tuple(sorted((int(left), int(right))))
        for triangle in triangles
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    )
    return sorted(edge for edge, count in counts.items() if count == 1)


def _project_to_boundary(
    point: np.ndarray,
    points: np.ndarray,
    boundary_edges: list[tuple[int, int]],
) -> tuple[float, int, int, float, np.ndarray]:
    candidates: list[tuple[float, int, int, float, np.ndarray]] = []
    for left, right in boundary_edges:
        vector = points[right] - points[left]
        fraction = float(
            np.clip(np.dot(point - points[left], vector) / np.dot(vector, vector), 0.0, 1.0)
        )
        projected = points[left] + fraction * vector
        distance = float(np.linalg.norm(point - projected))
        candidates.append((distance, left, right, fraction, projected))
    if not candidates:
        raise ValueError("the translator triangulation has no boundary")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0]


def translate_complete_trace(
    trace: pd.DataFrame,
    support: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, mtri.Triangulation]:
    points = _scaled_points(support, config)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    finder = triangulation.get_trifinder()
    scaling = config["translator"]["output_space_scaling"]
    rsrp_scale = float(scaling["relative_rsrp_db"])
    sinr_scale = float(scaling["sinr_db"])
    target_points = np.column_stack(
        [
            trace["relative_rsrp_db"].to_numpy(dtype=float) / rsrp_scale,
            trace["sinr_db"].to_numpy(dtype=float) / sinr_scale,
        ]
    )
    triangle_indices = np.asarray(finder(target_points[:, 0], target_points[:, 1]), dtype=int)
    controls = support[["commanded_gain_db", "commanded_noise_power_db"]].to_numpy(
        dtype=float
    )
    boundary_edges = _boundary_edges(triangulation.triangles)
    decimals = int(config["translator"]["command_rounding_decimal_places"])
    rows: list[dict[str, Any]] = []
    for command_index, ((trace_index, target), point, triangle_index) in enumerate(
        zip(trace.iterrows(), target_points, triangle_indices, strict=True)
    ):
        if triangle_index >= 0:
            vertices = triangulation.triangles[int(triangle_index)]
            weights = _barycentric(point, points[vertices])
            if (weights < -1e-8).any() or not np.isclose(weights.sum(), 1.0, atol=1e-8):
                raise ValueError(f"invalid in-hull barycentric weights at row {trace_index}")
            command = weights @ controls[vertices]
            projected = point
            clipped = False
            distance = 0.0
            vertex_values = [int(value) for value in vertices]
            weight_values = [float(value) for value in weights]
        else:
            distance, left, right, fraction, projected = _project_to_boundary(
                point, points, boundary_edges
            )
            command = (1.0 - fraction) * controls[left] + fraction * controls[right]
            clipped = True
            vertex_values = [left, right, -1]
            weight_values = [1.0 - fraction, fraction, 0.0]
        rows.append(
            {
                "command_index": command_index,
                "trace_row_index": int(trace_index),
                "trace_time_bin": int(target["time_bin"]),
                "trace_t_s": float(target["t_s"]),
                "target_relative_rsrp_db": float(target["relative_rsrp_db"]),
                "target_sinr_db": float(target["sinr_db"]),
                "projected_relative_rsrp_db": float(projected[0] * rsrp_scale),
                "projected_sinr_db": float(projected[1] * sinr_scale),
                "commanded_gain_db": round(float(command[0]), decimals),
                "commanded_noise_power_db": round(float(command[1]), decimals),
                "clipped": clipped,
                "clipping_distance_scaled": distance,
                "triangle_index": int(triangle_index),
                "vertex_0": vertex_values[0],
                "vertex_1": vertex_values[1],
                "vertex_2": vertex_values[2],
                "barycentric_0": weight_values[0],
                "barycentric_1": weight_values[1],
                "barycentric_2": weight_values[2],
            }
        )
    commands = pd.DataFrame(rows, columns=list(COMMAND_COLUMNS))
    if not commands["commanded_gain_db"].between(-18.0, 0.0).all():
        raise ValueError("a full-trace gain command exceeds validated controls")
    if not commands["commanded_noise_power_db"].between(-35.0, -17.0).all():
        raise ValueError("a full-trace noise command exceeds validated controls")
    return commands, triangulation


def freeze_phase3j_full_trace(
    *,
    config_path: str | Path,
    phase3i_decision_path: str | Path,
    phase3g_execution_medians_path: str | Path,
    phase3h_state_validation_path: str | Path,
    direct_trace_path: str | Path,
    pyproject_path: str | Path,
    uv_lock_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3i_file = Path(phase3i_decision_path).resolve()
    phase3g_file = Path(phase3g_execution_medians_path).resolve()
    phase3h_file = Path(phase3h_state_validation_path).resolve()
    trace_file = Path(direct_trace_path).resolve()
    pyproject_file = Path(pyproject_path).resolve()
    uv_lock_file = Path(uv_lock_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3J protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3j_config(config)
    frozen = config["frozen_inputs"]
    expected = {
        phase3i_file: frozen["phase3i_pass_decision_sha256"],
        phase3g_file: frozen["phase3g_execution_medians_sha256"],
        phase3h_file: frozen["phase3h_state_validation_sha256"],
        trace_file: frozen["direct_test1_trace_sha256"],
        pyproject_file: frozen["pyproject_sha256"],
        uv_lock_file: frozen["uv_lock_sha256"],
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"frozen Phase 3J input checksum mismatch: {path}")
    phase3i = _read_json(phase3i_file)
    if phase3i.get("decision_code") != frozen["required_phase3i_decision"]:
        raise ValueError("Phase 3I did not pass the representative trace")
    if phase3i.get("full_trace_protocol_freeze_authorized") is not True:
        raise ValueError("Phase 3I did not authorize the complete-trace protocol freeze")
    if phase3i.get("full_trace_replay_currently_authorized") is not False:
        raise ValueError("Phase 3I improperly authorized complete-trace hardware execution")
    if phase3i.get("final_test6_accessed") is not False:
        raise ValueError("Phase 3I accessed Test 6")

    trace = pd.read_csv(trace_file)
    target = config["target_trace"]
    if len(trace) != int(target["rows"]) or set(trace["session_id"]) != {
        target["session_id"]
    }:
        raise ValueError("the complete Test 1 trace identity or row count changed")
    if not np.allclose(np.diff(trace["t_s"]), float(target["sampling_interval_seconds"])):
        raise ValueError("the complete Test 1 trace is not uniformly sampled")
    support = _support_bank(pd.read_csv(phase3g_file), pd.read_csv(phase3h_file))
    commands, triangulation = translate_complete_trace(trace, support, config)
    clipped = commands["clipped"].astype(bool)
    clipped_fraction = float(clipped.mean())
    maximum_distance = (
        float(commands.loc[clipped, "clipping_distance_scaled"].max())
        if clipped.any()
        else 0.0
    )
    development_gate = config["development_support_gate"]
    support_gate_passed = bool(
        clipped_fraction <= float(development_gate["maximum_clipped_fraction"])
        and maximum_distance <= float(development_gate["maximum_clipping_distance_scaled"])
    )
    if not support_gate_passed:
        raise ValueError("the complete Test 1 trajectory is outside the frozen support gate")

    points = _scaled_points(support, config)
    support = support.copy()
    support["scaled_relative_rsrp"] = points[:, 0]
    support["scaled_sinr"] = points[:, 1]
    clipped_rows = commands.loc[clipped].copy()
    clipped_rows["relative_rsrp_clipping_error_db"] = (
        clipped_rows["projected_relative_rsrp_db"]
        - clipped_rows["target_relative_rsrp_db"]
    )
    clipped_rows["sinr_clipping_error_db"] = (
        clipped_rows["projected_sinr_db"] - clipped_rows["target_sinr_db"]
    )
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "pyproject_sha256": _sha256(pyproject_file),
        "uv_lock_sha256": _sha256(uv_lock_file),
    }
    support_report = {
        "schema_version": 1,
        "stage": "phase_3j_complete_test1_support_report",
        "target_rows": len(commands),
        "inside_rows": int((~clipped).sum()),
        "clipped_rows": int(clipped.sum()),
        "clipped_fraction": clipped_fraction,
        "maximum_clipping_distance_scaled": maximum_distance,
        "development_support_gate": development_gate,
        "development_support_gate_passed": support_gate_passed,
        "test6_support_gate_frozen_but_not_applied": config["test6_support_gate"],
        "test6_accessed": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "translator_support_nodes.csv", support)
    _write_csv(
        output / "complete_test1_target_trace.csv",
        trace.reset_index(names="trace_row_index"),
    )
    _write_csv(output / "full_trace_commands.csv", commands)
    _write_csv(output / "clipped_targets.csv", clipped_rows)
    _write_json(output / "support_report.json", support_report)
    _write_json(output / "test6_support_rules.json", config["test6_support_gate"])
    _write_json(
        output / "protocol.json",
        {
            "schema_version": 1,
            "stage": config["stage"],
            "protocol_revision": config["protocol_revision"],
            "evaluation_status": config["evaluation_status"],
            "analysis_repository_revision": _git_revision(),
            "input_sha256": {path.name: _sha256(path) for path in expected},
            "target_trace": config["target_trace"],
            "translator": {
                **config["translator"],
                "support_nodes": len(support),
                "triangles": len(triangulation.triangles),
            },
            "support_report": support_report,
            "clipping_evaluation": config["clipping_evaluation"],
            "execution": config["execution"],
            "runtime_gates": config["runtime_gates"],
            "fidelity_gates_per_execution": config["fidelity_gates_per_execution"],
            "metric_definitions": config["metric_definitions"],
            "repeatability_gates": config["repeatability_gates"],
            "model_update_policy": config["model_update_policy"],
            "test6_interpretation": config["test6_interpretation"],
            "test6_support_gate": config["test6_support_gate"],
            "decision_rules": config["decision_rules"],
            "claim_limits": config["claim_limits"],
            "reservation": config["reservation"],
            "software_environment": environment,
            "execution_authorized": False,
            "test6_access_authorized": False,
            "abc_authorized": False,
        },
    )
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3j_complete_test1_protocol_manifest",
            "analysis_repository_revision": _git_revision(),
            "command_rows": len(commands),
            "clipped_rows": int(clipped.sum()),
            "development_support_gate_passed": support_gate_passed,
            "execution_authorized": False,
            "test6_accessed": False,
            "software_environment": environment,
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
        "command_rows": str(len(commands)),
        "clipped_rows": str(int(clipped.sum())),
        "clipped_fraction": f"{clipped_fraction:.12g}",
        "maximum_clipping_distance_scaled": f"{maximum_distance:.12g}",
        "development_support_gate_passed": str(support_gate_passed).lower(),
        "execution_authorized": "false",
        "test6_accessed": "false",
    }


def _boolean_series(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"{label} contains missing Boolean values")
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} contains invalid Boolean values")
    return normalized == "true"


def _pearson_exact(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _wasserstein_equal_weight(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("equal-weight empirical Wasserstein inputs must be non-empty and equal")
    return float(np.mean(np.abs(np.sort(left) - np.sort(right))))


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray) -> float:
    differences = left[:, None, :] - right[None, :, :]
    return float(np.sqrt(np.square(differences).sum(axis=2)).mean())


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    value = (
        2.0 * _mean_pairwise_distance(left, right)
        - _mean_pairwise_distance(left, left)
        - _mean_pairwise_distance(right, right)
    )
    return max(0.0, float(value))


def _metric_bundle(rows: pd.DataFrame, *, target: str) -> dict[str, float]:
    if target not in {"original", "projected"}:
        raise ValueError(f"unknown Phase 3J metric target: {target}")
    prefix = "target" if target == "original" else "projected"
    ordered = rows.sort_values("command_index")
    target_rsrp = ordered[f"{prefix}_relative_rsrp_db"].to_numpy(dtype=float)
    target_sinr = ordered[f"{prefix}_sinr_db"].to_numpy(dtype=float)
    observed_rsrp = ordered["observed_relative_rsrp_db"].to_numpy(dtype=float)
    observed_sinr = ordered["ss_sinr_db"].to_numpy(dtype=float)
    consecutive = np.diff(ordered["command_index"].to_numpy(dtype=int)) == 1
    target_rsrp_increment = np.diff(target_rsrp)[consecutive]
    target_sinr_increment = np.diff(target_sinr)[consecutive]
    observed_rsrp_increment = np.diff(observed_rsrp)[consecutive]
    observed_sinr_increment = np.diff(observed_sinr)[consecutive]
    target_joint = np.column_stack([target_rsrp, target_sinr / 2.0])
    observed_joint = np.column_stack([observed_rsrp, observed_sinr / 2.0])
    return {
        "rows": float(len(ordered)),
        "relative_rsrp_mae_db": float(np.mean(np.abs(observed_rsrp - target_rsrp))),
        "sinr_mae_db": float(np.mean(np.abs(observed_sinr - target_sinr))),
        "relative_rsrp_pearson_correlation": _pearson_exact(
            target_rsrp, observed_rsrp
        ),
        "sinr_pearson_correlation": _pearson_exact(target_sinr, observed_sinr),
        "relative_rsrp_wasserstein1_db": _wasserstein_equal_weight(
            target_rsrp, observed_rsrp
        ),
        "sinr_wasserstein1_db": _wasserstein_equal_weight(target_sinr, observed_sinr),
        "scaled_joint_energy_distance": _energy_distance(target_joint, observed_joint),
        "target_relative_rsrp_lag1_correlation": _pearson_exact(
            target_rsrp[:-1][consecutive], target_rsrp[1:][consecutive]
        ),
        "observed_relative_rsrp_lag1_correlation": _pearson_exact(
            observed_rsrp[:-1][consecutive], observed_rsrp[1:][consecutive]
        ),
        "target_sinr_lag1_correlation": _pearson_exact(
            target_sinr[:-1][consecutive], target_sinr[1:][consecutive]
        ),
        "observed_sinr_lag1_correlation": _pearson_exact(
            observed_sinr[:-1][consecutive], observed_sinr[1:][consecutive]
        ),
        "relative_rsrp_increment_wasserstein1_db": _wasserstein_equal_weight(
            target_rsrp_increment, observed_rsrp_increment
        ),
        "sinr_increment_wasserstein1_db": _wasserstein_equal_weight(
            target_sinr_increment, observed_sinr_increment
        ),
    }


def _supported_temporal_rows(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values("command_index").copy()
    clipped = _boolean_series(ordered["clipped"], "clipped")
    ordered.loc[clipped, "command_index"] = -10_000 - np.arange(int(clipped.sum()))
    return ordered.loc[~clipped]


def _lag_diagnostics(rows: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    indexed = rows.set_index("command_index").sort_index()
    records: list[dict[str, Any]] = []
    for lag in lags:
        shifted = indexed[["observed_relative_rsrp_db", "ss_sinr_db"]].copy()
        shifted.index = shifted.index - lag
        joined = indexed.join(shifted, how="inner", rsuffix="_lagged")
        records.append(
            {
                "lag_seconds": lag,
                "paired_rows": len(joined),
                "relative_rsrp_correlation": _pearson_exact(
                    joined["target_relative_rsrp_db"].to_numpy(dtype=float),
                    joined["observed_relative_rsrp_db_lagged"].to_numpy(dtype=float),
                ),
                "sinr_correlation": _pearson_exact(
                    joined["target_sinr_db"].to_numpy(dtype=float),
                    joined["ss_sinr_db_lagged"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(records)


def _verify_protocol_checksums(protocol_root: Path) -> None:
    checksum_file = protocol_root / "SHA256SUMS.json"
    checksums = _read_json(checksum_file)
    for name, expected in checksums.items():
        path = protocol_root / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"Phase 3J protocol artifact checksum mismatch: {name}")


def _analyze_phase3j_execution(
    *,
    campaign: Path,
    execution_number: int,
    commands: pd.DataFrame,
    commands_sha256: str,
    config: dict[str, Any],
    config_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry_file = campaign / "phase3j_full_trace_telemetry.csv"
    anchors_file = campaign / "phase3j_anchor_telemetry.csv"
    state_file = campaign / "execution_state.json"
    for path in (telemetry_file, anchors_file, state_file):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe Phase 3J campaign artifact: {path}")
    telemetry = pd.read_csv(telemetry_file)
    anchors = pd.read_csv(anchors_file)
    state = _read_json(state_file)
    expected_seed = int(config["execution"]["oai_rng_seeds"][execution_number - 1])
    if state.get("stage") != config["stage"]:
        raise ValueError(f"execution {execution_number} has the wrong stage")
    if state.get("evaluation_status") != config["evaluation_status"]:
        raise ValueError(f"execution {execution_number} has the wrong evaluation status")
    if state.get("execution_number") != execution_number:
        raise ValueError(f"execution {execution_number} identity mismatch")
    if state.get("oai_rng_seed") != expected_seed:
        raise ValueError(f"execution {execution_number} RNG seed mismatch")
    if state.get("execution_completed") is not True or state.get("error") is not None:
        raise ValueError(f"execution {execution_number} did not complete cleanly")
    if state.get("commands_sha256") != commands_sha256:
        raise ValueError(f"execution {execution_number} command checksum mismatch")
    if state.get("research_protocol_sha256") != config_sha256:
        raise ValueError(f"execution {execution_number} protocol checksum mismatch")
    if state.get("test6_accessed") is not False:
        raise ValueError(f"execution {execution_number} accessed Test 6")
    if state.get("translator_update_authorized") is not False:
        raise ValueError(f"execution {execution_number} authorized translator updates")
    if state.get("gNB_untouched") is not True:
        raise ValueError(f"execution {execution_number} changed the gNB")
    if telemetry["command_index"].duplicated().any():
        raise ValueError(f"execution {execution_number} contains duplicate command rows")
    if not telemetry["command_index"].isin(commands["command_index"]).all():
        raise ValueError(f"execution {execution_number} contains unknown command rows")
    paired = commands.merge(
        telemetry,
        on="command_index",
        how="left",
        validate="one_to_one",
        suffixes=("_protocol", "_observed"),
    )
    observed = paired["rsrp_db_per_re_unquantized"].notna()
    paired = paired.loc[observed].copy()
    for column in (
        "trace_row_index",
        "trace_time_bin",
        "trace_t_s",
        "target_relative_rsrp_db",
        "target_sinr_db",
        "projected_relative_rsrp_db",
        "projected_sinr_db",
        "commanded_gain_db",
        "commanded_noise_power_db",
    ):
        protocol_values = paired[f"{column}_protocol"].to_numpy(dtype=float)
        observed_values = paired[f"{column}_observed"].to_numpy(dtype=float)
        if not np.allclose(protocol_values, observed_values, atol=1e-9, rtol=0.0):
            raise ValueError(f"execution {execution_number} changed {column}")
        paired[column] = paired[f"{column}_protocol"]
    clipped_protocol = _boolean_series(paired["clipped_protocol"], "protocol clipped")
    clipped_observed = _boolean_series(paired["clipped_observed"], "observed clipped")
    if not clipped_protocol.equals(clipped_observed):
        raise ValueError(f"execution {execution_number} changed clipping flags")
    paired["clipped"] = clipped_protocol.to_numpy()
    if not np.allclose(
        paired["commanded_gain_db"], paired["applied_gain_db"], atol=1e-6, rtol=0.0
    ) or not np.allclose(
        paired["commanded_noise_power_db"],
        paired["applied_noise_power_db"],
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"execution {execution_number} applied different controls")
    anchor_medians = (
        anchors.groupby("anchor_type", as_index=False)[
            ["rsrp_db_per_re_unquantized", "ss_sinr_db"]
        ]
        .median()
        .set_index("anchor_type")
    )
    if set(anchor_medians.index) != {"anchor_start", "anchor_end"}:
        raise ValueError(f"execution {execution_number} is missing an anchor")
    anchor_reference = float(anchor_medians["rsrp_db_per_re_unquantized"].mean())
    paired["observed_relative_rsrp_db"] = (
        paired["rsrp_db_per_re_unquantized"] - anchor_reference
    )
    paired["translator_relative_rsrp_error_db"] = (
        paired["projected_relative_rsrp_db"] - paired["target_relative_rsrp_db"]
    )
    paired["translator_sinr_error_db"] = (
        paired["projected_sinr_db"] - paired["target_sinr_db"]
    )
    paired["dynamic_relative_rsrp_error_db"] = (
        paired["observed_relative_rsrp_db"] - paired["projected_relative_rsrp_db"]
    )
    paired["dynamic_sinr_error_db"] = paired["ss_sinr_db"] - paired["projected_sinr_db"]
    paired["total_relative_rsrp_error_db"] = (
        paired["observed_relative_rsrp_db"] - paired["target_relative_rsrp_db"]
    )
    paired["total_sinr_error_db"] = paired["ss_sinr_db"] - paired["target_sinr_db"]
    total = _metric_bundle(paired, target="original")
    dynamic = _metric_bundle(paired, target="projected")
    supported = _metric_bundle(_supported_temporal_rows(paired), target="original")
    rsrp_lag_error = abs(
        total["observed_relative_rsrp_lag1_correlation"]
        - total["target_relative_rsrp_lag1_correlation"]
    )
    sinr_lag_error = abs(
        total["observed_sinr_lag1_correlation"]
        - total["target_sinr_lag1_correlation"]
    )
    rsrp_anchor_drift = float(
        anchor_medians.loc["anchor_end", "rsrp_db_per_re_unquantized"]
        - anchor_medians.loc["anchor_start", "rsrp_db_per_re_unquantized"]
    )
    sinr_anchor_drift = float(
        anchor_medians.loc["anchor_end", "ss_sinr_db"]
        - anchor_medians.loc["anchor_start", "ss_sinr_db"]
    )
    runtime = config["runtime_gates"]
    attached = _boolean_series(paired["attached"], "attached")
    channel_identity = set(
        zip(
            paired["channel_family"].astype(str),
            paired["channel_length"].astype(int),
            paired["nb_taps"].astype(int),
            strict=True,
        )
    )
    runtime_gate = bool(
        len(paired) >= int(runtime["minimum_paired_rows_per_execution"])
        and attached.all()
        and float(state["ping_success_fraction"])
        >= float(runtime["minimum_ping_success_fraction_per_execution"])
        and float(paired["command_completion_lateness_seconds"].max())
        <= float(runtime["maximum_command_completion_lateness_seconds"])
        and state.get("critical_failure_count") == 0
        and state.get("ue_restart_count") == 0
        and state.get("gnb_restart_count_change") == 0
        and state.get("rollback", {}).get("passed") is True
        and channel_identity == {("AWGN", 1, 1)}
        and np.allclose(paired["tap_energy_linear"], 1.0, atol=1e-9, rtol=0.0)
    )
    fidelity = config["fidelity_gates_per_execution"]
    fidelity_gate = bool(
        total["relative_rsrp_mae_db"]
        <= float(fidelity["maximum_total_relative_rsrp_mae_db"])
        and total["sinr_mae_db"] <= float(fidelity["maximum_total_sinr_mae_db"])
        and total["relative_rsrp_pearson_correlation"]
        >= float(fidelity["minimum_total_relative_rsrp_pearson_correlation"])
        and total["sinr_pearson_correlation"]
        >= float(fidelity["minimum_total_sinr_pearson_correlation"])
        and total["relative_rsrp_wasserstein1_db"]
        <= float(fidelity["maximum_total_relative_rsrp_wasserstein1_db"])
        and total["sinr_wasserstein1_db"]
        <= float(fidelity["maximum_total_sinr_wasserstein1_db"])
        and total["scaled_joint_energy_distance"]
        <= float(fidelity["maximum_scaled_joint_energy_distance"])
        and rsrp_lag_error
        <= float(fidelity["maximum_relative_rsrp_lag1_correlation_error"])
        and sinr_lag_error <= float(fidelity["maximum_sinr_lag1_correlation_error"])
        and total["relative_rsrp_increment_wasserstein1_db"]
        <= float(fidelity["maximum_relative_rsrp_increment_wasserstein1_db"])
        and total["sinr_increment_wasserstein1_db"]
        <= float(fidelity["maximum_sinr_increment_wasserstein1_db"])
        and abs(rsrp_anchor_drift)
        <= float(fidelity["maximum_anchor_start_end_relative_rsrp_drift_db"])
        and abs(sinr_anchor_drift)
        <= float(fidelity["maximum_anchor_start_end_sinr_drift_db"])
    )
    metrics = {
        "execution_number": execution_number,
        "execution_id": state["execution_id"],
        "oai_rng_seed": expected_seed,
        "paired_rows": len(paired),
        "missing_rows": len(commands) - len(paired),
        "clipped_rows": int(paired["clipped"].sum()),
        **{f"total_{key}": value for key, value in total.items()},
        **{f"dynamic_{key}": value for key, value in dynamic.items()},
        **{f"supported_only_{key}": value for key, value in supported.items()},
        "total_relative_rsrp_lag1_correlation_error": rsrp_lag_error,
        "total_sinr_lag1_correlation_error": sinr_lag_error,
        "anchor_rsrp_drift_db": rsrp_anchor_drift,
        "anchor_sinr_drift_db": sinr_anchor_drift,
        "maximum_command_completion_lateness_seconds": float(
            paired["command_completion_lateness_seconds"].max()
        ),
        "p95_command_completion_lateness_seconds": float(
            paired["command_completion_lateness_seconds"].quantile(0.95)
        ),
        "runtime_gate_passed": runtime_gate,
        "fidelity_gate_passed": fidelity_gate,
        "all_per_execution_gates_passed": runtime_gate and fidelity_gate,
    }
    paired.insert(0, "execution_number", execution_number)
    anchors_output = anchor_medians.reset_index()
    anchors_output.insert(0, "execution_number", execution_number)
    diagnostics = _lag_diagnostics(
        paired, [int(value) for value in fidelity["diagnostic_lags_seconds"]]
    )
    diagnostics.insert(0, "execution_number", execution_number)
    return metrics, paired, anchors_output, diagnostics


def analyze_phase3j_full_trace(
    *,
    campaign_dirs: list[str | Path],
    protocol_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    if len(campaign_dirs) != 3:
        raise ValueError("Phase 3J requires exactly three campaign directories")
    campaigns = [Path(value).resolve() for value in campaign_dirs]
    if len(set(campaigns)) != 3:
        raise ValueError("Phase 3J campaign directories must be distinct")
    protocol_root = Path(protocol_dir).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3J analysis output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3j_config(config)
    _verify_protocol_checksums(protocol_root)
    commands_file = protocol_root / "full_trace_commands.csv"
    protocol_file = protocol_root / "protocol.json"
    commands = pd.read_csv(commands_file)
    protocol = _read_json(protocol_file)
    if tuple(commands.columns) != COMMAND_COLUMNS or len(commands) != 305:
        raise ValueError("the frozen Phase 3J command table is invalid")
    if protocol.get("evaluation_status") != config["evaluation_status"]:
        raise ValueError("the Phase 3J protocol evaluation status changed")
    command_sha256 = _sha256(commands_file)
    config_sha256 = _sha256(config_file)
    metric_rows: list[dict[str, Any]] = []
    paired_tables: list[pd.DataFrame] = []
    anchor_tables: list[pd.DataFrame] = []
    lag_tables: list[pd.DataFrame] = []
    campaign_hashes: dict[str, dict[str, str]] = {}
    for execution_number, campaign in enumerate(campaigns, start=1):
        metrics, paired, anchors, lag_diagnostics = _analyze_phase3j_execution(
            campaign=campaign,
            execution_number=execution_number,
            commands=commands,
            commands_sha256=command_sha256,
            config=config,
            config_sha256=config_sha256,
        )
        metric_rows.append(metrics)
        paired_tables.append(paired)
        anchor_tables.append(anchors)
        lag_tables.append(lag_diagnostics)
        campaign_hashes[f"execution_{execution_number}"] = {
            name: _sha256(campaign / name)
            for name in (
                "execution_state.json",
                "phase3j_full_trace_telemetry.csv",
                "phase3j_anchor_telemetry.csv",
            )
        }
    execution_ids = [row["execution_id"] for row in metric_rows]
    if len(set(execution_ids)) != 3:
        raise ValueError("Phase 3J execution identifiers must be distinct")
    paired_all = pd.concat(paired_tables, ignore_index=True)
    common = paired_all.pivot(
        index="command_index",
        columns="execution_number",
        values=["observed_relative_rsrp_db", "ss_sinr_db"],
    ).dropna()
    if len(common) < int(config["runtime_gates"]["minimum_paired_rows_per_execution"]):
        raise ValueError("too few command indices are present in all three executions")
    rsrp_values = common["observed_relative_rsrp_db"].to_numpy(dtype=float)
    sinr_values = common["ss_sinr_db"].to_numpy(dtype=float)
    rsrp_std = np.std(rsrp_values, axis=1, ddof=1)
    sinr_std = np.std(sinr_values, axis=1, ddof=1)
    repeatability_table = pd.DataFrame(
        {
            "command_index": common.index.to_numpy(dtype=int),
            "relative_rsrp_sample_standard_deviation_db": rsrp_std,
            "sinr_sample_standard_deviation_db": sinr_std,
        }
    )
    rsrp_repeatability = float(np.sqrt(np.mean(np.square(rsrp_std))))
    sinr_repeatability = float(np.sqrt(np.mean(np.square(sinr_std))))
    repeatability = config["repeatability_gates"]
    repeatability_gate = bool(
        rsrp_repeatability
        <= float(repeatability["maximum_between_execution_rsrp_standard_deviation_db"])
        and sinr_repeatability
        <= float(repeatability["maximum_between_execution_sinr_standard_deviation_db"])
    )
    runtime_all = all(bool(row["runtime_gate_passed"]) for row in metric_rows)
    fidelity_all = all(bool(row["fidelity_gate_passed"]) for row in metric_rows)
    if not runtime_all:
        decision_key = "fail_runtime"
    elif not fidelity_all:
        decision_key = "fail_fidelity"
    elif not repeatability_gate:
        decision_key = "fail_repeatability"
    else:
        decision_key = "pass"
    decision_rule = config["decision_rules"][decision_key]
    passed = decision_key == "pass"
    result = {
        "schema_version": 1,
        "stage": "phase_3j_complete_test1_development_fidelity_and_repeatability_result",
        "evaluation_status": config["evaluation_status"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "config": config_sha256,
            "protocol": _sha256(protocol_file),
            "commands": command_sha256,
            "campaigns": campaign_hashes,
        },
        "campaign": {
            "executions": 3,
            "target_rows_per_execution": len(commands),
            "development_trace": "corrected_test_1_ASUS",
            "independent_final_validation": False,
            "test6_accessed": False,
        },
        "repeatability": {
            "common_command_indices": len(common),
            "relative_rsrp_rms_between_execution_standard_deviation_db": (
                rsrp_repeatability
            ),
            "sinr_rms_between_execution_standard_deviation_db": sinr_repeatability,
            "gate_passed": repeatability_gate,
        },
        "gates": {
            "all_execution_runtime_gates_passed": runtime_all,
            "all_execution_fidelity_gates_passed": fidelity_all,
            "repeatability_gate_passed": repeatability_gate,
            "all_gates_passed": passed,
        },
        "decision_code": decision_rule["code"],
        "next_action": decision_rule["next_action"],
        "model_release_freeze_authorized": passed,
        "translator_update_from_residuals_authorized": False,
        "test6_access_authorized": False,
        "test6_accessed": False,
        "abc_authorized": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "paired_full_trace_fidelity.csv", paired_all)
    _write_csv(output / "per_execution_metrics.csv", pd.DataFrame(metric_rows))
    _write_csv(output / "repeatability_by_command.csv", repeatability_table)
    _write_csv(output / "anchor_medians.csv", pd.concat(anchor_tables, ignore_index=True))
    _write_csv(output / "lag_diagnostics.csv", pd.concat(lag_tables, ignore_index=True))
    _write_json(output / "phase3j_full_trace_decision.json", result)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": decision_rule["code"],
        "executions": "3",
        "model_release_freeze_authorized": str(passed).lower(),
        "test6_access_authorized": "false",
        "test6_accessed": "false",
    }
