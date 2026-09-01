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
