from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.tri as mtri
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

COMMAND_COLUMNS = (
    "command_index",
    "trace_row_index",
    "trace_time_bin",
    "trace_t_s",
    "target_relative_rsrp_db",
    "target_sinr_db",
    "projected_relative_rsrp_db",
    "projected_sinr_db",
    "commanded_gain_db",
    "commanded_noise_power_db",
    "clipped",
    "clipping_distance_scaled",
    "triangle_index",
    "vertex_0",
    "vertex_1",
    "vertex_2",
    "barycentric_0",
    "barycentric_1",
    "barycentric_2",
)

MARKER_FIELD = re.compile(r"(\w+)=([^\s]+)")
LEADING_MONOTONIC = re.compile(r"^(\d+(?:\.\d+)?)")
REJECTED_RUNNER_SHA256 = "8c41175ad59a2eb6221c543d4136d9e0ef91a2e5b78ab903408852389e43fa3b"
UE_FAILURE_PATTERNS = {
    "pbch_decode_error": re.compile(r"Error decoding PBCH!", re.IGNORECASE),
    "random_access_failure": re.compile(r"RA (?:Procedure|procedure) failed"),
    "radio_link_failure": re.compile(r"RLF detected|radio link failure", re.IGNORECASE),
    "lost_sync": re.compile(r"LOST SYNC|out of sync", re.IGNORECASE),
}
GNB_FAILURE_PATTERNS = {
    "pusch_ul_failure": re.compile(r"Detected UL Failure on PUSCH after"),
    "random_access_failure": re.compile(r"RA (?:Procedure|procedure) failed"),
    "rlc_max_retx": re.compile(r"max RETX reached", re.IGNORECASE),
    "unhandled_rlf_indication": re.compile(
        r"RLF detected, but no callable RLF handler registered", re.IGNORECASE
    ),
    "radio_link_failure": re.compile(
        r"RLF detected(?!,\s*but no callable RLF handler registered)|radio link failure",
        re.IGNORECASE,
    ),
}


def validate_phase3i_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "phase_3i_representative_short_trace_replay":
        raise ValueError("unexpected Phase 3I stage")
    for flag in (
        "execution_authorized",
        "full_trace_replay_authorized",
        "final_evaluation_authorized",
        "abc_authorized",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false during protocol preparation")
    target = config["target_trace"]
    if target.get("session_id") != "corrected_test_1_ASUS":
        raise ValueError("Phase 3I must use the designated Test 1 development trace")
    if target.get("final_test6_access") is not False:
        raise ValueError("Test 6 access is prohibited")
    selection = config["window_selection"]
    if int(selection.get("length_rows", 0)) != 60:
        raise ValueError("the Phase 3I smoke segment must contain 60 rows")
    if int(selection.get("stride_rows", 0)) != 1:
        raise ValueError("window selection must examine every contiguous start row")
    if selection.get("require_every_target_inside_validated_hull") is not True:
        raise ValueError("the short trace must be entirely inside validated support")
    if selection.get("tie_break") != "earliest_start_row":
        raise ValueError("window-selection ties must choose the earliest start")
    translator = config["translator"]
    if translator.get("method") != "bounded_piecewise_affine_interpolation":
        raise ValueError("Phase 3I requires bounded piecewise-affine interpolation")
    if translator.get("extrapolation") != "prohibited":
        raise ValueError("translator extrapolation is prohibited")
    if float(translator.get("short_trace_maximum_clipped_fraction", -1)) != 0.0:
        raise ValueError("the short trace must contain no clipped targets")
    scaling = translator["output_space_scaling"]
    if float(scaling["relative_rsrp_db"]) <= 0 or float(scaling["sinr_db"]) <= 0:
        raise ValueError("translator scales must be positive")
    execution = config["execution"]
    if float(execution["command_interval_seconds"]) != 1.0:
        raise ValueError("the short trace must retain its one-second sampling interval")
    if int(execution["oai_rng_seed"]) <= 0:
        raise ValueError("a positive OAI RNG seed is required")
    for key in (
        "post_attachment_stabilization_seconds",
        "anchor_start_settling_seconds",
        "anchor_end_settling_seconds",
        "anchor_usable_seconds",
    ):
        if float(execution[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    fidelity = config["fidelity_gates"]
    if fidelity.get("primary_alignment") != "zero_lag_only":
        raise ValueError("zero lag must remain the primary fidelity alignment")
    if fidelity.get("lag_search_for_gate_selection") != "prohibited":
        raise ValueError("post-hoc lag selection is prohibited")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("reservation remains closed during protocol preparation")


def _support_bank(
    phase3g_execution_medians: pd.DataFrame,
    phase3h_state_validation: pd.DataFrame,
) -> pd.DataFrame:
    phase3g = phase3g_execution_medians.loc[
        phase3g_execution_medians["stage"].isin(["factorial", "boundary"])
    ].copy()
    grouped = (
        phase3g.groupby(
            ["commanded_gain_db", "commanded_noise_power_db"], as_index=False
        )
        .agg(
            observed_rsrp_db=("rsrp_db_per_re_unquantized", "mean"),
            observed_sinr_db=("ss_sinr_db", "mean"),
            execution_units=("execution_id", "size"),
        )
        .sort_values(["commanded_gain_db", "commanded_noise_power_db"])
    )
    if len(grouped) != 13 or not (grouped["execution_units"] == 3).all():
        raise ValueError("the Phase 3G 13-state execution-level support bank is incomplete")
    anchor = grouped.loc[
        (grouped["commanded_gain_db"] == -10)
        & (grouped["commanded_noise_power_db"] == -25),
        "observed_rsrp_db",
    ]
    if len(anchor) != 1:
        raise ValueError("the Phase 3G relative-RSRP anchor is missing")
    grouped["observed_relative_rsrp_db"] = grouped["observed_rsrp_db"] - float(
        anchor.iloc[0]
    )
    grouped["source"] = "phase3g_execution_mean"
    grouped["source_state"] = grouped.apply(
        lambda row: f"g{row['commanded_gain_db']:g}_n{row['commanded_noise_power_db']:g}",
        axis=1,
    )

    required_phase3h = {
        "state_id",
        "commanded_gain_db",
        "commanded_noise_power_db",
        "observed_mean_relative_rsrp_db",
        "observed_mean_sinr_db",
    }
    if not required_phase3h.issubset(phase3h_state_validation.columns):
        raise ValueError("the Phase 3H state-validation table is incomplete")
    phase3h = phase3h_state_validation.copy()
    if len(phase3h) != 7 or set(phase3h["state_id"]) != set("ABCDEFG"):
        raise ValueError("Phase 3H must supply the seven validated outer states")
    phase3h = phase3h.rename(
        columns={
            "state_id": "source_state",
            "observed_mean_relative_rsrp_db": "observed_relative_rsrp_db",
            "observed_mean_sinr_db": "observed_sinr_db",
        }
    )
    phase3h["source"] = "phase3h_sequence_mean"
    phase3h["execution_units"] = 3
    phase3h["observed_rsrp_db"] = np.nan

    columns = [
        "source",
        "source_state",
        "commanded_gain_db",
        "commanded_noise_power_db",
        "observed_relative_rsrp_db",
        "observed_sinr_db",
        "execution_units",
    ]
    support = pd.concat([grouped[columns], phase3h[columns]], ignore_index=True)
    support = support.sort_values(
        ["source", "commanded_gain_db", "commanded_noise_power_db"]
    ).reset_index(drop=True)
    if len(support) != 20:
        raise ValueError("the combined translator support must contain 20 control states")
    if support.groupby(["commanded_gain_db", "commanded_noise_power_db"]).ngroups != 20:
        raise ValueError("the combined translator support contains duplicate controls")
    numeric = support[
        [
            "commanded_gain_db",
            "commanded_noise_power_db",
            "observed_relative_rsrp_db",
            "observed_sinr_db",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("the translator support contains non-finite values")
    support.insert(0, "support_index", np.arange(len(support), dtype=int))
    return support


def _scaled_points(support: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    scaling = config["translator"]["output_space_scaling"]
    return np.column_stack(
        [
            support["observed_relative_rsrp_db"].to_numpy(dtype=float)
            / float(scaling["relative_rsrp_db"]),
            support["observed_sinr_db"].to_numpy(dtype=float)
            / float(scaling["sinr_db"]),
        ]
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _lag1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return _pearson(values[:-1], values[1:])


def _window_scores(
    trace: pd.DataFrame,
    inside: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    selection = config["window_selection"]
    length = int(selection["length_rows"])
    quantiles = np.asarray(selection["quantiles"], dtype=float)
    rsrp = trace["relative_rsrp_db"].to_numpy(dtype=float)
    sinr = trace["sinr_db"].to_numpy(dtype=float)
    full_quantiles = np.concatenate([np.quantile(rsrp, quantiles), np.quantile(sinr, quantiles)])
    scales = np.concatenate(
        [
            np.repeat(np.quantile(rsrp, 0.95) - np.quantile(rsrp, 0.05), len(quantiles)),
            np.repeat(np.quantile(sinr, 0.95) - np.quantile(sinr, 0.05), len(quantiles)),
        ]
    )
    if (scales <= 0).any():
        raise ValueError("the full-trace quantile normalization is degenerate")
    full_cross = _pearson(rsrp, sinr)
    full_rsrp_lag1 = _lag1(rsrp)
    full_sinr_lag1 = _lag1(sinr)
    weights = selection["score"]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(trace) - length + 1, int(selection["stride_rows"])):
        stop = start + length
        window_rsrp = rsrp[start:stop]
        window_sinr = sinr[start:stop]
        window_quantiles = np.concatenate(
            [np.quantile(window_rsrp, quantiles), np.quantile(window_sinr, quantiles)]
        )
        quantile_component = float(np.mean(np.abs(window_quantiles - full_quantiles) / scales))
        cross_component = float(abs(_pearson(window_rsrp, window_sinr) - full_cross))
        rsrp_lag1_component = float(abs(_lag1(window_rsrp) - full_rsrp_lag1))
        sinr_lag1_component = float(abs(_lag1(window_sinr) - full_sinr_lag1))
        score = (
            float(weights["marginal_quantile_mean_absolute_weight"]) * quantile_component
            + float(weights["rsrp_sinr_pearson_absolute_difference_weight"])
            * cross_component
            + float(weights["rsrp_lag1_autocorrelation_absolute_difference_weight"])
            * rsrp_lag1_component
            + float(weights["sinr_lag1_autocorrelation_absolute_difference_weight"])
            * sinr_lag1_component
        )
        rows.append(
            {
                "start_row": start,
                "end_row_inclusive": stop - 1,
                "all_targets_inside_hull": bool(inside[start:stop].all()),
                "inside_fraction": float(inside[start:stop].mean()),
                "selection_score": score,
                "quantile_component": quantile_component,
                "cross_correlation_component": cross_component,
                "rsrp_lag1_component": rsrp_lag1_component,
                "sinr_lag1_component": sinr_lag1_component,
            }
        )
    return pd.DataFrame(rows)


def _barycentric(point: np.ndarray, triangle_points: np.ndarray) -> np.ndarray:
    matrix = np.vstack([triangle_points.T, np.ones(3)])
    return np.linalg.solve(matrix, np.append(point, 1.0))


def _translate_inside_targets(
    targets: pd.DataFrame,
    support: pd.DataFrame,
    triangulation: mtri.Triangulation,
    triangle_indices: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    points = _scaled_points(support, config)
    controls = support[["commanded_gain_db", "commanded_noise_power_db"]].to_numpy(
        dtype=float
    )
    scaling = config["translator"]["output_space_scaling"]
    decimals = int(config["translator"]["command_rounding_decimal_places"])
    rows: list[dict[str, Any]] = []
    for command_index, (trace_index, target) in enumerate(targets.iterrows()):
        point = np.array(
            [
                float(target["relative_rsrp_db"]) / float(scaling["relative_rsrp_db"]),
                float(target["sinr_db"]) / float(scaling["sinr_db"]),
            ]
        )
        triangle_index = int(triangle_indices[command_index])
        if triangle_index < 0:
            raise ValueError("the selected short trace contains an out-of-hull target")
        vertices = triangulation.triangles[triangle_index]
        weights = _barycentric(point, points[vertices])
        if (weights < -1e-8).any() or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
            raise ValueError("invalid barycentric weights for an in-hull target")
        command = weights @ controls[vertices]
        rows.append(
            {
                "command_index": command_index,
                "trace_row_index": int(trace_index),
                "trace_time_bin": int(target["time_bin"]),
                "trace_t_s": float(target["t_s"]),
                "target_relative_rsrp_db": float(target["relative_rsrp_db"]),
                "target_sinr_db": float(target["sinr_db"]),
                "projected_relative_rsrp_db": float(target["relative_rsrp_db"]),
                "projected_sinr_db": float(target["sinr_db"]),
                "commanded_gain_db": round(float(command[0]), decimals),
                "commanded_noise_power_db": round(float(command[1]), decimals),
                "clipped": False,
                "clipping_distance_scaled": 0.0,
                "triangle_index": triangle_index,
                "vertex_0": int(vertices[0]),
                "vertex_1": int(vertices[1]),
                "vertex_2": int(vertices[2]),
                "barycentric_0": float(weights[0]),
                "barycentric_1": float(weights[1]),
                "barycentric_2": float(weights[2]),
            }
        )
    commands = pd.DataFrame(rows, columns=list(COMMAND_COLUMNS))
    if not (
        commands["commanded_gain_db"].between(-18.0, 0.0).all()
        and commands["commanded_noise_power_db"].between(-35.0, -17.0).all()
    ):
        raise ValueError("an interpolated short-trace command exceeds validated controls")
    return commands


def freeze_phase3i_short_trace(
    *,
    config_path: str | Path,
    phase3h_decision_path: str | Path,
    phase3g_execution_medians_path: str | Path,
    phase3h_state_validation_path: str | Path,
    direct_trace_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    decision_file = Path(phase3h_decision_path).resolve()
    phase3g_file = Path(phase3g_execution_medians_path).resolve()
    phase3h_file = Path(phase3h_state_validation_path).resolve()
    trace_file = Path(direct_trace_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3I protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3i_config(config)
    frozen = config["frozen_inputs"]
    expected = {
        decision_file: frozen["phase3h_decision_sha256"],
        phase3g_file: frozen["phase3g_execution_medians_sha256"],
        phase3h_file: frozen["phase3h_state_validation_sha256"],
        trace_file: frozen["direct_test1_trace_sha256"],
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"frozen input checksum mismatch: {path}")
    decision = _read_json(decision_file)
    if decision.get("decision_code") != frozen["required_phase3h_decision"]:
        raise ValueError("Phase 3H did not authorize short-trace protocol preparation")
    if decision.get("short_trace_protocol_freeze_authorized") is not True:
        raise ValueError("Phase 3H did not open the short-trace protocol-freeze gate")
    if decision.get("short_trace_replay_currently_authorized") is not False:
        raise ValueError("Phase 3H improperly authorized hardware replay")
    if decision.get("final_test6_accessed") is not False:
        raise ValueError("Phase 3H accessed Test 6")

    phase3g = pd.read_csv(phase3g_file)
    phase3h = pd.read_csv(phase3h_file)
    trace = pd.read_csv(trace_file)
    target = config["target_trace"]
    if len(trace) != int(target["rows"]):
        raise ValueError("the designated Test 1 trace row count changed")
    if set(trace["session_id"]) != {target["session_id"]}:
        raise ValueError("the designated Test 1 trace identity changed")
    if not np.allclose(np.diff(trace["t_s"]), float(target["sampling_interval_seconds"])):
        raise ValueError("the designated Test 1 trace is not uniformly sampled")

    support = _support_bank(phase3g, phase3h)
    points = _scaled_points(support, config)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    finder = triangulation.get_trifinder()
    scaling = config["translator"]["output_space_scaling"]
    triangle_indices = np.asarray(
        finder(
            trace["relative_rsrp_db"].to_numpy(dtype=float)
            / float(scaling["relative_rsrp_db"]),
            trace["sinr_db"].to_numpy(dtype=float) / float(scaling["sinr_db"]),
        ),
        dtype=int,
    )
    inside = triangle_indices >= 0
    scores = _window_scores(trace, inside, config)
    eligible = scores.loc[scores["all_targets_inside_hull"]].sort_values(
        ["selection_score", "start_row"]
    )
    if eligible.empty:
        raise ValueError("no 60-second Test 1 window lies completely inside support")
    selected_score = eligible.iloc[0]
    start = int(selected_score["start_row"])
    length = int(config["window_selection"]["length_rows"])
    selected = trace.iloc[start : start + length].copy()
    selected_triangles = triangle_indices[start : start + length]
    commands = _translate_inside_targets(
        selected, support, triangulation, selected_triangles, config
    )
    if float(commands["clipped"].mean()) > float(
        config["translator"]["short_trace_maximum_clipped_fraction"]
    ):
        raise ValueError("the selected short trace exceeds the clipping limit")

    support = support.copy()
    support["scaled_relative_rsrp"] = points[:, 0]
    support["scaled_sinr"] = points[:, 1]
    output.mkdir(parents=True)
    _write_csv(output / "translator_support_nodes.csv", support)
    _write_csv(output / "window_selection_scores.csv", scores)
    _write_csv(output / "selected_target_trace.csv", selected.reset_index(names="trace_row_index"))
    _write_csv(output / "short_trace_commands.csv", commands)
    _write_json(
        output / "protocol.json",
        {
            "schema_version": 1,
            "stage": config["stage"],
            "protocol_revision": config["protocol_revision"],
            "analysis_repository_revision": _git_revision(),
            "input_sha256": {path.name: _sha256(path) for path in expected},
            "selection": {
                "start_row": start,
                "end_row_inclusive": start + length - 1,
                "start_trace_second": float(selected["t_s"].iloc[0]),
                "end_trace_second": float(selected["t_s"].iloc[-1]),
                "rows": length,
                "selection_score": float(selected_score["selection_score"]),
                "inside_fraction": float(selected_score["inside_fraction"]),
                "clipped_fraction": float(commands["clipped"].mean()),
            },
            "translator": {
                **config["translator"],
                "support_nodes": len(support),
                "triangles": len(triangulation.triangles),
            },
            "execution": config["execution"],
            "runtime_gates": config["runtime_gates"],
            "fidelity_gates": config["fidelity_gates"],
            "decision_rules": config["decision_rules"],
            "claim_limits": config["claim_limits"],
            "reservation": config["reservation"],
            "execution_authorized": False,
            "full_trace_replay_authorized": False,
            "final_test6_accessed": False,
            "abc_authorized": False,
        },
    )
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3i_short_trace_protocol_manifest",
            "support_nodes": len(support),
            "trace_rows": length,
            "selected_start_row": start,
            "selected_end_row_inclusive": start + length - 1,
            "clipped_rows": int(commands["clipped"].sum()),
            "execution_authorized": False,
            "full_trace_replay_authorized": False,
            "final_test6_accessed": False,
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
        "selected_start_row": str(start),
        "selected_end_row_inclusive": str(start + length - 1),
        "trace_rows": str(length),
        "clipped_rows": str(int(commands["clipped"].sum())),
        "execution_authorized": "false",
        "full_trace_replay_authorized": "false",
        "final_test6_accessed": "false",
    }


def _marker_rows(log_text: str, marker: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for line in log_text.splitlines():
        position = line.find(marker)
        if position < 0:
            continue
        fields = dict(MARKER_FIELD.findall(line[position + len(marker) :]))
        if "utc_second" in fields:
            rows[int(fields["utc_second"])] = fields
    return rows


def _observation_suffix(log_text: str, cutoff_monotonic: float) -> str:
    kept: list[str] = []
    for line in log_text.splitlines():
        match = LEADING_MONOTONIC.match(line)
        if match and float(match.group(1)) >= cutoff_monotonic:
            kept.append(line)
    return "\n".join(kept)


def _failure_counts(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in patterns.items()}


def recover_phase3i_short_trace(
    *,
    campaign_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    campaign = Path(campaign_dir).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3I recovery output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3i_config(config)
    paths = {
        "state": campaign / "execution_state.json",
        "events": campaign / "phase3i-command-events.json",
        "windows": campaign / "phase3i-anchor-windows.json",
        "pings": campaign / "phase3i-ping-checks.json",
        "ue_log": campaign / "phase3i-ue.log",
        "gnb_log": campaign / "phase3i-gnb.log",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe Phase 3I {name}: {path}")
    state = _read_json(paths["state"])
    events = json.loads(paths["events"].read_text())
    windows = json.loads(paths["windows"].read_text())
    pings = json.loads(paths["pings"].read_text())
    if state.get("execution_completed") is not False:
        raise ValueError("recovery requires the fail-closed rejected execution state")
    if state.get("error") != "applied gain mismatch at command 0":
        raise ValueError("recovery is not authorized for this execution error")
    if state.get("runner_sha256") != REJECTED_RUNNER_SHA256:
        raise ValueError("the rejected runner identity is not authorized for recovery")
    if state.get("research_protocol_sha256") != _sha256(config_file):
        raise ValueError("the rejected execution used a different scientific protocol")
    if state.get("rollback", {}).get("passed") is not True:
        raise ValueError("the rejected execution did not pass rollback")
    if not isinstance(events, list) or len(events) != 60:
        raise ValueError("recovery requires all 60 command events")
    if [int(row["command_index"]) for row in events] != list(range(60)):
        raise ValueError("recovery command indices are incomplete or reordered")
    if not isinstance(windows, list) or len(windows) != 2:
        raise ValueError("recovery requires both anchor windows")
    if not isinstance(pings, list) or not pings:
        raise ValueError("recovery requires recorded ping checks")

    ue_log = paths["ue_log"].read_text(errors="replace")
    gnb_log = paths["gnb_log"].read_text(errors="replace")
    ue_rows = _marker_rows(ue_log, "UE_RADIO_DEBUG_V1")
    channel_rows = _marker_rows(ue_log, "RFSIM_CHANNEL_DEBUG_V1")
    telemetry: list[dict[str, Any]] = []
    for event in events:
        second = int(event["sample_utc_second"])
        ue = ue_rows.get(second)
        channel = channel_rows.get(second + 1)
        if ue is None or channel is None:
            raise ValueError(
                f"missing recovered UE or next-second channel row for command "
                f"{event['command_index']}"
            )
        gain = float(channel["applied_gain_db"])
        noise = float(channel["noise_power_db"])
        if not math.isclose(gain, float(event["commanded_gain_db"]), abs_tol=1e-6):
            raise ValueError(f"recovered gain mismatch at command {event['command_index']}")
        if not math.isclose(
            noise, float(event["commanded_noise_power_db"]), abs_tol=1e-6
        ):
            raise ValueError(f"recovered noise mismatch at command {event['command_index']}")
        if float(event["command_completion_lateness_seconds"]) > float(
            config["runtime_gates"]["maximum_command_completion_lateness_seconds"]
        ):
            raise ValueError(f"late recovered command {event['command_index']}")
        if float(event["command_complete_epoch"]) > second + 0.5:
            raise ValueError(f"recovered command missed midpoint {event['command_index']}")
        if channel.get("model") != "rfsimu_channel_enB0":
            raise ValueError("the recovered trace did not use the active AWGN model")
        if channel.get("channel_length") != "1" or channel.get("nb_taps") != "1":
            raise ValueError("the recovered trace did not retain one-tap AWGN")
        if not math.isclose(float(channel["tap_energy_linear"]), 1.0, abs_tol=1e-9):
            raise ValueError("the recovered AWGN tap energy changed")
        telemetry.append(
            {
                **event,
                "channel_verification_utc_second": second + 1,
                "channel_verification_emitted_epoch_us": channel["emitted_epoch_us"],
                "ue_measurement_emitted_epoch_us": ue["emitted_epoch_us"],
                "applied_gain_db": gain,
                "applied_noise_power_db": noise,
                "channel_family": "AWGN",
                "channel_model_name": channel["model"],
                "channel_snapshot_id": channel["channel_snapshot_id"],
                "channel_snapshot_timestamp_ns": channel[
                    "channel_snapshot_timestamp_ns"
                ],
                "tap_energy_linear": channel["tap_energy_linear"],
                "tap_fingerprint_fnv1a64": channel["tap_fingerprint_fnv1a64"],
                "channel_length": channel["channel_length"],
                "nb_taps": channel["nb_taps"],
                "nb_tx": channel["nb_tx"],
                "nb_rx": channel["nb_rx"],
                "rsrp_digital_power_linear": ue["rsrp_digital_power_linear"],
                "rsrp_db_per_re_unquantized": ue["rsrp_db_per_re_unquantized"],
                "ss_rsrp_dbm_integer": ue["ss_rsrp_dbm_integer"],
                "ss_sinr_db": ue["ss_sinr_db"],
                "attached": True,
            }
        )

    anchors: list[dict[str, Any]] = []
    for anchor_type, window in zip(("anchor_start", "anchor_end"), windows, strict=True):
        start = float(window["usable_start_epoch"])
        end = float(window["usable_end_epoch"])
        for second, ue in sorted(ue_rows.items()):
            if not start <= second + 0.5 < end:
                continue
            channel = channel_rows.get(second)
            if channel is None:
                continue
            gain = float(channel["applied_gain_db"])
            noise = float(channel["noise_power_db"])
            if not math.isclose(gain, -10.0, abs_tol=1e-6) or not math.isclose(
                noise, -25.0, abs_tol=1e-6
            ):
                raise ValueError(f"{anchor_type} controls changed")
            anchors.append(
                {
                    "anchor_type": anchor_type,
                    "utc_second": second,
                    "rsrp_db_per_re_unquantized": ue["rsrp_db_per_re_unquantized"],
                    "ss_sinr_db": ue["ss_sinr_db"],
                    "applied_gain_db": gain,
                    "applied_noise_power_db": noise,
                    "channel_model_name": channel["model"],
                    "channel_length": channel["channel_length"],
                    "nb_taps": channel["nb_taps"],
                    "tap_energy_linear": channel["tap_energy_linear"],
                }
            )
    anchor_counts = pd.DataFrame(anchors)["anchor_type"].value_counts().to_dict()
    if anchor_counts.get("anchor_start", 0) < 7 or anchor_counts.get("anchor_end", 0) < 7:
        raise ValueError(f"insufficient recovered anchors: {anchor_counts}")

    first_marker = re.search(
        r"(?m)^(\d+(?:\.\d+)?).*RFSIM_CHANNEL_DEBUG_V1 .*emitted_epoch_us=(\d+)",
        ue_log,
    )
    if first_marker is None:
        raise ValueError("cannot establish the log monotonic-to-epoch relationship")
    boot_epoch = int(first_marker.group(2)) / 1_000_000 - float(first_marker.group(1))
    cutoff_monotonic = float(windows[0]["usable_start_epoch"]) - boot_epoch
    ue_observation = _observation_suffix(ue_log, cutoff_monotonic)
    gnb_observation = _observation_suffix(gnb_log, cutoff_monotonic)
    failures = {
        "ue": _failure_counts(ue_observation, UE_FAILURE_PATTERNS),
        "gnb": _failure_counts(gnb_observation, GNB_FAILURE_PATTERNS),
    }
    critical_failure_count = sum(
        count for domain in failures.values() for count in domain.values()
    )
    if critical_failure_count != 0:
        raise ValueError(f"critical failure in recovered observation window: {failures}")
    if ue_log.count("== Starting NR UE soft modem") != 1:
        raise ValueError("the recovered UE log does not contain exactly one process start")
    if not all(bool(item["passed"]) for item in pings):
        raise ValueError("the recovered run contains a failed ping")
    if not all(
        bool(item["attached"])
        for window in windows
        for item in window["attachment_checks"]
    ):
        raise ValueError("the recovered run contains anchor attachment loss")
    gnb_before = int(state["rollback"]["gnb_restart_count_before"])
    gnb_after = int(state["rollback"]["gnb_restart_count_after"])
    recovered_state = {
        **state,
        "execution_completed": True,
        "error": None,
        "paired_trace_rows": len(telemetry),
        "anchor_rows": len(anchors),
        "ping_success_fraction": 1.0,
        "critical_failure_count": critical_failure_count,
        "ue_restart_count": 0,
        "gnb_restart_count_change": gnb_after - gnb_before,
        "gnb_health": "healthy_after_rollback",
        "recovered_from_fail_closed_execution": True,
        "source_execution_state_sha256": _sha256(paths["state"]),
        "channel_verification_alignment_seconds": 1,
        "primary_kpi_alignment_seconds": 0,
        "recovery_repository_revision": _git_revision(),
    }
    recovery_report = {
        "schema_version": 1,
        "stage": "phase_3i_timestamp_alignment_recovery",
        "recovery_repository_revision": _git_revision(),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "source_error": state["error"],
        "command_events": len(events),
        "paired_trace_rows": len(telemetry),
        "anchor_rows": len(anchors),
        "same_second_channel_matches": 0,
        "next_second_channel_matches": len(telemetry),
        "channel_verification_alignment_seconds": 1,
        "primary_kpi_alignment_seconds": 0,
        "maximum_command_completion_lateness_seconds": max(
            float(row["command_completion_lateness_seconds"]) for row in events
        ),
        "failure_observation_cutoff_monotonic": cutoff_monotonic,
        "failure_marker_counts": failures,
        "critical_failure_count": critical_failure_count,
        "ping_successes": len(pings),
        "ping_checks": len(pings),
        "rollback_passed": True,
        "hardware_rerun_used": False,
        "scientific_target_or_gate_changed": False,
        "full_trace_replay_currently_authorized": False,
        "final_test6_accessed": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "phase3i_short_trace_telemetry.csv", pd.DataFrame(telemetry))
    _write_csv(output / "phase3i_anchor_telemetry.csv", pd.DataFrame(anchors))
    _write_json(output / "execution_state.json", recovered_state)
    _write_json(output / "recovery_report.json", recovery_report)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "paired_rows": str(len(telemetry)),
        "hardware_rerun_used": "false",
        "full_trace_replay_currently_authorized": "false",
        "final_test6_accessed": "false",
    }


def analyze_phase3i_short_trace(
    *,
    campaign_dir: str | Path,
    protocol_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    campaign = Path(campaign_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3I analysis output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3i_config(config)
    commands_file = protocol_root / "short_trace_commands.csv"
    protocol_file = protocol_root / "protocol.json"
    telemetry_file = campaign / "phase3i_short_trace_telemetry.csv"
    anchors_file = campaign / "phase3i_anchor_telemetry.csv"
    state_file = campaign / "execution_state.json"
    commands = pd.read_csv(commands_file)
    telemetry = pd.read_csv(telemetry_file)
    anchors = pd.read_csv(anchors_file)
    state = _read_json(state_file)
    if tuple(commands.columns) != COMMAND_COLUMNS or len(commands) != 60:
        raise ValueError("the frozen short-trace command table is invalid")
    if state.get("execution_completed") is not True or state.get("error") is not None:
        raise ValueError("the short-trace campaign did not complete cleanly")
    if state.get("final_test6_accessed") is not False:
        raise ValueError("the short-trace campaign accessed Test 6")
    if state.get("full_trace_replay_authorized") is not False:
        raise ValueError("the hardware runner cannot authorize full replay")
    if state.get("commands_sha256") != _sha256(commands_file):
        raise ValueError("the campaign command checksum does not match the protocol")
    if state.get("research_protocol_sha256") != _sha256(config_file):
        raise ValueError("the campaign protocol checksum does not match the evaluator")
    if state.get("gNB_untouched") is not True:
        raise ValueError("the short-trace campaign changed the gNB")
    required = {
        "command_index",
        "target_relative_rsrp_db",
        "target_sinr_db",
        "projected_relative_rsrp_db",
        "projected_sinr_db",
        "commanded_gain_db",
        "commanded_noise_power_db",
        "applied_gain_db",
        "applied_noise_power_db",
        "rsrp_db_per_re_unquantized",
        "ss_sinr_db",
        "command_completion_lateness_seconds",
        "attached",
    }
    if not required.issubset(telemetry.columns):
        raise ValueError("the short-trace telemetry is missing required columns")
    if telemetry["command_index"].duplicated().any():
        raise ValueError("the short-trace telemetry contains duplicate command rows")
    paired = commands.merge(
        telemetry,
        on=[
            "command_index",
            "target_relative_rsrp_db",
            "target_sinr_db",
            "projected_relative_rsrp_db",
            "projected_sinr_db",
            "commanded_gain_db",
            "commanded_noise_power_db",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("_protocol", "_observed"),
    )
    observed = paired["rsrp_db_per_re_unquantized"].notna()
    paired_observed = paired.loc[observed].copy()
    if not np.allclose(
        paired_observed["commanded_gain_db"], paired_observed["applied_gain_db"]
    ) or not np.allclose(
        paired_observed["commanded_noise_power_db"],
        paired_observed["applied_noise_power_db"],
    ):
        raise ValueError("applied controls differ from the frozen commands")
    anchor_medians = (
        anchors.groupby("anchor_type", as_index=False)[
            ["rsrp_db_per_re_unquantized", "ss_sinr_db"]
        ]
        .median()
        .set_index("anchor_type")
    )
    if set(anchor_medians.index) != {"anchor_start", "anchor_end"}:
        raise ValueError("both short-trace anchors are required")
    anchor_reference = float(anchor_medians["rsrp_db_per_re_unquantized"].mean())
    paired_observed["observed_relative_rsrp_db"] = (
        paired_observed["rsrp_db_per_re_unquantized"] - anchor_reference
    )
    paired_observed["relative_rsrp_error_db"] = (
        paired_observed["observed_relative_rsrp_db"]
        - paired_observed["projected_relative_rsrp_db"]
    )
    paired_observed["sinr_error_db"] = (
        paired_observed["ss_sinr_db"] - paired_observed["projected_sinr_db"]
    )
    rsrp_mae = float(paired_observed["relative_rsrp_error_db"].abs().mean())
    sinr_mae = float(paired_observed["sinr_error_db"].abs().mean())
    rsrp_correlation = _pearson(
        paired_observed["projected_relative_rsrp_db"].to_numpy(),
        paired_observed["observed_relative_rsrp_db"].to_numpy(),
    )
    sinr_correlation = _pearson(
        paired_observed["projected_sinr_db"].to_numpy(),
        paired_observed["ss_sinr_db"].to_numpy(),
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
    fidelity = config["fidelity_gates"]
    runtime_gate = bool(
        len(paired_observed) >= int(runtime["minimum_paired_trace_rows"])
        and paired_observed["attached"].astype(bool).all()
        and float(state["ping_success_fraction"])
        >= float(runtime["minimum_ping_success_fraction"])
        and float(paired_observed["command_completion_lateness_seconds"].max())
        <= float(runtime["maximum_command_completion_lateness_seconds"])
        and state.get("critical_failure_count") == 0
        and state.get("ue_restart_count") == 0
        and state.get("gnb_restart_count_change") == 0
        and state.get("rollback", {}).get("passed") is True
    )
    fidelity_gate = bool(
        rsrp_mae <= float(fidelity["maximum_relative_rsrp_mae_db"])
        and sinr_mae <= float(fidelity["maximum_sinr_mae_db"])
        and rsrp_correlation
        >= float(fidelity["minimum_relative_rsrp_pearson_correlation"])
        and sinr_correlation >= float(fidelity["minimum_sinr_pearson_correlation"])
        and abs(rsrp_anchor_drift)
        <= float(fidelity["maximum_anchor_start_end_relative_rsrp_drift_db"])
        and abs(sinr_anchor_drift)
        <= float(fidelity["maximum_anchor_start_end_sinr_drift_db"])
    )
    lag_rows: list[dict[str, Any]] = []
    for lag in fidelity["diagnostic_lags_seconds"]:
        lag = int(lag)
        if lag >= 0:
            target_part = paired_observed.iloc[: len(paired_observed) - lag or None]
            observed_part = paired_observed.iloc[lag:]
        else:
            target_part = paired_observed.iloc[-lag:]
            observed_part = paired_observed.iloc[:lag]
        lag_rows.append(
            {
                "lag_seconds": lag,
                "paired_rows": len(target_part),
                "relative_rsrp_correlation": _pearson(
                    target_part["projected_relative_rsrp_db"].to_numpy(),
                    observed_part["observed_relative_rsrp_db"].to_numpy(),
                ),
                "sinr_correlation": _pearson(
                    target_part["projected_sinr_db"].to_numpy(),
                    observed_part["ss_sinr_db"].to_numpy(),
                ),
            }
        )
    if not runtime_gate:
        decision_key = "fail_runtime"
    elif not fidelity_gate:
        decision_key = "fail_fidelity"
    else:
        decision_key = "pass"
    decision_rule = config["decision_rules"][decision_key]
    result = {
        "schema_version": 1,
        "stage": "phase_3i_representative_short_trace_result",
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "campaign_execution_state": _sha256(state_file),
            "campaign_telemetry": _sha256(telemetry_file),
            "campaign_anchors": _sha256(anchors_file),
            "protocol": _sha256(protocol_file),
            "commands": _sha256(commands_file),
            "config": _sha256(config_file),
        },
        "campaign": {
            "target_rows": len(commands),
            "paired_rows": len(paired_observed),
            "clipped_rows": int(commands["clipped"].astype(bool).sum()),
            "primary_alignment_seconds": 0,
            "lag_search_used_for_gate_selection": False,
        },
        "metrics": {
            "relative_rsrp_mae_db": rsrp_mae,
            "sinr_mae_db": sinr_mae,
            "relative_rsrp_pearson_correlation": rsrp_correlation,
            "sinr_pearson_correlation": sinr_correlation,
            "anchor_rsrp_drift_db": rsrp_anchor_drift,
            "anchor_sinr_drift_db": sinr_anchor_drift,
            "maximum_command_completion_lateness_seconds": float(
                paired_observed["command_completion_lateness_seconds"].max()
            ),
            "p95_command_completion_lateness_seconds": float(
                paired_observed["command_completion_lateness_seconds"].quantile(0.95)
            ),
        },
        "gates": {
            "runtime_gate_passed": runtime_gate,
            "fidelity_gate_passed": fidelity_gate,
            "all_gates_passed": runtime_gate and fidelity_gate,
        },
        "decision_code": decision_rule["code"],
        "next_action": decision_rule["next_action"],
        "full_trace_protocol_freeze_authorized": runtime_gate and fidelity_gate,
        "full_trace_replay_currently_authorized": False,
        "final_test6_accessed": False,
        "abc_authorized": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "paired_short_trace_fidelity.csv", paired_observed)
    _write_csv(output / "anchor_medians.csv", anchor_medians.reset_index())
    _write_csv(output / "lag_diagnostics.csv", pd.DataFrame(lag_rows))
    _write_json(output / "phase3i_short_trace_decision.json", result)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": decision_rule["code"],
        "paired_rows": str(len(paired_observed)),
        "full_trace_protocol_freeze_authorized": str(
            runtime_gate and fidelity_gate
        ).lower(),
        "full_trace_replay_currently_authorized": "false",
        "final_test6_accessed": "false",
    }
