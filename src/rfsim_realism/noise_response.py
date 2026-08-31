from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = (
    "rsrp_db_per_re_unquantized",
    "ss_rsrp_dbm_integer",
    "ss_sinr_db",
)

NUMERIC_COLUMNS = (
    "repetition",
    "position",
    "oai_rng_seed",
    "commanded_noise_power_db",
    "applied_noise_power_db",
    "commanded_gain_db",
    "applied_gain_db",
    "tap_energy_linear",
    "channel_length",
    "nb_taps",
    "nb_tx",
    "nb_rx",
    *METRICS,
)

REQUIRED_COLUMNS = {
    "execution_id",
    "channel_family",
    "channel_model_name",
    "channel_snapshot_id",
    "tap_fingerprint_fnv1a64",
    "attached",
    *NUMERIC_COLUMNS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"missing or unsafe {label}: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _expected_execution_id(row: dict[str, Any]) -> str:
    noise = float(row["noise_power_db"])
    if not noise.is_integer() or noise >= 0:
        raise ValueError("the frozen noise states must be negative integer-valued dB controls")
    return f"r{int(row['repetition'])}-p{int(row['position'])}-n{abs(int(noise))}"


def validate_noise_response_analysis_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("noise-response analysis schema_version must be 1")
    if spec.get("stage") != "corrected_rfsim_noise_response_analysis_specification":
        raise ValueError("unexpected noise-response analysis stage")
    if spec.get("status") != "frozen_before_confirmatory_evaluation":
        raise ValueError("the analysis specification is not frozen")

    disclosure = spec.get("timing_disclosure") or {}
    if disclosure.get("bootstrap_intervals_inspected_before_this_freeze") is not False:
        raise ValueError("bootstrap intervals must not precede their specification")
    if disclosure.get("decision_gate_changed_after_execution") is not False:
        raise ValueError("the predeclared decision gate must remain unchanged")

    inputs = spec.get("frozen_inputs") or {}
    required_inputs = {
        "raw_archive_sha256",
        "telemetry_sha256",
        "execution_state_sha256",
        "protocol_sha256",
        "hardware_freeze_sha256",
        "development_route_means_sha256",
        "phase3d_decision_sha256",
    }
    for name in required_inputs:
        value = str(inputs.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid frozen checksum: {name}")

    analysis = spec.get("analysis") or {}
    if analysis.get("unit") != "execution":
        raise ValueError("the analysis unit must be the execution")
    if analysis.get("within_execution_summary") != "median":
        raise ValueError("each execution must be summarized by its median")
    if analysis.get("state_summary") != "mean_of_three_execution_medians":
        raise ValueError("unexpected state summary")
    if analysis.get("metrics") != list(METRICS):
        raise ValueError("unexpected response metrics")
    if analysis.get("monotonic_metric") != "ss_sinr_db":
        raise ValueError("SINR must determine the monotonicity gate")
    if float(analysis.get("maximum_allowed_upward_sinr_step_db", math.nan)) != 0.5:
        raise ValueError("the frozen monotonicity tolerance must be 0.5 dB")
    bootstrap = analysis.get("bootstrap") or {}
    if bootstrap.get("resampling_unit") != "execution_median":
        raise ValueError("bootstrap resampling must use execution medians")
    if bootstrap.get("groups_resampled_independently") is not True:
        raise ValueError("state groups must be resampled independently")
    if int(bootstrap.get("repetitions", 0)) < 1000:
        raise ValueError("at least 1000 bootstrap repetitions are required")
    if int(bootstrap.get("seed", -1)) < 0:
        raise ValueError("the bootstrap seed must be nonnegative")
    if float(bootstrap.get("confidence_level", 0)) != 0.95:
        raise ValueError("the frozen confidence level must be 0.95")
    if bootstrap.get("interval") != "percentile":
        raise ValueError("the frozen interval must be percentile bootstrap")

    comparison = spec.get("development_comparison") or {}
    if comparison.get("final_test6_access") is not False:
        raise ValueError("Test 6 must remain unopened")
    if comparison.get("fixed_noise_selection_authorized") is not False:
        raise ValueError("this analysis cannot select a fixed noise value")
    claims = spec.get("claim_limits") or {}
    for name in ("absolute_environmental_noise_calibration", "absolute_rsrp_calibration"):
        if claims.get(name) != "prohibited":
            raise ValueError(f"{name} must remain prohibited")


def _validate_input_checksums(paths: dict[str, Path], spec: dict[str, Any]) -> None:
    expected = spec["frozen_inputs"]
    mapping = {
        "raw_archive": "raw_archive_sha256",
        "telemetry": "telemetry_sha256",
        "execution_state": "execution_state_sha256",
        "protocol": "protocol_sha256",
        "hardware_freeze": "hardware_freeze_sha256",
        "development_route_means": "development_route_means_sha256",
        "phase3d_decision": "phase3d_decision_sha256",
    }
    for path_name, checksum_name in mapping.items():
        if _sha256(paths[path_name]) != expected[checksum_name]:
            raise ValueError(f"frozen input checksum mismatch: {path_name}")


def _validate_protocol_and_freeze(
    protocol: dict[str, Any], freeze: dict[str, Any], state: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    if protocol.get("stage") != "corrected_rfsim_noise_response_validation_protocol":
        raise ValueError("unexpected corrected-noise protocol stage")
    design = protocol.get("design") or {}
    if design.get("noise_power_db_states") != [-60.0, -40.0, -30.0, -25.0, -20.0]:
        raise ValueError("unexpected corrected-noise states")
    if int(design.get("independent_executions_per_state", 0)) != 3:
        raise ValueError("three execution units per noise state are required")
    if protocol.get("analysis_unit") != "execution":
        raise ValueError("the protocol analysis unit must be the execution")
    if "final UPV validation" not in (protocol.get("not_authorized") or []):
        raise ValueError("the protocol must keep final UPV validation unauthorized")

    if freeze.get("stage") != "corrected_rfsim_noise_hardware_execution_freeze":
        raise ValueError("unexpected hardware-freeze stage")
    if (freeze.get("authorization") or {}).get("final_upv_test6_access") is not False:
        raise ValueError("the hardware freeze must keep Test 6 inaccessible")
    plan = freeze.get("execution_plan") or []
    expected_noise = [
        float(value) for repetition in design["state_order_by_repetition"] for value in repetition
    ]
    if [float(row.get("noise_power_db", math.nan)) for row in plan] != expected_noise:
        raise ValueError("hardware execution order differs from the protocol")
    if len(plan) != 15 or len({int(row["oai_rng_seed"]) for row in plan}) != 15:
        raise ValueError("the hardware plan must contain 15 unique execution seeds")

    revisions = freeze["revision_identity"]
    configurations = freeze["configuration_identity"]
    images = freeze["image_identity"]
    rollback = state.get("rollback") or {}
    gates = {
        "stage": state.get("stage") == "corrected_rfsim_noise_response_validation",
        "execution_completed": state.get("execution_completed") is True,
        "error_absent": state.get("error") is None,
        "oai_revision": state.get("oai_revision") == revisions["oai_revision"],
        "profile_revision": state.get("profile_revision")
        == revisions["execution_profile_revision"],
        "runner_sha256": state.get("runner_sha256") == revisions["runner_sha256"],
        "compose_sha256": state.get("compose_sha256") == configurations["compose_sha256"],
        "channel_config_sha256": state.get("channel_config_sha256")
        == configurations["channel_config_sha256"],
        "ue_config_sha256": state.get("ue_config_sha256") == configurations["ue_config_sha256"],
        "attach_config_sha256": state.get("attach_config_sha256")
        == configurations["derived_attach_minus60_config_sha256"],
        "debug_image": state.get("debug_image") == images["corrected_tag"],
        "debug_image_id": state.get("debug_image_id") == images["corrected_image_id"],
        "debug_image_revision": state.get("debug_image_revision_label")
        == revisions["oai_revision"],
        "gNB_untouched": state.get("gNB_untouched") is True,
        "rollback_passed": rollback.get("passed") is True,
        "rollback_attached": rollback.get("attached") is True,
        "rollback_image": rollback.get("restored_image_id") == images["rollback_image_id"],
        "rollback_gnb_not_restarted": rollback.get("gnb_restart_count_before")
        == rollback.get("gnb_restart_count_after")
        == 0,
        "execution_plan": state.get("execution_plan")
        == [
            {
                "repetition": int(row["repetition"]),
                "position": int(row["position"]),
                "noise_power_db": float(row["noise_power_db"]),
                "oai_rng_seed": int(row["oai_rng_seed"]),
            }
            for row in plan
        ],
    }
    return plan, {name: bool(value) for name, value in gates.items()}


def _execution_state_gates(
    summary: dict[str, Any], expected: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, bool]:
    rules = protocol["valid_execution_rules"]
    command = summary.get("applied_command_result") or {}
    attach = summary.get("channel_identity_at_attach") or {}
    failures = summary.get("failure_marker_counts") or {}
    ue_failures = failures.get("ue") or {}
    gnb_failures = failures.get("gnb") or {}
    expected_id = _expected_execution_id(expected)
    noise = float(expected["noise_power_db"])
    return {
        "execution_id": summary.get("execution_id") == expected_id,
        "repetition": int(summary.get("repetition", -1)) == int(expected["repetition"]),
        "position": int(summary.get("position", -1)) == int(expected["position"]),
        "seed": int(summary.get("oai_rng_seed", -1)) == int(expected["oai_rng_seed"]),
        "commanded_noise": float(summary.get("commanded_noise_power_db", math.nan)) == noise,
        "applied_command_verified": command.get("verified") is True,
        "applied_command_parameter": command.get("parameter") == "noise_power_dB",
        "applied_command_model": command.get("model_type") == "AWGN",
        "applied_command_value": math.isclose(
            float(command.get("observed", math.nan)), noise, abs_tol=1e-9
        )
        and math.isclose(float(command.get("requested", math.nan)), noise, abs_tol=1e-9),
        "attach_at_minus60": attach.get("reachable") is True
        and attach.get("model_type") == "AWGN"
        and math.isclose(float(attach.get("observed", math.nan)), -60.0, abs_tol=1e-9),
        "continuous_attachment": summary.get("continuous_attachment") is True,
        "paired_samples": int(summary.get("paired_radio_samples", 0))
        >= int(rules["paired_radio_samples_minimum"]),
        "ping": float(summary.get("ping_success_fraction", 0.0))
        >= float(rules["ping_success_fraction_minimum"]),
        "critical_failures": int(summary.get("critical_failure_count", -1))
        <= int(rules["critical_pbch_failure_count_maximum"]),
        "critical_pbch_failures": int(summary.get("critical_pbch_failure_count", -1))
        <= int(rules["critical_pbch_failure_count_maximum"]),
        "critical_pusch_failures": int(summary.get("critical_pusch_failure_count", -1))
        <= int(rules["critical_pusch_failure_count_maximum"]),
        "failure_markers": sum(int(value) for value in ue_failures.values()) == 0
        and sum(int(value) for value in gnb_failures.values()) == 0,
        "ue_not_restarted": int(summary.get("ue_restart_count", -1))
        <= int(rules["unintended_ue_restart_count_maximum"]),
        "gnb_not_restarted": int(summary.get("gnb_restart_count", -1))
        <= int(rules["gnb_restart_count_change_maximum"]),
        "gnb_healthy": summary.get("gnb_health") == rules["gnb_health"],
    }


def _validate_telemetry(
    telemetry: pd.DataFrame,
    plan: list[dict[str, Any]],
    state: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    missing = sorted(REQUIRED_COLUMNS - set(telemetry.columns))
    if missing:
        raise ValueError(f"corrected-noise telemetry is missing columns: {missing}")
    for column in NUMERIC_COLUMNS:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    if not np.isfinite(telemetry[list(NUMERIC_COLUMNS)].to_numpy(float)).all():
        raise ValueError("corrected-noise telemetry contains non-finite required values")
    telemetry["attached"] = _truthy(telemetry["attached"])

    summaries = {
        str(summary.get("execution_id")): summary for summary in (state.get("executions") or [])
    }
    if len(summaries) != len(plan):
        raise ValueError("execution-state summary count differs from the frozen plan")

    execution_results: list[dict[str, Any]] = []
    expected_ids: list[str] = []
    rules = protocol["valid_execution_rules"]
    for expected in plan:
        execution_id = _expected_execution_id(expected)
        expected_ids.append(execution_id)
        frame = telemetry.loc[telemetry["execution_id"].astype(str).eq(execution_id)].copy()
        if frame.empty:
            raise ValueError(f"missing telemetry execution: {execution_id}")
        noise = float(expected["noise_power_db"])
        identity_gates = {
            "row_count": len(frame) >= int(rules["paired_radio_samples_minimum"]),
            "repetition": set(frame["repetition"].astype(int)) == {int(expected["repetition"])},
            "position": set(frame["position"].astype(int)) == {int(expected["position"])},
            "seed": set(frame["oai_rng_seed"].astype(int)) == {int(expected["oai_rng_seed"])},
            "commanded_noise": np.allclose(frame["commanded_noise_power_db"], noise),
            "applied_noise": np.allclose(frame["applied_noise_power_db"], noise),
            "zero_gain": np.allclose(frame["commanded_gain_db"], 0.0)
            and np.allclose(frame["applied_gain_db"], 0.0),
            "continuous_attachment": bool(frame["attached"].all()),
            "awgn_identity": set(frame["channel_family"].astype(str)) == {"AWGN"}
            and set(frame["channel_model_name"].astype(str)) == {"rfsimu_channel_enB0"},
            "one_snapshot": frame["channel_snapshot_id"].astype(str).nunique() == 1,
            "one_fingerprint": frame["tap_fingerprint_fnv1a64"].astype(str).nunique() == 1,
            "unit_channel": set(frame["channel_length"].astype(int)) == {1}
            and set(frame["nb_taps"].astype(int)) == {1}
            and set(frame["nb_tx"].astype(int)) == {1}
            and set(frame["nb_rx"].astype(int)) == {1}
            and np.allclose(frame["tap_energy_linear"], 1.0),
        }
        summary = summaries.get(execution_id) or {}
        state_gates = _execution_state_gates(summary, expected, protocol)
        gates = {**identity_gates, **state_gates}
        execution_results.append(
            {
                "execution_id": execution_id,
                "repetition": int(expected["repetition"]),
                "position": int(expected["position"]),
                "oai_rng_seed": int(expected["oai_rng_seed"]),
                "commanded_noise_power_db": noise,
                "paired_radio_samples": len(frame),
                **{f"{metric}_median": float(frame[metric].median()) for metric in METRICS},
                "gate_results": {name: bool(value) for name, value in gates.items()},
                "execution_pass": all(gates.values()),
            }
        )

    if sorted(telemetry["execution_id"].astype(str).unique()) != sorted(expected_ids):
        raise ValueError("telemetry contains unexpected execution identities")
    return telemetry, execution_results


def _bootstrap_pairwise(execution_medians: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    bootstrap = spec["analysis"]["bootstrap"]
    repetitions = int(bootstrap["repetitions"])
    confidence = float(bootstrap["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(int(bootstrap["seed"]))
    states = sorted(execution_medians["commanded_noise_power_db"].unique())
    records: list[dict[str, Any]] = []
    for metric in METRICS:
        column = f"{metric}_median"
        for lower_state, higher_state in combinations(states, 2):
            lower = execution_medians.loc[
                execution_medians["commanded_noise_power_db"].eq(lower_state), column
            ].to_numpy(float)
            higher = execution_medians.loc[
                execution_medians["commanded_noise_power_db"].eq(higher_state), column
            ].to_numpy(float)
            lower_draws = rng.choice(lower, size=(repetitions, len(lower)), replace=True).mean(
                axis=1
            )
            higher_draws = rng.choice(higher, size=(repetitions, len(higher)), replace=True).mean(
                axis=1
            )
            difference = higher_draws - lower_draws
            records.append(
                {
                    "metric": metric,
                    "lower_noise_power_db": float(lower_state),
                    "higher_noise_power_db": float(higher_state),
                    "direction": "higher_noise_power_state_minus_lower_noise_power_state",
                    "difference_of_mean_execution_medians": float(higher.mean() - lower.mean()),
                    "bootstrap_ci_lower": float(np.quantile(difference, alpha)),
                    "bootstrap_ci_upper": float(np.quantile(difference, 1.0 - alpha)),
                    "confidence_level": confidence,
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": int(bootstrap["seed"]),
                }
            )
    return pd.DataFrame(records)


def evaluate_corrected_noise_response(
    *,
    raw_archive_path: str | Path,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    protocol_path: str | Path,
    hardware_freeze_path: str | Path,
    analysis_spec_path: str | Path,
    development_route_means_path: str | Path,
    phase3d_decision_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "raw_archive": _require_file(raw_archive_path, "raw archive"),
        "telemetry": _require_file(telemetry_path, "corrected-noise telemetry"),
        "execution_state": _require_file(execution_state_path, "execution state"),
        "protocol": _require_file(protocol_path, "corrected-noise protocol"),
        "hardware_freeze": _require_file(hardware_freeze_path, "hardware freeze"),
        "analysis_spec": _require_file(analysis_spec_path, "analysis specification"),
        "development_route_means": _require_file(
            development_route_means_path, "development route means"
        ),
        "phase3d_decision": _require_file(phase3d_decision_path, "Phase 3D decision"),
    }
    spec = _read_json(paths["analysis_spec"])
    validate_noise_response_analysis_spec(spec)
    _validate_input_checksums(paths, spec)
    protocol = _read_json(paths["protocol"])
    freeze = _read_json(paths["hardware_freeze"])
    state = _read_json(paths["execution_state"])
    plan, state_gates = _validate_protocol_and_freeze(protocol, freeze, state)

    telemetry = pd.read_csv(
        paths["telemetry"],
        dtype={"channel_snapshot_id": str, "tap_fingerprint_fnv1a64": str},
    )
    telemetry, execution_results = _validate_telemetry(telemetry, plan, state, protocol)
    execution_medians = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "gate_results"}
            for row in execution_results
        ]
    )

    state_records: list[dict[str, Any]] = []
    for noise, frame in execution_medians.groupby("commanded_noise_power_db", sort=True):
        row: dict[str, Any] = {
            "commanded_noise_power_db": float(noise),
            "valid_executions": int(frame["execution_pass"].sum()),
            "execution_count": len(frame),
        }
        for metric in METRICS:
            values = frame[f"{metric}_median"]
            row[f"{metric}_mean_execution_median"] = float(values.mean())
            row[f"{metric}_between_execution_std"] = float(values.std(ddof=1))
            row[f"{metric}_execution_median_min"] = float(values.min())
            row[f"{metric}_execution_median_max"] = float(values.max())
        state_records.append(row)
    state_summary = pd.DataFrame(state_records)

    tolerance = float(spec["analysis"]["maximum_allowed_upward_sinr_step_db"])
    monotonic_results: list[dict[str, Any]] = []
    for repetition, frame in execution_medians.groupby("repetition", sort=True):
        ordered = frame.sort_values("commanded_noise_power_db")
        sinr = ordered["ss_sinr_db_median"].to_numpy(float)
        steps = np.diff(sinr)
        monotonic_results.append(
            {
                "repetition": int(repetition),
                "ordered_noise_power_db": ordered["commanded_noise_power_db"].tolist(),
                "ordered_sinr_median_db": sinr.tolist(),
                "successive_sinr_steps_db": steps.tolist(),
                "maximum_upward_step_db": float(max(0.0, steps.max(initial=-math.inf))),
                "pass": bool(np.all(steps <= tolerance)),
            }
        )

    pairwise = _bootstrap_pairwise(execution_medians, spec)
    development = pd.read_csv(paths["development_route_means"])
    required_development = {"route_relative_rsrp_db", "route_sinr_db", "supported", "fold_id"}
    if not required_development.issubset(development.columns):
        raise ValueError("development route means are missing required columns")
    supported = development.loc[_truthy(development["supported"])].copy()
    if supported.empty:
        raise ValueError("no supported development route means are available")
    phase3d = _read_json(paths["phase3d_decision"])
    final_evaluation = phase3d.get("final_evaluation") or {}
    if final_evaluation.get("payload_opened") is not False:
        raise ValueError("the Phase 3D final payload must remain unopened")

    centered_rsrp = telemetry["rsrp_db_per_re_unquantized"] - telemetry.groupby("execution_id")[
        "rsrp_db_per_re_unquantized"
    ].transform("median")
    upv_sinr = supported["route_sinr_db"].astype(float)
    upv_relative_rsrp = supported["route_relative_rsrp_db"].astype(float)
    rfsim_sinr_min = float(state_summary["ss_sinr_db_mean_execution_median"].min())
    rfsim_sinr_max = float(state_summary["ss_sinr_db_mean_execution_median"].max())
    upv_sinr_median = float(upv_sinr.median())
    closest_index = (
        (state_summary["ss_sinr_db_mean_execution_median"] - upv_sinr_median).abs().idxmin()
    )
    closest = state_summary.loc[closest_index]

    all_executions_pass = all(row["execution_pass"] for row in execution_results)
    monotonic_pass = all(row["pass"] for row in monotonic_results)
    control_valid = all(state_gates.values()) and all_executions_pass and monotonic_pass
    decision = "corrected_control_valid" if control_valid else "implementation_or_transport_invalid"
    next_action = (
        "freeze_separate_gain_noise_replay_protocol_using_development_data"
        if control_valid
        else "stop_before_replay_design_and_audit_failed_gates"
    )
    result = {
        "schema_version": 1,
        "stage": "corrected_rfsim_noise_response_evaluation",
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "state_gate_results": state_gates,
        "execution_results": execution_results,
        "monotonicity_results": monotonic_results,
        "development_comparison": {
            "development_rows": len(supported),
            "development_folds": int(supported["fold_id"].nunique()),
            "upv_route_sinr_db": {
                "minimum": float(upv_sinr.min()),
                "median": upv_sinr_median,
                "maximum": float(upv_sinr.max()),
            },
            "upv_route_relative_rsrp_db": {
                "minimum": float(upv_relative_rsrp.min()),
                "median": float(upv_relative_rsrp.median()),
                "maximum": float(upv_relative_rsrp.max()),
                "q05_to_q95_span": float(
                    upv_relative_rsrp.quantile(0.95) - upv_relative_rsrp.quantile(0.05)
                ),
            },
            "rfsim_validated_state_mean_sinr_db": {
                "minimum": rfsim_sinr_min,
                "maximum": rfsim_sinr_max,
            },
            "closest_validated_noise_state_to_upv_development_median": {
                "noise_power_db": float(closest["commanded_noise_power_db"]),
                "mean_execution_median_sinr_db": float(closest["ss_sinr_db_mean_execution_median"]),
                "absolute_gap_db": float(
                    abs(closest["ss_sinr_db_mean_execution_median"] - upv_sinr_median)
                ),
            },
            "upv_development_median_inside_validated_sinr_range": bool(
                rfsim_sinr_min <= upv_sinr_median <= rfsim_sinr_max
            ),
            "noise_only_rfsim_relative_rsrp_q05_to_q95_span_db": float(
                centered_rsrp.quantile(0.95) - centered_rsrp.quantile(0.05)
            ),
            "descriptive_only": True,
            "final_test6_accessed": False,
        },
        "control_gate_pass": control_valid,
        "decision_code": decision,
        "next_action": next_action,
        "fixed_noise_selection_authorized": False,
        "additional_reservation_should_be_requested_now": False,
        "abc_authorized": False,
        "claim_limits": spec["claim_limits"],
    }
    return result, execution_medians, state_summary, pairwise


def write_corrected_noise_response_evaluation(
    *,
    raw_archive_path: str | Path,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    protocol_path: str | Path,
    hardware_freeze_path: str | Path,
    analysis_spec_path: str | Path,
    development_route_means_path: str | Path,
    phase3d_decision_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"corrected-noise evaluation output already exists: {output}")
    result, execution_medians, state_summary, pairwise = evaluate_corrected_noise_response(
        raw_archive_path=raw_archive_path,
        telemetry_path=telemetry_path,
        execution_state_path=execution_state_path,
        protocol_path=protocol_path,
        hardware_freeze_path=hardware_freeze_path,
        analysis_spec_path=analysis_spec_path,
        development_route_means_path=development_route_means_path,
        phase3d_decision_path=phase3d_decision_path,
    )
    output.mkdir(parents=True)
    paths = {
        "evaluation": output / "corrected_noise_response_evaluation.json",
        "execution_medians": output / "execution_medians.csv",
        "state_summary": output / "state_summary.csv",
        "pairwise_differences": output / "pairwise_differences.csv",
    }
    paths["evaluation"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    execution_medians.to_csv(paths["execution_medians"], index=False)
    state_summary.to_csv(paths["state_summary"], index=False)
    pairwise.to_csv(paths["pairwise_differences"], index=False)
    checksums = {path.name: _sha256(path) for path in paths.values()}
    checksum_path = output / "SHA256SUMS.json"
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return {**{name: str(path) for name, path in paths.items()}, "checksums": str(checksum_path)}
