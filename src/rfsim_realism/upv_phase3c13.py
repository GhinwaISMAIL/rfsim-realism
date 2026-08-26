from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

FINGERPRINT = re.compile(r"^[0-9a-fA-F]{16}$")
STRING_COLUMNS = {
    "replay_id",
    "channel_family",
    "channel_model_name",
    "channel_snapshot_id",
    "tap_fingerprint_fnv1a64",
    "attached",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"missing or unsafe {label}: {resolved}")
    return resolved


def validate_phase3c13_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3C13 schema_version must be 1")
    if config.get("stage") != "phase_3c13_static_tdlb_scalar_pilot":
        raise ValueError("unexpected Phase 3C13 stage")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("the offline Phase 3C13 freeze cannot authorize execution or ABC")

    frozen = config.get("frozen_inputs") or {}
    expected = {
        "oai_revision": "70508ebaf52f2aae420566d380c6537f2efb9f0c",
        "profile_revision": "181f9daa5d633ffa2cc74cb72e233387bef37298",
        "profile_runner_sha256": (
            "eb5bb8edaa25030bd35589a5a4313593529b3be1317e35e9306c4867b1f152a1"
        ),
        "phase3c1_awgn_result_sha256": (
            "491b6c933032f28f36803343f4ef4f03d51086f91ce655e087b6ba115971b68b"
        ),
        "phase3c1_awgn_telemetry_sha256": (
            "12ce943762697fbc6f16b3f1151cdea8f8095ad47db6a07652813db71190b61b"
        ),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"unexpected frozen {key}")

    design = config.get("design") or {}
    if design.get("control") != "phase3c1_static_awgn_scalar_transfer":
        raise ValueError("AWGN must remain the structural control")
    if design.get("primary_candidate") != "static_TDL_B_plus_time_varying_scalar_gain":
        raise ValueError("unexpected primary channel candidate")
    if design.get("continuous_multipath_or_doppler") != "not_modelled":
        raise ValueError("the static TDL-B claim limit must be explicit")

    pilot = config.get("pilot") or {}
    if pilot.get("channel_family") != "TDL_B":
        raise ValueError("the pilot channel family must be TDL_B")
    if int(pilot.get("tdl_rms_delay_spread_ns", 0)) != 30:
        raise ValueError("the TDL-B RMS delay spread must be 30 ns")
    if float(pilot.get("fixed_noise_power_db", math.nan)) != -30.0:
        raise ValueError("noise must remain fixed at -30 dB")
    if pilot.get("gain_db_sequence") != [0.0, -2.0, -4.0, -2.0, 0.0]:
        raise ValueError("unexpected scalar gain sequence")
    if pilot.get("rng_seeds") != [31001, 31002, 31003, 31004, 31005]:
        raise ValueError("unexpected or incomplete independent RNG seed list")
    if int(pilot.get("independent_executions", 0)) != 5:
        raise ValueError("exactly five independent TDL-B executions are required")
    if float(pilot.get("segment_duration_seconds", 0)) != 10.0:
        raise ValueError("each gain segment must last 10 seconds")
    if float(pilot.get("settling_seconds_per_segment", -1)) != 3.0:
        raise ValueError("each gain segment must exclude three settling seconds")

    acceptance = config.get("acceptance") or {}
    if int(acceptance.get("valid_executions_required", 0)) != 5:
        raise ValueError("the acceptance gate must require five valid executions")
    if float(acceptance.get("attachment_fraction_required", 0)) != 1.0:
        raise ValueError("continuous attachment is required")
    if int(acceptance.get("critical_failure_count_maximum", -1)) != 0:
        raise ValueError("critical failure markers must have zero tolerance")
    if int(acceptance.get("distinct_fingerprints_required", 0)) != 5:
        raise ValueError("five distinct channel fingerprints are required")
    slope = acceptance.get("float_rsrp_transfer_slope_range") or []
    if len(slope) != 2 or not float(slope[0]) <= 1.0 <= float(slope[1]):
        raise ValueError("the RSRP transfer-slope interval must contain one")
    if float(acceptance.get("between_execution_baseline_rsrp_range_max_db", 0)) <= 0:
        raise ValueError("the between-execution RSRP stability gate must be positive")

    claims = config.get("claim_limits") or {}
    if claims.get("absolute_rsrp_calibration") != "prohibited":
        raise ValueError("absolute RSRP calibration remains prohibited")
    if claims.get("fully_time_varying_tdl_b") != "prohibited":
        raise ValueError("fully time-varying TDL-B claims remain prohibited")
    if claims.get("mcs_bler_realism") != "not_tested_by_this_pilot":
        raise ValueError("this pilot does not establish MCS/BLER realism")

    reservation = config.get("reservation") or {}
    if reservation.get("gate_state") != "open_for_build_and_smoke_only" or not bool(
        reservation.get("reservation_should_be_requested_now")
    ):
        raise ValueError("Phase 3C13 must request only the build-and-smoke reservation")
    if reservation.get("pilot_execution_authorized") is not False:
        raise ValueError("the five-execution pilot remains unauthorized before the build")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def _attached(series: pd.Series) -> pd.Series:
    result = (
        series.astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )
    if result.isna().any():
        raise ValueError("attached must contain only true/false or 1/0")
    return result.astype(bool)


def _state_gates(state: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    frozen = config["frozen_inputs"]
    pilot = config["pilot"]
    summaries = state.get("replays") or []
    return {
        "stage": state.get("stage") == "phase_3c13_static_tdlb_scalar_pilot_execution",
        "execution_completed": state.get("execution_completed") is True,
        "error_absent": state.get("error") is None,
        "oai_revision": state.get("oai_revision") == frozen["oai_revision"],
        "profile_revision": state.get("profile_revision") == frozen["profile_revision"],
        "runner_sha256": state.get("runner_sha256") == frozen["profile_runner_sha256"],
        "channel_family": state.get("channel_family") == pilot["channel_family"],
        "delay_spread": int(state.get("tdl_rms_delay_spread_ns", -1))
        == int(pilot["tdl_rms_delay_spread_ns"]),
        "rng_seeds": state.get("rng_seeds") == pilot["rng_seeds"],
        "replay_count": len(summaries) == int(pilot["independent_executions"]),
        "distinct_fingerprints": int(state.get("distinct_fingerprint_count", -1))
        == int(config["acceptance"]["distinct_fingerprints_required"]),
        "rollback": (state.get("rollback") or {}).get("passed") is True,
        "gnb_untouched": state.get("gNB_untouched") is True,
    }


def _segment_summary(
    replay_id: str,
    replay: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pilot = config["pilot"]
    acceptance = config["acceptance"]
    rows: list[dict[str, Any]] = []
    duration = float(pilot["segment_duration_seconds"])
    settling = float(pilot["settling_seconds_per_segment"])
    for index, (label, gain) in enumerate(
        zip(pilot["segment_labels"], pilot["gain_db_sequence"], strict=True)
    ):
        selected = replay.loc[
            replay["t_s"].ge(index * duration + settling) & replay["t_s"].lt((index + 1) * duration)
        ]
        coverage = len(selected) >= int(acceptance["minimum_analysis_rows_per_segment"])
        rows.append(
            {
                "replay_id": replay_id,
                "segment_index": index,
                "segment_label": label,
                "expected_gain_db": float(gain),
                "analysis_rows": len(selected),
                "coverage_pass": coverage,
                "commanded_gain_max_abs_error_db": (
                    float((selected["commanded_gain_db"] - gain).abs().max())
                    if coverage
                    else math.inf
                ),
                "applied_gain_max_abs_error_db": (
                    float((selected["applied_gain_db"] - gain).abs().max())
                    if coverage
                    else math.inf
                ),
                "float_rsrp_median_db": (
                    float(selected["rsrp_db_per_re_unquantized"].median()) if coverage else math.nan
                ),
                "integer_rsrp_median_dbm": (
                    float(selected["ss_rsrp_dbm_integer"].median()) if coverage else math.nan
                ),
                "sinr_median_db": (
                    float(selected["ss_sinr_db"].median()) if coverage else math.nan
                ),
            }
        )
    segments = pd.DataFrame(rows)
    baseline_float = float(
        segments.loc[segments["expected_gain_db"].eq(0), "float_rsrp_median_db"].mean()
    )
    baseline_integer = float(
        segments.loc[segments["expected_gain_db"].eq(0), "integer_rsrp_median_dbm"].mean()
    )
    baseline_sinr = float(segments.loc[segments["expected_gain_db"].eq(0), "sinr_median_db"].mean())
    x = segments["expected_gain_db"].to_numpy(float)
    y = segments["float_rsrp_median_db"].to_numpy(float) - baseline_float
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    integer_relative = segments["integer_rsrp_median_dbm"].to_numpy(float) - baseline_integer
    hysteresis = float(
        max(
            float(group["float_rsrp_median_db"].max()) - float(group["float_rsrp_median_db"].min())
            for _, group in segments.groupby("expected_gain_db", sort=True)
        )
    )
    return rows, {
        "baseline_float_rsrp_median_db": baseline_float,
        "baseline_sinr_median_db": baseline_sinr,
        "float_rsrp_transfer_slope": float(slope),
        "float_rsrp_transfer_intercept_db": float(intercept),
        "float_rsrp_transfer_r_squared": r_squared,
        "float_rsrp_delta_max_abs_error_db": float(np.max(np.abs(y - x))),
        "integer_rsrp_delta_max_abs_error_db": float(np.max(np.abs(integer_relative - x))),
        "repeated_level_hysteresis_max_abs_db": hysteresis,
    }


def evaluate_static_tdlb_pilot(
    *,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    telemetry_file = _require_file(telemetry_path, "Phase 3C13 telemetry")
    state_file = _require_file(execution_state_path, "Phase 3C13 execution state")
    config_file = _require_file(config_path, "Phase 3C13 configuration")
    config = _read_yaml(config_file)
    validate_phase3c13_config(config)
    state = _read_json(state_file)
    state_gates = _state_gates(state, config)
    telemetry = pd.read_csv(
        telemetry_file,
        dtype={
            "tap_fingerprint_fnv1a64": str,
            "channel_snapshot_id": str,
        },
    )
    required = set(config["acceptance"]["required_telemetry_fields"])
    missing = sorted(required - set(telemetry.columns))
    if missing:
        raise ValueError(f"Phase 3C13 telemetry is missing columns: {missing}")
    numeric = sorted(required - STRING_COLUMNS)
    for column in numeric:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    if not np.isfinite(telemetry[numeric].to_numpy(float)).all():
        raise ValueError("Phase 3C13 telemetry contains non-finite required values")
    telemetry["attached"] = _attached(telemetry["attached"])
    fingerprints = telemetry["tap_fingerprint_fnv1a64"].astype(str)
    if not fingerprints.map(lambda value: bool(FINGERPRINT.fullmatch(value))).all():
        raise ValueError("tap fingerprints must be 16 hexadecimal digits")

    pilot = config["pilot"]
    acceptance = config["acceptance"]
    expected_ids = [f"tdlb-{index}" for index in range(1, 6)]
    expected_seed = dict(zip(expected_ids, pilot["rng_seeds"], strict=True))
    observed_ids = sorted(telemetry["replay_id"].astype(str).unique())
    state_summaries = {str(row.get("replay_id")): row for row in state.get("replays") or []}
    segment_results: list[dict[str, Any]] = []
    replay_results: list[dict[str, Any]] = []
    for replay_id in observed_ids:
        replay = telemetry.loc[telemetry["replay_id"].astype(str).eq(replay_id)].copy()
        segments, metrics = _segment_summary(replay_id, replay, config)
        segment_results.extend(segments)
        segment_frame = pd.DataFrame(segments)
        tap_mean = float(replay["tap_energy_linear"].mean())
        tap_cv = (
            float(replay["tap_energy_linear"].std(ddof=0) / tap_mean) if tap_mean > 0 else math.inf
        )
        seeds = set(replay["oai_rng_seed"].astype(int))
        replay_fingerprints = set(replay["tap_fingerprint_fnv1a64"].astype(str))
        summary = state_summaries.get(replay_id) or {}
        slope_bounds = acceptance["float_rsrp_transfer_slope_range"]
        tolerance = float(acceptance["commanded_applied_gain_max_abs_error_db"])
        gates = {
            "coverage": bool(segment_frame["coverage_pass"].all()),
            "commanded_gain": bool(
                segment_frame["commanded_gain_max_abs_error_db"].max() <= tolerance
            ),
            "applied_gain": bool(segment_frame["applied_gain_max_abs_error_db"].max() <= tolerance),
            "fixed_noise": bool(
                (replay["noise_power_db"] - pilot["fixed_noise_power_db"]).abs().max() <= tolerance
            ),
            "attachment": float(replay["attached"].mean())
            >= float(acceptance["attachment_fraction_required"]),
            "float_slope": float(slope_bounds[0])
            <= metrics["float_rsrp_transfer_slope"]
            <= float(slope_bounds[1]),
            "float_r_squared": metrics["float_rsrp_transfer_r_squared"]
            >= float(acceptance["float_rsrp_transfer_r_squared_minimum"]),
            "float_delta": metrics["float_rsrp_delta_max_abs_error_db"]
            <= float(acceptance["float_rsrp_delta_max_abs_error_db"]),
            "integer_delta": metrics["integer_rsrp_delta_max_abs_error_db"]
            <= float(acceptance["integer_rsrp_delta_max_abs_error_db"]),
            "hysteresis": metrics["repeated_level_hysteresis_max_abs_db"]
            <= float(acceptance["repeated_level_hysteresis_max_abs_db"]),
            "tap_energy_static": tap_cv
            <= float(acceptance["tap_energy_coefficient_of_variation_maximum"]),
            "one_snapshot": replay["channel_snapshot_id"].astype(str).nunique() == 1,
            "one_fingerprint": len(replay_fingerprints) == 1,
            "expected_seed": seeds == {expected_seed.get(replay_id)},
            "tdlb_identity": set(replay["channel_family"].astype(str)) == {"TDL_B"}
            and set(replay["channel_model_name"].astype(str)) == {"rfsimu_channel_enB0"},
            "multipath_dimensions": int(replay["channel_length"].min())
            >= int(acceptance["channel_length_minimum"])
            and int(replay["nb_taps"].min()) >= int(acceptance["nb_taps_minimum"]),
            "state_summary": summary.get("continuous_attachment") is True
            and summary.get("operational_runtime_pass") is True
            and int(summary.get("ue_restart_count", -1))
            <= int(acceptance["ue_restart_count_maximum"])
            and summary.get("gnb_health") == acceptance["gnb_health_required"]
            and int(summary.get("critical_failure_count", -1))
            <= int(acceptance["critical_failure_count_maximum"])
            and summary.get("tap_fingerprint_fnv1a64")
            == (next(iter(replay_fingerprints)) if len(replay_fingerprints) == 1 else None)
            and int(summary.get("oai_rng_seed", -1)) == expected_seed.get(replay_id),
        }
        gates = {name: bool(value) for name, value in gates.items()}
        replay_results.append(
            {
                "replay_id": replay_id,
                "oai_rng_seed": next(iter(seeds)) if len(seeds) == 1 else None,
                "tap_fingerprint_fnv1a64": (
                    next(iter(replay_fingerprints)) if len(replay_fingerprints) == 1 else None
                ),
                **metrics,
                "tap_energy_coefficient_of_variation": tap_cv,
                "attachment_fraction": float(replay["attached"].mean()),
                "gate_results": gates,
                "replay_pass": all(gates.values()),
            }
        )

    baseline_rsrp = [row["baseline_float_rsrp_median_db"] for row in replay_results]
    baseline_sinr = [row["baseline_sinr_median_db"] for row in replay_results]
    replay_fingerprints = [row["tap_fingerprint_fnv1a64"] for row in replay_results]
    cross_gates = {
        "exact_replay_identity": observed_ids == expected_ids,
        "five_distinct_fingerprints": len(set(replay_fingerprints))
        == int(acceptance["distinct_fingerprints_required"]),
        "baseline_rsrp_range": bool(baseline_rsrp)
        and max(baseline_rsrp) - min(baseline_rsrp)
        <= float(acceptance["between_execution_baseline_rsrp_range_max_db"]),
        "baseline_sinr_range": bool(baseline_sinr)
        and max(baseline_sinr) - min(baseline_sinr)
        <= float(acceptance["between_execution_baseline_sinr_range_max_db"]),
        "execution_state": all(state_gates.values()),
    }
    cross_gates = {name: bool(value) for name, value in cross_gates.items()}
    overall = (
        len(replay_results) == int(acceptance["valid_executions_required"])
        and all(row["replay_pass"] for row in replay_results)
        and all(cross_gates.values())
    )
    if overall:
        decision_code = "static_tdlb_scalar_candidate_accepted"
        next_action = "complete_the_execution_level_AWGN_control_before_distribution_claims"
    elif not cross_gates["five_distinct_fingerprints"]:
        decision_code = "static_tdlb_realization_gate_failed"
        next_action = "audit_RNG_and_treat_static_TDL_B_as_uncontrolled"
    elif not cross_gates["baseline_rsrp_range"] or not cross_gates["baseline_sinr_range"]:
        decision_code = "static_tdlb_execution_variance_too_large"
        next_action = "retain_TDL_B_as_sensitivity_analysis_only"
    else:
        decision_code = "static_tdlb_scalar_candidate_rejected"
        next_action = "retain_AWGN_as_primary_and_stop_static_TDL_B_escalation"
    return {
        "schema_version": 1,
        "stage": "phase_3c13_static_tdlb_scalar_pilot_evaluation",
        "input_sha256": {
            "telemetry": _sha256(telemetry_file),
            "execution_state": _sha256(state_file),
            "config": _sha256(config_file),
        },
        "state_gate_results": state_gates,
        "segment_results": segment_results,
        "replay_results": replay_results,
        "cross_execution": {
            "baseline_rsrp_range_db": (
                max(baseline_rsrp) - min(baseline_rsrp) if baseline_rsrp else math.nan
            ),
            "baseline_sinr_range_db": (
                max(baseline_sinr) - min(baseline_sinr) if baseline_sinr else math.nan
            ),
            "gate_results": cross_gates,
        },
        "pilot_gate_pass": overall,
        "decision_code": decision_code,
        "next_action": next_action,
        "additional_reservation_should_be_requested_now": False,
        "abc_authorized": False,
        "claim_limits": config["claim_limits"],
    }


def write_static_tdlb_pilot_evaluation(
    *,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3C13 evaluation already exists: {output}")
    result = evaluate_static_tdlb_pilot(
        telemetry_path=telemetry_path,
        execution_state_path=execution_state_path,
        config_path=config_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output
