from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .upv_phase3d import (
    _load_development_sessions,
    _read_json,
    _read_yaml,
    _sha256,
    _write_csv,
    _write_json,
    validate_phase3d_config,
)


def validate_phase3g_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3G schema_version must be 1")
    if config.get("stage") != "phase_3g_direct_measured_trace_control_preparation":
        raise ValueError("unexpected Phase 3G stage")
    if any(
        bool(config.get(name))
        for name in ("execution_authorized", "final_evaluation_authorized", "abc_authorized")
    ):
        raise ValueError("Phase 3G preparation cannot authorize execution, final access, or ABC")
    frozen = config.get("frozen_inputs") or {}
    for name in (
        "archive_sha256",
        "phase3d_config_sha256",
        "phase3f_result_sha256",
        "scalar_control_result_sha256",
        "corrected_noise_result_sha256",
    ):
        value = str(frozen.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid Phase 3G frozen checksum: {name}")
    if frozen.get("required_phase3f_decision") != "cross_session_exchangeability_not_established":
        raise ValueError("Phase 3G requires the Phase 3F exchangeability result")
    target = config.get("target_trace") or {}
    if target.get("session_id") != "corrected_test_1_ASUS":
        raise ValueError("Phase 3G must use the predesignated Test 1 development trace")
    if target.get("final_test6_access") is not False:
        raise ValueError("Phase 3G must keep Test 6 inaccessible")
    if target.get("interpolation") != "prohibited":
        raise ValueError("Phase 3G cannot interpolate the measured trace")
    mapping = config.get("provisional_mapping") or {}
    if mapping.get("status") != "engineering_hypothesis_for_response_design_only":
        raise ValueError("the provisional mapping must remain an engineering hypothesis")
    if mapping.get("replay_authorized") is not False:
        raise ValueError("the provisional mapping cannot authorize replay")

    design = config.get("hardware_design") or {}
    if design.get("channel_family") != "AWGN":
        raise ValueError("the bounded response design must use AWGN")
    if design.get("safety_gain_states_db") != [-8.0, -12.0, -16.0, -18.0]:
        raise ValueError("unexpected gain safety states")
    if design.get("safety_noise_states_db") != [-18.0, -17.0]:
        raise ValueError("unexpected noise safety states")
    if design.get("factorial_gain_states_db") != [-8.0, -10.0, -12.0]:
        raise ValueError("unexpected factorial gain states")
    if design.get("factorial_noise_states_db") != [-28.0, -25.0, -22.0]:
        raise ValueError("unexpected factorial noise states")
    if int(design.get("factorial_executions_per_state", 0)) != 3:
        raise ValueError("the factorial requires three execution units per state")
    if int(design.get("boundary_executions_per_pair", 0)) != 3:
        raise ValueError("the boundary pairs require three execution units")

    gates = config.get("analysis_gates") or {}
    if float(gates.get("maximum_condition_number", 0)) != 10.0:
        raise ValueError("the Jacobian condition-number gate must remain 10")
    if int(gates.get("bootstrap_repetitions", 0)) < 1000:
        raise ValueError("the analysis requires at least 1000 execution bootstraps")
    claims = config.get("claim_limits") or {}
    if claims.get("direct_trace_replay_currently_authorized") is not False:
        raise ValueError("direct trace replay must remain unauthorized")
    prohibited = {
        name: value
        for name, value in claims.items()
        if name != "direct_trace_replay_currently_authorized"
    }
    if any(value != "prohibited" for value in prohibited.values()):
        raise ValueError("Phase 3G claim limits must remain prohibited")
    reservation = config.get("reservation") or {}
    if reservation.get("request_now") is not False:
        raise ValueError("a reservation cannot be requested before runner/profile freeze")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must remain at least 30 minutes")


def derive_provisional_controls(
    frame: pd.DataFrame,
    *,
    gain_response_slope: float,
    noise_intercept: float,
    noise_slope: float,
    gain_to_sinr_coefficient: float = 1.0,
) -> pd.DataFrame:
    required = {"relative_rsrp_db", "sinr_db"}
    if not required.issubset(frame.columns):
        raise ValueError("the target trace lacks relative RSRP or SINR")
    if gain_response_slope <= 0 or noise_slope >= 0:
        raise ValueError("unexpected provisional response directions")
    result = frame.copy()
    maximum_rsrp = float(result["relative_rsrp_db"].max())
    result["provisional_gain_db"] = (
        result["relative_rsrp_db"] - maximum_rsrp
    ) / gain_response_slope
    result["provisional_noise_power_db"] = (
        result["sinr_db"]
        - noise_intercept
        - gain_to_sinr_coefficient * result["provisional_gain_db"]
    ) / noise_slope
    return result


def _quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "q01": float(series.quantile(0.01)),
        "q05": float(series.quantile(0.05)),
        "median": float(series.median()),
        "q95": float(series.quantile(0.95)),
        "q99": float(series.quantile(0.99)),
        "maximum": float(series.max()),
    }


def _execution_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    design = config["hardware_design"]
    rows = []
    seed = 44001
    position = 1
    for gain in design["safety_gain_states_db"]:
        rows.append(
            {
                "stage": "gain_safety",
                "stage_position": position,
                "repetition": 1,
                "gain_db": float(gain),
                "noise_power_db": float(design["safety_gain_noise_power_db"]),
                "oai_rng_seed": seed,
            }
        )
        seed += 1
        position += 1
    position = 1
    for noise in design["safety_noise_states_db"]:
        rows.append(
            {
                "stage": "noise_safety",
                "stage_position": position,
                "repetition": 1,
                "gain_db": float(design["safety_noise_gain_db"]),
                "noise_power_db": float(noise),
                "oai_rng_seed": seed,
            }
        )
        seed += 1
        position += 1

    factorial = [
        (float(gain), float(noise))
        for gain in design["factorial_gain_states_db"]
        for noise in design["factorial_noise_states_db"]
    ]
    rng = np.random.default_rng(int(config["analysis_gates"]["bootstrap_seed"]))
    for repetition in range(1, int(design["factorial_executions_per_state"]) + 1):
        order = rng.permutation(len(factorial))
        for stage_position, index in enumerate(order, start=1):
            gain, noise = factorial[int(index)]
            rows.append(
                {
                    "stage": "factorial",
                    "stage_position": stage_position,
                    "repetition": repetition,
                    "gain_db": gain,
                    "noise_power_db": noise,
                    "oai_rng_seed": seed,
                }
            )
            seed += 1
    boundary = [tuple(map(float, value)) for value in design["boundary_pairs_gain_noise_db"]]
    for repetition in range(1, int(design["boundary_executions_per_pair"]) + 1):
        order = rng.permutation(len(boundary))
        for stage_position, index in enumerate(order, start=1):
            gain, noise = boundary[int(index)]
            rows.append(
                {
                    "stage": "boundary",
                    "stage_position": stage_position,
                    "repetition": repetition,
                    "gain_db": gain,
                    "noise_power_db": noise,
                    "oai_rng_seed": seed,
                }
            )
            seed += 1
    return rows


def prepare_phase3g_direct_trace(
    *,
    archive_path: str | Path,
    phase3d_config_path: str | Path,
    phase3f_result_path: str | Path,
    scalar_control_result_path: str | Path,
    corrected_noise_result_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    archive = Path(archive_path).resolve()
    phase3d_file = Path(phase3d_config_path).resolve()
    phase3f_file = Path(phase3f_result_path).resolve()
    scalar_file = Path(scalar_control_result_path).resolve()
    noise_file = Path(corrected_noise_result_path).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3G output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3g_config(config)
    phase3d = _read_yaml(phase3d_file)
    validate_phase3d_config(phase3d)
    paths = {
        "archive_sha256": archive,
        "phase3d_config_sha256": phase3d_file,
        "phase3f_result_sha256": phase3f_file,
        "scalar_control_result_sha256": scalar_file,
        "corrected_noise_result_sha256": noise_file,
    }
    for name, path in paths.items():
        if _sha256(path) != config["frozen_inputs"][name]:
            raise ValueError(f"Phase 3G frozen input checksum mismatch: {name}")
    phase3f = _read_json(phase3f_file)
    if phase3f.get("decision_code") != config["frozen_inputs"]["required_phase3f_decision"]:
        raise ValueError("unexpected Phase 3F result")
    scalar = _read_json(scalar_file)
    if (scalar.get("decision") or {}).get("deterministic_scalar_gate_state") != config[
        "frozen_inputs"
    ]["required_scalar_decision"]:
        raise ValueError("unexpected scalar-control result")
    noise = _read_json(noise_file)
    if noise.get("decision_code") != config["frozen_inputs"]["required_noise_decision"]:
        raise ValueError("unexpected corrected-noise result")
    if (noise.get("development_comparison") or {}).get("final_test6_accessed") is not False:
        raise ValueError("the corrected-noise result must keep Test 6 inaccessible")

    sessions, quality = _load_development_sessions(archive, phase3d)
    session_id = config["target_trace"]["session_id"]
    if session_id not in sessions:
        raise ValueError("the frozen direct-trace session is unavailable")
    if phase3d["final_evaluation"]["source_path"] in set(quality["source_path"]):
        raise AssertionError("the final payload entered Phase 3G")
    target = sessions[session_id].copy()
    slopes = [
        float(value["float_rsrp_transfer_slope"])
        for value in scalar["evaluation"]["replay_results"]
    ]
    gain_slope = float(np.mean(slopes))
    states = pd.DataFrame(noise["state_mean_execution_medians"])
    active = states.loc[states["noise_power_db"].isin([-40.0, -30.0, -25.0, -20.0])]
    noise_slope, noise_intercept = np.polyfit(
        active["noise_power_db"].to_numpy(float),
        active["ss_sinr_db"].to_numpy(float),
        1,
    )
    mapped = derive_provisional_controls(
        target,
        gain_response_slope=gain_slope,
        noise_intercept=float(noise_intercept),
        noise_slope=float(noise_slope),
        gain_to_sinr_coefficient=float(
            config["provisional_mapping"]["noise"]["assumed_gain_to_sinr_coefficient"]
        ),
    )
    plan = _execution_plan(config)
    stage_counts = pd.Series([value["stage"] for value in plan]).value_counts().to_dict()
    envelope = {
        "schema_version": 1,
        "stage": "phase_3g_direct_trace_control_envelope",
        "target_session_id": session_id,
        "target_rows": len(mapped),
        "target_duration_seconds": float(mapped["t_s"].max() - mapped["t_s"].min() + 1.0),
        "target_relative_rsrp_db": _quantiles(mapped["relative_rsrp_db"]),
        "target_sinr_db": _quantiles(mapped["sinr_db"]),
        "provisional_gain_db": _quantiles(mapped["provisional_gain_db"]),
        "provisional_noise_power_db": _quantiles(mapped["provisional_noise_power_db"]),
        "response_model": {
            "gain_response_slope": gain_slope,
            "noise_to_sinr_intercept": float(noise_intercept),
            "noise_to_sinr_slope": float(noise_slope),
            "assumed_gain_to_sinr_coefficient": 1.0,
            "status": "unvalidated_two_dimensional_engineering_hypothesis",
        },
        "hardware_plan": {
            "executions": len(plan),
            "stage_counts": {name: int(value) for name, value in stage_counts.items()},
            "execution_authorized": False,
            "runner_and_profile_freeze_required": True,
        },
        "direct_replay_authorized": False,
        "final_test6_accessed": False,
        "reservation_requested": False,
        "claim_limits": config["claim_limits"],
    }
    output.mkdir(parents=True)
    _write_csv(output / "direct_test1_target_trace.csv", mapped)
    _write_csv(output / "bounded_response_execution_plan.csv", pd.DataFrame(plan))
    _write_json(output / "control_envelope.json", envelope)
    _write_json(
        output / "response_protocol.json",
        {
            "schema_version": 1,
            "stage": "phase_3g_bounded_gain_noise_response_protocol",
            "hardware_design": config["hardware_design"],
            "valid_execution_rules": config["valid_execution_rules"],
            "analysis_gates": config["analysis_gates"],
            "stopping_rules": config["stopping_rules"],
            "decision_rules": config["decision_rules"],
            "reservation": config["reservation"],
            "execution_plan_rows": len(plan),
            "execution_authorized": False,
            "final_test6_accessed": False,
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
        "target_rows": str(len(mapped)),
        "hardware_plan_executions": str(len(plan)),
        "direct_replay_authorized": "false",
        "final_test6_accessed": "false",
        "reservation_requested": "false",
    }
