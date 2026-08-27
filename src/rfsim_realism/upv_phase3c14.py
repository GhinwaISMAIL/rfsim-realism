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

from .upv_phase3c13 import _attached, _segment_summary

FINGERPRINT = re.compile(r"^[0-9a-fA-F]{16}$")
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
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


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def validate_phase3c14_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3C14 schema_version must be 1")
    if config.get("stage") != "phase_3c14_execution_level_awgn_control":
        raise ValueError("unexpected Phase 3C14 stage")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("the offline Phase 3C14 freeze cannot authorize execution or ABC")

    frozen = config.get("frozen_inputs") or {}
    expected = {
        "oai_revision": "70508ebaf52f2aae420566d380c6537f2efb9f0c",
        "profile_revision": "90bfa2f470ba0f796123ba363bd27b4988a4518e",
        "profile_runner_sha256": (
            "01eb1944b6bdc5c88373612b4590b37343a93f1b76bc894ad916d60f45d6970b"
        ),
        "phase3c13_evaluation_sha256": (
            "19ea9a549edb152d3199d1dadca084f5852bbe4cb430540e5a807bbf6fd76859"
        ),
        "phase3c13_result_sha256": (
            "6121b2561efd61bcbe6d6496adfd182a848bb1092e08ddc8f1f52f9458f973c2"
        ),
        "phase3c13_decision": "static_tdlb_execution_variance_too_large",
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"unexpected frozen {key}")

    design = config.get("design") or {}
    if design.get("primary_channel_after_phase3c13") != "AWGN_plus_time_varying_scalar_gain":
        raise ValueError("AWGN must be the frozen primary channel after Phase 3C13")
    if design.get("static_TDL_B_status") != "sensitivity_analysis_only":
        raise ValueError("the Phase 3C13 TDL-B decision must remain explicit")
    if int(design.get("fresh_executions_required", 0)) != 5:
        raise ValueError("five fresh execution units are required")

    control = config.get("control") or {}
    if control.get("channel_family") != "AWGN":
        raise ValueError("the execution-level control must use AWGN")
    if float(control.get("fixed_noise_power_db", math.nan)) != -30.0:
        raise ValueError("noise must remain fixed at -30 dB")
    if control.get("gain_db_sequence") != [0.0, -2.0, -4.0, -2.0, 0.0]:
        raise ValueError("unexpected scalar gain sequence")
    if control.get("process_seeds") != [32001, 32002, 32003, 32004, 32005]:
        raise ValueError("unexpected process seed list")
    if int(control.get("independent_executions", 0)) != 5:
        raise ValueError("exactly five AWGN executions are required")
    if float(control.get("segment_duration_seconds", 0)) != 10.0:
        raise ValueError("each gain segment must last ten seconds")
    if float(control.get("settling_seconds_per_segment", -1)) != 3.0:
        raise ValueError("three settling seconds must be excluded per segment")

    acceptance = config.get("acceptance") or {}
    if int(acceptance.get("valid_executions_required", 0)) != 5:
        raise ValueError("the gate must require five valid executions")
    if float(acceptance.get("attachment_fraction_required", 0)) != 1.0:
        raise ValueError("continuous attachment is required")
    if int(acceptance.get("channel_length_required", 0)) != 1:
        raise ValueError("the AWGN identity path must have channel length one")
    if int(acceptance.get("nb_taps_required", 0)) != 1:
        raise ValueError("the AWGN identity path must have one tap")
    if float(acceptance.get("tap_energy_target", math.nan)) != 1.0:
        raise ValueError("the AWGN identity tap must have unit energy")
    slope = acceptance.get("float_rsrp_transfer_slope_range") or []
    if len(slope) != 2 or not float(slope[0]) <= 1.0 <= float(slope[1]):
        raise ValueError("the RSRP transfer-slope interval must contain one")
    if float(acceptance.get("between_execution_baseline_rsrp_range_max_db", 0)) <= 0:
        raise ValueError("the between-execution RSRP range gate must be positive")

    claims = config.get("claim_limits") or {}
    if claims.get("absolute_rsrp_calibration") != "prohibited":
        raise ValueError("absolute RSRP calibration remains prohibited")
    if claims.get("UPV_distribution_support") != "not_tested_by_this_control":
        raise ValueError("this control cannot claim UPV distribution support")
    if claims.get("abc") != "prohibited":
        raise ValueError("ABC remains prohibited")

    reservation = config.get("reservation") or {}
    if reservation.get("gate_state") != "closed_pending_protocol_freeze":
        raise ValueError("the offline reservation gate must remain closed")
    if bool(reservation.get("reservation_should_be_requested_now")):
        raise ValueError("the offline freeze cannot request a reservation")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def _validate_prerequisites(
    config: dict[str, Any], tdlb_evaluation: Path, tdlb_result: Path
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    if _sha256(tdlb_evaluation) != frozen["phase3c13_evaluation_sha256"]:
        raise ValueError("Phase 3C13 evaluation checksum mismatch")
    if _sha256(tdlb_result) != frozen["phase3c13_result_sha256"]:
        raise ValueError("Phase 3C13 result checksum mismatch")
    evaluation = _read_json(tdlb_evaluation)
    result = _read_json(tdlb_result)
    if evaluation.get("decision_code") != frozen["phase3c13_decision"]:
        raise ValueError("unexpected Phase 3C13 evaluation decision")
    if result.get("decision") != frozen["phase3c13_decision"]:
        raise ValueError("unexpected Phase 3C13 recorded decision")
    if evaluation.get("abc_authorized") is not False:
        raise ValueError("Phase 3C13 must not authorize ABC")
    return evaluation


def _validate_execution_identity(
    state: dict[str, Any],
    config: dict[str, Any],
    amended_identity: dict[str, str] | None = None,
) -> dict[str, bool]:
    frozen = config["frozen_inputs"]
    control = config["control"]
    summaries = state.get("replays") or []
    image_id = str(state.get("debug_image_id", ""))
    expected_profile = frozen["profile_revision"]
    expected_runner = frozen["profile_runner_sha256"]
    if amended_identity is not None:
        expected_profile = amended_identity["profile_revision"]
        expected_runner = amended_identity["runner_sha256"]
    gates = {
        "stage": state.get("stage") == "phase_3c14_awgn_execution_control",
        "execution_completed": state.get("execution_completed") is True,
        "error_absent": state.get("error") is None,
        "oai_revision": state.get("oai_revision") == frozen["oai_revision"],
        "profile_revision": state.get("profile_revision") == expected_profile,
        "runner_sha256": state.get("runner_sha256") == expected_runner,
        "channel_family": state.get("channel_family") == control["channel_family"],
        "process_seeds": state.get("rng_seeds") == control["process_seeds"],
        "replay_count": len(summaries) == int(control["independent_executions"]),
        "image_id_format": bool(IMAGE_ID.fullmatch(image_id)),
        "image_revision_label": state.get("debug_image_revision_label") == frozen["oai_revision"],
        "gNB_untouched": state.get("gNB_untouched") is True,
        "rollback": (state.get("rollback") or {}).get("passed") is True,
    }
    if amended_identity is not None:
        gates.update(
            {
                "original_image": state.get("original_image") == amended_identity["original_image"],
                "original_image_id": state.get("original_image_id")
                == amended_identity["original_image_id"],
                "compose_sha256": state.get("compose_sha256") == amended_identity["compose_sha256"],
                "channel_config_sha256": state.get("channel_config_sha256")
                == amended_identity["channel_config_sha256"],
                "ue_config_sha256": state.get("ue_config_sha256")
                == amended_identity["ue_config_sha256"],
            }
        )
    return {name: bool(value) for name, value in gates.items()}


def _identity_from_amendment(amendment: dict[str, Any]) -> dict[str, str]:
    if amendment.get("schema_version") != 1:
        raise ValueError("Phase 3C14 identity amendment schema_version must be 1")
    if amendment.get("stage") != "phase_3c14_awgn_rollback_identity_correction":
        raise ValueError("unexpected Phase 3C14 identity amendment stage")
    if amendment.get("decision") != "corrected_five_execution_control_authorized":
        raise ValueError("the corrected Phase 3C14 execution is not authorized")
    if amendment.get("scientific_design_unchanged") is not True:
        raise ValueError("the Phase 3C14 scientific design must remain unchanged")
    if amendment.get("thresholds_and_claim_limits_unchanged") is not True:
        raise ValueError("Phase 3C14 thresholds and claim limits must remain unchanged")
    identity = amendment.get("corrected_execution_identity") or {}
    expected = {
        "profile_revision": "cf27748ee6c3592cb4ee1581ac47bc50e52739ef",
        "runner_sha256": ("608009e9aeab7eedd4c4595452723db07d8950c57fbc2cc2c82c2e743fd9212f"),
        "original_image": "ghinwa555/oai-nr-ue-chan:v4",
        "original_image_id": (
            "sha256:7d66805b1da7bf6704821b9975b93af6db557874e6a0f17e12831da158a1f01f"
        ),
        "compose_sha256": ("db5aade37a4613a95c3f9682cdddf3bc5bc73d74f398c004105547c80b8d0260"),
        "channel_config_sha256": (
            "8814d9dd7f05ae96093a4f2a327e176f638a0fff8030136a844d1e0950179d72"
        ),
        "ue_config_sha256": ("d7f10f47440e67a9395391b11797473dc24a63c90d2faad9292c216fc3a6734e"),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"unexpected amended {key}")
    return expected


def evaluate_awgn_execution_control(
    *,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    config_path: str | Path,
    tdlb_evaluation_path: str | Path,
    tdlb_result_path: str | Path,
    identity_amendment_path: str | Path | None = None,
) -> dict[str, Any]:
    telemetry_file = _require_file(telemetry_path, "Phase 3C14 telemetry")
    state_file = _require_file(execution_state_path, "Phase 3C14 execution state")
    config_file = _require_file(config_path, "Phase 3C14 configuration")
    tdlb_evaluation_file = _require_file(tdlb_evaluation_path, "Phase 3C13 evaluation")
    tdlb_result_file = _require_file(tdlb_result_path, "Phase 3C13 result")
    config = _read_yaml(config_file)
    validate_phase3c14_config(config)
    tdlb_evaluation = _validate_prerequisites(config, tdlb_evaluation_file, tdlb_result_file)
    state = _read_json(state_file)
    amendment_file = (
        _require_file(identity_amendment_path, "Phase 3C14 identity amendment")
        if identity_amendment_path is not None
        else None
    )
    amended_identity = (
        _identity_from_amendment(_read_json(amendment_file)) if amendment_file is not None else None
    )
    state_gates = _validate_execution_identity(state, config, amended_identity)

    telemetry = pd.read_csv(
        telemetry_file,
        dtype={"tap_fingerprint_fnv1a64": str, "channel_snapshot_id": str},
    )
    required = set(config["acceptance"]["required_telemetry_fields"])
    missing = sorted(required - set(telemetry.columns))
    if missing:
        raise ValueError(f"Phase 3C14 telemetry is missing columns: {missing}")
    numeric = sorted(required - STRING_COLUMNS)
    for column in numeric:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    if not np.isfinite(telemetry[numeric].to_numpy(float)).all():
        raise ValueError("Phase 3C14 telemetry contains non-finite required values")
    telemetry["attached"] = _attached(telemetry["attached"])
    fingerprints = telemetry["tap_fingerprint_fnv1a64"].astype(str)
    if not fingerprints.map(lambda value: bool(FINGERPRINT.fullmatch(value))).all():
        raise ValueError("tap fingerprints must be 16 hexadecimal digits")

    control = config["control"]
    acceptance = config["acceptance"]
    expected_ids = [f"awgn-{index}" for index in range(1, 6)]
    expected_seed = dict(zip(expected_ids, control["process_seeds"], strict=True))
    observed_ids = sorted(telemetry["replay_id"].astype(str).unique())
    state_summaries = {str(row.get("replay_id")): row for row in state.get("replays") or []}
    segment_results: list[dict[str, Any]] = []
    replay_results: list[dict[str, Any]] = []
    for replay_id in observed_ids:
        replay = telemetry.loc[telemetry["replay_id"].astype(str).eq(replay_id)].copy()
        segments, metrics = _segment_summary(
            replay_id,
            replay,
            {
                "pilot": control,
                "acceptance": acceptance,
            },
        )
        segment_results.extend(segments)
        segment_frame = pd.DataFrame(segments)
        seeds = set(replay["oai_rng_seed"].astype(int))
        replay_fingerprints = set(replay["tap_fingerprint_fnv1a64"].astype(str))
        summary = state_summaries.get(replay_id) or {}
        slope_bounds = acceptance["float_rsrp_transfer_slope_range"]
        tolerance = float(acceptance["commanded_applied_gain_max_abs_error_db"])
        tap_error = float(
            (replay["tap_energy_linear"] - acceptance["tap_energy_target"]).abs().max()
        )
        gates = {
            "coverage": bool(segment_frame["coverage_pass"].all()),
            "commanded_gain": bool(
                segment_frame["commanded_gain_max_abs_error_db"].max() <= tolerance
            ),
            "applied_gain": bool(segment_frame["applied_gain_max_abs_error_db"].max() <= tolerance),
            "fixed_noise": bool(
                (replay["noise_power_db"] - control["fixed_noise_power_db"]).abs().max()
                <= tolerance
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
            "unit_tap_energy": tap_error <= float(acceptance["tap_energy_max_abs_error"]),
            "one_snapshot": replay["channel_snapshot_id"].astype(str).nunique()
            == int(acceptance["unique_snapshots_per_execution_required"]),
            "one_fingerprint": len(replay_fingerprints)
            == int(acceptance["unique_fingerprints_per_execution_required"]),
            "expected_seed": seeds == {expected_seed.get(replay_id)},
            "awgn_identity": set(replay["channel_family"].astype(str)) == {"AWGN"}
            and set(replay["channel_model_name"].astype(str)) == {"rfsimu_channel_enB0"}
            and set(replay["channel_length"].astype(int))
            == {int(acceptance["channel_length_required"])}
            and set(replay["nb_taps"].astype(int)) == {int(acceptance["nb_taps_required"])},
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
                "tap_energy_max_abs_error": tap_error,
                "attachment_fraction": float(replay["attached"].mean()),
                "gate_results": gates,
                "replay_pass": all(gates.values()),
            }
        )

    baseline_rsrp = [row["baseline_float_rsrp_median_db"] for row in replay_results]
    baseline_sinr = [row["baseline_sinr_median_db"] for row in replay_results]
    replay_fingerprints = [row["tap_fingerprint_fnv1a64"] for row in replay_results]
    rsrp_range = max(baseline_rsrp) - min(baseline_rsrp) if baseline_rsrp else math.nan
    sinr_range = max(baseline_sinr) - min(baseline_sinr) if baseline_sinr else math.nan
    identity_gates = {
        "exact_replay_identity": observed_ids == expected_ids,
        "one_common_awgn_fingerprint": len(set(replay_fingerprints))
        == int(acceptance["common_fingerprints_across_executions_required"]),
        "execution_state": all(state_gates.values()),
    }
    stability_gates = {
        "baseline_rsrp_range": bool(baseline_rsrp)
        and rsrp_range <= float(acceptance["between_execution_baseline_rsrp_range_max_db"]),
        "baseline_sinr_range": bool(baseline_sinr)
        and sinr_range <= float(acceptance["between_execution_baseline_sinr_range_max_db"]),
    }
    identity_gates = {name: bool(value) for name, value in identity_gates.items()}
    stability_gates = {name: bool(value) for name, value in stability_gates.items()}
    overall = (
        len(replay_results) == int(acceptance["valid_executions_required"])
        and all(row["replay_pass"] for row in replay_results)
        and all(identity_gates.values())
        and all(stability_gates.values())
    )
    if overall:
        decision_code = "awgn_execution_control_accepted"
        next_action = "proceed_to_relative_RSRP_and_SINR_distribution_support_analysis"
    elif not all(identity_gates.values()):
        decision_code = "awgn_identity_gate_failed"
        next_action = "stop_and_audit_generated_channel_and_debug_image_identity"
    elif not all(stability_gates.values()):
        decision_code = "awgn_execution_variance_too_large"
        next_action = "audit_uncontrolled_execution_state_before_distribution_support"
    else:
        decision_code = "awgn_execution_control_rejected"
        next_action = "stop_before_distribution_support_and_audit_the_scalar_runtime_path"

    tdlb_cross = tdlb_evaluation["cross_execution"]
    tdlb_rsrp_range = float(tdlb_cross["baseline_rsrp_range_db"])
    tdlb_sinr_range = float(tdlb_cross["baseline_sinr_range_db"])
    input_sha256 = {
        "telemetry": _sha256(telemetry_file),
        "execution_state": _sha256(state_file),
        "config": _sha256(config_file),
        "tdlb_evaluation": _sha256(tdlb_evaluation_file),
        "tdlb_result": _sha256(tdlb_result_file),
    }
    if amendment_file is not None:
        input_sha256["identity_amendment"] = _sha256(amendment_file)
    return {
        "schema_version": 1,
        "stage": "phase_3c14_awgn_execution_control_evaluation",
        "input_sha256": input_sha256,
        "state_gate_results": state_gates,
        "segment_results": segment_results,
        "replay_results": replay_results,
        "cross_execution": {
            "baseline_rsrp_range_db": rsrp_range,
            "baseline_sinr_range_db": sinr_range,
            "identity_gate_results": identity_gates,
            "stability_gate_results": stability_gates,
        },
        "static_tdlb_comparison": {
            "tdlb_baseline_rsrp_range_db": tdlb_rsrp_range,
            "tdlb_baseline_sinr_range_db": tdlb_sinr_range,
            "awgn_to_tdlb_baseline_rsrp_range_ratio": (
                rsrp_range / tdlb_rsrp_range if tdlb_rsrp_range > 0 else math.nan
            ),
            "awgn_to_tdlb_baseline_sinr_range_ratio": (
                sinr_range / tdlb_sinr_range if tdlb_sinr_range > 0 else math.nan
            ),
            "descriptive_only": True,
        },
        "control_gate_pass": overall,
        "decision_code": decision_code,
        "next_action": next_action,
        "additional_reservation_should_be_requested_now": False,
        "abc_authorized": False,
        "claim_limits": config["claim_limits"],
    }


def write_awgn_execution_control_evaluation(
    *,
    telemetry_path: str | Path,
    execution_state_path: str | Path,
    config_path: str | Path,
    tdlb_evaluation_path: str | Path,
    tdlb_result_path: str | Path,
    output_path: str | Path,
    identity_amendment_path: str | Path | None = None,
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3C14 evaluation already exists: {output}")
    result = evaluate_awgn_execution_control(
        telemetry_path=telemetry_path,
        execution_state_path=execution_state_path,
        config_path=config_path,
        tdlb_evaluation_path=tdlb_evaluation_path,
        tdlb_result_path=tdlb_result_path,
        identity_amendment_path=identity_amendment_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output
