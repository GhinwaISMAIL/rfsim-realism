from __future__ import annotations

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
from .upv_phase3g import validate_phase3g_config

RESPONSE_COLUMNS = ("rsrp_db_per_re_unquantized", "ss_sinr_db")


def _design_matrix(frame: pd.DataFrame, gain_center: float, noise_center: float) -> np.ndarray:
    gain = frame["commanded_gain_db"].to_numpy(float) - gain_center
    noise = frame["commanded_noise_power_db"].to_numpy(float) - noise_center
    return np.column_stack((np.ones(len(frame)), gain, noise, gain * noise))


def _fit_response(
    frame: pd.DataFrame, gain_center: float, noise_center: float
) -> tuple[np.ndarray, np.ndarray]:
    design = _design_matrix(frame, gain_center, noise_center)
    response = frame[list(RESPONSE_COLUMNS)].to_numpy(float)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients, design @ coefficients


def _condition_number(coefficients: np.ndarray) -> float:
    jacobian = np.array(
        [
            [coefficients[1, 0], coefficients[2, 0]],
            [coefficients[1, 1], coefficients[2, 1]],
        ]
    )
    return float(np.linalg.cond(jacobian))


def _validate_campaign(
    campaign: Path,
    telemetry: pd.DataFrame,
    state: dict[str, Any],
    direct_config: dict[str, Any],
    execution_config: dict[str, Any],
) -> None:
    if state.get("execution_completed") is not True or state.get("error") is not None:
        raise ValueError("the bounded response campaign did not complete cleanly")
    if state.get("failed_execution") is not None:
        raise ValueError("the bounded response campaign contains a failed execution")
    if int(state.get("completed_execution_count", 0)) != 45:
        raise ValueError("the bounded response campaign must contain 45 executions")
    if int(state.get("planned_execution_count", 0)) != 45:
        raise ValueError("the frozen execution plan must contain 45 executions")
    if state.get("final_test6_accessed") is not False:
        raise ValueError("the pre-designated final Test 6 payload was accessed")
    if state.get("abc_authorized") is not False:
        raise ValueError("ABC must remain unauthorized")
    if state.get("direct_trace_replay_authorized") is not False:
        raise ValueError("direct trace replay cannot be authorized by the hardware runner")
    if state.get("gNB_untouched") is not True:
        raise ValueError("the campaign changed the gNB")
    rollback = state.get("rollback") or {}
    if rollback.get("passed") is not True or rollback.get("attached") is not True:
        raise ValueError("the campaign rollback did not restore an attached UE")
    if rollback.get("restored_image_id") != rollback.get("expected_image_id"):
        raise ValueError("the rollback image identity does not match")

    frozen_profile = execution_config["frozen_profile"]
    frozen_oai = execution_config["frozen_oai"]
    frozen_research = execution_config["frozen_research"]
    expected = {
        "profile_revision": frozen_profile["revision"],
        "runner_sha256": frozen_profile["runner_sha256"],
        "execution_plan_sha256": frozen_profile["execution_plan_sha256"],
        "oai_revision": frozen_oai["revision"],
        "research_revision": frozen_research["repository_revision"],
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(f"campaign provenance mismatch: {name}")
    if state.get("research_protocol_sha256") != _sha256(Path(direct_config["_config_path"])):
        raise ValueError("campaign research protocol checksum mismatch")

    required = {
        "execution_index",
        "execution_id",
        "stage",
        "repetition",
        "commanded_gain_db",
        "applied_gain_db",
        "commanded_noise_power_db",
        "applied_noise_power_db",
        "channel_family",
        "channel_length",
        "nb_taps",
        "tap_energy_linear",
        "attached",
        *RESPONSE_COLUMNS,
    }
    if not required.issubset(telemetry.columns):
        raise ValueError("campaign telemetry is missing required fields")
    if len(telemetry) != 675:
        raise ValueError("campaign telemetry must contain 675 paired rows")
    if telemetry["execution_index"].nunique() != 45:
        raise ValueError("campaign telemetry does not contain 45 execution units")
    if not telemetry["attached"].astype(bool).all():
        raise ValueError("campaign telemetry contains detached samples")
    if set(telemetry["channel_family"]) != {"AWGN"}:
        raise ValueError("campaign telemetry contains a non-AWGN channel")
    if set(telemetry["channel_length"].astype(int)) != {1}:
        raise ValueError("campaign telemetry has an unexpected channel length")
    if set(telemetry["nb_taps"].astype(int)) != {1}:
        raise ValueError("campaign telemetry has an unexpected tap count")
    if not np.allclose(telemetry["tap_energy_linear"], 1.0):
        raise ValueError("campaign telemetry has an unexpected tap energy")
    if not np.allclose(telemetry["commanded_gain_db"], telemetry["applied_gain_db"]):
        raise ValueError("applied gain does not match the command")
    if not np.allclose(telemetry["commanded_noise_power_db"], telemetry["applied_noise_power_db"]):
        raise ValueError("applied noise does not match the command")
    counts = telemetry.groupby("execution_index").size()
    if not (counts == 15).all():
        raise ValueError("each execution must contain exactly 15 paired samples")

    for execution in state["executions"]:
        execution_id = execution["execution_id"]
        for suffix, checksum_field in (("ue.log", "ue_log_sha256"), ("gnb.log", "gnb_log_sha256")):
            path = campaign / f"{execution_id}-{suffix}"
            if not path.is_file() or _sha256(path) != execution[checksum_field]:
                raise ValueError(f"campaign log checksum mismatch: {path.name}")
        if execution.get("continuous_attachment") is not True:
            raise ValueError(f"execution lost attachment: {execution_id}")
        if int(execution.get("paired_radio_samples", 0)) != 15:
            raise ValueError(f"execution has incomplete telemetry: {execution_id}")
        if float(execution.get("ping_success_fraction", 0.0)) != 1.0:
            raise ValueError(f"execution has packet loss: {execution_id}")


def _bootstrap_response(
    factorial: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    gain_center: float,
    noise_center: float,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factorial_groups = [
        group.sort_values("execution_index")[list(RESPONSE_COLUMNS)].to_numpy(float)
        for _, group in factorial.groupby(
            ["commanded_gain_db", "commanded_noise_power_db"], sort=True
        )
    ]
    boundary_groups = [
        group.sort_values("execution_index")[list(RESPONSE_COLUMNS)].to_numpy(float)
        for _, group in boundary.groupby(
            ["commanded_gain_db", "commanded_noise_power_db"], sort=True
        )
    ]
    factorial_values = np.stack(factorial_groups)
    boundary_values = np.stack(boundary_groups)
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

    rng = np.random.default_rng(seed)
    factorial_choices = rng.integers(0, 3, size=(repetitions, 9, 3))
    factorial_samples = factorial_values[np.arange(9)[None, :, None], factorial_choices].mean(
        axis=2
    )
    coefficients = np.einsum("ij,bjk->bik", projection, factorial_samples)

    jacobians = np.empty((repetitions, 2, 2), dtype=float)
    jacobians[:, 0, 0] = coefficients[:, 1, 0]
    jacobians[:, 0, 1] = coefficients[:, 2, 0]
    jacobians[:, 1, 0] = coefficients[:, 1, 1]
    jacobians[:, 1, 1] = coefficients[:, 2, 1]
    condition_numbers = np.linalg.cond(jacobians)

    boundary_choices = rng.integers(0, 3, size=(repetitions, 4, 3))
    boundary_samples = boundary_values[np.arange(4)[None, :, None], boundary_choices].mean(axis=2)
    boundary_predictions = np.einsum("ij,bjk->bik", boundary_design, coefficients)
    maximum_boundary_errors = np.abs(boundary_samples - boundary_predictions).max(axis=1)
    return coefficients, condition_numbers, maximum_boundary_errors


def analyze_phase3g_bounded_response(
    *,
    campaign_dir: str | Path,
    archive_path: str | Path,
    direct_config_path: str | Path,
    execution_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    campaign = Path(campaign_dir).resolve()
    archive = Path(archive_path).resolve()
    direct_file = Path(direct_config_path).resolve()
    execution_file = Path(execution_config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3G response output already exists: {output}")
    direct_config = _read_yaml(direct_file)
    direct_config["_config_path"] = str(direct_file)
    validate_phase3g_config(direct_config)
    execution_config = _read_yaml(execution_file)
    if execution_config.get("stage") != "phase_3g_bounded_gain_noise_response_execution":
        raise ValueError("unexpected Phase 3G execution stage")

    state_file = campaign / "execution_state.json"
    telemetry_file = campaign / "phase3g_bounded_response_telemetry.csv"
    state = _read_json(state_file)
    telemetry = pd.read_csv(telemetry_file)
    _validate_campaign(campaign, telemetry, state, direct_config, execution_config)

    medians = (
        telemetry.groupby(
            [
                "execution_index",
                "execution_id",
                "stage",
                "stage_position",
                "repetition",
                "oai_rng_seed",
                "commanded_gain_db",
                "commanded_noise_power_db",
            ],
            as_index=False,
        )[list(RESPONSE_COLUMNS)]
        .median()
        .sort_values("execution_index")
    )
    factorial = medians.loc[medians["stage"] == "factorial"].copy()
    boundary = medians.loc[medians["stage"] == "boundary"].copy()
    if len(factorial) != 27 or len(boundary) != 12:
        raise ValueError("campaign stage counts do not match the frozen design")
    factorial_counts = factorial.groupby(["commanded_gain_db", "commanded_noise_power_db"]).size()
    boundary_counts = boundary.groupby(["commanded_gain_db", "commanded_noise_power_db"]).size()
    if len(factorial_counts) != 9 or not (factorial_counts == 3).all():
        raise ValueError("the factorial does not contain three executions per state")
    if len(boundary_counts) != 4 or not (boundary_counts == 3).all():
        raise ValueError("the boundary design does not contain three executions per state")

    design = direct_config["hardware_design"]
    gates = direct_config["analysis_gates"]
    gain_center = float(np.mean(design["factorial_gain_states_db"]))
    noise_center = float(np.mean(design["factorial_noise_states_db"]))
    coefficients, factorial_predictions = _fit_response(factorial, gain_center, noise_center)
    condition_number = _condition_number(coefficients)
    factorial["predicted_rsrp_db_per_re_unquantized"] = factorial_predictions[:, 0]
    factorial["predicted_ss_sinr_db"] = factorial_predictions[:, 1]
    factorial["relative_rsrp_db"] = factorial["rsrp_db_per_re_unquantized"] - coefficients[0, 0]

    boundary_design = _design_matrix(boundary, gain_center, noise_center)
    boundary_predictions = boundary_design @ coefficients
    boundary["predicted_rsrp_db_per_re_unquantized"] = boundary_predictions[:, 0]
    boundary["predicted_ss_sinr_db"] = boundary_predictions[:, 1]
    boundary["rsrp_error_db"] = (
        boundary["rsrp_db_per_re_unquantized"] - boundary["predicted_rsrp_db_per_re_unquantized"]
    )
    boundary["sinr_error_db"] = boundary["ss_sinr_db"] - boundary["predicted_ss_sinr_db"]
    boundary_states = (
        boundary.groupby(["commanded_gain_db", "commanded_noise_power_db"], as_index=False)
        .agg(
            observed_rsrp_db_per_re_unquantized=(
                "rsrp_db_per_re_unquantized",
                "mean",
            ),
            predicted_rsrp_db_per_re_unquantized=(
                "predicted_rsrp_db_per_re_unquantized",
                "mean",
            ),
            observed_ss_sinr_db=("ss_sinr_db", "mean"),
            predicted_ss_sinr_db=("predicted_ss_sinr_db", "mean"),
        )
        .sort_values(["commanded_gain_db", "commanded_noise_power_db"])
    )
    boundary_states["absolute_rsrp_error_db"] = np.abs(
        boundary_states["observed_rsrp_db_per_re_unquantized"]
        - boundary_states["predicted_rsrp_db_per_re_unquantized"]
    )
    boundary_states["absolute_sinr_error_db"] = np.abs(
        boundary_states["observed_ss_sinr_db"] - boundary_states["predicted_ss_sinr_db"]
    )

    bootstrap_coefficients, bootstrap_condition, bootstrap_boundary_errors = _bootstrap_response(
        factorial,
        boundary,
        gain_center=gain_center,
        noise_center=noise_center,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(gates["bootstrap_seed"]),
    )
    coefficient_definitions = [
        ("intercept", 0, None),
        ("gain", 1, "by_gain"),
        ("noise", 2, "by_noise"),
        ("gain_noise_interaction", 3, None),
    ]
    response_definitions = [
        ("relative_RSRP", 0, "relative_RSRP"),
        ("SINR", 1, "SINR"),
    ]
    coefficient_rows = []
    for response_name, response_index, range_prefix in response_definitions:
        for term_name, term_index, range_suffix in coefficient_definitions:
            values = bootstrap_coefficients[:, term_index, response_index]
            row = {
                "response": response_name,
                "term": term_name,
                "estimate": float(coefficients[term_index, response_index]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
            if range_suffix is not None:
                lower, upper = gates["coefficient_ranges"][f"{range_prefix}_{range_suffix}"]
                row["gate_low"] = float(lower)
                row["gate_high"] = float(upper)
                row["point_estimate_passed"] = bool(lower <= row["estimate"] <= upper)
                row["confidence_interval_inside_gate"] = bool(
                    lower <= row["ci_low"] and row["ci_high"] <= upper
                )
            coefficient_rows.append(row)
    coefficient_table = pd.DataFrame(coefficient_rows)

    named = {(row["response"], row["term"]): row for row in coefficient_rows}
    coefficient_point_gate = all(
        named[key]["point_estimate_passed"]
        for key in (
            ("relative_RSRP", "gain"),
            ("relative_RSRP", "noise"),
            ("SINR", "gain"),
            ("SINR", "noise"),
        )
    )
    coefficient_uncertainty_gate = all(
        named[key]["confidence_interval_inside_gate"]
        for key in (
            ("relative_RSRP", "gain"),
            ("relative_RSRP", "noise"),
            ("SINR", "gain"),
            ("SINR", "noise"),
        )
    )
    interaction_limit = float(gates["maximum_absolute_interaction_coefficient"])
    interaction_point_gate = all(
        abs(named[(response, "gain_noise_interaction")]["estimate"]) <= interaction_limit
        for response in ("relative_RSRP", "SINR")
    )
    interaction_uncertainty_gate = all(
        max(
            abs(named[(response, "gain_noise_interaction")]["ci_low"]),
            abs(named[(response, "gain_noise_interaction")]["ci_high"]),
        )
        <= interaction_limit
        for response in ("relative_RSRP", "SINR")
    )
    maximum_rsrp_error = float(boundary_states["absolute_rsrp_error_db"].max())
    maximum_sinr_error = float(boundary_states["absolute_sinr_error_db"].max())
    boundary_rsrp_bootstrap_high = float(np.quantile(bootstrap_boundary_errors[:, 0], 0.975))
    boundary_sinr_bootstrap_high = float(np.quantile(bootstrap_boundary_errors[:, 1], 0.975))
    boundary_point_gate = maximum_rsrp_error <= float(
        gates["boundary_mapping_maximum_absolute_rsrp_error_db"]
    ) and maximum_sinr_error <= float(gates["boundary_mapping_maximum_absolute_sinr_error_db"])
    boundary_uncertainty_gate = boundary_rsrp_bootstrap_high <= float(
        gates["boundary_mapping_maximum_absolute_rsrp_error_db"]
    ) and boundary_sinr_bootstrap_high <= float(
        gates["boundary_mapping_maximum_absolute_sinr_error_db"]
    )
    condition_bootstrap_high = float(np.quantile(bootstrap_condition, 0.975))
    condition_point_gate = condition_number <= float(gates["maximum_condition_number"])
    condition_uncertainty_gate = condition_bootstrap_high <= float(
        gates["maximum_condition_number"]
    )
    response_supported = all(
        (
            coefficient_point_gate,
            coefficient_uncertainty_gate,
            interaction_point_gate,
            interaction_uncertainty_gate,
            boundary_point_gate,
            boundary_uncertainty_gate,
            condition_point_gate,
            condition_uncertainty_gate,
        )
    )
    decision_key = "response_supported" if response_supported else "nonidentifiable_or_nonlinear"
    decision = direct_config["decision_rules"][decision_key]

    result = {
        "schema_version": 1,
        "stage": "phase_3g_bounded_gain_noise_response_result",
        "protocol_revision": direct_config["protocol_revision"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "campaign_archive": _sha256(archive),
            "execution_state": _sha256(state_file),
            "paired_telemetry": _sha256(telemetry_file),
            "direct_response_config": _sha256(direct_file),
            "execution_config": _sha256(execution_file),
        },
        "campaign": {
            "planned_executions": 45,
            "completed_executions": 45,
            "paired_samples": len(telemetry),
            "all_executions_attached": True,
            "all_execution_ping_success_fraction": 1.0,
            "rollback_passed": True,
            "original_ue_image_restored": True,
            "gNB_untouched": True,
        },
        "model": {
            "responses": ["relative_RSRP", "SINR"],
            "controls": ["gain", "noise"],
            "fit_unit": "execution_median",
            "gain_center_db": gain_center,
            "noise_center_db": noise_center,
            "terms": ["intercept", "gain", "noise", "gain_noise_interaction"],
        },
        "jacobian": {
            "relative_RSRP_by_gain": float(coefficients[1, 0]),
            "relative_RSRP_by_noise": float(coefficients[2, 0]),
            "SINR_by_gain": float(coefficients[1, 1]),
            "SINR_by_noise": float(coefficients[2, 1]),
            "condition_number": condition_number,
            "condition_number_ci_low": float(np.quantile(bootstrap_condition, 0.025)),
            "condition_number_ci_high": condition_bootstrap_high,
        },
        "interaction": {
            "relative_RSRP": float(coefficients[3, 0]),
            "SINR": float(coefficients[3, 1]),
            "maximum_absolute_allowed": interaction_limit,
        },
        "boundary_validation": {
            "maximum_absolute_rsrp_error_db": maximum_rsrp_error,
            "maximum_absolute_sinr_error_db": maximum_sinr_error,
            "bootstrap_maximum_absolute_rsrp_error_ci": [
                float(np.quantile(bootstrap_boundary_errors[:, 0], 0.025)),
                boundary_rsrp_bootstrap_high,
            ],
            "bootstrap_maximum_absolute_sinr_error_ci": [
                float(np.quantile(bootstrap_boundary_errors[:, 1], 0.025)),
                boundary_sinr_bootstrap_high,
            ],
        },
        "gates": {
            "coefficient_point_estimates_passed": coefficient_point_gate,
            "coefficient_uncertainty_passed": coefficient_uncertainty_gate,
            "condition_number_point_estimate_passed": condition_point_gate,
            "condition_number_uncertainty_passed": condition_uncertainty_gate,
            "interaction_point_estimates_passed": interaction_point_gate,
            "interaction_uncertainty_passed": interaction_uncertainty_gate,
            "boundary_mapping_point_estimates_passed": boundary_point_gate,
            "boundary_mapping_uncertainty_passed": boundary_uncertainty_gate,
            "all_analysis_gates_passed": response_supported,
        },
        "decision_code": decision["code"],
        "next_action": decision["next_action"],
        "direct_trace_replay_authorized": False,
        "final_test6_accessed": False,
        "abc_authorized": False,
        "claim_limits": direct_config["claim_limits"],
    }
    output.mkdir(parents=True)
    _write_json(output / "phase3g_response_decision.json", result)
    _write_csv(output / "execution_medians.csv", medians)
    _write_csv(output / "factorial_fit.csv", factorial)
    _write_csv(output / "response_coefficients.csv", coefficient_table)
    _write_csv(output / "boundary_execution_validation.csv", boundary)
    _write_csv(output / "boundary_state_validation.csv", boundary_states)
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3g_bounded_gain_noise_response_analysis_manifest",
            "factorial_executions": len(factorial),
            "boundary_executions": len(boundary),
            "bootstrap_repetitions": int(gates["bootstrap_repetitions"]),
            "bootstrap_seed": int(gates["bootstrap_seed"]),
            "final_test6_accessed": False,
            "direct_trace_replay_authorized": False,
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
        "decision": decision["code"],
        "completed_executions": "45",
        "paired_samples": str(len(telemetry)),
        "direct_trace_replay_authorized": "false",
        "final_test6_accessed": "false",
        "abc_authorized": "false",
    }
