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

PLAN_COLUMNS = (
    "sequence_index",
    "repetition",
    "oai_rng_seed",
    "segment_index",
    "segment_type",
    "state_id",
    "position",
    "gain_db",
    "noise_power_db",
    "expected_relative_rsrp_db",
    "expected_sinr_db",
)


def validate_phase3h_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "phase_3h_dynamic_staircase_translation_validation":
        raise ValueError("unexpected Phase 3H stage")
    for flag in (
        "execution_authorized",
        "short_trace_replay_authorized",
        "full_trace_replay_authorized",
        "final_evaluation_authorized",
        "abc_authorized",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false during the protocol freeze")
    if config["target_trace"]["session_id"] != "corrected_test_1_ASUS":
        raise ValueError("the staircase must use the designated Test 1 development trace")
    if config["target_trace"].get("final_test6_access") is not False:
        raise ValueError("Test 6 access is prohibited")
    translator = config["translator"]
    if translator.get("extrapolation") != "prohibited":
        raise ValueError("the translator must prohibit extrapolation")
    if translator.get("post_validation_method") != "bounded_piecewise_affine_interpolation":
        raise ValueError("the frozen translator must use bounded piecewise interpolation")

    states = config["validation_states"]
    state_ids = [state["id"] for state in states]
    if len(states) != 7 or len(set(state_ids)) != 7:
        raise ValueError("the staircase must contain seven unique validation states")
    pairs = {(float(state["gain_db"]), float(state["noise_power_db"])) for state in states}
    if len(pairs) != 7:
        raise ValueError("validation control pairs must be unique")
    if not all(
        -18.0 <= gain <= 0.0 and -35.0 <= noise <= -17.0 for gain, noise in pairs
    ):
        raise ValueError("a validation state exceeds the tested operational controls")
    if not all(
        np.isfinite(
            [
                state["gain_db"],
                state["noise_power_db"],
                state["expected_relative_rsrp_db"],
                state["expected_sinr_db"],
            ]
        ).all()
        for state in states
    ):
        raise ValueError("a validation state contains a non-finite value")

    sequences = config["sequences"]
    if len(sequences) != 3:
        raise ValueError("the staircase requires three complete sequence units")
    if len({int(sequence["oai_rng_seed"]) for sequence in sequences}) != 3:
        raise ValueError("sequence RNG seeds must be unique")
    position_sets = {state_id: set() for state_id in state_ids}
    for expected_repetition, sequence in enumerate(sequences, start=1):
        if int(sequence["repetition"]) != expected_repetition:
            raise ValueError("sequence repetitions must be consecutive")
        order = sequence["state_order"]
        if len(order) != 7 or set(order) != set(state_ids):
            raise ValueError("each sequence must visit every validation state exactly once")
        for position, state_id in enumerate(order, start=1):
            position_sets[state_id].add(position)
    if not all(len(positions) == 3 for positions in position_sets.values()):
        raise ValueError("every state must occupy three distinct sequence positions")
    timing = config["timing"]
    if timing.get("return_to_anchor_between_validation_states") is not True:
        raise ValueError("every validation state must be approached from the anchor")
    for name in (
        "state_settling_seconds",
        "state_usable_seconds",
        "anchor_start_settling_seconds",
        "anchor_usable_seconds",
        "anchor_reset_seconds_between_states",
    ):
        if float(timing[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    design = config["statistical_design"]
    if design.get("independent_unit") != "complete_staircase_after_clean_ue_recreation":
        raise ValueError("the independent unit must be the complete staircase")
    if int(design.get("sequence_repetitions", 0)) != 3:
        raise ValueError("exactly three complete staircases are required")
    if design.get("individual_radio_samples_are_independent_repetitions") is not False:
        raise ValueError("radio samples cannot be treated as independent repetitions")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("the reservation gate must remain closed during the freeze")


def _build_plan(config: dict[str, Any]) -> pd.DataFrame:
    states = {state["id"]: state for state in config["validation_states"]}
    channel = config["channel"]
    rows: list[dict[str, Any]] = []
    for sequence_index, sequence in enumerate(config["sequences"], start=1):
        common = {
            "sequence_index": sequence_index,
            "repetition": int(sequence["repetition"]),
            "oai_rng_seed": int(sequence["oai_rng_seed"]),
        }
        rows.append(
            {
                **common,
                "segment_index": 1,
                "segment_type": "anchor_start",
                "state_id": "anchor",
                "position": 0,
                "gain_db": float(channel["anchor_gain_db"]),
                "noise_power_db": float(channel["anchor_noise_power_db"]),
                "expected_relative_rsrp_db": float(
                    channel["anchor_expected_relative_rsrp_db"]
                ),
                "expected_sinr_db": float(channel["anchor_expected_sinr_db"]),
            }
        )
        for position, state_id in enumerate(sequence["state_order"], start=1):
            state = states[state_id]
            rows.append(
                {
                    **common,
                    "segment_index": position + 1,
                    "segment_type": "validation",
                    "state_id": state_id,
                    "position": position,
                    "gain_db": float(state["gain_db"]),
                    "noise_power_db": float(state["noise_power_db"]),
                    "expected_relative_rsrp_db": float(
                        state["expected_relative_rsrp_db"]
                    ),
                    "expected_sinr_db": float(state["expected_sinr_db"]),
                }
            )
        rows.append(
            {
                **common,
                "segment_index": 9,
                "segment_type": "anchor_end",
                "state_id": "anchor",
                "position": 8,
                "gain_db": float(channel["anchor_gain_db"]),
                "noise_power_db": float(channel["anchor_noise_power_db"]),
                "expected_relative_rsrp_db": float(
                    channel["anchor_expected_relative_rsrp_db"]
                ),
                "expected_sinr_db": float(channel["anchor_expected_sinr_db"]),
            }
        )
    return pd.DataFrame(rows, columns=list(PLAN_COLUMNS))


def freeze_phase3h_dynamic_staircase(
    *,
    config_path: str | Path,
    diagnosis_path: str | Path,
    execution_medians_path: str | Path,
    direct_trace_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    diagnosis_file = Path(diagnosis_path).resolve()
    medians_file = Path(execution_medians_path).resolve()
    trace_file = Path(direct_trace_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3H protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3h_config(config)
    frozen = config["frozen_inputs"]
    expected_hashes = {
        diagnosis_file: frozen["diagnosis_sha256"],
        medians_file: frozen["execution_medians_sha256"],
        trace_file: frozen["direct_test1_trace_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen input checksum mismatch: {path}")
    diagnosis = _read_json(diagnosis_file)
    if diagnosis.get("diagnosis_code") != frozen["required_diagnosis"]:
        raise ValueError("the Phase 3G diagnosis does not authorize this preparation")
    if diagnosis.get("final_test6_accessed") is not False:
        raise ValueError("the Phase 3G diagnosis accessed Test 6")
    trace = pd.read_csv(trace_file)
    if len(trace) != int(config["target_trace"]["rows"]):
        raise ValueError("the designated Test 1 trace row count changed")
    if set(trace["session_id"]) != {config["target_trace"]["session_id"]}:
        raise ValueError("the designated Test 1 trace identity changed")
    medians = pd.read_csv(medians_file)
    development = medians.loc[medians["stage"].isin(["factorial", "boundary"])]
    if len(development) != 39 or development.groupby(
        ["commanded_gain_db", "commanded_noise_power_db"]
    ).ngroups != 13:
        raise ValueError("the 13-state Phase 3G development bank is incomplete")

    plan = _build_plan(config)
    targets = pd.DataFrame(config["validation_states"]).rename(
        columns={"gain_db": "commanded_gain_db", "noise_power_db": "commanded_noise_power_db"}
    )
    output.mkdir(parents=True)
    _write_csv(output / "dynamic_staircase_plan.csv", plan)
    _write_csv(output / "validation_targets.csv", targets)
    protocol = {
        "schema_version": 1,
        "stage": config["stage"],
        "protocol_revision": config["protocol_revision"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "config": _sha256(config_file),
            "phase3g_diagnosis": _sha256(diagnosis_file),
            "phase3g_execution_medians": _sha256(medians_file),
            "direct_test1_target_trace": _sha256(trace_file),
        },
        "design": {
            "sequence_units": 3,
            "segments_per_sequence": 9,
            "validation_states_per_sequence": 7,
            "ue_recreations": 3,
            "state_observations": 21,
            "return_to_anchor_between_states": True,
            "individual_radio_samples_are_independent_repetitions": False,
        },
        "timing": config["timing"],
        "execution_gates": config["execution_gates"],
        "translator": config["translator"],
        "claim_limits": config["claim_limits"],
        "reservation": config["reservation"],
        "execution_authorized": False,
        "short_trace_replay_authorized": False,
        "full_trace_replay_authorized": False,
        "final_test6_accessed": False,
        "abc_authorized": False,
    }
    _write_json(output / "protocol.json", protocol)
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3h_dynamic_staircase_protocol_manifest",
            "plan_rows": len(plan),
            "sequence_units": 3,
            "validation_states": 7,
            "reservation_requested": False,
            "execution_authorized": False,
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
        "plan_rows": str(len(plan)),
        "sequence_units": "3",
        "ue_recreations": "3",
        "reservation_request_now": "false",
        "execution_authorized": "false",
        "final_test6_accessed": "false",
    }


def analyze_phase3h_dynamic_staircase(
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
        raise FileExistsError(f"Phase 3H analysis output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3h_config(config)
    plan_file = protocol_root / "dynamic_staircase_plan.csv"
    protocol_file = protocol_root / "protocol.json"
    telemetry_file = campaign / "phase3h_dynamic_staircase_telemetry.csv"
    state_file = campaign / "execution_state.json"
    plan = pd.read_csv(plan_file)
    telemetry = pd.read_csv(telemetry_file)
    state = _read_json(state_file)
    if tuple(plan.columns) != PLAN_COLUMNS or len(plan) != 27:
        raise ValueError("the frozen dynamic staircase plan is invalid")
    if state.get("execution_completed") is not True or state.get("error") is not None:
        raise ValueError("the dynamic staircase campaign did not complete cleanly")
    if int(state.get("completed_sequence_count", 0)) != 3:
        raise ValueError("the campaign does not contain three complete staircase units")
    if state.get("final_test6_accessed") is not False:
        raise ValueError("the staircase campaign accessed Test 6")
    if state.get("short_trace_replay_authorized") is not False:
        raise ValueError("the hardware runner cannot authorize trace replay")
    if state.get("gNB_untouched") is not True:
        raise ValueError("the staircase campaign changed the gNB")
    if state.get("execution_plan_sha256") != _sha256(plan_file):
        raise ValueError("the staircase plan checksum does not match the campaign")
    if state.get("research_protocol_sha256") != _sha256(config_file):
        raise ValueError("the research protocol checksum does not match the campaign")
    required = {
        "sequence_index",
        "sequence_id",
        "segment_index",
        "segment_type",
        "state_id",
        "commanded_gain_db",
        "applied_gain_db",
        "commanded_noise_power_db",
        "applied_noise_power_db",
        "rsrp_db_per_re_unquantized",
        "ss_sinr_db",
        "attached",
    }
    if not required.issubset(telemetry.columns):
        raise ValueError("the campaign telemetry is missing required columns")
    if telemetry["sequence_index"].nunique() != 3:
        raise ValueError("telemetry does not contain three sequence units")
    if not telemetry["attached"].astype(bool).all():
        raise ValueError("telemetry contains detached observations")
    if not np.allclose(telemetry["commanded_gain_db"], telemetry["applied_gain_db"]):
        raise ValueError("applied gain differs from the command")
    if not np.allclose(
        telemetry["commanded_noise_power_db"], telemetry["applied_noise_power_db"]
    ):
        raise ValueError("applied noise differs from the command")

    minimum_samples = int(config["valid_segment_rules"]["minimum_paired_radio_samples"])
    segment_counts = telemetry.groupby(["sequence_index", "segment_index"]).size()
    if len(segment_counts) != 27 or (segment_counts < minimum_samples).any():
        raise ValueError("one or more staircase segments has insufficient telemetry")
    medians = (
        telemetry.groupby(
            [
                "sequence_index",
                "sequence_id",
                "segment_index",
                "segment_type",
                "state_id",
                "commanded_gain_db",
                "commanded_noise_power_db",
            ],
            as_index=False,
        )[["rsrp_db_per_re_unquantized", "ss_sinr_db"]]
        .median()
        .sort_values(["sequence_index", "segment_index"])
    )
    anchor_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []
    targets = {state["id"]: state for state in config["validation_states"]}
    gates = config["execution_gates"]
    for sequence_index, group in medians.groupby("sequence_index", sort=True):
        start = group.loc[group["segment_type"] == "anchor_start"].iloc[0]
        end = group.loc[group["segment_type"] == "anchor_end"].iloc[0]
        rsrp_drift = float(
            end["rsrp_db_per_re_unquantized"] - start["rsrp_db_per_re_unquantized"]
        )
        sinr_drift = float(end["ss_sinr_db"] - start["ss_sinr_db"])
        anchor_reference = float(
            (start["rsrp_db_per_re_unquantized"] + end["rsrp_db_per_re_unquantized"])
            / 2.0
        )
        anchor_rows.append(
            {
                "sequence_index": int(sequence_index),
                "anchor_start_rsrp": float(start["rsrp_db_per_re_unquantized"]),
                "anchor_end_rsrp": float(end["rsrp_db_per_re_unquantized"]),
                "rsrp_drift_db": rsrp_drift,
                "absolute_rsrp_drift_db": abs(rsrp_drift),
                "anchor_start_sinr": float(start["ss_sinr_db"]),
                "anchor_end_sinr": float(end["ss_sinr_db"]),
                "sinr_drift_db": sinr_drift,
                "absolute_sinr_drift_db": abs(sinr_drift),
            }
        )
        selected = group.loc[group["segment_type"] == "validation"].copy()
        selected["observed_relative_rsrp_db"] = (
            selected["rsrp_db_per_re_unquantized"] - anchor_reference
        )
        selected["expected_relative_rsrp_db"] = selected["state_id"].map(
            lambda state_id: float(targets[state_id]["expected_relative_rsrp_db"])
        )
        selected["expected_sinr_db"] = selected["state_id"].map(
            lambda state_id: float(targets[state_id]["expected_sinr_db"])
        )
        selected["relative_rsrp_error_db"] = (
            selected["observed_relative_rsrp_db"]
            - selected["expected_relative_rsrp_db"]
        )
        selected["sinr_error_db"] = selected["ss_sinr_db"] - selected["expected_sinr_db"]
        validation_rows.append(selected)
    anchor_drift = pd.DataFrame(anchor_rows)
    validation = pd.concat(validation_rows, ignore_index=True)
    state_validation = (
        validation.groupby("state_id", as_index=False)
        .agg(
            commanded_gain_db=("commanded_gain_db", "first"),
            commanded_noise_power_db=("commanded_noise_power_db", "first"),
            expected_relative_rsrp_db=("expected_relative_rsrp_db", "first"),
            observed_mean_relative_rsrp_db=("observed_relative_rsrp_db", "mean"),
            expected_sinr_db=("expected_sinr_db", "first"),
            observed_mean_sinr_db=("ss_sinr_db", "mean"),
            relative_rsrp_error_db=("relative_rsrp_error_db", "mean"),
            sinr_error_db=("sinr_error_db", "mean"),
            relative_rsrp_sequence_sd_db=("observed_relative_rsrp_db", "std"),
            sinr_sequence_sd_db=("ss_sinr_db", "std"),
        )
        .sort_values("state_id")
    )
    state_validation["absolute_relative_rsrp_error_db"] = state_validation[
        "relative_rsrp_error_db"
    ].abs()
    state_validation["absolute_sinr_error_db"] = state_validation["sinr_error_db"].abs()
    operational_gate = bool(
        (
            anchor_drift["absolute_rsrp_drift_db"]
            <= float(gates["maximum_anchor_start_end_relative_rsrp_drift_db"])
        ).all()
        and (
            anchor_drift["absolute_sinr_drift_db"]
            <= float(gates["maximum_anchor_start_end_sinr_drift_db"])
        ).all()
        and all(sequence.get("valid") is True for sequence in state["sequences"])
    )
    translation_gate = bool(
        (
            state_validation["absolute_relative_rsrp_error_db"]
            <= float(gates["maximum_absolute_mean_relative_rsrp_error_db"])
        ).all()
        and (
            state_validation["absolute_sinr_error_db"]
            <= float(gates["maximum_absolute_mean_sinr_error_db"])
        ).all()
    )
    if not operational_gate:
        decision_key = "fail_operation"
    elif not translation_gate:
        decision_key = "fail_translation"
    else:
        decision_key = "pass"
    decision_rule = config["decision_rules"][decision_key]
    result = {
        "schema_version": 1,
        "stage": "phase_3h_dynamic_staircase_translation_validation_result",
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "campaign_execution_state": _sha256(state_file),
            "campaign_telemetry": _sha256(telemetry_file),
            "protocol": _sha256(protocol_file),
            "plan": _sha256(plan_file),
            "config": _sha256(config_file),
        },
        "campaign": {
            "sequence_units": 3,
            "segments": 27,
            "validation_state_observations": 21,
            "individual_radio_samples_are_independent_repetitions": False,
            "gNB_untouched": True,
        },
        "maximum_errors": {
            "relative_rsrp_db": float(
                state_validation["absolute_relative_rsrp_error_db"].max()
            ),
            "sinr_db": float(state_validation["absolute_sinr_error_db"].max()),
        },
        "maximum_anchor_drift": {
            "relative_rsrp_db": float(anchor_drift["absolute_rsrp_drift_db"].max()),
            "sinr_db": float(anchor_drift["absolute_sinr_drift_db"].max()),
        },
        "gates": {
            "operational_gate_passed": operational_gate,
            "translation_gate_passed": translation_gate,
            "all_gates_passed": operational_gate and translation_gate,
            "confidence_intervals_are_diagnostic_only": True,
        },
        "decision_code": decision_rule["code"],
        "next_action": decision_rule["next_action"],
        "short_trace_protocol_freeze_authorized": operational_gate and translation_gate,
        "short_trace_replay_currently_authorized": False,
        "full_trace_replay_currently_authorized": False,
        "final_test6_accessed": False,
        "abc_authorized": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "segment_medians.csv", medians)
    _write_csv(output / "sequence_anchor_drift.csv", anchor_drift)
    _write_csv(output / "sequence_state_validation.csv", validation)
    _write_csv(output / "state_translation_validation.csv", state_validation)
    _write_json(output / "phase3h_staircase_decision.json", result)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": decision_rule["code"],
        "sequence_units": "3",
        "short_trace_protocol_freeze_authorized": str(
            operational_gate and translation_gate
        ).lower(),
        "short_trace_replay_currently_authorized": "false",
        "final_test6_accessed": "false",
    }
