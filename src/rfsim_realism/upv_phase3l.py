from __future__ import annotations

import math
import shutil
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
from .upv_phase3i import COMMAND_COLUMNS
from .upv_phase3j import (
    _analyze_phase3j_execution,
    _verify_protocol_checksums,
)


def validate_phase3l_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3L schema_version must be 1")
    if config.get("stage") != "phase_3l_posthoc_test6_exploratory_replay":
        raise ValueError("unexpected Phase 3L stage")
    if config.get("evaluation_status") != "posthoc_exploratory_not_confirmatory_validation":
        raise ValueError("Phase 3L must remain explicitly exploratory")
    for flag in (
        "hardware_execution_authorized",
        "translator_update_authorized",
        "confirmatory_support_pass_claimed",
        "abc_authorized",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false during Phase 3L preparation")
    official = config["official_confirmatory_result"]
    if official.get("support_gate_passed") is not False:
        raise ValueError("the frozen Phase 3K support failure must remain unchanged")
    if official.get("status") != "unsupported_under_version_1_protocol":
        raise ValueError("the official Test 6 status changed")
    if official.get("immutable") is not True:
        raise ValueError("the official Test 6 result must be immutable")
    if int(official.get("target_rows", 0)) != 297:
        raise ValueError("the frozen Test 6 target row count changed")
    if int(official.get("clipped_rows", 0)) != 21:
        raise ValueError("the frozen Test 6 clipped row count changed")
    if not math.isclose(float(official.get("clipped_fraction", 0)), 21 / 297):
        raise ValueError("the frozen Test 6 clipped fraction changed")

    rationale = config["exploratory_rationale"]
    if rationale.get("selected_after_test6_support_result") is not True:
        raise ValueError("the post hoc status must be disclosed")
    if rationale.get("changes_confirmatory_result") is not False:
        raise ValueError("Phase 3L may not change the confirmatory result")
    if rationale.get("new_support_thresholds_defined") is not False:
        raise ValueError("Phase 3L may not introduce new support thresholds")
    if rationale.get("translator_changed") is not False:
        raise ValueError("Phase 3L must retain the frozen translator")

    classification = config["radio_state_classification"]
    if classification.get("frozen_xapp_threshold_available") is not False:
        raise ValueError("no xApp classification threshold was frozen")
    if classification.get("primary_classification_claim") != "prohibited":
        raise ValueError("threshold-specific classification claims are prohibited")

    execution = config["execution"]
    if int(execution.get("exploratory_executions", 0)) != 1:
        raise ValueError("Phase 3L authorizes one time-efficient exploratory execution")
    if execution.get("oai_rng_seeds") != [48001]:
        raise ValueError("the Phase 3L exploratory seed changed")
    if int(execution.get("target_rows", 0)) != 297:
        raise ValueError("the Phase 3L target row count changed")
    if execution.get("commands_may_adapt_during_execution") is not False:
        raise ValueError("Phase 3L commands may not adapt")

    runtime = config["runtime_gates"]
    if int(runtime.get("minimum_paired_rows_per_execution", 0)) != 292:
        raise ValueError("the proportional Test 6 paired-row requirement changed")
    if runtime.get("missing_row_interpolation") != "prohibited":
        raise ValueError("Phase 3L telemetry interpolation is prohibited")
    references = config["reference_fidelity_limits_not_acceptance_gates"]
    if references.get("source") != "unchanged_phase3j_development_limits":
        raise ValueError("Phase 3L fidelity references must remain inherited")
    if references.get("lag_search_for_gate_selection") != "prohibited":
        raise ValueError("post-hoc lag selection is prohibited")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("a reservation is premature before the runner freeze")


def _verify_bundle(directory: Path, expected_manifest_sha256: str) -> dict[str, str]:
    manifest_file = directory / "SHA256SUMS.json"
    if _sha256(manifest_file) != expected_manifest_sha256:
        raise ValueError(f"bundle checksum-manifest mismatch: {directory}")
    checksums = _read_json(manifest_file)
    for name, digest in checksums.items():
        path = directory / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
            raise ValueError(f"bundle artifact checksum mismatch: {path}")
    return {str(key): str(value) for key, value in checksums.items()}


def _interval_rows(clipped: pd.DataFrame) -> pd.DataFrame:
    frame = clipped.sort_values("command_index").copy()
    frame["relative_rsrp_clipping_error_db"] = (
        frame["target_relative_rsrp_db"] - frame["projected_relative_rsrp_db"]
    )
    frame["sinr_clipping_error_db"] = frame["target_sinr_db"] - frame["projected_sinr_db"]
    groups = (frame["command_index"].diff().fillna(1).ne(1)).cumsum()
    records: list[dict[str, Any]] = []
    for interval_number, (_, group) in enumerate(frame.groupby(groups, sort=True), start=1):
        records.append(
            {
                "interval_number": interval_number,
                "start_command_index": int(group["command_index"].min()),
                "end_command_index": int(group["command_index"].max()),
                "duration_rows": len(group),
                "mean_absolute_relative_rsrp_error_db": float(
                    group["relative_rsrp_clipping_error_db"].abs().mean()
                ),
                "maximum_absolute_relative_rsrp_error_db": float(
                    group["relative_rsrp_clipping_error_db"].abs().max()
                ),
                "mean_absolute_sinr_error_db": float(
                    group["sinr_clipping_error_db"].abs().mean()
                ),
                "maximum_absolute_sinr_error_db": float(
                    group["sinr_clipping_error_db"].abs().max()
                ),
            }
        )
    return pd.DataFrame(records)


def _threshold_sensitivity(commands: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    clipped = commands.loc[commands["clipped"].astype(bool)].copy()
    ranges = config["radio_state_classification"]["diagnostic_only_sensitivity_thresholds"]
    records: list[dict[str, Any]] = []
    definitions = (
        (
            "sinr_db",
            "target_sinr_db",
            "projected_sinr_db",
            ranges["sinr_db_integers"],
        ),
        (
            "relative_rsrp_db",
            "target_relative_rsrp_db",
            "projected_relative_rsrp_db",
            ranges["relative_rsrp_db_integers"],
        ),
    )
    for metric, target, projected, bounds in definitions:
        for threshold in range(int(bounds[0]), int(bounds[1]) + 1):
            changed = (clipped[target] >= threshold) != (clipped[projected] >= threshold)
            records.append(
                {
                    "metric": metric,
                    "threshold_db": threshold,
                    "changed_classification_rows": int(changed.sum()),
                    "changed_command_indices": ",".join(
                        str(value)
                        for value in clipped.loc[changed, "command_index"].astype(int)
                    ),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(records)


def freeze_phase3l_exploratory_protocol(
    *,
    config_path: str | Path,
    phase3k_config_path: str | Path,
    model_release_dir: str | Path,
    support_result_dir: str | Path,
    phase3j_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3k_file = Path(phase3k_config_path).resolve()
    release_dir = Path(model_release_dir).resolve()
    support_dir = Path(support_result_dir).resolve()
    phase3j_file = Path(phase3j_config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3L protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3l_config(config)
    frozen = config["frozen_inputs"]
    expected_files = {
        phase3k_file: frozen["phase3k_config_sha256"],
        phase3j_file: frozen["phase3j_config_sha256"],
        support_dir / "test6_support_decision.json": frozen[
            "phase3k_support_decision_sha256"
        ],
        support_dir / "test6_commands.csv": frozen["test6_commands_sha256"],
        support_dir / "test6_clipped_targets.csv": frozen[
            "test6_clipped_targets_sha256"
        ],
    }
    for path, digest in expected_files.items():
        if _sha256(path) != digest:
            raise ValueError(f"frozen Phase 3L input checksum mismatch: {path}")
    _verify_bundle(release_dir, frozen["phase3k_model_release_checksums_sha256"])
    _verify_bundle(support_dir, frozen["phase3k_support_result_checksums_sha256"])
    decision = _read_json(support_dir / "test6_support_decision.json")
    if decision.get("decision_code") != frozen["required_phase3k_decision"]:
        raise ValueError("the official Phase 3K support decision changed")
    if decision.get("gates", {}).get("trajectory_support_gate_passed") is not False:
        raise ValueError("Phase 3L requires the disclosed Phase 3K support failure")
    if decision.get("translator_changed") is not False:
        raise ValueError("the Phase 3K translator was changed")

    commands_file = support_dir / "test6_commands.csv"
    commands = pd.read_csv(commands_file)
    if tuple(commands.columns) != COMMAND_COLUMNS or len(commands) != 297:
        raise ValueError("the frozen Test 6 command table is invalid")
    clipped = commands.loc[commands["clipped"].astype(bool)].copy()
    if len(clipped) != 21:
        raise ValueError("the frozen Test 6 clipping set changed")
    intervals = _interval_rows(clipped)
    sensitivity = _threshold_sensitivity(commands, config)
    clipping_errors = pd.DataFrame(
        {
            "relative_rsrp_db": (
                commands["target_relative_rsrp_db"]
                - commands["projected_relative_rsrp_db"]
            ),
            "sinr_db": commands["target_sinr_db"] - commands["projected_sinr_db"],
        }
    )
    diagnosis = {
        "schema_version": 1,
        "stage": "phase_3l_posthoc_clipping_diagnosis",
        "target_rows": len(commands),
        "inside_rows": int((~commands["clipped"].astype(bool)).sum()),
        "clipped_rows": len(clipped),
        "clipped_fraction": float(len(clipped) / len(commands)),
        "clipped_intervals": len(intervals),
        "longest_clipped_interval_rows": int(intervals["duration_rows"].max()),
        "clipped_point_errors": {
            "relative_rsrp_mean_absolute_db": float(
                clipping_errors.loc[clipped.index, "relative_rsrp_db"].abs().mean()
            ),
            "relative_rsrp_maximum_absolute_db": float(
                clipping_errors.loc[clipped.index, "relative_rsrp_db"].abs().max()
            ),
            "sinr_mean_absolute_db": float(
                clipping_errors.loc[clipped.index, "sinr_db"].abs().mean()
            ),
            "sinr_maximum_absolute_db": float(
                clipping_errors.loc[clipped.index, "sinr_db"].abs().max()
            ),
        },
        "complete_trace_projection_only_errors": {
            "relative_rsrp_mae_db": float(clipping_errors["relative_rsrp_db"].abs().mean()),
            "relative_rsrp_rmse_db": float(
                np.sqrt(np.square(clipping_errors["relative_rsrp_db"]).mean())
            ),
            "sinr_mae_db": float(clipping_errors["sinr_db"].abs().mean()),
            "sinr_rmse_db": float(np.sqrt(np.square(clipping_errors["sinr_db"]).mean())),
        },
        "official_phase3k_status": "unsupported_under_version_1_protocol",
        "exploratory_replay_status": "protocol_frozen_hardware_not_yet_authorized",
    }
    protocol = {
        "schema_version": 1,
        "stage": config["stage"],
        "protocol_revision": config["protocol_revision"],
        "evaluation_status": config["evaluation_status"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {path.name: digest for path, digest in expected_files.items()},
        "official_confirmatory_result": config["official_confirmatory_result"],
        "exploratory_rationale": config["exploratory_rationale"],
        "clipping_diagnosis": diagnosis,
        "radio_state_classification": config["radio_state_classification"],
        "execution": config["execution"],
        "runtime_gates": config["runtime_gates"],
        "reference_fidelity_limits_not_acceptance_gates": config[
            "reference_fidelity_limits_not_acceptance_gates"
        ],
        "metric_definitions": config["metric_definitions"],
        "decision_rules": config["decision_rules"],
        "claim_limits": config["claim_limits"],
        "hardware_execution_authorized": False,
        "translator_update_authorized": False,
        "confirmatory_support_pass_claimed": False,
        "reservation_requested": False,
    }
    output.mkdir(parents=True)
    shutil.copyfile(commands_file, output / "exploratory_test6_commands.csv")
    _write_csv(output / "clipping_intervals.csv", intervals)
    _write_csv(output / "classification_threshold_sensitivity.csv", sensitivity)
    _write_json(output / "clipping_diagnosis.json", diagnosis)
    _write_json(output / "protocol.json", protocol)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "target_rows": str(len(commands)),
        "clipped_rows": str(len(clipped)),
        "clipped_intervals": str(len(intervals)),
        "official_support_status": "unsupported_under_version_1_protocol",
        "hardware_execution_authorized": "false",
        "reservation_requested": "false",
    }


def _analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "fidelity_gates_per_execution": config[
            "reference_fidelity_limits_not_acceptance_gates"
        ],
    }


def analyze_phase3l_exploratory_replay(
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
        raise FileExistsError(f"Phase 3L analysis output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3l_config(config)
    _verify_protocol_checksums(protocol_root)
    commands_file = protocol_root / "exploratory_test6_commands.csv"
    commands = pd.read_csv(commands_file)
    if tuple(commands.columns) != COMMAND_COLUMNS or len(commands) != 297:
        raise ValueError("the frozen Phase 3L command table is invalid")
    metrics, paired, anchors, lag_diagnostics = _analyze_phase3j_execution(
        campaign=campaign,
        execution_number=1,
        commands=commands,
        commands_sha256=_sha256(commands_file),
        config=_analysis_config(config),
        config_sha256=_sha256(config_file),
        telemetry_name="phase3l_test6_telemetry.csv",
        anchors_name="phase3l_anchor_telemetry.csv",
        expected_test6_accessed=True,
    )
    state = _read_json(campaign / "execution_state.json")
    if state.get("exploratory_replay") is not True:
        raise ValueError("the Phase 3L execution is not labelled exploratory")
    if state.get("frozen_v1_support_gate_passed") is not False:
        raise ValueError("the execution state changed the frozen support verdict")
    runtime_valid = bool(metrics["runtime_gate_passed"])
    rule = config["decision_rules"]["runtime_valid" if runtime_valid else "runtime_invalid"]
    result = {
        "schema_version": 1,
        "stage": "phase_3l_posthoc_test6_exploratory_replay_result",
        "evaluation_status": config["evaluation_status"],
        "analysis_repository_revision": _git_revision(),
        "input_sha256": {
            "config": _sha256(config_file),
            "protocol": _sha256(protocol_root / "protocol.json"),
            "commands": _sha256(commands_file),
            "campaign": {
                name: _sha256(campaign / name)
                for name in (
                    "execution_state.json",
                    "phase3l_test6_telemetry.csv",
                    "phase3l_anchor_telemetry.csv",
                )
            },
        },
        "official_phase3k_support_status": "unsupported_under_version_1_protocol",
        "confirmatory_support_pass_claimed": False,
        "exploratory_execution": {
            "target_rows": len(commands),
            "paired_rows": int(metrics["paired_rows"]),
            "clipped_rows": int(metrics["clipped_rows"]),
            "runtime_valid": runtime_valid,
            "phase3j_reference_fidelity_limits_met": bool(metrics["fidelity_gate_passed"]),
            "reference_limits_are_acceptance_gates": False,
        },
        "decision_code": rule["code"],
        "next_action": rule["next_action"],
        "translator_changed": False,
        "threshold_specific_xapp_behavior_claimed": False,
        "independent_final_validation_claimed": False,
        "claim_limits": config["claim_limits"],
    }
    output.mkdir(parents=True)
    _write_csv(output / "paired_test6_fidelity.csv", paired)
    _write_csv(output / "exploratory_execution_metrics.csv", pd.DataFrame([metrics]))
    _write_csv(output / "anchor_medians.csv", anchors)
    _write_csv(output / "lag_diagnostics.csv", lag_diagnostics)
    _write_json(output / "phase3l_exploratory_decision.json", result)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": str(rule["code"]),
        "runtime_valid": str(runtime_valid).lower(),
        "confirmatory_support_pass_claimed": "false",
    }
