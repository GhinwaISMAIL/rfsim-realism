from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _software_revision() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "tracked_worktree_dirty": None}
    return {"revision": revision, "tracked_worktree_dirty": bool(dirty)}


def validate_upv_support_v2_protocol(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("UPV support v2 protocol schema_version must be 1")
    if config.get("stage") != "protocol_preparation_only":
        raise ValueError("UPV support v2 must remain protocol preparation only")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("the offline v2 protocol cannot authorize execution or ABC")
    frozen = config.get("frozen_inputs") or {}
    if frozen.get("required_phase3a_decision") != (
        "insufficient_metadata_absolute_rsrp_not_identified"
    ):
        raise ValueError("unexpected Phase 3A decision prerequisite")
    if not bool(frozen.get("phase2_snapshot_must_remain_unchanged")):
        raise ValueError("Phase 2 must remain an unchanged snapshot")
    kernel = config.get("kernel") or {}
    if kernel.get("name") != "rbf":
        raise ValueError("UPV support v2 requires the RBF kernel")
    if kernel.get("primary_estimator") != "biased_mmd_squared_v_statistic":
        raise ValueError("UPV support v2 requires biased MMD squared")
    if kernel.get("posthoc_clipping") != "prohibited":
        raise ValueError("UPV support v2 prohibits post-hoc clipping")
    if (kernel.get("legacy_unbiased_estimator") or {}).get("role") != "diagnostic_only":
        raise ValueError("the legacy unbiased estimator must be diagnostic only")
    preprocessing = config.get("preprocessing") or {}
    if bool(preprocessing.get("distance_calculation_authorized")):
        raise ValueError("distances cannot be calculated before selecting a branch")
    if preprocessing.get("branch_status") != "unresolved":
        raise ValueError("the measurement branch must remain unresolved")
    repetitions = config.get("repetition_gate") or {}
    if int(repetitions.get("existing_bank_executions_per_state", 0)) >= int(
        repetitions.get("minimum_independent_executions_per_state", 0)
    ):
        raise ValueError("existing-bank repetition gate unexpectedly passes")
    if bool(repetitions.get("abc_allowed_with_existing_bank")):
        raise ValueError("the existing bank cannot authorize ABC")
    probe = config.get("positive_ploss_safety_probe") or {}
    if probe.get("label") != "positive_ploss_safety_and_interaction_probe":
        raise ValueError("the positive-ploss design must be labelled as a safety probe")
    if bool(probe.get("final_support_extension")):
        raise ValueError("the safety probe cannot be called a final support extension")
    if probe.get("ploss_values") != [0.0, 2.5]:
        raise ValueError("the first safety probe must use ploss values 0 and 2.5")
    if probe.get("noise_power_dB_values") != [-12.5, -10.0, -7.5]:
        raise ValueError("unexpected safety-probe noise values")
    if bool((config.get("adaptive_localization") or {}).get(
        "authorized_before_safety_probe"
    )):
        raise ValueError("adaptive localization cannot precede the safety probe")
    reservation = config.get("reservation") or {}
    if bool(reservation.get("request_now")):
        raise ValueError("the offline protocol cannot request a reservation")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")
    if config.get("protocol_revision") == "2.1":
        _validate_v2_1_clarifications(config)


def _validate_v2_1_clarifications(config: dict[str, Any]) -> None:
    branches = config.get("measurement_branches") or {}
    positive = branches.get("no_offset_positive_ploss_valid") or {}
    trigger = positive.get("trigger") or {}
    required_trigger_terms = {
        "nemo_oai_rsrp_definitions_comparable_without_offset",
        "positive_ploss_accepted_as_controlled_simulator_gain_only",
    }
    if set(trigger.get("all_of") or []) != required_trigger_terms:
        raise ValueError(
            "the positive-ploss trigger must establish no-offset comparability "
            "and simulator-gain-only interpretation"
        )
    if positive.get("physical_propagation_loss_interpretation") != "prohibited":
        raise ValueError("positive ploss cannot be interpreted as propagation loss")

    offset = config.get("offset_contract") or {}
    if offset.get("transformed_quantity") != "OAI_ss_rsrp_dbm":
        raise ValueError("the offset contract must transform OAI SS-RSRP")
    if offset.get("reference_quantity") != "NEMO_NR_SpCell_SSB_RSRP":
        raise ValueError("the offset contract must target NEMO SSB RSRP")
    if offset.get("equation") != (
        "RSRP_OAI_to_NEMO = RSRP_OAI + delta_OAI_to_NEMO"
    ):
        raise ValueError("the OAI-to-NEMO offset sign is not explicit")
    if offset.get("delta_definition") != (
        "delta_OAI_to_NEMO = RSRP_NEMO - RSRP_OAI"
    ):
        raise ValueError("the offset delta definition is not explicit")
    if offset.get("upv_calibration_or_validation_fit") != "prohibited":
        raise ValueError("the equivalence offset cannot be fitted to UPV analysis bins")
    required_offset_records = {
        "estimation_method",
        "uncertainty_interval_and_level",
        "applicable_radio_configuration",
        "equipment_and_antenna_configuration",
        "source_data_and_checksums",
    }
    if set(offset.get("required_records") or []) != required_offset_records:
        raise ValueError("the offset provenance record is incomplete")

    relative = config.get("relative_rsrp_diagnostic") or {}
    if relative.get("status") != "diagnostic_only":
        raise ValueError("relative RSRP must remain diagnostic only")
    if relative.get("centering_statistic") != "within_unit_median":
        raise ValueError("relative RSRP must use a frozen within-unit median")
    if relative.get("absolute_location_information") != "removed":
        raise ValueError("relative RSRP must remove absolute location information")
    if relative.get("physical_ploss_inference") != "prohibited":
        raise ValueError("relative RSRP cannot identify physical propagation loss")
    if not relative.get("upv_equation") or not relative.get("rfsim_equation"):
        raise ValueError("relative RSRP equations must be explicit for both systems")

    gate = config.get("probe_quality_gate") or {}
    if gate.get("authorization_status") != "blocked_pending_branch_and_parser":
        raise ValueError("the safety probe must remain blocked")
    if gate.get("analysis_window_seconds") != [15.0, 175.0]:
        raise ValueError("unexpected safety-probe analysis window")
    if int(gate.get("minimum_usable_telemetry_seconds", 0)) != 120:
        raise ValueError("the safety probe requires 120 usable telemetry seconds")
    attachment = gate.get("attachment") or {}
    if attachment.get("required_fraction") != 1.0:
        raise ValueError("every expected UE must remain attached")
    failures = gate.get("failure_limits") or {}
    if failures.get("pbch_failure_events_maximum") != 0:
        raise ValueError("PBCH failure limit must be zero")
    if failures.get("pusch_ul_failure_events_maximum") != 0:
        raise ValueError("fatal PUSCH failure limit must be zero")
    if failures.get("parser_status") != "required_not_yet_implemented":
        raise ValueError("unimplemented failure counters must fail closed")
    direction = gate.get("rsrp_direction_test") or {}
    if direction.get("test") != "exact_one_sided_execution_level_permutation":
        raise ValueError("RSRP direction requires the frozen exact permutation test")
    if float(direction.get("alpha", 1.0)) != 0.05:
        raise ValueError("RSRP direction alpha must be 0.05")
    if direction.get("required_direction") != "RSRP(ploss=2.5)>RSRP(ploss=0.0)":
        raise ValueError("the expected positive-gain RSRP direction is not explicit")
    stopping = set(gate.get("immediate_stopping_conditions") or [])
    required_stops = {
        "ue_detach_or_rnti_change",
        "channel_readback_mismatch",
        "pbch_failure_event",
        "fatal_pusch_ul_failure_event",
        "rfsim_crash_nonfinite_value_or_numeric_overflow",
    }
    if stopping != required_stops:
        raise ValueError("the immediate stopping conditions are incomplete")


def build_upv_support_v2_plan(
    *,
    phase3a_decision: str | Path,
    phase3a_gate: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    decision_path = Path(phase3a_decision).resolve()
    gate_path = Path(phase3a_gate).resolve()
    protocol_path = Path(config_path).resolve()
    for path in [decision_path, gate_path, protocol_path]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe protocol input: {path}")
    config = _read_yaml(protocol_path)
    validate_upv_support_v2_protocol(config)
    decision = _read_json(decision_path)
    gate = _read_json(gate_path)
    required = str(config["frozen_inputs"]["required_phase3a_decision"])
    if decision.get("decision_code") != required:
        raise ValueError("Phase 3A decision does not match the v2 prerequisite")
    if bool(decision.get("absolute_rsrp_calibration_authorized")):
        raise ValueError("the supplied Phase 3A decision unexpectedly authorizes RSRP")
    if bool(decision.get("abc_authorized")):
        raise ValueError("the supplied Phase 3A decision unexpectedly authorizes ABC")
    if gate.get("decision_code") != required:
        raise ValueError("Phase 3A reservation gate does not match its decision")
    if bool(gate.get("reservation_should_be_requested_now")):
        raise ValueError("the supplied Phase 3A reservation gate is open")

    probe = config["positive_ploss_safety_probe"]
    state_count = len(probe["ploss_values"]) * len(probe["noise_power_dB_values"])
    software = _software_revision()
    plan = {
        "schema_version": 1,
        "plan_id": config["name"],
        "stage": config["stage"],
        "analysis_implementation_revision": software["revision"],
        "tracked_worktree_dirty_at_start": software["tracked_worktree_dirty"],
        "input_sha256": {
            "phase3a_decision": _sha256(decision_path),
            "phase3a_gate": _sha256(gate_path),
            "protocol_config": _sha256(protocol_path),
        },
        "frozen_inputs": config["frozen_inputs"],
        "execution_authorized": False,
        "distance_calculation_authorized": False,
        "abc_authorized": False,
        "measurement_branch_status": "unresolved",
        "measurement_branches": config["measurement_branches"],
        "preprocessing": config["preprocessing"],
        "kernel": config["kernel"],
        "repetition_gate": config["repetition_gate"],
        "conditional_safety_probe": {
            **probe,
            "state_count": state_count,
            "minimum_executions": state_count * int(
                probe["minimum_repetitions_per_state"]
            ),
            "preferred_executions": state_count * int(
                probe["preferred_repetitions_per_state"]
            ),
        },
        "adaptive_localization": config["adaptive_localization"],
        "reservation": {
            **config["reservation"],
            "reservation_should_be_requested_now": False,
        },
        "author_metadata_request": config["author_metadata_request"],
        "next_action": (
            "obtain or assess independent measurement metadata; do not compute v2 "
            "distances or request POWDER until a branch is selected"
        ),
    }
    for key in (
        "protocol_revision",
        "offset_contract",
        "relative_rsrp_diagnostic",
        "probe_quality_gate",
        "external_request",
    ):
        if key in config:
            plan[key] = config[key]
    return plan


def write_upv_support_v2_plan(
    *,
    phase3a_decision: str | Path,
    phase3a_gate: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"UPV support v2 plan already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_upv_support_v2_plan(
        phase3a_decision=phase3a_decision,
        phase3a_gate=phase3a_gate,
        config_path=config_path,
    )
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return output
