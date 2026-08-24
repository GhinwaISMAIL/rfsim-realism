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
    return {
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
