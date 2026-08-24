from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .upv_protocol import _normal_member_path


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


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def _git_revision(repository: Path) -> tuple[str, bool]:
    if not (repository / ".git").exists():
        raise ValueError(f"not a Git checkout: {repository}")
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
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git checkout {repository}: {error}") from error
    return revision, bool(dirty)


def _implementation_revision() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision, dirty = _git_revision(repository)
    except ValueError:
        return {"revision": "unavailable", "tracked_worktree_dirty": None}
    return {"revision": revision, "tracked_worktree_dirty": dirty}


def validate_measurement_audit_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("measurement audit schema_version must be 1")
    if config.get("stage") != "phase_3a_measurement_equivalence_and_link_budget_audit":
        raise ValueError("unexpected measurement audit stage")
    frozen = config.get("frozen_inputs") or {}
    if frozen.get("phase2_decision_code") != "systematic_rsrp_offset_but_sinr_support":
        raise ValueError("Phase 3A requires the frozen Phase 2 Decision 4 result")
    if frozen.get("phase2_estimator") != "unbiased_mmd_squared":
        raise ValueError("the Phase 2 snapshot must retain its unbiased MMD estimator")
    if not bool(frozen.get("phase2_must_remain_unchanged")):
        raise ValueError("Phase 2 must remain an unchanged frozen snapshot")
    future = config.get("future_mmd_protocol") or {}
    if future.get("primary_estimator") != "biased_mmd_squared_v_statistic":
        raise ValueError("future ABC must use the nonnegative biased MMD squared estimator")
    if future.get("clipping_unbiased_estimates") != "prohibited":
        raise ValueError("clipping unbiased MMD estimates must be prohibited")
    offset = config.get("offset_policy") or {}
    if not bool(offset.get("observed_phase2_gap_must_not_determine_delta")):
        raise ValueError("the Phase 2 RSRP gap must not determine an equivalence offset")
    if not bool(offset.get("verified_offset_requires_independent_link_budget_evidence")):
        raise ValueError("offset verification must require independent evidence")
    probe = config.get("positive_ploss_probe") or {}
    if probe.get("interpretation") != "positive_ploss_safety_and_interaction_probe_only":
        raise ValueError("the first positive-ploss experiment must be labelled as a safety probe")
    if not bool(probe.get("not_final_support_extension")):
        raise ValueError("the positive-ploss safety probe is not a final support extension")
    reservation = config.get("reservation_policy") or {}
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")
    if bool(reservation.get("request_now")):
        raise ValueError("the frozen Phase 3A audit must start offline")
    assertions = config.get("source_assertions") or []
    if not assertions or {item.get("source") for item in assertions} != {"oai", "profile"}:
        raise ValueError("source assertions must cover both OAI and the POWDER profile")


def _source_trace(
    config: dict[str, Any],
    repositories: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_states: dict[str, object] = {}
    for name, repository in repositories.items():
        revision, dirty = _git_revision(repository)
        expected = str(config["pinned_sources"][name]["expected_revision"])
        if revision != expected:
            raise ValueError(f"{name} revision {revision} does not match {expected}")
        if dirty:
            raise ValueError(f"tracked files are dirty in the {name} source checkout")
        source_states[name] = {
            "source_id": f"{name}_source_checkout",
            "revision": revision,
            "tracked_worktree_dirty": dirty,
        }

    rows: list[dict[str, object]] = []
    for assertion_index, assertion in enumerate(config["source_assertions"], start=1):
        source = str(assertion["source"])
        relative = str(assertion["path"])
        path = repositories[source] / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe source evidence: {path}")
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        for snippet_index, snippet_value in enumerate(assertion["snippets"], start=1):
            snippet = str(snippet_value)
            count = text.count(snippet)
            if count < 1:
                raise ValueError(f"source assertion not found in {path}: {snippet}")
            first_offset = text.index(snippet)
            line_number = text[:first_offset].count("\n") + 1
            rows.append({
                "assertion_id": assertion_index,
                "snippet_id": snippet_index,
                "source": source,
                "revision": source_states[source]["revision"],
                "relative_path": relative,
                "file_sha256": _sha256(path),
                "fact": assertion["fact"],
                "snippet": snippet,
                "occurrence_count": count,
                "first_line": line_number,
                "line_text": lines[line_number - 1].strip(),
                "verified": True,
            })
    return pd.DataFrame(rows), source_states


def _radio_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for archive_path in archive.namelist():
        normalized = _normal_member_path(archive_path)
        if len(PurePosixPath(normalized).parts) != 2:
            continue
        if not normalized.lower().endswith(".csv"):
            continue
        result.append((archive_path, normalized))
    return sorted(result, key=lambda item: item[1])


def _numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.to_numeric(
        raw[column].astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _device(source_path: str) -> str:
    lowered = PurePosixPath(source_path).name.lower()
    if "asus" in lowered:
        return "ASUS"
    if "s25" in lowered:
        return "S25"
    return "unknown"


def _test_id(source_path: str) -> int:
    match = re.search(r"Test[_ ]?(\d+)", source_path, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot infer test ID from {source_path}")
    return int(match.group(1))


def _minimum_resolution(values: pd.Series) -> float:
    unique = np.unique(values.dropna().to_numpy(float))
    if len(unique) < 2:
        return math.nan
    differences = np.diff(unique)
    positive = differences[differences > 1e-9]
    return float(np.min(positive)) if len(positive) else math.nan


def _upv_inventory(
    archive_path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fields = {str(key): str(value) for key, value in config["upv_fields"].items()}
    rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    measurement_types: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = _radio_members(archive)
        if not members:
            raise ValueError("UPV archive does not contain UE radio CSV files")
        for archive_name, source_path in members:
            raw = pd.read_csv(
                io.BytesIO(archive.read(archive_name)),
                sep=";",
                dtype=str,
                low_memory=False,
            )
            row: dict[str, object] = {
                "source_path": source_path,
                "test_id": _test_id(source_path),
                "device": _device(source_path),
                "csv_rows": len(raw),
            }
            for name, column in fields.items():
                if column not in raw:
                    row[f"{name}_column_present"] = False
                    row[f"{name}_nonmissing_rows"] = 0
                    continue
                row[f"{name}_column_present"] = True
                row[f"{name}_nonmissing_rows"] = int(raw[column].notna().sum())

            rsrp = _numeric(raw, fields["rsrp"])
            pathloss = _numeric(raw, fields["pathloss"])
            valid_reference = pd.DataFrame({
                "rsrp_dbm": rsrp,
                "pathloss_db": pathloss,
            }).dropna()
            row.update({
                "rsrp_min_dbm": float(rsrp.min()) if rsrp.notna().any() else math.nan,
                "rsrp_median_dbm": float(rsrp.median()) if rsrp.notna().any() else math.nan,
                "rsrp_max_dbm": float(rsrp.max()) if rsrp.notna().any() else math.nan,
                "rsrp_observed_resolution_db": _minimum_resolution(rsrp),
                "pathloss_plus_rsrp_complete_rows": len(valid_reference),
            })
            type_column = fields["measurement_type"]
            types = (
                raw[type_column].dropna().astype(str).str.strip()
                if type_column in raw
                else pd.Series(dtype=str)
            )
            source_types = sorted(value for value in types.unique() if value)
            measurement_types.update(source_types)
            row["measurement_type_values"] = "|".join(source_types)
            if valid_reference.empty:
                reference_rows.append({
                    "source_path": source_path,
                    "test_id": row["test_id"],
                    "device": row["device"],
                    "complete_rows": 0,
                    "implied_reference_power_mean_dbm": math.nan,
                    "implied_reference_power_sd_db": math.nan,
                    "implied_reference_power_median_dbm": math.nan,
                    "status": "unavailable_pathloss_column_has_no_values",
                })
            else:
                implied = valid_reference["rsrp_dbm"] + valid_reference["pathloss_db"]
                reference_rows.append({
                    "source_path": source_path,
                    "test_id": row["test_id"],
                    "device": row["device"],
                    "complete_rows": len(implied),
                    "implied_reference_power_mean_dbm": float(implied.mean()),
                    "implied_reference_power_sd_db": float(implied.std(ddof=1)),
                    "implied_reference_power_median_dbm": float(implied.median()),
                    "status": "available",
                })
            rows.append(row)

    inventory = pd.DataFrame(rows).sort_values(["test_id", "device"]).reset_index(drop=True)
    reference = pd.DataFrame(reference_rows).sort_values(
        ["test_id", "device"]
    ).reset_index(drop=True)
    total_rsrp = int(inventory["rsrp_nonmissing_rows"].sum())
    total_pathloss = int(inventory["pathloss_nonmissing_rows"].sum())
    total_complete = int(inventory["pathloss_plus_rsrp_complete_rows"].sum())
    summary = {
        "radio_csv_files": len(inventory),
        "devices": sorted(inventory["device"].unique()),
        "tests": sorted(int(value) for value in inventory["test_id"].unique()),
        "serving_rsrp_rows": total_rsrp,
        "serving_measurement_type_rows": int(
            inventory["measurement_type_nonmissing_rows"].sum()
        ),
        "serving_pathloss_rows": total_pathloss,
        "pathloss_plus_rsrp_complete_rows": total_complete,
        "measurement_type_values": sorted(measurement_types),
        "all_observed_measurement_types_are_ssb": measurement_types == {"SSB"},
        "pathloss_reference_power_diagnostic_available": total_complete > 0,
    }
    return inventory, reference, summary


def _measurement_crosswalk(upv_summary: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "system": "UPV / NEMO",
            "quantity": "RSRP (NR SpCell)",
            "signal_basis": "SSB (confirmed by every nonmissing Measurement type label)",
            "receive_antenna_treatment": "not documented in archive",
            "filtering_or_aggregation": "NEMO/device behavior not documented in archive",
            "absolute_power_reference": "not recoverable; exported Pathloss column is empty",
            "observed_rows": upv_summary["serving_rsrp_rows"],
            "equivalence_status": "partially_aligned_signal_basis_absolute_scale_unresolved",
        },
        {
            "system": "OAI NR UE",
            "quantity": "measurements.ssb_rsrp_dBm",
            "signal_basis": "SS-RSRP over SSS REs, dBm per RE",
            "receive_antenna_treatment": "power averaged across receive antennas and 127 SSS REs",
            "filtering_or_aggregation": (
                "integer dBm Layer-1 sample; code states no Layer-3 filtering"
            ),
            "absolute_power_reference": (
                "digital power corrected by FFT normalization and RF gain metadata"
            ),
            "observed_rows": None,
            "equivalence_status": "definition_traced_absolute_scale_depends_on_rfsim_gain_metadata",
        },
        {
            "system": "RFsim UE_RADIO_V1",
            "quantity": "ss_rsrp_dbm",
            "signal_basis": "serving OAI SSB RSRP",
            "receive_antenna_treatment": "inherits OAI receive-antenna averaging",
            "filtering_or_aggregation": (
                "arithmetic mean of integer-dBm Layer-1 samples per UTC second"
            ),
            "absolute_power_reference": "OAI result shifted once by configured rsrp_offset_dB",
            "observed_rows": None,
            "equivalence_status": (
                "structurally_traceable_empirical_offset_not_physical_calibration"
            ),
        },
    ])


def _link_budget_ledger(config: dict[str, Any], upv_summary: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in config["known_link_budget_values"]:
        rows.append({
            "quantity": item["quantity"],
            "value": item["value"],
            "unit": item["unit"],
            "provenance": item["provenance"],
            "interpretation": item["interpretation"],
            "status": "known_but_not_sufficient_for_independent_offset",
        })
    for quantity in config["unknown_link_budget_values"]:
        rows.append({
            "quantity": quantity,
            "value": None,
            "unit": None,
            "provenance": "not present in supplied archive or pinned profile",
            "interpretation": "required metadata remains unresolved",
            "status": "unknown",
        })
    rows.append({
        "quantity": "upv_pathloss_plus_rsrp_reference_power",
        "value": None,
        "unit": "dBm",
        "provenance": "UPV Pathloss (NR SpCell) + RSRP (NR SpCell)",
        "interpretation": (
            f"cannot compute: {upv_summary['serving_pathloss_rows']} nonmissing pathloss rows "
            f"and {upv_summary['pathloss_plus_rsrp_complete_rows']} complete pairs"
        ),
        "status": "unavailable",
    })
    return pd.DataFrame(rows)


def _offset_dossier(
    config: dict[str, Any],
    phase2_gate: dict[str, Any],
    upv_summary: dict[str, object],
) -> dict[str, object]:
    gate_reason = str(
        phase2_gate.get("gate_before_next_reservation", {}).get("reason", "")
    )
    gap_match = re.search(r"([0-9]+(?:\.[0-9]+)?) dB below", gate_reason)
    phase2_gap = float(gap_match.group(1)) if gap_match else None
    return {
        "schema_version": 1,
        "transform_definition": config["offset_policy"]["transform"],
        "current_rfsim_reporting_offset_db": -56.0,
        "application_count": 1,
        "application_mechanism": (
            "profile patch sets RFsim rx_gain to -rsrp_offset_dB; the OAI SS-RSRP "
            "formula subtracts rx_gain minus rx_gain_offset"
        ),
        "changes_iq_samples": False,
        "changes_sinr": False,
        "origin": (
            "empirical mapping of the healthy zero-loss RFsim baseline from about -41 dBm "
            "to a -97 dBm real-data reference, as documented by the pinned profile"
        ),
        "independent_of_phase2_upv_gap": True,
        "independently_physically_verified": False,
        "phase2_gap_db_for_audit_context_only": phase2_gap,
        "phase2_gap_used_to_select_offset": False,
        "upv_pathloss_reference_check": {
            "available": upv_summary["pathloss_reference_power_diagnostic_available"],
            "complete_rows": upv_summary["pathloss_plus_rsrp_complete_rows"],
        },
        "uncertainty_db": None,
        "status": "empirical_discrepancy_transform_not_verified_physical_offset",
        "permitted_use": "provenance-preserved diagnostic comparison",
        "prohibited_use": "claiming absolute physical propagation-loss calibration",
    }


def _decision(
    config: dict[str, Any],
    phase2_manifest: dict[str, Any],
    phase2_gate: dict[str, Any],
    upv_summary: dict[str, object],
) -> dict[str, object]:
    if phase2_manifest.get("decision_code") != config["frozen_inputs"][
        "phase2_decision_code"
    ]:
        raise ValueError("Phase 2 manifest decision does not match the frozen Phase 3A input")
    if phase2_gate.get("gate_before_next_reservation", {}).get(
        "reservation_should_be_requested_now"
    ) is not False:
        raise ValueError("Phase 2 reservation gate is not closed")
    missing_absolute_metadata = bool(
        not upv_summary["pathloss_reference_power_diagnostic_available"]
        or config["unknown_link_budget_values"]
    )
    if missing_absolute_metadata:
        code = "insufficient_metadata_absolute_rsrp_not_identified"
        action = config["decision_policy"]["insufficient_metadata_action"]
    else:
        code = "measurement_equivalence_requires_expert_review"
        action = "review_complete_link_budget_before_selecting_a_branch"
    return {
        "schema_version": 1,
        "decision_code": code,
        "action": action,
        "evidence": {
            "upv_measurement_type_values": upv_summary["measurement_type_values"],
            "upv_all_observed_measurement_types_are_ssb": upv_summary[
                "all_observed_measurement_types_are_ssb"
            ],
            "upv_serving_rsrp_rows": upv_summary["serving_rsrp_rows"],
            "upv_serving_pathloss_rows": upv_summary["serving_pathloss_rows"],
            "upv_pathloss_plus_rsrp_complete_rows": upv_summary[
                "pathloss_plus_rsrp_complete_rows"
            ],
            "oai_measurement_formula_traced": True,
            "rfsim_offset_application_traced": True,
            "rfsim_offset_is_empirical_not_independently_physical": True,
            "positive_ploss_is_numerically_a_path_gain": True,
        },
        "absolute_rsrp_calibration_authorized": False,
        "sinr_and_relative_rsrp_analysis_authorized": True,
        "new_measurement_equivalence_offset_applied": False,
        "abc_authorized": False,
        "reservation_should_be_requested_now": False,
        "reservation_notice_lead_time_minutes": int(
            config["reservation_policy"]["preparation_lead_time_minutes"]
        ),
        "next_external_action": {
            "recipient": "UPV dataset authors",
            "questions": [
                "What ss-PBCH-BlockPower or equivalent SSB EPRE was signalled during each test?",
                (
                    "Does NEMO RSRP (NR SpCell) represent beam/SSB RSRP, cell-level "
                    "RSRP, or an antenna-combined value, and what filtering is applied?"
                ),
                (
                    "Why is Pathloss (NR SpCell) present as a column but empty in all "
                    "UE CSV files, and can populated exports be supplied?"
                ),
                (
                    "What gNB conducted power, antenna/cable gains, EIRP, and UE "
                    "calibration or combining assumptions apply?"
                ),
            ],
        },
        "offline_work_while_waiting": [
            "retain SINR and relative RSRP variation as non-absolute diagnostics",
            (
                "prepare upv-support-v2 with biased MMD squared but do not execute it "
                "without a frozen transform decision"
            ),
            "freeze the positive-ploss safety-probe protocol without requesting POWDER time",
        ],
    }


def _reservation_gate(config: dict[str, Any], decision: dict[str, object]) -> dict[str, object]:
    probe = config["positive_ploss_probe"]
    first = probe["first_stage"]
    state_count = len(first["ploss_values"]) * len(first["noise_power_dB_values"])
    return {
        "schema_version": 1,
        "decision_code": decision["decision_code"],
        "reservation_should_be_requested_now": False,
        "preparation_lead_time_minutes": config["reservation_policy"][
            "preparation_lead_time_minutes"
        ],
        "current_blocker": (
            "absolute RSRP measurement equivalence is not identified and the UPV pathloss "
            "field has no values"
        ),
        "notification_rule": (
            "tell the user to start the reservation at least 30 minutes before the first "
            "frozen POWDER-dependent action"
        ),
        "conditional_safety_probe": {
            "label": probe["interpretation"],
            "not_final_support_extension": probe["not_final_support_extension"],
            "ploss_values": first["ploss_values"],
            "noise_power_dB_values": first["noise_power_dB_values"],
            "state_count": state_count,
            "minimum_repetitions_per_state": first["minimum_repetitions_per_state"],
            "minimum_executions": state_count * first["minimum_repetitions_per_state"],
            "preferred_repetitions_per_state": first["preferred_repetitions_per_state"],
            "preferred_executions": state_count * first["preferred_repetitions_per_state"],
            "trigger": (
                "only after the measurement-equivalence branch is selected, positive ploss "
                "is accepted as a simulator gain control, and this exact design is re-frozen"
            ),
        },
        "adaptive_localization": probe["adaptive_localization_after_safe_monotone_first_stage"],
        "slope_warning": probe["slope_warning"],
    }


def _input_inventory(paths: dict[str, Path], source_states: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name in ["upv_archive", "phase2_manifest", "phase2_gate", "config"]:
        path = paths[name]
        rows.append({
            "input": name,
            "source_id": path.name,
            "sha256": _sha256(path),
            "git_revision": None,
            "tracked_worktree_dirty": None,
        })
    for name in ["oai", "profile"]:
        state = source_states[name]
        rows.append({
            "input": f"{name}_source_checkout",
            "source_id": state["source_id"],
            "sha256": None,
            "git_revision": state["revision"],
            "tracked_worktree_dirty": state["tracked_worktree_dirty"],
        })
    return pd.DataFrame(rows)


def run_measurement_equivalence_audit(
    *,
    upv_archive: str | Path,
    phase2_manifest: str | Path,
    phase2_gate: str | Path,
    oai_source: str | Path,
    profile_source: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    paths = {
        "upv_archive": Path(upv_archive).resolve(),
        "phase2_manifest": Path(phase2_manifest).resolve(),
        "phase2_gate": Path(phase2_gate).resolve(),
        "oai_source": Path(oai_source).resolve(),
        "profile_source": Path(profile_source).resolve(),
        "config": Path(config_path).resolve(),
        "output": Path(output_dir).resolve(),
    }
    for name in ["upv_archive", "phase2_manifest", "phase2_gate", "config"]:
        if not paths[name].is_file() or paths[name].is_symlink():
            raise ValueError(f"missing or unsafe input: {paths[name]}")
    for name in ["oai_source", "profile_source"]:
        if not paths[name].is_dir() or paths[name].is_symlink():
            raise ValueError(f"missing or unsafe source checkout: {paths[name]}")
    if paths["output"].exists():
        raise FileExistsError(f"measurement audit output already exists: {paths['output']}")

    config = _read_yaml(paths["config"])
    validate_measurement_audit_config(config)
    phase2 = _read_json(paths["phase2_manifest"])
    gate2 = _read_json(paths["phase2_gate"])
    repositories = {"oai": paths["oai_source"], "profile": paths["profile_source"]}
    source_trace, source_states = _source_trace(config, repositories)
    upv_inventory, pathloss_reference, upv_summary = _upv_inventory(
        paths["upv_archive"], config
    )
    crosswalk = _measurement_crosswalk(upv_summary)
    ledger = _link_budget_ledger(config, upv_summary)
    dossier = _offset_dossier(config, gate2, upv_summary)
    decision = _decision(config, phase2, gate2, upv_summary)
    reservation = _reservation_gate(config, decision)
    inputs = _input_inventory(paths, source_states)
    software = _implementation_revision()

    staging = paths["output"].parent / f".{paths['output'].name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_csv(staging / "input_inventory.csv", inputs)
        _write_csv(staging / "source_code_trace.csv", source_trace)
        _write_csv(staging / "upv_field_completeness.csv", upv_inventory)
        _write_csv(staging / "upv_pathloss_reference_audit.csv", pathloss_reference)
        _write_csv(staging / "measurement_definition_crosswalk.csv", crosswalk)
        _write_csv(staging / "link_budget_ledger.csv", ledger)
        _write_json(staging / "offset_dossier.json", dossier)
        _write_json(staging / "phase3a_decision.json", decision)
        _write_json(staging / "reservation_gate_v2.json", reservation)
        protocol_update = {
            "schema_version": 1,
            "phase2_snapshot_unchanged": True,
            "phase2_estimator": config["frozen_inputs"]["phase2_estimator"],
            "next_protocol_version": "upv-support-v2",
            "future_mmd_protocol": config["future_mmd_protocol"],
            "offset_policy": config["offset_policy"],
            "abc_permitted": False,
        }
        _write_json(staging / "protocol_update_v2.json", protocol_update)

        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "name": config["name"],
            "stage": config["stage"],
            "analysis_implementation_revision": software["revision"],
            "tracked_worktree_dirty_at_start": software["tracked_worktree_dirty"],
            "frozen_inputs": config["frozen_inputs"],
            "source_states": source_states,
            "input_sha256": {
                row["input"]: row["sha256"]
                for row in inputs.to_dict("records")
                if isinstance(row["sha256"], str)
            },
            "upv_summary": upv_summary,
            "decision_code": decision["decision_code"],
            "absolute_rsrp_calibration_authorized": False,
            "abc_performed": False,
            "reservation_should_be_requested_now": False,
            "output_sha256_before_manifest": output_hashes,
        }
        _write_json(staging / "analysis_manifest.json", manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.json"
        }
        _write_json(staging / "SHA256SUMS.json", checksums)
        staging.replace(paths["output"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(paths["output"]),
        "decision_code": decision["decision_code"],
        "upv_serving_rsrp_rows": upv_summary["serving_rsrp_rows"],
        "upv_serving_pathloss_rows": upv_summary["serving_pathloss_rows"],
        "absolute_rsrp_calibration_authorized": False,
        "abc_performed": False,
        "reservation_should_be_requested_now": False,
    }
