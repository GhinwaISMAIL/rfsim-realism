from __future__ import annotations

import math
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .upv_phase3d import (
    _aggregate_session,
    _archive_members,
    _read_json,
    _read_yaml,
    _sha256,
    _sha256_bytes,
    _write_csv,
    _write_json,
    validate_phase3d_config,
)
from .upv_phase3j import translate_complete_trace, validate_phase3j_config
from .upv_protocol import _load_radio_csv, build_route_table


def validate_phase3k_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3K schema_version must be 1")
    if config.get("stage") != "phase_3k_frozen_model_release_and_unseen_target_support_check":
        raise ValueError("unexpected Phase 3K stage")
    if config.get("evaluation_status") != (
        "predesignated_untouched_session_target_trajectory_validation"
    ):
        raise ValueError("unexpected Phase 3K evaluation status")
    for flag in (
        "offline_test6_access_authorized_before_release_commit",
        "hardware_execution_authorized",
        "translator_update_authorized",
        "abc_authorized",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain false in the Phase 3K protocol")

    required = config["required_phase3j_result"]
    if required.get("decision_code") != "complete_test1_development_replay_passed":
        raise ValueError("Phase 3K requires the passed Phase 3J decision")
    if required.get("all_gates_passed") is not True:
        raise ValueError("all Phase 3J gates must have passed")
    if int(required.get("independent_executions", 0)) != 3:
        raise ValueError("Phase 3K requires three Phase 3J executions")

    freeze = config["model_freeze"]
    if freeze.get("translator_method") != "bounded_piecewise_affine_interpolation":
        raise ValueError("the frozen translator method changed")
    if freeze.get("translator_updates_from_phase3j_or_test6") != "prohibited":
        raise ValueError("Phase 3J and Test 6 may not update the translator")
    if freeze.get("extrapolation") != "prohibited":
        raise ValueError("Phase 3K prohibits extrapolation")
    if freeze.get("post_hoc_lag_selection") != "prohibited":
        raise ValueError("Phase 3K prohibits post-hoc lag selection")

    source = config["test6_source"]
    if source.get("member_path") != "Test_6/Test_6_ASUS.csv":
        raise ValueError("the predesignated Test 6 source changed")
    if source.get("session_id") != "corrected_test_6_ASUS":
        raise ValueError("the predesignated Test 6 session changed")
    if source.get("use") != "observed_target_trajectory_not_prediction_target":
        raise ValueError("Test 6 must be interpreted as an observed target trajectory")
    for key in ("archive_sha256", "member_sha256"):
        if len(str(source.get(key, ""))) != 64:
            raise ValueError(f"Test 6 {key} must be frozen")

    preprocessing = config["preprocessing"]
    if preprocessing.get("source") != "unchanged_from_phase3d_config":
        raise ValueError("Test 6 preprocessing must remain identical to Phase 3D")
    if preprocessing.get("interpolation") != "prohibited":
        raise ValueError("Test 6 interpolation is prohibited")
    if preprocessing.get("runner_input_requirement") != "one_contiguous_one_hz_sequence":
        raise ValueError("the Test 6 runner input rule changed")

    gate = config["test6_support_gate"]
    if float(gate.get("maximum_clipped_fraction", -1)) != 0.05:
        raise ValueError("the Test 6 clipped-fraction gate changed")
    if float(gate.get("maximum_clipping_distance_scaled", -1)) != 1.0:
        raise ValueError("the Test 6 clipping-distance gate changed")
    if gate.get("hardware_execution_if_unsupported") != "prohibited":
        raise ValueError("unsupported Test 6 trajectories may not be executed")
    amendment = gate["pre_access_runtime_gate_amendment"]
    reference_rows = int(amendment["reference_target_rows"])
    legacy_rows = int(amendment["legacy_absolute_minimum_paired_rows"])
    fraction = float(amendment["minimum_paired_fraction"])
    if reference_rows != 305 or legacy_rows != 299:
        raise ValueError("the disclosed Phase 3J paired-row reference changed")
    if not math.isclose(fraction, legacy_rows / reference_rows, abs_tol=1e-15):
        raise ValueError("the Test 6 paired-coverage fraction is inconsistent")
    if amendment.get("thresholds_selected_from_test6_kpi_distribution") is not False:
        raise ValueError("the runtime amendment may not inspect Test 6 KPI values")

    replay = config["test6_replay_if_supported"]
    if int(replay.get("repetitions", 0)) != 3:
        raise ValueError("final replay must use three executions")
    if len(set(replay.get("oai_rng_seeds", []))) != 3:
        raise ValueError("final replay seeds must be distinct")
    if replay.get("hardware_authorization_requires_separate_freeze") is not True:
        raise ValueError("offline support may not authorize hardware")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("the reservation gate must remain closed")


def _verify_bundle(directory: Path, expected_manifest_sha256: str) -> dict[str, str]:
    manifest_path = directory / "SHA256SUMS.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"bundle checksum-manifest mismatch: {directory}")
    manifest = _read_json(manifest_path)
    for name, digest in manifest.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"bundle file checksum mismatch: {path}")
    return {str(name): str(digest) for name, digest in manifest.items()}


def _verify_preprocessing_equivalence(
    phase3k_config: dict[str, Any], phase3d_config: dict[str, Any]
) -> None:
    phase3k = phase3k_config["preprocessing"]
    phase3d = phase3d_config["preprocessing"]
    shared = (
        "temporal_aggregation_seconds",
        "aggregation_statistic",
        "interpolation",
        "maximum_gap_seconds",
        "minimum_sequence_rows",
        "relative_rsrp_definition",
        "sinr_definition",
        "paired_missing_value_rule",
        "preserve_synchronized_pairs",
    )
    for key in shared:
        if phase3k.get(key) != phase3d.get(key):
            raise ValueError(f"Test 6 preprocessing differs from Phase 3D: {key}")
    if phase3k.get("long_gap_rule") != phase3d.get("long_gap_rule"):
        raise ValueError("Test 6 long-gap handling differs from Phase 3D")


def _verify_test6_source_identity(
    phase3k_config: dict[str, Any], phase3d_config: dict[str, Any]
) -> None:
    source = phase3k_config["test6_source"]
    final = phase3d_config["final_evaluation"]
    if source["archive_sha256"] != phase3d_config["source"]["expected_sha256"]:
        raise ValueError("Test 6 archive identity differs from the Phase 3D freeze")
    expected = {
        "member_path": final["source_path"],
        "member_sha256": final["source_sha256"],
        "corrected_test_id": final["corrected_test_id"],
        "device": final["device"],
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"Test 6 source identity differs from Phase 3D: {key}")


def _repository_state(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"revision": revision, "tracked_worktree_dirty": bool(status)}


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def freeze_phase3k_model_release(
    *,
    config_path: str | Path,
    phase3d_config_path: str | Path,
    phase3j_config_path: str | Path,
    phase3j_protocol_dir: str | Path,
    phase3j_result_dir: str | Path,
    profile_runner_path: str | Path,
    profile_engine_path: str | Path,
    pyproject_path: str | Path,
    uv_lock_path: str | Path,
    output_dir: str | Path,
    require_clean_repositories: bool = True,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3d_file = Path(phase3d_config_path).resolve()
    phase3j_file = Path(phase3j_config_path).resolve()
    protocol_dir = Path(phase3j_protocol_dir).resolve()
    result_dir = Path(phase3j_result_dir).resolve()
    profile_runner = Path(profile_runner_path).resolve()
    profile_engine = Path(profile_engine_path).resolve()
    pyproject = Path(pyproject_path).resolve()
    uv_lock = Path(uv_lock_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3K model release already exists: {output}")

    config = _read_yaml(config_file)
    validate_phase3k_config(config)
    phase3d_config = _read_yaml(phase3d_file)
    phase3j_config = _read_yaml(phase3j_file)
    validate_phase3d_config(phase3d_config)
    validate_phase3j_config(phase3j_config)
    _verify_preprocessing_equivalence(config, phase3d_config)
    _verify_test6_source_identity(config, phase3d_config)
    frozen = config["frozen_inputs"]
    expected_files = {
        phase3d_file: frozen["phase3d_config"]["sha256"],
        phase3j_file: frozen["phase3j_config"]["sha256"],
        pyproject: frozen["software_environment"]["pyproject_sha256"],
        uv_lock: frozen["software_environment"]["uv_lock_sha256"],
        profile_runner: frozen["profile"]["full_trace_runner_sha256"],
        profile_engine: frozen["profile"]["replay_engine_sha256"],
    }
    for path, expected in expected_files.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen Phase 3K input checksum mismatch: {path}")

    protocol_checksums = _verify_bundle(
        protocol_dir, frozen["phase3j_protocol_bundle"]["checksums_sha256"]
    )
    result_checksums = _verify_bundle(
        result_dir, frozen["phase3j_result_bundle"]["checksums_sha256"]
    )
    if protocol_checksums.get("protocol.json") != (
        frozen["phase3j_protocol_bundle"]["protocol_sha256"]
    ):
        raise ValueError("the Phase 3J frozen protocol identity changed")
    if protocol_checksums.get("translator_support_nodes.csv") != (
        frozen["phase3j_protocol_bundle"]["translator_support_nodes_sha256"]
    ):
        raise ValueError("the Phase 3J translator support changed")
    if protocol_checksums.get("test6_support_rules.json") != (
        frozen["phase3j_protocol_bundle"]["original_test6_support_rules_sha256"]
    ):
        raise ValueError("the original Phase 3J Test 6 support rules changed")
    if result_checksums.get("phase3j_full_trace_decision.json") != (
        frozen["phase3j_result_bundle"]["decision_sha256"]
    ):
        raise ValueError("the Phase 3J decision identity changed")
    if result_checksums.get("per_execution_metrics.csv") != (
        frozen["phase3j_result_bundle"]["per_execution_metrics_sha256"]
    ):
        raise ValueError("the Phase 3J execution metrics changed")

    decision = _read_json(result_dir / "phase3j_full_trace_decision.json")
    required = config["required_phase3j_result"]
    if decision.get("decision_code") != required["decision_code"]:
        raise ValueError("Phase 3J did not produce the required decision")
    if decision.get("gates", {}).get("all_gates_passed") is not True:
        raise ValueError("Phase 3J did not pass every frozen gate")
    if decision.get("model_release_freeze_authorized") is not True:
        raise ValueError("Phase 3J did not authorize a model release")
    if decision.get("test6_accessed") is not False:
        raise ValueError("the Phase 3J Test 6 access lock is not intact")
    if decision.get("translator_update_from_residuals_authorized") is not False:
        raise ValueError("Phase 3J improperly authorized translator updates")
    if int(decision.get("campaign", {}).get("executions", 0)) != int(
        required["independent_executions"]
    ):
        raise ValueError("the Phase 3J execution count changed")

    research_root = Path(__file__).resolve().parents[2]
    profile_root = profile_runner.parents[1]
    research_state = _repository_state(research_root)
    profile_state = _repository_state(profile_root)
    if profile_state["revision"] != frozen["profile"]["expected_revision"]:
        raise ValueError("the frozen profile repository revision changed")
    if require_clean_repositories and (
        research_state["tracked_worktree_dirty"] or profile_state["tracked_worktree_dirty"]
    ):
        raise ValueError("model release requires clean tracked repositories")

    inventory_rows = [
        {
            "role": "phase3k_config",
            "path": _display_path(config_file, research_root),
            "sha256": _sha256(config_file),
        },
        *[
            {
                "role": f"frozen_input_{index:02d}",
                "path": _display_path(path, research_root),
                "sha256": digest,
            }
            for index, (path, digest) in enumerate(expected_files.items(), start=1)
        ],
        {
            "role": "phase3j_protocol_checksums",
            "path": _display_path(protocol_dir / "SHA256SUMS.json", research_root),
            "sha256": _sha256(protocol_dir / "SHA256SUMS.json"),
        },
        {
            "role": "phase3j_result_checksums",
            "path": _display_path(result_dir / "SHA256SUMS.json", research_root),
            "sha256": _sha256(result_dir / "SHA256SUMS.json"),
        },
    ]
    release = {
        "schema_version": 1,
        "stage": "phase_3k_immutable_model_release",
        "protocol_revision": config["protocol_revision"],
        "evaluation_status": config["evaluation_status"],
        "analysis_repository": research_state,
        "profile_repository": profile_state,
        "phase3j_decision_code": decision["decision_code"],
        "phase3j_all_gates_passed": True,
        "translator": config["model_freeze"],
        "test6_source_identity": {
            key: config["test6_source"][key]
            for key in (
                "archive_sha256",
                "member_path",
                "member_sha256",
                "session_id",
                "device",
            )
        },
        "test6_preprocessing": config["preprocessing"],
        "test6_support_gate": config["test6_support_gate"],
        "test6_replay_if_supported": config["test6_replay_if_supported"],
        "claim_limits": config["claim_limits"],
        "test6_payload_opened": False,
        "hardware_execution_authorized": False,
        "translator_update_authorized": False,
        "reservation_requested": False,
    }
    authorization = {
        "schema_version": 1,
        "stage": "phase_3k_offline_test6_access_authorization",
        "status_before_release_commit": "inactive_pending_commit",
        "activation_condition": (
            "every model-release file is committed at HEAD with matching checksums "
            "and the tracked research worktree is clean"
        ),
        "authorized_operation_after_activation": (
            "open_only_Test_6/Test_6_ASUS.csv_for_the_predeclared_offline_support_check"
        ),
        "prohibited_operations": [
            "translator_fitting_or_update",
            "threshold_update",
            "post_hoc_alignment_selection",
            "hardware_execution",
            "powder_reservation_request",
        ],
        "test6_payload_opened": False,
        "hardware_execution_authorized": False,
    }
    output.mkdir(parents=True)
    _write_csv(output / "frozen_file_inventory.csv", pd.DataFrame(inventory_rows))
    _write_json(output / "model_release.json", release)
    _write_json(output / "offline_test6_access_authorization.json", authorization)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "phase3j_decision": str(decision["decision_code"]),
        "test6_payload_opened": "false",
        "offline_test6_access": "pending_release_commit",
        "hardware_execution_authorized": "false",
        "reservation_requested": "false",
    }


def _require_committed_files(paths: list[Path]) -> None:
    repository = Path(__file__).resolve().parents[2]
    state = _repository_state(repository)
    if state["tracked_worktree_dirty"]:
        raise ValueError("Test 6 access requires a clean tracked research worktree")
    for path in sorted(paths):
        relative = path.relative_to(repository)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative.as_posix()],
            cwd=repository,
            check=True,
        )


def check_phase3k_test6_support(
    *,
    config_path: str | Path,
    phase3d_config_path: str | Path,
    phase3j_config_path: str | Path,
    model_release_dir: str | Path,
    translator_support_path: str | Path,
    archive_path: str | Path,
    output_dir: str | Path,
    require_committed_release: bool = True,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3d_file = Path(phase3d_config_path).resolve()
    phase3j_file = Path(phase3j_config_path).resolve()
    release_dir = Path(model_release_dir).resolve()
    support_file = Path(translator_support_path).resolve()
    archive_file = Path(archive_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3K support output already exists: {output}")

    config = _read_yaml(config_file)
    validate_phase3k_config(config)
    phase3d_config = _read_yaml(phase3d_file)
    phase3j_config = _read_yaml(phase3j_file)
    validate_phase3d_config(phase3d_config)
    validate_phase3j_config(phase3j_config)
    _verify_preprocessing_equivalence(config, phase3d_config)
    _verify_test6_source_identity(config, phase3d_config)
    frozen = config["frozen_inputs"]
    if _sha256(phase3d_file) != frozen["phase3d_config"]["sha256"]:
        raise ValueError("the frozen Phase 3D preprocessing config changed")
    if _sha256(phase3j_file) != frozen["phase3j_config"]["sha256"]:
        raise ValueError("the frozen Phase 3J translator config changed")
    if _sha256(support_file) != (
        frozen["phase3j_protocol_bundle"]["translator_support_nodes_sha256"]
    ):
        raise ValueError("the frozen translator support nodes changed")
    release_checksums = _read_json(release_dir / "SHA256SUMS.json")
    for name, digest in release_checksums.items():
        if _sha256(release_dir / name) != digest:
            raise ValueError(f"model-release bundle checksum mismatch: {name}")
    release = _read_json(release_dir / "model_release.json")
    if release.get("test6_payload_opened") is not False:
        raise ValueError("the pre-access model release is invalid")
    if release.get("translator_update_authorized") is not False:
        raise ValueError("the model release improperly authorizes translator updates")
    if require_committed_release:
        release_files = [path for path in release_dir.iterdir() if path.is_file()]
        _require_committed_files(
            [
                Path(__file__).resolve(),
                config_file,
                phase3d_file,
                phase3j_file,
                support_file,
                *release_files,
            ]
        )

    source = config["test6_source"]
    if _sha256(archive_file) != source["archive_sha256"]:
        raise ValueError("UPV source archive checksum mismatch")
    with zipfile.ZipFile(archive_file) as archive:
        members = _archive_members(archive)
        if source["member_path"] not in members:
            raise ValueError("the frozen Test 6 member is missing")
        payload = archive.read(members[source["member_path"]])
    if _sha256_bytes(payload) != source["member_sha256"]:
        raise ValueError("the frozen Test 6 member checksum changed")

    radio, raw_quality = _load_radio_csv(payload)
    route = build_route_table(
        radio,
        source_path=source["member_path"],
        corrected_test_id=int(source["corrected_test_id"]),
        bin_sizes_m=[15],
        minimum_step_m_for_heading=0.1,
        direction_sectors=8,
    )
    valid_pci = route["serving_pci"].dropna()
    pci_fraction = float((valid_pci == float(source["serving_pci"])).mean())
    pci_gate_passed = bool(pci_fraction >= float(source["minimum_serving_pci_fraction"]))
    trace = _aggregate_session(route, session_id=source["session_id"], config=phase3d_config)
    sequence_count = int(trace["sequence_id"].nunique())
    one_hz = bool(
        len(trace) > 1 and np.allclose(np.diff(trace["t_s"].to_numpy(float)), 1.0)
    )
    contiguous_gate_passed = sequence_count == 1 and one_hz
    support = pd.read_csv(support_file)
    commands, _ = translate_complete_trace(trace, support, phase3j_config)
    clipped = commands["clipped"].astype(bool)
    clipped_fraction = float(clipped.mean())
    maximum_distance = (
        float(commands.loc[clipped, "clipping_distance_scaled"].max())
        if clipped.any()
        else 0.0
    )
    gate = config["test6_support_gate"]
    clip_fraction_gate_passed = bool(
        clipped_fraction <= float(gate["maximum_clipped_fraction"])
    )
    clipping_distance_gate_passed = bool(
        maximum_distance <= float(gate["maximum_clipping_distance_scaled"])
    )
    amendment = gate["pre_access_runtime_gate_amendment"]
    minimum_paired_fraction = float(amendment["minimum_paired_fraction"])
    minimum_paired_rows = math.ceil(minimum_paired_fraction * len(trace))
    supported = bool(
        pci_gate_passed
        and contiguous_gate_passed
        and clip_fraction_gate_passed
        and clipping_distance_gate_passed
    )
    decision_rule = config["decision_rules"]["supported" if supported else "unsupported"]
    clipped_rows = commands.loc[clipped].copy()
    clipped_rows["relative_rsrp_clipping_error_db"] = (
        clipped_rows["projected_relative_rsrp_db"]
        - clipped_rows["target_relative_rsrp_db"]
    )
    clipped_rows["sinr_clipping_error_db"] = (
        clipped_rows["projected_sinr_db"] - clipped_rows["target_sinr_db"]
    )
    decision = {
        "schema_version": 1,
        "stage": "phase_3k_test6_offline_support_result",
        "evaluation_status": config["evaluation_status"],
        "decision_code": decision_rule["code"],
        "next_action": decision_rule["next_action"],
        "test6_payload_opened": True,
        "test6_used_for_fitting_selection_or_threshold_tuning": False,
        "test6_use": source["use"],
        "target_rows": len(trace),
        "sequence_count": sequence_count,
        "one_contiguous_one_hz_sequence": contiguous_gate_passed,
        "serving_pci_fraction": pci_fraction,
        "serving_pci_gate_passed": pci_gate_passed,
        "inside_rows": int((~clipped).sum()),
        "clipped_rows": int(clipped.sum()),
        "clipped_fraction": clipped_fraction,
        "maximum_clipping_distance_scaled": maximum_distance,
        "gates": {
            "clipped_fraction_gate_passed": clip_fraction_gate_passed,
            "clipping_distance_gate_passed": clipping_distance_gate_passed,
            "trajectory_support_gate_passed": supported,
        },
        "runtime_gate_for_future_execution": {
            "minimum_paired_fraction": minimum_paired_fraction,
            "minimum_paired_rows": minimum_paired_rows,
            "target_rows": len(trace),
        },
        "translator_changed": False,
        "hardware_execution_authorized": False,
        "reservation_requested": False,
        "claim_limits": config["claim_limits"],
    }
    source_audit = {
        "schema_version": 1,
        "stage": "phase_3k_test6_source_audit",
        "archive_path": str(archive_file),
        "archive_sha256": _sha256(archive_file),
        "member_path": source["member_path"],
        "member_sha256": _sha256_bytes(payload),
        "member_payload_opened": True,
        "other_archive_member_payloads_opened_by_this_command": 0,
        "raw_quality": raw_quality,
        "route_rows": len(route),
        "aggregated_target_rows": len(trace),
        "prior_inspection_disclosure": source["prior_inspection_disclosure"],
    }
    output.mkdir(parents=True)
    _write_csv(output / "test6_target_trace.csv", trace)
    _write_csv(output / "test6_commands.csv", commands)
    _write_csv(output / "test6_clipped_targets.csv", clipped_rows)
    _write_json(output / "test6_support_decision.json", decision)
    _write_json(output / "source_audit.json", source_audit)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision_code": str(decision["decision_code"]),
        "target_rows": str(len(trace)),
        "clipped_rows": str(int(clipped.sum())),
        "clipped_fraction": f"{clipped_fraction:.12g}",
        "maximum_clipping_distance_scaled": f"{maximum_distance:.12g}",
        "minimum_paired_rows_for_future_execution": str(minimum_paired_rows),
        "hardware_execution_authorized": "false",
        "reservation_requested": "false",
    }
