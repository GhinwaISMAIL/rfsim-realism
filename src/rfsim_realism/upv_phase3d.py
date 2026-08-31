from __future__ import annotations

import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .mmd_abc import biased_rbf_mmd2, median_heuristic_bandwidth
from .upv_protocol import (
    _load_radio_csv,
    _normal_member_path,
    build_route_table,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
    temporary.replace(path)


def _git_revision() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
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
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "tracked_worktree_dirty": None}
    return {"revision": revision, "tracked_worktree_dirty": bool(status)}


def validate_phase3d_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3D schema_version must be 1")
    if config.get("stage") != "phase_3d_offline_radio_process_feasibility":
        raise ValueError("unexpected Phase 3D stage")
    if any(
        bool(config.get(key))
        for key in ("execution_authorized", "abc_authorized", "final_evaluation_authorized")
    ):
        raise ValueError("Phase 3D development cannot authorize execution, ABC, or final access")

    source = config.get("source") or {}
    if len(str(source.get("expected_sha256") or "")) != 64:
        raise ValueError("the source archive checksum must be frozen")
    if source.get("required_phase3c15_decision") != ("gain_only_rejected_noise_control_required"):
        raise ValueError("Phase 3D requires the Phase 3C15 gain-only rejection")

    development = config.get("development") or {}
    sessions = development.get("sessions") or []
    if development.get("cross_validation") != "leave_one_complete_session_out":
        raise ValueError("Phase 3D requires leave-one-complete-session-out development")
    if len(sessions) != 5:
        raise ValueError("Phase 3D requires five frozen development sessions")
    source_paths = [str(value.get("source_path") or "") for value in sessions]
    if len(set(source_paths)) != len(source_paths) or any(not value for value in source_paths):
        raise ValueError("development source paths must be present and unique")
    corrected_ids = [int(value.get("corrected_test_id", 0)) for value in sessions]
    if corrected_ids != [1, 2, 3, 4, 5]:
        raise ValueError("development corrected test IDs must remain 1 through 5")
    if any(len(str(value.get("source_sha256") or "")) != 64 for value in sessions):
        raise ValueError("every development source checksum must be frozen")

    final = config.get("final_evaluation") or {}
    if final.get("source_path") in source_paths:
        raise ValueError("the final session cannot appear in development")
    if final.get("status") != "locked_not_opened_by_phase3d_development_analysis":
        raise ValueError("the final session must remain locked")
    if "not represented as never previously parsed" not in str(
        final.get("prior_inspection_disclosure") or ""
    ):
        raise ValueError("the prior-inspection disclosure cannot be removed")
    if len(str(final.get("source_sha256") or "")) != 64:
        raise ValueError("the final session checksum must be frozen")

    preprocessing = config.get("preprocessing") or {}
    if preprocessing.get("interpolation") != "prohibited":
        raise ValueError("Phase 3D prohibits temporal interpolation")
    if preprocessing.get("long_gap_rule") != "split_sequence_and_never_bridge":
        raise ValueError("long gaps must split sequences")
    if not bool(preprocessing.get("preserve_session_boundaries")):
        raise ValueError("session boundaries must be preserved")
    if not bool(preprocessing.get("preserve_synchronized_pairs")):
        raise ValueError("RSRP and SINR pairs must remain synchronized")
    spatial = preprocessing.get("spatial_conditioning") or {}
    if spatial.get("fitted_inside_each_cross_validation_fold") is not True:
        raise ValueError("spatial conditioning must be fitted inside every fold")
    if spatial.get("extrapolation") != "prohibited":
        raise ValueError("unsupported spatial extrapolation is prohibited")

    models = config.get("models") or {}
    expected_candidates = {
        "gaussian_1state",
        "gaussian_2state",
        "student_t_1state",
        "student_t_2state",
        "gamma_gaussian_1state",
        "gamma_gaussian_2state",
    }
    observed_candidates = {str(value.get("id") or "") for value in models.get("candidates") or []}
    if observed_candidates != expected_candidates:
        raise ValueError("the frozen one-state and two-state candidates are incomplete")
    hmm = models.get("hmm") or {}
    if int(hmm.get("initializations", 0)) < 2:
        raise ValueError("multiple HMM initializations are required")
    occupancy = float(hmm.get("minimum_state_occupancy", 0))
    if not 0 < occupancy < 0.5:
        raise ValueError("state occupancy threshold must be in (0, 0.5)")

    evaluation = config.get("evaluation") or {}
    if int(evaluation.get("generation_repetitions", 0)) < 20:
        raise ValueError("at least twenty synthetic repetitions are required")
    if evaluation.get("joint_distribution", {}).get("statistic") != (
        "biased_rbf_mmd_squared_v_statistic"
    ):
        raise ValueError("Phase 3D requires nonnegative biased MMD squared")

    rules = config.get("decision_rules") or {}
    if int(rules.get("required_development_folds", 0)) != len(sessions):
        raise ValueError("the decision rule must require every development fold")
    superiority = rules.get("hmm_better_than_bootstrap") or {}
    for key in (
        "minimum_median_joint_mmd_relative_improvement",
        "minimum_median_temporal_relative_improvement",
    ):
        if float(superiority.get(key, 0)) <= 0:
            raise ValueError("HMM superiority margins must be positive")

    claims = config.get("claim_limits") or {}
    required_prohibitions = {
        "physical_channel_reconstruction",
        "multipath_or_doppler_reconstruction",
        "causal_interference_state_interpretation",
        "absolute_rsrp_calibration",
        "physical_ploss_inference",
        "absolute_noise_power_calibration",
        "attachment_distribution_validation",
        "final_device_or_population_generalization_before_final_evaluation",
        "abc_primary_pipeline",
    }
    if any(claims.get(key) != "prohibited" for key in required_prohibitions):
        raise ValueError("Phase 3D claim limits are incomplete")

    reservation = config.get("reservation") or {}
    if bool(reservation.get("request_now")):
        raise ValueError("the offline phase cannot request POWDER")
    if reservation.get("gate_state") != "closed_for_offline_development":
        raise ValueError("the reservation gate must remain closed")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        normalized = _normal_member_path(info.filename)
        if normalized in result:
            raise ValueError(f"duplicate normalized archive member: {normalized}")
        result[normalized] = info
    return result


def _verify_prerequisites(
    archive_path: Path,
    phase3c15_result_path: Path,
    config: dict[str, Any],
) -> dict[str, zipfile.ZipInfo]:
    if _sha256(archive_path) != config["source"]["expected_sha256"]:
        raise ValueError("UPV archive checksum mismatch")
    phase3c15 = _read_json(phase3c15_result_path)
    if phase3c15.get("decision") != config["source"]["required_phase3c15_decision"]:
        raise ValueError("Phase 3C15 prerequisite is not satisfied")
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive)
    expected = [
        *[value["source_path"] for value in config["development"]["sessions"]],
        config["final_evaluation"]["source_path"],
    ]
    missing = sorted(set(expected) - set(members))
    if missing:
        raise ValueError(f"UPV archive is missing frozen sessions: {missing}")
    return members


def _repository_file(relative: str) -> Path:
    return Path(__file__).resolve().parents[2] / relative


def write_phase3d_protocol_freeze(
    *,
    archive_path: str | Path,
    phase3c15_result_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    phase3c15 = Path(phase3c15_result_path).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if not archive.is_file() or not phase3c15.is_file() or not config_file.is_file():
        raise ValueError("Phase 3D freeze inputs must be regular files")
    config = _read_yaml(config_file)
    validate_phase3d_config(config)
    members = _verify_prerequisites(archive, phase3c15, config)

    development_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as source:
        normalized = _archive_members(source)
        for session in config["development"]["sessions"]:
            payload = source.read(normalized[session["source_path"]])
            if _sha256_bytes(payload) != session["source_sha256"]:
                raise ValueError(f"development member checksum mismatch: {session['source_path']}")
            development_rows.append(
                {
                    "source_path": session["source_path"],
                    "source_sha256": session["source_sha256"],
                    "corrected_test_id": session["corrected_test_id"],
                    "device": config["development"]["device"],
                    "trim_last_seconds": session["trim_last_seconds"],
                    "role": "development_cross_validation",
                }
            )

    final = config["final_evaluation"]
    final_info = members[final["source_path"]]
    final_lock = {
        "schema_version": 1,
        "stage": "phase_3d_final_evaluation_access_lock",
        "source_path": final["source_path"],
        "source_sha256_from_frozen_phase1_inventory": final["source_sha256"],
        "archive_member_size_bytes": final_info.file_size,
        "archive_member_crc32": f"0x{final_info.CRC:08x}",
        "payload_opened_by_freeze": False,
        "payload_opened_by_development_analysis": False,
        "status": final["status"],
        "access_rule": final["access_rule"],
        "prior_inspection_disclosure": final["prior_inspection_disclosure"],
    }

    tracked_files = {
        "protocol_report": "reports/UPV_PHASE3D_RADIO_PROCESS_PROTOCOL.md",
        "evaluator": "src/rfsim_realism/upv_phase3d.py",
        "tests": "tests/test_upv_phase3d.py",
        "cli": "src/rfsim_realism/cli.py",
        "makefile": "Makefile",
    }
    file_records = {
        key: {"path": value, "sha256": _sha256(_repository_file(value))}
        for key, value in tracked_files.items()
    }
    try:
        config_display_path = str(config_file.relative_to(_repository_file(".")))
    except ValueError:
        config_display_path = str(config_file)
    file_records["config"] = {
        "path": config_display_path,
        "sha256": _sha256(config_file),
    }
    freeze = {
        "schema_version": 1,
        "stage": "phase_3d_radio_process_protocol_freeze",
        "protocol_revision": config["protocol_revision"],
        "research_question": config["research_question"],
        "repository": _git_revision(),
        "files": file_records,
        "source_archive": {
            "path": str(archive),
            "sha256": config["source"]["expected_sha256"],
        },
        "phase3c15_result": {
            "path": str(phase3c15),
            "sha256": _sha256(phase3c15),
            "required_decision": config["source"]["required_phase3c15_decision"],
        },
        "development_sessions": len(development_rows),
        "final_evaluation_payload_opened": False,
        "execution_authorized": False,
        "abc_authorized": False,
        "final_evaluation_authorized": False,
        "reservation": config["reservation"],
    }
    source_audit = {
        "schema_version": 1,
        "stage": "phase_3d_radio_process_source_audit",
        "archive_checksum_verified": True,
        "development_member_checksums_verified": True,
        "development_member_payloads_opened": len(development_rows),
        "final_member_presence_verified_from_zip_directory": True,
        "final_member_payload_opened": False,
        "prior_global_inspection_disclosed": True,
        "phase3c15_prerequisite_verified": True,
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "development_split.csv", pd.DataFrame(development_rows))
    _write_json(output / "final_evaluation_lock.json", final_lock)
    _write_json(output / "source_audit.json", source_audit)
    _write_json(output / "protocol_freeze.json", freeze)
    generated = [
        "development_split.csv",
        "final_evaluation_lock.json",
        "protocol_freeze.json",
        "source_audit.json",
    ]
    checksums = {name: _sha256(output / name) for name in generated}
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "development_sessions": len(development_rows),
        "final_evaluation_locked": True,
        "reservation_requested": False,
    }


def _verify_protocol_freeze(
    protocol_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    checksums = _read_json(protocol_dir / "SHA256SUMS.json")
    for name, expected in checksums.items():
        if _sha256(protocol_dir / name) != expected:
            raise ValueError(f"protocol bundle checksum mismatch: {name}")
    freeze = _read_json(protocol_dir / "protocol_freeze.json")
    if freeze.get("final_evaluation_payload_opened") is not False:
        raise ValueError("the final evaluation access lock is not intact")
    if freeze["files"]["config"]["sha256"] != _sha256(config_path):
        raise ValueError("the analysis config differs from the frozen protocol")
    evaluator = _repository_file(freeze["files"]["evaluator"]["path"])
    if _sha256(evaluator) != freeze["files"]["evaluator"]["sha256"]:
        raise ValueError("the evaluator differs from the frozen protocol")
    return freeze


def _aggregate_session(
    route: pd.DataFrame,
    *,
    session_id: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    preprocessing = config["preprocessing"]
    duration = float(preprocessing["temporal_aggregation_seconds"])
    maximum_gap = float(preprocessing["maximum_gap_seconds"])
    minimum_rows = int(preprocessing["minimum_sequence_rows"])
    frame = route.copy().sort_values("seconds_of_day").reset_index(drop=True)
    origin = float(frame["seconds_of_day"].iloc[0])
    frame["time_bin"] = np.floor((frame["seconds_of_day"] - origin) / duration).astype(int)
    aggregated = (
        frame.groupby("time_bin", sort=True)[
            ["route_fraction", "rsrp_dbm", "sinr_db", "serving_pci"]
        ]
        .median()
        .reset_index()
    )
    aggregated = aggregated.dropna(subset=["route_fraction", "rsrp_dbm", "sinr_db"])
    if aggregated.empty:
        raise ValueError(f"development session has no paired aggregated rows: {session_id}")
    aggregated["relative_rsrp_db"] = aggregated["rsrp_dbm"] - float(aggregated["rsrp_dbm"].median())
    aggregated["t_s"] = aggregated["time_bin"].astype(float) * duration
    gap = aggregated["t_s"].diff().fillna(0.0)
    aggregated["sequence_number"] = (gap > maximum_gap).cumsum().astype(int)
    sizes = aggregated.groupby("sequence_number").size()
    valid = set(sizes[sizes >= minimum_rows].index.astype(int))
    aggregated = aggregated[aggregated["sequence_number"].isin(valid)].copy()
    if aggregated.empty:
        raise ValueError(f"development session has no sufficiently long sequence: {session_id}")
    aggregated["session_id"] = session_id
    aggregated["sequence_id"] = [
        f"{session_id}:segment-{int(value)}" for value in aggregated["sequence_number"]
    ]
    return aggregated[
        [
            "session_id",
            "sequence_id",
            "time_bin",
            "t_s",
            "route_fraction",
            "rsrp_dbm",
            "relative_rsrp_db",
            "sinr_db",
            "serving_pci",
        ]
    ].reset_index(drop=True)


def _load_development_sessions(
    archive_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    final_path = config["final_evaluation"]["source_path"]
    sessions: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        for frozen in config["development"]["sessions"]:
            source_path = frozen["source_path"]
            if source_path == final_path:
                raise AssertionError("the final evaluation member cannot be opened")
            payload = archive.read(members[source_path])
            if _sha256_bytes(payload) != frozen["source_sha256"]:
                raise ValueError(f"development member checksum mismatch: {source_path}")
            radio, raw_quality = _load_radio_csv(payload)
            trim = float(frozen["trim_last_seconds"])
            if trim > 0:
                cutoff = float(radio["seconds_of_day"].max()) - trim
                radio = radio[radio["seconds_of_day"] <= cutoff].copy()
            route = build_route_table(
                radio,
                source_path=source_path,
                corrected_test_id=int(frozen["corrected_test_id"]),
                bin_sizes_m=[15],
                minimum_step_m_for_heading=0.1,
                direction_sectors=8,
            )
            valid_pci = route["serving_pci"].dropna()
            pci_fraction = float((valid_pci == float(config["source"]["serving_pci"])).mean())
            if pci_fraction < float(config["source"]["minimum_serving_pci_fraction"]):
                raise ValueError(f"serving PCI gate failed: {source_path}")
            session_id = f"corrected_test_{int(frozen['corrected_test_id'])}_ASUS"
            aggregated = _aggregate_session(route, session_id=session_id, config=config)
            sessions[session_id] = aggregated
            quality_rows.append(
                {
                    "session_id": session_id,
                    "source_path": source_path,
                    "source_sha256": frozen["source_sha256"],
                    "corrected_test_id": int(frozen["corrected_test_id"]),
                    "raw_complete_radio_triplets": raw_quality["complete_radio_triplets"],
                    "raw_complete_radio_gps_rows": raw_quality["complete_radio_gps_rows"],
                    "aggregated_rows": len(aggregated),
                    "sequence_count": aggregated["sequence_id"].nunique(),
                    "serving_pci_fraction": pci_fraction,
                    "trim_last_seconds": trim,
                }
            )
    if final_path not in members:
        raise ValueError("the locked final member is missing")
    return sessions, pd.DataFrame(quality_rows).sort_values("corrected_test_id")


def _spatial_bin(values: pd.Series, width: float) -> pd.Series:
    maximum = max(math.ceil(1.0 / width) - 1, 0)
    return np.floor(values.clip(0.0, 1.0) / width).astype(int).clip(upper=maximum)


def _fit_route_means(
    sessions: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    spatial = config["preprocessing"]["spatial_conditioning"]
    width = float(spatial["bin_width"])
    minimum_sessions = int(spatial["minimum_training_sessions_per_bin"])
    rows: list[pd.DataFrame] = []
    for session_id, frame in sessions.items():
        value = frame.copy()
        value["spatial_bin"] = _spatial_bin(value["route_fraction"], width)
        grouped = (
            value.groupby("spatial_bin", sort=True)[["relative_rsrp_db", "sinr_db"]]
            .median()
            .reset_index()
        )
        grouped["session_id"] = session_id
        rows.append(grouped)
    per_session = pd.concat(rows, ignore_index=True)
    summary = (
        per_session.groupby("spatial_bin", sort=True)
        .agg(
            supporting_sessions=("session_id", "nunique"),
            route_relative_rsrp_db=("relative_rsrp_db", "median"),
            route_sinr_db=("sinr_db", "median"),
        )
        .reset_index()
    )
    summary["supported"] = summary["supporting_sessions"] >= minimum_sessions
    summary["bin_start_fraction"] = summary["spatial_bin"] * width
    summary["bin_end_fraction"] = (summary["spatial_bin"] + 1) * width
    return summary


def _apply_route_means(
    frame: pd.DataFrame,
    route_means: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    width = float(config["preprocessing"]["spatial_conditioning"]["bin_width"])
    value = frame.copy()
    value["spatial_bin"] = _spatial_bin(value["route_fraction"], width)
    means = route_means[route_means["supported"]][
        ["spatial_bin", "route_relative_rsrp_db", "route_sinr_db"]
    ]
    value = value.merge(means, on="spatial_bin", how="left", validate="many_to_one")
    value["spatial_supported"] = value["route_relative_rsrp_db"].notna()
    total = len(value)
    value = value[value["spatial_supported"]].copy()
    value["rsrp_residual_db"] = value["relative_rsrp_db"] - value["route_relative_rsrp_db"]
    value["sinr_residual_db"] = value["sinr_db"] - value["route_sinr_db"]
    retained_sequences = value.groupby("sequence_id").size()
    minimum_rows = int(config["preprocessing"]["minimum_sequence_rows"])
    keep = set(retained_sequences[retained_sequences >= minimum_rows].index)
    value = value[value["sequence_id"].isin(keep)].copy()
    return value, {
        "input_rows": total,
        "supported_rows": len(value),
        "unsupported_rows": total - len(value),
        "unsupported_fraction": 1.0 - len(value) / total if total else 1.0,
        "retained_sequences": value["sequence_id"].nunique(),
    }


def _feature_sequences(frame: pd.DataFrame, columns: list[str]) -> list[np.ndarray]:
    result = []
    for _, sequence in frame.groupby("sequence_id", sort=False):
        values = sequence.sort_values("t_s")[columns].to_numpy(float)
        if len(values):
            result.append(values)
    return result


def _balanced_scale(sessions: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    scales = []
    pooled = []
    for frame in sessions.values():
        values = frame[["relative_rsrp_db", "sinr_db"]].to_numpy(float)
        center = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        centers.append(center)
        scales.append(scale)
        pooled.append(values)
    center = np.median(np.asarray(centers), axis=0)
    scale = np.median(np.asarray(scales), axis=0)
    pooled_values = np.vstack(pooled)
    fallback = np.std(pooled_values, axis=0, ddof=1)
    scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
    return center, scale


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    stable = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(stable.squeeze())
    return np.squeeze(stable, axis=axis)


def _digamma(value: float) -> float:
    result = 0.0
    x = float(value)
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    inverse2 = inverse * inverse
    return (
        result
        + math.log(x)
        - 0.5 * inverse
        - inverse2 * (1.0 / 12.0 - inverse2 * (1.0 / 120.0 - inverse2 / 252.0))
    )


def _trigamma(value: float) -> float:
    result = 0.0
    x = float(value)
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0
    inverse = 1.0 / x
    inverse2 = inverse * inverse
    return (
        result
        + inverse
        + 0.5 * inverse2
        + inverse2 * inverse / 6.0
        - (inverse2 * inverse2 * inverse / 30.0)
    )


def _weighted_gamma(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    values = np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Gamma weights must have positive mass")
    mean = float(np.sum(weights * values) / total)
    log_mean = float(np.sum(weights * np.log(values)) / total)
    statistic = max(math.log(mean) - log_mean, 1e-12)
    shape = (3.0 - statistic + math.sqrt((statistic - 3.0) ** 2 + 24.0 * statistic)) / (
        12.0 * statistic
    )
    shape = min(max(shape, 0.05), 1e5)
    for _ in range(50):
        function = math.log(shape) - _digamma(shape) - statistic
        derivative = 1.0 / shape - _trigamma(shape)
        candidate = shape - function / derivative
        if not math.isfinite(candidate) or candidate <= 0:
            candidate = shape / 2.0
        candidate = min(max(candidate, 0.05), 1e5)
        if abs(candidate - shape) <= 1e-9 * max(shape, 1.0):
            shape = candidate
            break
        shape = candidate
    return shape, max(mean / shape, 1e-12)


def _regular_covariance(
    values: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    regularization: float,
    *,
    denominator: float | None = None,
) -> np.ndarray:
    centered = values - mean
    total = float(weights.sum()) if denominator is None else float(denominator)
    covariance = (centered * weights[:, None]).T @ centered / max(total, 1e-12)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(values.shape[1]) * regularization
    return covariance


def _fit_emission(
    values: np.ndarray,
    weights: np.ndarray,
    emission: str,
    regularization: float,
    *,
    degrees_of_freedom: float = 5.0,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 1e-9:
        raise ValueError("emission weights have insufficient mass")
    if emission == "multivariate_gaussian_db":
        mean = np.sum(values * weights[:, None], axis=0) / total
        covariance = _regular_covariance(values, weights, mean, regularization)
        return {"mean": mean, "covariance": covariance}
    if emission == "multivariate_student_t_db":
        if previous is None:
            mean = np.sum(values * weights[:, None], axis=0) / total
            covariance = _regular_covariance(values, weights, mean, regularization)
        else:
            mean = np.asarray(previous["mean"], dtype=float)
            covariance = np.asarray(previous["covariance"], dtype=float)
        inverse = np.linalg.inv(covariance)
        centered = values - mean
        distance = np.einsum("ni,ij,nj->n", centered, inverse, centered)
        latent_scale = (degrees_of_freedom + values.shape[1]) / (degrees_of_freedom + distance)
        effective = weights * latent_scale
        mean = np.sum(values * effective[:, None], axis=0) / max(effective.sum(), 1e-12)
        covariance = _regular_covariance(
            values,
            effective,
            mean,
            regularization,
            denominator=total,
        )
        return {
            "mean": mean,
            "covariance": covariance,
            "degrees_of_freedom": float(degrees_of_freedom),
        }
    if emission == "gamma_relative_power_gaussian_sinr":
        coefficient = math.log(10.0) / 10.0
        relative_power = np.exp(np.clip(values[:, 0] * coefficient, -50.0, 50.0))
        shape, scale = _weighted_gamma(relative_power, weights)
        sinr_mean = float(np.sum(weights * values[:, 1]) / total)
        sinr_variance = float(
            np.sum(weights * (values[:, 1] - sinr_mean) ** 2) / total + regularization
        )
        return {
            "rsrp_gamma_shape": shape,
            "rsrp_gamma_scale": scale,
            "sinr_mean": sinr_mean,
            "sinr_variance": sinr_variance,
        }
    raise ValueError(f"unsupported emission: {emission}")


def _emission_logpdf(
    values: np.ndarray,
    parameters: dict[str, Any],
    emission: str,
) -> np.ndarray:
    dimension = values.shape[1]
    if emission in {"multivariate_gaussian_db", "multivariate_student_t_db"}:
        mean = np.asarray(parameters["mean"], dtype=float)
        covariance = np.asarray(parameters["covariance"], dtype=float)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("emission covariance is not positive definite")
        inverse = np.linalg.inv(covariance)
        centered = values - mean
        distance = np.einsum("ni,ij,nj->n", centered, inverse, centered)
        if emission == "multivariate_gaussian_db":
            return -0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + distance)
        degrees = float(parameters["degrees_of_freedom"])
        constant = (
            math.lgamma((degrees + dimension) / 2.0)
            - math.lgamma(degrees / 2.0)
            - 0.5 * (dimension * math.log(degrees * math.pi) + log_determinant)
        )
        return constant - 0.5 * (degrees + dimension) * np.log1p(distance / degrees)
    if emission == "gamma_relative_power_gaussian_sinr":
        coefficient = math.log(10.0) / 10.0
        log_power = np.clip(values[:, 0] * coefficient, -50.0, 50.0)
        power = np.exp(log_power)
        shape = float(parameters["rsrp_gamma_shape"])
        scale = float(parameters["rsrp_gamma_scale"])
        gamma_logpdf = (
            (shape - 1.0) * log_power
            - power / scale
            - math.lgamma(shape)
            - shape * math.log(scale)
            + math.log(coefficient)
            + log_power
        )
        variance = float(parameters["sinr_variance"])
        gaussian_logpdf = -0.5 * (
            math.log(2.0 * math.pi * variance)
            + (values[:, 1] - float(parameters["sinr_mean"])) ** 2 / variance
        )
        return gamma_logpdf + gaussian_logpdf
    raise ValueError(f"unsupported emission: {emission}")


def _expected_rsrp(parameters: dict[str, Any], emission: str) -> float:
    if emission in {"multivariate_gaussian_db", "multivariate_student_t_db"}:
        return float(np.asarray(parameters["mean"])[0])
    shape = float(parameters["rsrp_gamma_shape"])
    scale = float(parameters["rsrp_gamma_scale"])
    return (10.0 / math.log(10.0)) * (_digamma(shape) + math.log(scale))


def _sample_emission(
    rng: np.random.Generator,
    parameters: dict[str, Any],
    emission: str,
) -> np.ndarray:
    if emission == "multivariate_gaussian_db":
        return rng.multivariate_normal(parameters["mean"], parameters["covariance"])
    if emission == "multivariate_student_t_db":
        degrees = float(parameters["degrees_of_freedom"])
        gaussian = rng.multivariate_normal(np.zeros(2), parameters["covariance"])
        scale = math.sqrt(rng.chisquare(degrees) / degrees)
        return np.asarray(parameters["mean"], dtype=float) + gaussian / scale
    if emission == "gamma_relative_power_gaussian_sinr":
        power = rng.gamma(
            float(parameters["rsrp_gamma_shape"]),
            float(parameters["rsrp_gamma_scale"]),
        )
        rsrp = 10.0 * math.log10(max(float(power), np.finfo(float).tiny))
        sinr = rng.normal(
            float(parameters["sinr_mean"]),
            math.sqrt(float(parameters["sinr_variance"])),
        )
        return np.asarray([rsrp, sinr])
    raise ValueError(f"unsupported emission: {emission}")


def _sequence_posteriors(
    sequence: np.ndarray,
    model: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    states = int(model["states"])
    emissions = np.column_stack(
        [_emission_logpdf(sequence, value, model["emission"]) for value in model["emissions"]]
    )
    log_initial = np.log(np.maximum(np.asarray(model["initial"]), 1e-300))
    log_transition = np.log(np.maximum(np.asarray(model["transition"]), 1e-300))
    alpha = np.empty((len(sequence), states), dtype=float)
    constants = np.empty(len(sequence), dtype=float)
    alpha[0] = log_initial + emissions[0]
    constants[0] = float(_logsumexp(alpha[0]))
    alpha[0] -= constants[0]
    for index in range(1, len(sequence)):
        alpha[index] = emissions[index] + _logsumexp(
            alpha[index - 1][:, None] + log_transition,
            axis=0,
        )
        constants[index] = float(_logsumexp(alpha[index]))
        alpha[index] -= constants[index]
    beta = np.zeros((len(sequence), states), dtype=float)
    for index in range(len(sequence) - 2, -1, -1):
        beta[index] = (
            _logsumexp(
                log_transition + emissions[index + 1][None, :] + beta[index + 1][None, :],
                axis=1,
            )
            - constants[index + 1]
        )
    log_gamma = alpha + beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    xi = np.empty((max(len(sequence) - 1, 0), states, states), dtype=float)
    for index in range(len(sequence) - 1):
        value = (
            alpha[index][:, None]
            + log_transition
            + emissions[index + 1][None, :]
            + beta[index + 1][None, :]
        )
        value -= float(_logsumexp(value))
        xi[index] = np.exp(value)
    return float(constants.sum()), gamma, xi


def _initial_model(
    values: np.ndarray,
    candidate: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    states = int(candidate["states"])
    emission = str(candidate["emission"])
    hmm = config["models"]["hmm"]
    regularization = float(hmm["covariance_regularization"])
    degrees = float(candidate.get("degrees_of_freedom", 5.0))
    if states == 1:
        weights = np.ones(len(values), dtype=float)
        return {
            "states": 1,
            "emission": emission,
            "initial": np.ones(1),
            "transition": np.ones((1, 1)),
            "emissions": [
                _fit_emission(
                    values,
                    weights,
                    emission,
                    regularization,
                    degrees_of_freedom=degrees,
                )
            ],
        }
    score = values[:, 0] + rng.normal(0.0, max(np.std(values[:, 0]) * 0.05, 1e-6), len(values))
    threshold = float(np.quantile(score, rng.uniform(0.35, 0.65)))
    hard = (score > threshold).astype(int)
    weights = np.column_stack([1.0 - hard, hard]) * 0.9 + 0.05
    self_probability = float(rng.uniform(0.80, 0.97))
    transition = np.asarray(
        [[self_probability, 1.0 - self_probability], [1.0 - self_probability, self_probability]]
    )
    return {
        "states": states,
        "emission": emission,
        "initial": np.asarray([0.5, 0.5]),
        "transition": transition,
        "emissions": [
            _fit_emission(
                values,
                weights[:, state],
                emission,
                regularization,
                degrees_of_freedom=degrees,
            )
            for state in range(states)
        ],
    }


def _align_model(model: dict[str, Any]) -> dict[str, Any]:
    order = np.argsort([_expected_rsrp(value, model["emission"]) for value in model["emissions"]])
    model["initial"] = np.asarray(model["initial"])[order]
    model["transition"] = np.asarray(model["transition"])[np.ix_(order, order)]
    model["emissions"] = [model["emissions"][int(index)] for index in order]
    model["state_expected_relative_rsrp"] = [
        _expected_rsrp(value, model["emission"]) for value in model["emissions"]
    ]
    return model


def _fit_hmm(
    sequences: list[np.ndarray],
    candidate: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    values = np.vstack(sequences)
    hmm = config["models"]["hmm"]
    states = int(candidate["states"])
    attempts = 1 if states == 1 else int(hmm["initializations"])
    regularization = float(hmm["covariance_regularization"])
    transition_floor = float(hmm["transition_probability_floor"])
    maximum_iterations = int(hmm["maximum_iterations"])
    tolerance = float(hmm["relative_log_likelihood_tolerance"])
    degrees = float(candidate.get("degrees_of_freedom", 5.0))
    best: dict[str, Any] | None = None
    for attempt in range(attempts):
        rng = np.random.default_rng(seed + attempt * 1009)
        model = _initial_model(values, candidate, config, rng)
        previous_likelihood = -math.inf
        converged = False
        try:
            iterations_run = 0
            for _iteration in range(1, maximum_iterations + 1):
                iterations_run = _iteration
                likelihood = 0.0
                gammas = []
                xis = []
                for sequence in sequences:
                    value, gamma, xi = _sequence_posteriors(sequence, model)
                    likelihood += value
                    gammas.append(gamma)
                    xis.append(xi)
                initial = np.mean([value[0] for value in gammas], axis=0)
                transition_counts = np.sum(
                    [value.sum(axis=0) for value in xis],
                    axis=0,
                )
                transition = np.maximum(transition_counts, transition_floor)
                transition /= transition.sum(axis=1, keepdims=True)
                all_gamma = np.vstack(gammas)
                emissions = [
                    _fit_emission(
                        values,
                        all_gamma[:, state],
                        model["emission"],
                        regularization,
                        degrees_of_freedom=degrees,
                        previous=model["emissions"][state],
                    )
                    for state in range(states)
                ]
                model.update(
                    {
                        "initial": initial / initial.sum(),
                        "transition": transition,
                        "emissions": emissions,
                    }
                )
                relative_change = abs(likelihood - previous_likelihood) / max(
                    abs(previous_likelihood), 1.0
                )
                if math.isfinite(previous_likelihood) and relative_change <= tolerance:
                    converged = True
                    break
                previous_likelihood = likelihood
            final_likelihood = 0.0
            final_gammas = []
            for sequence in sequences:
                value, gamma, _ = _sequence_posteriors(sequence, model)
                final_likelihood += value
                final_gammas.append(gamma)
            occupancy = np.vstack(final_gammas).mean(axis=0)
            model.update(
                {
                    "log_likelihood": float(final_likelihood),
                    "converged": converged,
                    "iterations": iterations_run,
                    "occupancy": occupancy,
                    "attempt": attempt,
                }
            )
            model = _align_model(model)
            _, aligned_gammas, _ = zip(
                *[_sequence_posteriors(sequence, model) for sequence in sequences],
                strict=True,
            )
            model["occupancy"] = np.vstack(aligned_gammas).mean(axis=0)
            model["expected_dwell_rows"] = 1.0 / np.maximum(
                1.0 - np.diag(model["transition"]),
                1e-12,
            )
            if not math.isfinite(model["log_likelihood"]):
                continue
            if best is None or model["log_likelihood"] > best["log_likelihood"]:
                best = model
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
    if best is None:
        raise ValueError(f"all HMM fits failed: {candidate['id']}")
    minimum_occupancy = float(hmm["minimum_state_occupancy"])
    minimum_dwell = float(hmm["minimum_expected_dwell_rows"])
    best["eligible"] = bool(
        np.min(best["occupancy"]) >= minimum_occupancy
        and np.min(best["expected_dwell_rows"]) >= minimum_dwell
        and np.all(np.isfinite(best["occupancy"]))
    )
    best["candidate_id"] = candidate["id"]
    return best


def _hmm_log_score(sequences: list[np.ndarray], model: dict[str, Any]) -> float:
    likelihood = sum(_sequence_posteriors(sequence, model)[0] for sequence in sequences)
    rows = sum(len(sequence) for sequence in sequences)
    return float(likelihood / rows)


def _sample_hmm(
    model: dict[str, Any],
    lengths: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    result = []
    states = int(model["states"])
    for length in lengths:
        values = np.empty((length, 2), dtype=float)
        state = int(rng.choice(states, p=np.asarray(model["initial"])))
        for index in range(length):
            values[index] = _sample_emission(
                rng,
                model["emissions"][state],
                model["emission"],
            )
            if index + 1 < length:
                state = int(rng.choice(states, p=np.asarray(model["transition"])[state]))
        result.append(values)
    return result


def _sample_blocks(
    source_sequences: list[np.ndarray],
    lengths: list[int],
    block_length: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    eligible = [value for value in source_sequences if len(value) >= block_length]
    if not eligible:
        raise ValueError("no training sequence is long enough for block resampling")
    result = []
    for length in lengths:
        blocks = []
        rows = 0
        while rows < length:
            sequence = eligible[int(rng.integers(0, len(eligible)))]
            start = int(rng.integers(0, len(sequence)))
            indices = (start + np.arange(block_length)) % len(sequence)
            block = sequence[indices]
            blocks.append(block)
            rows += len(block)
        result.append(np.vstack(blocks)[:length])
    return result


def _subsample(values: np.ndarray, maximum_rows: int) -> np.ndarray:
    if len(values) <= maximum_rows:
        return values
    indices = np.linspace(0, len(values) - 1, maximum_rows).round().astype(int)
    return values[indices]


def _acf(sequences: list[np.ndarray], feature: int, lag: int) -> float:
    numerator = 0.0
    denominator = 0.0
    all_values = np.concatenate([value[:, feature] for value in sequences])
    mean = float(np.mean(all_values))
    variance = float(np.sum((all_values - mean) ** 2))
    if variance <= 1e-12:
        return 0.0
    for sequence in sequences:
        if len(sequence) <= lag:
            continue
        numerator += float(
            np.sum((sequence[:-lag, feature] - mean) * (sequence[lag:, feature] - mean))
        )
        denominator += float(np.sum((sequence[:, feature] - mean) ** 2))
    return numerator / denominator if denominator > 0 else 0.0


def _increments(sequences: list[np.ndarray], feature: int) -> np.ndarray:
    values = [np.diff(sequence[:, feature]) for sequence in sequences if len(sequence) > 1]
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def _wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return math.inf
    size = max(len(left), len(right))
    quantiles = (np.arange(size, dtype=float) + 0.5) / size
    left_values = np.quantile(left, quantiles)
    right_values = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_values - right_values)))


def _temporal_error(
    observed: list[np.ndarray],
    generated: list[np.ndarray],
    scale: np.ndarray,
    lags: list[int],
) -> float:
    acf_errors = []
    increment_errors = []
    for feature in range(2):
        difference = np.asarray(
            [_acf(observed, feature, lag) - _acf(generated, feature, lag) for lag in lags]
        )
        acf_errors.append(float(np.sqrt(np.mean(difference**2))))
        increment_errors.append(
            _wasserstein(
                _increments(observed, feature),
                _increments(generated, feature),
            )
            / float(scale[feature])
        )
    return float(np.mean([*acf_errors, *increment_errors]))


def _joint_mmd(
    observed: list[np.ndarray],
    generated: list[np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
    maximum_rows: int,
) -> float:
    left = _subsample((np.vstack(observed) - center) / scale, maximum_rows)
    right = _subsample((np.vstack(generated) - center) / scale, maximum_rows)
    return biased_rbf_mmd2(
        left,
        right,
        bandwidth=bandwidth,
        maximum_samples=maximum_rows,
    )


def _add_route_mean(
    residuals: list[np.ndarray],
    route_sequences: list[np.ndarray],
) -> list[np.ndarray]:
    if len(residuals) != len(route_sequences):
        raise ValueError("generated residual and route sequence counts differ")
    return [value + route for value, route in zip(residuals, route_sequences, strict=True)]


def _real_pairwise_reference(
    sessions: dict[str, pd.DataFrame],
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
    maximum_rows: int,
    lags: list[int],
) -> tuple[float, float]:
    identifiers = sorted(sessions)
    joint = []
    temporal = []
    for left_index, left_id in enumerate(identifiers):
        left = _feature_sequences(sessions[left_id], ["relative_rsrp_db", "sinr_db"])
        for right_id in identifiers[left_index + 1 :]:
            right = _feature_sequences(sessions[right_id], ["relative_rsrp_db", "sinr_db"])
            joint.append(_joint_mmd(left, right, center, scale, bandwidth, maximum_rows))
            temporal.append(_temporal_error(left, right, scale, lags))
    if not joint:
        raise ValueError("pairwise reference requires at least two training sessions")
    return float(np.quantile(joint, 0.9)), float(np.quantile(temporal, 0.9))


def _evaluate_generator(
    *,
    observed: list[np.ndarray],
    route_sequences: list[np.ndarray],
    generator: Any,
    repetitions: int,
    seed: int,
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
    maximum_rows: int,
    lags: list[int],
) -> dict[str, float]:
    lengths = [len(value) for value in observed]
    joint = []
    temporal = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(seed + repetition * 7919)
        residuals = generator(lengths, rng)
        generated = _add_route_mean(residuals, route_sequences)
        joint.append(_joint_mmd(observed, generated, center, scale, bandwidth, maximum_rows))
        temporal.append(_temporal_error(observed, generated, scale, lags))
    return {
        "joint_mmd_squared_mean": float(np.mean(joint)),
        "joint_mmd_squared_median": float(np.median(joint)),
        "joint_mmd_squared_p10": float(np.quantile(joint, 0.1)),
        "joint_mmd_squared_p90": float(np.quantile(joint, 0.9)),
        "temporal_error_mean": float(np.mean(temporal)),
        "temporal_error_median": float(np.median(temporal)),
        "temporal_error_p10": float(np.quantile(temporal, 0.1)),
        "temporal_error_p90": float(np.quantile(temporal, 0.9)),
    }


def _jsonable_emission(value: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        elif isinstance(item, np.floating):
            result[key] = float(item)
        else:
            result[key] = item
    return result


def _model_record(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": model["candidate_id"],
        "states": model["states"],
        "emission": model["emission"],
        "initial": np.asarray(model["initial"]).tolist(),
        "transition": np.asarray(model["transition"]).tolist(),
        "emissions": [_jsonable_emission(value) for value in model["emissions"]],
        "log_likelihood": model["log_likelihood"],
        "converged": model["converged"],
        "iterations": model["iterations"],
        "attempt": model["attempt"],
        "occupancy": np.asarray(model["occupancy"]).tolist(),
        "expected_dwell_rows": np.asarray(model["expected_dwell_rows"]).tolist(),
        "state_expected_relative_rsrp": model["state_expected_relative_rsrp"],
        "eligible": model["eligible"],
    }


def _matching_one_state(candidate_id: str) -> str | None:
    return candidate_id.replace("_2state", "_1state") if candidate_id.endswith("_2state") else None


def _decision(
    results: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["decision_rules"]
    expected_folds = int(rules["required_development_folds"])
    baseline = results[results["candidate_id"] == config["block_bootstrap"]["id"]].copy()
    if len(baseline) != expected_folds:
        raise ValueError("the block-bootstrap baseline is incomplete")
    baseline = baseline.set_index("fold_id")
    superiority = rules["hmm_better_than_bootstrap"]
    predictive_requirement = rules["two_state_predictive_requirement"]
    candidate_records = []
    passing = []
    for candidate_id in sorted(
        value for value in results["candidate_id"].unique() if value.endswith("_2state")
    ):
        candidate = results[results["candidate_id"] == candidate_id].set_index("fold_id")
        if len(candidate) != expected_folds:
            continue
        aligned = candidate.join(
            baseline[["joint_mmd_squared_mean", "temporal_error_mean"]],
            rsuffix="_baseline",
        )
        joint_improvement = 1.0 - (
            aligned["joint_mmd_squared_mean"] / aligned["joint_mmd_squared_mean_baseline"]
        )
        temporal_improvement = 1.0 - (
            aligned["temporal_error_mean"] / aligned["temporal_error_mean_baseline"]
        )
        one_state_id = _matching_one_state(candidate_id)
        one_state = results[results["candidate_id"] == one_state_id].set_index("fold_id")
        predictive = (
            candidate["predictive_log_score_per_row"] - one_state["predictive_log_score_per_row"]
        )
        eligible = bool(candidate["model_eligible"].all())
        record = {
            "candidate_id": candidate_id,
            "median_joint_relative_improvement": float(np.median(joint_improvement)),
            "median_temporal_relative_improvement": float(np.median(temporal_improvement)),
            "joint_fold_wins": int((joint_improvement > 0).sum()),
            "temporal_fold_wins": int((temporal_improvement > 0).sum()),
            "median_predictive_log_score_improvement_nats_per_row": float(np.median(predictive)),
            "all_folds_model_eligible": eligible,
        }
        record["passes"] = bool(
            eligible
            and record["median_joint_relative_improvement"]
            >= float(superiority["minimum_median_joint_mmd_relative_improvement"])
            and record["median_temporal_relative_improvement"]
            >= float(superiority["minimum_median_temporal_relative_improvement"])
            and record["joint_fold_wins"] >= int(superiority["minimum_joint_metric_fold_wins"])
            and record["temporal_fold_wins"]
            >= int(superiority["minimum_temporal_metric_fold_wins"])
            and record["median_predictive_log_score_improvement_nats_per_row"]
            >= float(
                predictive_requirement[
                    "minimum_median_log_score_improvement_nats_per_row_over_matching_one_state"
                ]
            )
        )
        candidate_records.append(record)
        if record["passes"]:
            passing.append(record)

    bootstrap_rule = rules["block_bootstrap_support"]
    joint_supported = int(
        (baseline["joint_mmd_squared_mean"] <= baseline["joint_reference_p90"]).sum()
    )
    temporal_supported = int(
        (baseline["temporal_error_mean"] <= baseline["temporal_reference_p90"]).sum()
    )
    bootstrap_pass = bool(
        joint_supported >= int(bootstrap_rule["minimum_joint_supported_folds"])
        and temporal_supported >= int(bootstrap_rule["minimum_temporal_supported_folds"])
    )
    if passing:
        selected = sorted(
            passing,
            key=lambda value: (
                value["median_joint_relative_improvement"]
                + value["median_temporal_relative_improvement"],
                value["median_predictive_log_score_improvement_nats_per_row"],
            ),
            reverse=True,
        )[0]
        code = rules["outcomes"]["stable_hmm_advantage"]["code"]
        selected_process = selected["candidate_id"]
        next_action = rules["outcomes"]["stable_hmm_advantage"]["next_action"]
    elif bootstrap_pass:
        code = rules["outcomes"]["bootstrap_supported_without_hmm_advantage"]["code"]
        selected_process = config["block_bootstrap"]["id"]
        next_action = rules["outcomes"]["bootstrap_supported_without_hmm_advantage"]["next_action"]
    else:
        code = rules["outcomes"]["no_stable_process"]["code"]
        selected_process = None
        next_action = rules["outcomes"]["no_stable_process"]["next_action"]
    return {
        "decision_code": code,
        "selected_process": selected_process,
        "next_action": next_action,
        "hmm_candidate_comparisons": candidate_records,
        "block_bootstrap": {
            "joint_supported_folds": joint_supported,
            "temporal_supported_folds": temporal_supported,
            "passes": bootstrap_pass,
        },
        "powder_reservation_authorized": False,
        "final_evaluation_authorized": False,
        "abc_authorized": False,
    }


def analyze_phase3d_radio_process(
    *,
    archive_path: str | Path,
    phase3c15_result_path: str | Path,
    protocol_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    phase3c15 = Path(phase3c15_result_path).resolve()
    protocol = Path(protocol_dir).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    config = _read_yaml(config_file)
    validate_phase3d_config(config)
    freeze = _verify_protocol_freeze(protocol, config_file)
    _verify_prerequisites(archive, phase3c15, config)
    sessions, quality = _load_development_sessions(archive, config)
    if config["final_evaluation"]["source_path"] in set(quality["source_path"]):
        raise AssertionError("the final evaluation payload entered development")

    candidates = config["models"]["candidates"]
    repetitions = int(config["evaluation"]["generation_repetitions"])
    seed = int(config["evaluation"]["seed"])
    maximum_rows = int(config["evaluation"]["joint_distribution"]["maximum_rows_per_trace"])
    lags = [
        int(value) for value in config["evaluation"]["temporal"]["autocorrelation_lags_seconds"]
    ]
    block_length = int(config["block_bootstrap"]["block_length_rows"])
    maximum_unsupported = float(
        config["decision_rules"]["maximum_unsupported_position_fraction_per_fold"]
    )
    fold_rows = []
    route_rows = []
    model_records: dict[str, Any] = {}
    session_ids = sorted(sessions)
    for fold_index, holdout_id in enumerate(session_ids, start=1):
        fold_id = f"fold_{fold_index}_{holdout_id}"
        training = {key: value for key, value in sessions.items() if key != holdout_id}
        route_means = _fit_route_means(training, config)
        route_means["fold_id"] = fold_id
        route_rows.append(route_means)
        training_detrended = {}
        for session_id, frame in training.items():
            transformed, audit = _apply_route_means(frame, route_means, config)
            if audit["unsupported_fraction"] > maximum_unsupported:
                raise ValueError(f"training spatial support gate failed: {fold_id}/{session_id}")
            training_detrended[session_id] = transformed
        holdout, support = _apply_route_means(sessions[holdout_id], route_means, config)
        if support["unsupported_fraction"] > maximum_unsupported:
            raise ValueError(f"holdout spatial support gate failed: {fold_id}")

        residual_training = []
        for frame in training_detrended.values():
            residual_training.extend(
                _feature_sequences(frame, ["rsrp_residual_db", "sinr_residual_db"])
            )
        residual_holdout = _feature_sequences(holdout, ["rsrp_residual_db", "sinr_residual_db"])
        observed = _feature_sequences(holdout, ["relative_rsrp_db", "sinr_db"])
        route_sequences = _feature_sequences(holdout, ["route_relative_rsrp_db", "route_sinr_db"])
        center, scale = _balanced_scale(training)
        training_targets = np.vstack(
            [frame[["relative_rsrp_db", "sinr_db"]].to_numpy(float) for frame in training.values()]
        )
        bandwidth = median_heuristic_bandwidth((training_targets - center) / scale)
        joint_reference, temporal_reference = _real_pairwise_reference(
            training,
            center,
            scale,
            bandwidth,
            maximum_rows,
            lags,
        )

        def generate_blocks(
            lengths: list[int],
            rng: np.random.Generator,
            source: list[np.ndarray] = residual_training,
            block: int = block_length,
        ) -> list[np.ndarray]:
            return _sample_blocks(source, lengths, block, rng)

        baseline_metrics = _evaluate_generator(
            observed=observed,
            route_sequences=route_sequences,
            generator=generate_blocks,
            repetitions=repetitions,
            seed=seed + fold_index * 100000,
            center=center,
            scale=scale,
            bandwidth=bandwidth,
            maximum_rows=maximum_rows,
            lags=lags,
        )
        fold_rows.append(
            {
                "fold_id": fold_id,
                "holdout_session_id": holdout_id,
                "candidate_id": config["block_bootstrap"]["id"],
                "states": 0,
                "emission": "empirical_paired_blocks",
                "model_eligible": True,
                "predictive_log_score_per_row": math.nan,
                "supported_position_fraction": 1.0 - support["unsupported_fraction"],
                "joint_reference_p90": joint_reference,
                "temporal_reference_p90": temporal_reference,
                **baseline_metrics,
            }
        )

        model_records[fold_id] = {}
        for candidate_index, candidate in enumerate(candidates, start=1):
            model = _fit_hmm(
                residual_training,
                candidate,
                config,
                seed=seed + fold_index * 1000000 + candidate_index * 10000,
            )
            model_records[fold_id][candidate["id"]] = _model_record(model)
            metrics = _evaluate_generator(
                observed=observed,
                route_sequences=route_sequences,
                generator=lambda lengths, rng, fitted=model: _sample_hmm(fitted, lengths, rng),
                repetitions=repetitions,
                seed=seed + fold_index * 100000 + candidate_index * 1000,
                center=center,
                scale=scale,
                bandwidth=bandwidth,
                maximum_rows=maximum_rows,
                lags=lags,
            )
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "holdout_session_id": holdout_id,
                    "candidate_id": candidate["id"],
                    "states": candidate["states"],
                    "emission": candidate["emission"],
                    "model_eligible": model["eligible"],
                    "predictive_log_score_per_row": _hmm_log_score(residual_holdout, model),
                    "supported_position_fraction": 1.0 - support["unsupported_fraction"],
                    "joint_reference_p90": joint_reference,
                    "temporal_reference_p90": temporal_reference,
                    **metrics,
                }
            )

    results = pd.DataFrame(fold_rows)
    route_table = pd.concat(route_rows, ignore_index=True)
    decision = _decision(results, config)
    result = {
        "schema_version": 1,
        "stage": "phase_3d_offline_radio_process_result",
        "research_question": config["research_question"],
        "protocol_revision": config["protocol_revision"],
        "protocol_repository": freeze["repository"],
        "analysis_repository": _git_revision(),
        "development_sessions": session_ids,
        "development_folds": len(session_ids),
        "final_evaluation": {
            "source_path": config["final_evaluation"]["source_path"],
            "payload_opened": False,
            "authorized": False,
            "prior_inspection_disclosure": config["final_evaluation"][
                "prior_inspection_disclosure"
            ],
        },
        **decision,
        "reservation": config["reservation"],
        "claim_limits": config["claim_limits"],
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "development_input_quality.csv", quality)
    _write_csv(output / "fold_model_metrics.csv", results)
    _write_csv(output / "fold_route_means.csv", route_table)
    _write_json(output / "fitted_models.json", model_records)
    _write_json(output / "phase3d_decision.json", result)
    analysis_manifest = {
        "schema_version": 1,
        "stage": "phase_3d_analysis_manifest",
        "archive_sha256": _sha256(archive),
        "config_sha256": _sha256(config_file),
        "protocol_freeze_sha256": _sha256(protocol / "protocol_freeze.json"),
        "final_evaluation_payload_opened": False,
        "folds": len(session_ids),
        "generation_repetitions_per_candidate_fold": repetitions,
        "reservation_requested": False,
    }
    _write_json(output / "analysis_manifest.json", analysis_manifest)
    generated = [
        "analysis_manifest.json",
        "development_input_quality.csv",
        "fitted_models.json",
        "fold_model_metrics.csv",
        "fold_route_means.csv",
        "phase3d_decision.json",
    ]
    _write_json(
        output / "SHA256SUMS.json",
        {name: _sha256(output / name) for name in generated},
    )
    return result
