from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .mmd_abc import biased_rbf_mmd2, median_heuristic_bandwidth
from .upv_protocol import _load_radio_csv, _normal_member_path, build_route_table
from .upv_support import (
    _aggregate,
    _candidate_label,
    _circular_block_sample,
    _evenly_spaced,
    _load_rfsim,
    _read_json,
    _selection_rows,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


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


def validate_phase3b_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3B schema_version must be 1")
    if config.get("stage") != "existing_bank_nonabsolute_support_diagnostic_only":
        raise ValueError("Phase 3B must remain an existing-bank diagnostic")
    if config.get("selected_measurement_branch") != "insufficient_metadata":
        raise ValueError("Phase 3B requires the insufficient-metadata branch")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("Phase 3B cannot authorize execution or ABC")
    targets = sorted(config.get("targets") or [], key=lambda item: int(item["order"]))
    if [str(item.get("name")) for item in targets] != ["relative_RSRP", "SINR"]:
        raise ValueError("Phase 3B targets must be relative_RSRP then SINR")
    relative, sinr = targets
    if relative.get("transform") != (
        "subtract_within_unit_median_after_one_second_aggregation"
    ):
        raise ValueError("relative RSRP requires within-unit median centring")
    if relative.get("prohibited_inference") != "physical_propagation_loss":
        raise ValueError("relative RSRP cannot identify physical propagation loss")
    if sinr.get("scale_location") != "empirical_device_conditioned_distribution":
        raise ValueError("SINR must be treated as an empirical device-conditioned target")
    if sinr.get("prohibited_inference") != "absolute_noise_power":
        raise ValueError("SINR cannot authorize absolute noise-power calibration")
    kernel = config.get("kernel") or {}
    if kernel.get("estimator") != "biased_mmd_squared_v_statistic":
        raise ValueError("Phase 3B requires biased MMD squared")
    if kernel.get("posthoc_clipping") != "prohibited":
        raise ValueError("Phase 3B prohibits MMD clipping")
    reference = config.get("balanced_reference") or {}
    if not bool(reference.get("validation_filename_sensitivity_and_S25_excluded_from_fit")):
        raise ValueError("non-primary units must not affect Phase 3B preprocessing")
    if reference.get("zero_mad_fallback") != (
        "iqr_divided_by_1.349_then_pooled_standard_deviation_ddof0"
    ):
        raise ValueError("Phase 3B requires the frozen degenerate-scale fallback")
    if int(reference.get("comparison_rows_per_distribution", 0)) < 2:
        raise ValueError("Phase 3B requires a fixed comparison sample count")
    claim_limits = config.get("claim_limits") or {}
    required_prohibitions = {
        "absolute_rsrp_calibration",
        "physical_ploss_inference",
        "absolute_noise_power_calibration",
        "joint_identifiability",
        "abc",
    }
    if any(claim_limits.get(key) != "prohibited" for key in required_prohibitions):
        raise ValueError("Phase 3B claim prohibitions are incomplete")
    reservation = config.get("reservation") or {}
    if bool(reservation.get("request_now")) or reservation.get("gate_state") != "closed":
        raise ValueError("the Phase 3B reservation gate must start closed")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation preparation requires at least 30 minutes")


def _load_archive_route(
    archive: Path,
    *,
    source_path: str,
    corrected_test_id: int,
    trim_last_seconds: float,
    phase1_config: dict[str, Any],
) -> pd.DataFrame:
    payload: bytes | None = None
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if _normal_member_path(member.filename) == source_path:
                payload = source.read(member)
                break
    if payload is None:
        raise ValueError(f"UPV archive does not contain {source_path}")
    radio, _ = _load_radio_csv(payload)
    if trim_last_seconds > 0:
        cutoff = float(radio["seconds_of_day"].max()) - trim_last_seconds
        radio = radio.loc[radio["seconds_of_day"].le(cutoff)].copy()
    route_config = phase1_config["route"]
    return build_route_table(
        radio,
        source_path=source_path,
        corrected_test_id=corrected_test_id,
        bin_sizes_m=[int(value) for value in route_config["bin_sizes_m"]],
        minimum_step_m_for_heading=float(route_config["minimum_step_m_for_heading"]),
        direction_sectors=int(route_config["direction_sectors"]),
    )


def _assign_primary_regions(
    route: pd.DataFrame,
    split: pd.DataFrame,
    *,
    bin_size_m: int,
    roles: list[str],
) -> pd.DataFrame:
    role_by_bin = (
        split.loc[split["bin_size_m"].eq(bin_size_m)]
        .set_index("route_bin_id")["locked_role"]
        .astype(str)
        .to_dict()
    )
    result = route.copy()
    result["locked_role"] = (
        result[f"route_bin_{bin_size_m}m"].map(role_by_bin).fillna("unselected")
    )
    result.loc[~result["locked_role"].isin(roles), "locked_role"] = "unselected"
    result["reference_distance_m"] = 0.0
    return result


def _xy(frame: pd.DataFrame, latitude_origin: float) -> tuple[np.ndarray, np.ndarray]:
    x = frame["longitude_deg"].to_numpy(float) * 111320.0 * math.cos(latitude_origin)
    y = frame["latitude_deg"].to_numpy(float) * 110540.0
    return x, y


def transfer_locked_regions(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    roles: list[str],
    maximum_distance_m: float,
    minimum_rows: int,
    minimum_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transfer frozen regions by nearest GPS point within the same direction sector."""
    result = target.copy()
    result["locked_role"] = "unmatched"
    result["reference_distance_m"] = np.nan
    latitude_origin = math.radians(float(pd.concat(
        [target["latitude_deg"], reference["latitude_deg"]]
    ).mean()))
    target_x, target_y = _xy(target, latitude_origin)
    reference_x, reference_y = _xy(reference, latitude_origin)
    for direction, target_indices in result.groupby("direction_sector").groups.items():
        reference_indices = np.flatnonzero(
            reference["direction_sector"].astype(str).to_numpy() == str(direction)
        )
        if not len(reference_indices):
            continue
        indices = np.asarray(list(target_indices), dtype=int)
        distances = np.hypot(
            target_x[indices, None] - reference_x[reference_indices][None, :],
            target_y[indices, None] - reference_y[reference_indices][None, :],
        )
        nearest_local = np.argmin(distances, axis=1)
        nearest_reference = reference_indices[nearest_local]
        nearest_distance = distances[np.arange(len(indices)), nearest_local]
        result.loc[indices, "locked_role"] = reference.iloc[nearest_reference][
            "locked_role"
        ].to_numpy()
        result.loc[indices, "reference_distance_m"] = nearest_distance

    diagnostics: list[dict[str, object]] = []
    maximum_mask = result["reference_distance_m"].le(maximum_distance_m)
    for role in roles:
        assigned = result["locked_role"].eq(role)
        denominator = int(assigned.sum())
        retained = result.loc[assigned & maximum_mask]
        fraction = len(retained) / denominator if denominator else 0.0
        valid = denominator > 0 and len(retained) >= minimum_rows and fraction >= minimum_fraction
        diagnostics.append({
            "locked_role": role,
            "nearest_role_rows": denominator,
            "rows_within_maximum_distance": len(retained),
            "retained_fraction": fraction,
            "median_reference_distance_m": float(
                retained["reference_distance_m"].median()
            ) if len(retained) else math.nan,
            "p95_reference_distance_m": float(
                retained["reference_distance_m"].quantile(0.95)
            ) if len(retained) else math.nan,
            "unit_valid": valid,
        })
        if not valid:
            result.loc[assigned, "locked_role"] = "invalid_transfer"
    result.loc[~maximum_mask & result["locked_role"].isin(roles), "locked_role"] = (
        "outside_transfer_distance"
    )
    return result, pd.DataFrame(diagnostics)


def _aggregate_units(
    route: pd.DataFrame,
    *,
    source_key: str,
    source_role: str,
    device: str,
    roles: list[str],
    duration: float,
    minimum_aggregated_rows: int,
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, object]]]:
    units: dict[tuple[str, str], pd.DataFrame] = {}
    inventory: list[dict[str, object]] = []
    for role in roles:
        raw = route.loc[route["locked_role"].eq(role)].dropna(
            subset=["seconds_of_day", "rsrp_dbm", "sinr_db"]
        ).sort_values("seconds_of_day")
        if raw.empty:
            continue
        aggregated = _aggregate(
            raw,
            time_column="seconds_of_day",
            origin=float(raw["seconds_of_day"].min()),
            duration=duration,
            feature_columns=["rsrp_dbm", "sinr_db"],
        )
        aggregated["relative_rsrp_db"] = (
            aggregated["rsrp_dbm"] - float(aggregated["rsrp_dbm"].median())
        )
        valid = len(aggregated) >= minimum_aggregated_rows
        if valid:
            units[(source_key, role)] = aggregated
        inventory.append({
            "source_key": source_key,
            "source_role": source_role,
            "device": device,
            "source_path": str(raw["source_path"].iloc[0]),
            "locked_role": role,
            "raw_complete_rows": len(raw),
            "aggregated_rows": len(aggregated),
            "unit_valid": valid,
            "duration_seconds": float(
                raw["seconds_of_day"].max() - raw["seconds_of_day"].min()
            ),
            "relative_rsrp_median_db": float(aggregated["relative_rsrp_db"].median()),
            "sinr_median_db": float(aggregated["sinr_db"].median()),
        })
    return units, inventory


def _robust_scale(values: np.ndarray, factor: float) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) * factor)
    if scale <= np.finfo(float).eps:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale <= np.finfo(float).eps:
        scale = float(np.std(values, ddof=0))
    if scale <= np.finfo(float).eps:
        raise ValueError("balanced reference has zero robust scale")
    return center, scale


def _transformed(frame: pd.DataFrame, column: str, center: float, scale: float) -> np.ndarray:
    return ((frame[column].to_numpy(float) - center) / scale)[:, None]


def _metric_support(
    *,
    metric: str,
    upv_column: str,
    rfsim_column: str,
    units: dict[tuple[str, str], pd.DataFrame],
    unit_metadata: pd.DataFrame,
    simulation: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    primary = units[("primary", "calibration")]
    balanced_count = int(
        config["balanced_reference"]["comparison_rows_per_distribution"]
    )
    if len(primary) < balanced_count or any(
        len(frame) < balanced_count for frame in simulation.values()
    ):
        raise ValueError(f"{metric} balanced reference has too few rows")
    if any(len(frame) < balanced_count for frame in units.values()):
        raise ValueError(f"{metric} evaluation unit has too few rows")
    balanced_frames: list[pd.DataFrame] = []
    primary_balanced = _evenly_spaced(primary.rename(columns={upv_column: "value"}), balanced_count)
    primary_balanced["reference_source"] = "UPV_primary_calibration"
    balanced_frames.append(primary_balanced[["reference_source", "value"]])
    simulation_balanced: dict[str, pd.DataFrame] = {}
    for execution_id in selected["execution_id"]:
        balanced = _evenly_spaced(
            simulation[execution_id].rename(columns={rfsim_column: "value"}),
            balanced_count,
        )
        balanced["reference_source"] = execution_id
        simulation_balanced[execution_id] = balanced
        balanced_frames.append(balanced[["reference_source", "value"]])
    balanced_reference = pd.concat(balanced_frames, ignore_index=True)
    factor = float(config["balanced_reference"]["normalized_mad_factor"])
    center, scale = _robust_scale(balanced_reference["value"].to_numpy(float), factor)
    reference_values = ((balanced_reference[["value"]].to_numpy(float) - center) / scale)
    bandwidth = median_heuristic_bandwidth(
        reference_values, maximum_samples=len(reference_values)
    )

    repeatability_rows: list[dict[str, object]] = []
    for (ploss, noise), group in selected.groupby(["ploss", "noise_power_dB"], sort=True):
        for left_id, right_id in itertools.combinations(group["execution_id"], 2):
            left = _transformed(simulation_balanced[left_id], "value", center, scale)
            right = _transformed(simulation_balanced[right_id], "value", center, scale)
            repeatability_rows.append({
                "metric": metric,
                "candidate_id": _candidate_label(float(ploss), float(noise)),
                "ploss": ploss,
                "noise_power_dB": noise,
                "left_execution_id": left_id,
                "right_execution_id": right_id,
                "mmd_squared": biased_rbf_mmd2(
                    left, right, bandwidth=bandwidth, maximum_samples=balanced_count
                ),
            })
    repeatability = pd.DataFrame(repeatability_rows)
    threshold_quantile = float(config["support_rules"]["repeatability_threshold_quantile"])
    threshold = float(np.quantile(repeatability["mmd_squared"], threshold_quantile))

    metadata = unit_metadata.set_index(["source_key", "locked_role"])
    execution_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    bootstrap = config["bootstrap"]
    bootstrap_repetitions = int(bootstrap["repetitions"])
    block_length = int(bootstrap["block_length_aggregated_rows"])
    confidence = float(bootstrap["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(int(bootstrap["seed"]) + (0 if metric == "relative_RSRP" else 1))
    for source_key, role in sorted(units):
        upv = units[(source_key, role)]
        upv_balanced = _evenly_spaced(
            upv.rename(columns={upv_column: "value"}), balanced_count
        )
        upv_values = _transformed(upv_balanced, "value", center, scale)
        for item in selected.itertuples(index=False):
            simulated = simulation_balanced[item.execution_id]
            simulated_values = _transformed(simulated, "value", center, scale)
            execution_rows.append({
                "metric": metric,
                "source_key": source_key,
                "source_role": metadata.loc[(source_key, role), "source_role"],
                "device": metadata.loc[(source_key, role), "device"],
                "locked_role": role,
                "candidate_id": _candidate_label(item.ploss, item.noise_power_dB),
                "execution_id": item.execution_id,
                "repetition": item.repetition,
                "ploss": item.ploss,
                "noise_power_dB": item.noise_power_dB,
                "balanced_rows": balanced_count,
                "mmd_squared": biased_rbf_mmd2(
                    upv_values,
                    simulated_values,
                    bandwidth=bandwidth,
                    maximum_samples=balanced_count,
                ),
                "upv_p10": float(upv_balanced["value"].quantile(0.10)),
                "upv_median": float(upv_balanced["value"].median()),
                "upv_p90": float(upv_balanced["value"].quantile(0.90)),
                "rfsim_p10": float(simulated["value"].quantile(0.10)),
                "rfsim_median": float(simulated["value"].median()),
                "rfsim_p90": float(simulated["value"].quantile(0.90)),
            })
        execution_frame = pd.DataFrame(execution_rows)
        current = execution_frame.loc[
            execution_frame["metric"].eq(metric)
            & execution_frame["source_key"].eq(source_key)
            & execution_frame["locked_role"].eq(role)
        ]
        for (ploss, noise), group in current.groupby(["ploss", "noise_power_dB"], sort=True):
            execution_ids = group["execution_id"].tolist()
            bootstrap_values = np.empty(bootstrap_repetitions)
            upv_full = _transformed(
                upv.rename(columns={upv_column: "value"}), "value", center, scale
            )
            simulation_full = {
                execution_id: _transformed(
                    simulation[execution_id].rename(columns={rfsim_column: "value"}),
                    "value",
                    center,
                    scale,
                )
                for execution_id in execution_ids
            }
            for index in range(bootstrap_repetitions):
                upv_sample = _circular_block_sample(
                    upv_full,
                    target_count=balanced_count,
                    block_length=block_length,
                    rng=rng,
                )
                distances = []
                for execution_id in execution_ids:
                    simulated_sample = _circular_block_sample(
                        simulation_full[execution_id],
                        target_count=balanced_count,
                        block_length=block_length,
                        rng=rng,
                    )
                    distances.append(biased_rbf_mmd2(
                        upv_sample,
                        simulated_sample,
                        bandwidth=bandwidth,
                        maximum_samples=balanced_count,
                    ))
                bootstrap_values[index] = float(np.mean(distances))
            candidate_rows.append({
                "metric": metric,
                "source_key": source_key,
                "source_role": metadata.loc[(source_key, role), "source_role"],
                "device": metadata.loc[(source_key, role), "device"],
                "locked_role": role,
                "candidate_id": _candidate_label(float(ploss), float(noise)),
                "ploss": ploss,
                "noise_power_dB": noise,
                "execution_count": len(group),
                "mean_execution_mmd_squared": float(group["mmd_squared"].mean()),
                "minimum_execution_mmd_squared": float(group["mmd_squared"].min()),
                "maximum_execution_mmd_squared": float(group["mmd_squared"].max()),
                "conditional_bootstrap_ci_low": float(np.quantile(bootstrap_values, tail)),
                "conditional_bootstrap_ci_high": float(
                    np.quantile(bootstrap_values, 1.0 - tail)
                ),
                "repeatability_threshold_mmd_squared": threshold,
                "supported": bool(float(group["mmd_squared"].mean()) <= threshold),
            })
    executions = pd.DataFrame(execution_rows)
    candidates = pd.DataFrame(candidate_rows)
    candidates["rank_within_unit"] = candidates.groupby(
        ["metric", "source_key", "locked_role"]
    )["mean_execution_mmd_squared"].rank(method="first").astype(int)
    candidates = candidates.sort_values(
        ["metric", "source_key", "locked_role", "rank_within_unit"]
    ).reset_index(drop=True)
    summary = {
        "metric": metric,
        "balanced_rows_per_source": balanced_count,
        "balanced_reference_source_count": 1 + len(selected),
        "center": center,
        "scale": scale,
        "bandwidth": bandwidth,
        "repeatability_pair_count": len(repeatability),
        "repeatability_threshold_quantile": threshold_quantile,
        "repeatability_threshold_mmd_squared": threshold,
        "minimum_repeatability_mmd_squared": float(repeatability["mmd_squared"].min()),
        "maximum_repeatability_mmd_squared": float(repeatability["mmd_squared"].max()),
    }
    balanced_reference.insert(0, "metric", metric)
    return executions, candidates, repeatability, summary, balanced_reference


def _phase3b_decision(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    primary = candidates.loc[
        candidates["source_key"].eq("primary")
        & candidates["locked_role"].eq("calibration")
    ]
    metric_details: dict[str, dict[str, object]] = {}
    supported_sets: dict[str, set[str]] = {}
    control_limits = {
        "ploss": (float(selected["ploss"].min()), float(selected["ploss"].max())),
        "noise_power_dB": (
            float(selected["noise_power_dB"].min()),
            float(selected["noise_power_dB"].max()),
        ),
    }
    for metric in ["relative_RSRP", "SINR"]:
        metric_rows = primary.loc[primary["metric"].eq(metric)].sort_values(
            "mean_execution_mmd_squared"
        )
        best = metric_rows.iloc[0]
        supported = set(metric_rows.loc[metric_rows["supported"], "candidate_id"])
        supported_sets[metric] = supported
        validation = candidates.loc[
            candidates["metric"].eq(metric)
            & candidates["source_key"].eq("primary")
            & candidates["locked_role"].str.startswith("spatial_validation_")
            & candidates["candidate_id"].eq(best["candidate_id"])
        ]
        metric_details[metric] = {
            "best_candidate_id": best["candidate_id"],
            "best_mean_mmd_squared": float(best["mean_execution_mmd_squared"]),
            "repeatability_threshold_mmd_squared": float(
                best["repeatability_threshold_mmd_squared"]
            ),
            "primary_calibration_supported": bool(best["supported"]),
            "supported_candidate_ids": sorted(supported),
            "best_candidate_boundary": bool(
                float(best["ploss"]) in control_limits["ploss"]
                or float(best["noise_power_dB"]) in control_limits["noise_power_dB"]
            ),
            "locked_validation_regions_evaluated": len(validation),
            "locked_validation_regions_supported": int(validation["supported"].sum()),
            "locked_validation_all_supported": bool(
                len(validation) > 0 and validation["supported"].all()
            ),
        }
    relative_supported = bool(supported_sets["relative_RSRP"])
    sinr_supported = bool(supported_sets["SINR"])
    common = sorted(supported_sets["relative_RSRP"] & supported_sets["SINR"])
    if relative_supported and sinr_supported and common:
        code = "nonabsolute_support_with_common_candidate_repetitions_required"
        next_action = (
            "Plan independent repetitions of the common diagnostic states; do not "
            "interpret them as physical ploss/noise estimates."
        )
    elif relative_supported and sinr_supported:
        code = "separate_metric_support_without_common_candidate"
        next_action = "Revise the control/discrepancy design before collecting new executions."
    elif not relative_supported and sinr_supported:
        code = "relative_rsrp_shape_mismatch"
        next_action = "Revise fading/channel-variability structure before requesting POWDER time."
    elif relative_supported and not sinr_supported:
        code = "sinr_empirical_mismatch"
        next_action = (
            "Inspect the effective SINR/noise control boundary without claiming absolute "
            "noise-power calibration."
        )
    else:
        code = "gross_nonabsolute_joint_mismatch"
        next_action = "Revise the simulator discrepancy model before any new campaign."
    decision = {
        "schema_version": 1,
        "decision_code": code,
        "selected_measurement_branch": "insufficient_metadata",
        "metric_support": metric_details,
        "common_supported_candidate_ids": common,
        "next_action": next_action,
        "abc_performed": False,
        "new_execution_authorized": False,
        "claim_limits": config["claim_limits"],
        "interpretation": (
            "diagnostic support for offset-invariant relative RSRP and empirical, "
            "device-conditioned SINR only"
        ),
    }
    reservation = {
        "schema_version": 1,
        "decision_code": code,
        "reservation_should_be_requested_now": False,
        "preparation_lead_time_minutes": int(
            config["reservation"]["preparation_lead_time_minutes"]
        ),
        "gate_state": "closed",
        "blocking_conditions": [
            "Phase 3B is diagnostic and does not authorize execution.",
            "Only two independent executions exist per state.",
            "The PBCH/PUSCH parser is not yet validated on real logs from the pinned OAI revision.",
            "Any next campaign must be derived from the frozen Phase 3B decision "
            "and committed first.",
        ],
        "next_action": next_action,
    }
    return decision, reservation


def _distribution_diagnostics(
    units: dict[tuple[str, str], pd.DataFrame],
    unit_metadata: pd.DataFrame,
    simulation: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(
        *,
        metric: str,
        values: np.ndarray,
        source_kind: str,
        source_key: str,
        locked_role: str,
        device: str,
        candidate_id: str,
        ploss: float,
        noise_power_db: float,
    ) -> None:
        q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append({
            "metric": metric,
            "source_kind": source_kind,
            "source_key": source_key,
            "locked_role": locked_role,
            "device": device,
            "candidate_id": candidate_id,
            "ploss": ploss,
            "noise_power_dB": noise_power_db,
            "sample_count": len(values),
            "unique_value_count": len(np.unique(values)),
            "constant_distribution": bool(np.ptp(values) <= np.finfo(float).eps),
            "minimum": float(np.min(values)),
            "p10": float(q10),
            "p25": float(q25),
            "median": float(median),
            "p75": float(q75),
            "p90": float(q90),
            "maximum": float(np.max(values)),
            "iqr": float(q75 - q25),
            "standard_deviation_ddof1": float(np.std(values, ddof=1)),
        })

    metadata = unit_metadata.set_index(["source_key", "locked_role"])
    for source_key, role in sorted(units):
        frame = units[(source_key, role)]
        device = str(metadata.loc[(source_key, role), "device"])
        add_row(
            metric="relative_RSRP",
            values=frame["relative_rsrp_db"].to_numpy(float),
            source_kind="UPV",
            source_key=source_key,
            locked_role=role,
            device=device,
            candidate_id="",
            ploss=math.nan,
            noise_power_db=math.nan,
        )
        add_row(
            metric="SINR",
            values=frame["sinr_db"].to_numpy(float),
            source_kind="UPV",
            source_key=source_key,
            locked_role=role,
            device=device,
            candidate_id="",
            ploss=math.nan,
            noise_power_db=math.nan,
        )
    selected_by_execution = selected.set_index("execution_id")
    for execution_id, frame in sorted(simulation.items()):
        item = selected_by_execution.loc[execution_id]
        candidate_id = _candidate_label(
            float(item["ploss"]), float(item["noise_power_dB"])
        )
        add_row(
            metric="relative_RSRP",
            values=frame["relative_rsrp_db"].to_numpy(float),
            source_kind="RFsim",
            source_key=execution_id,
            locked_role="analysis_window_15_175s",
            device="OAI_UE",
            candidate_id=candidate_id,
            ploss=float(item["ploss"]),
            noise_power_db=float(item["noise_power_dB"]),
        )
        add_row(
            metric="SINR",
            values=frame["ss_sinr_db"].to_numpy(float),
            source_kind="RFsim",
            source_key=execution_id,
            locked_role="analysis_window_15_175s",
            device="OAI_UE",
            candidate_id=candidate_id,
            ploss=float(item["ploss"]),
            noise_power_db=float(item["noise_power_dB"]),
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "source_kind", "source_key", "locked_role"]
    ).reset_index(drop=True)


def _sensitivity_summary(candidates: pd.DataFrame) -> dict[str, object]:
    details: dict[str, object] = {}
    for source_key in ["primary", "filename_sensitivity", "s25_robustness"]:
        source_rows = candidates.loc[candidates["source_key"].eq(source_key)]
        source_details: dict[str, object] = {}
        for metric in ["relative_RSRP", "SINR"]:
            metric_rows = source_rows.loc[source_rows["metric"].eq(metric)]
            calibration = metric_rows.loc[metric_rows["locked_role"].eq("calibration")]
            best = calibration.sort_values("mean_execution_mmd_squared").iloc[0]
            role_support = metric_rows.groupby("locked_role")["supported"].any()
            source_details[metric] = {
                "calibration_best_candidate_id": best["candidate_id"],
                "calibration_best_mean_mmd_squared": float(
                    best["mean_execution_mmd_squared"]
                ),
                "calibration_supported": bool(best["supported"]),
                "evaluated_region_count": len(role_support),
                "regions_with_any_supported_candidate": int(role_support.sum()),
            }
        details[source_key] = source_details
    return {
        "schema_version": 1,
        "preprocessing_fit_source": "primary_ASUS_calibration_only",
        "nonprimary_sources_affect_fit_or_ranking": False,
        "sources": details,
    }


def analyze_phase3b_support(
    *,
    route_observations: str | Path,
    locked_split: str | Path,
    upv_archive: str | Path,
    phase1_config: str | Path,
    selection_manifest: str | Path,
    campaign_state: str | Path,
    executions_root: str | Path,
    phase3a_decision: str | Path,
    phase3a_gate: str | Path,
    public_evidence: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    paths = {
        "route_observations": Path(route_observations).resolve(),
        "locked_split": Path(locked_split).resolve(),
        "upv_archive": Path(upv_archive).resolve(),
        "phase1_config": Path(phase1_config).resolve(),
        "selection_manifest": Path(selection_manifest).resolve(),
        "campaign_state": Path(campaign_state).resolve(),
        "executions_root": Path(executions_root).resolve(),
        "phase3a_decision": Path(phase3a_decision).resolve(),
        "phase3a_gate": Path(phase3a_gate).resolve(),
        "public_evidence": Path(public_evidence).resolve(),
        "config": Path(config_path).resolve(),
        "output": Path(output_dir).resolve(),
    }
    for name, path in paths.items():
        if name in {"executions_root", "output"}:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe Phase 3B input: {path}")
    if not paths["executions_root"].is_dir() or paths["executions_root"].is_symlink():
        raise ValueError("executions root must be a real directory")
    if paths["output"].exists():
        raise FileExistsError(f"Phase 3B output already exists: {paths['output']}")

    config = _read_yaml(paths["config"])
    validate_phase3b_config(config)
    expected_evidence = str(config["frozen_inputs"]["public_evidence_addendum_sha256"])
    if _sha256(paths["public_evidence"]) != expected_evidence:
        raise ValueError("public-evidence addendum checksum does not match the frozen protocol")
    phase3a_decision_document = _read_json(paths["phase3a_decision"])
    phase3a_gate_document = _read_json(paths["phase3a_gate"])
    required_decision = str(config["frozen_inputs"]["required_phase3a_decision"])
    if phase3a_decision_document.get("decision_code") != required_decision:
        raise ValueError("Phase 3A decision prerequisite is not satisfied")
    if bool(phase3a_decision_document.get("absolute_rsrp_calibration_authorized")):
        raise ValueError("Phase 3A unexpectedly authorizes absolute RSRP")
    if bool(phase3a_gate_document.get("reservation_should_be_requested_now")):
        raise ValueError("Phase 3A reservation gate is unexpectedly open")

    phase1 = _read_yaml(paths["phase1_config"])
    primary_route = pd.read_csv(paths["route_observations"])
    split = pd.read_csv(paths["locked_split"])
    roles = [str(value) for value in config["upv_units"]["required_roles"]]
    bin_size = int(config["upv_units"]["primary_bin_size_m"])
    primary_route = _assign_primary_regions(
        primary_route, split, bin_size_m=bin_size, roles=roles
    )
    transfer = config["region_transfer"]
    routes: dict[str, pd.DataFrame] = {"primary": primary_route}
    transfer_frames: list[pd.DataFrame] = []
    primary_diagnostics = []
    for role in roles:
        count = int(primary_route["locked_role"].eq(role).sum())
        primary_diagnostics.append({
            "locked_role": role,
            "nearest_role_rows": count,
            "rows_within_maximum_distance": count,
            "retained_fraction": 1.0,
            "median_reference_distance_m": 0.0,
            "p95_reference_distance_m": 0.0,
            "unit_valid": count > 0,
        })
    primary_transfer = pd.DataFrame(primary_diagnostics)
    primary_transfer.insert(0, "source_key", "primary")
    transfer_frames.append(primary_transfer)
    for source_key in ["filename_sensitivity", "s25_robustness"]:
        unit = config["upv_units"][source_key]
        route = _load_archive_route(
            paths["upv_archive"],
            source_path=str(unit["source_path"]),
            corrected_test_id=int(unit["route_corrected_test_id"]),
            trim_last_seconds=float(unit["trim_last_seconds"]),
            phase1_config=phase1,
        )
        transferred, diagnostics = transfer_locked_regions(
            route,
            primary_route,
            roles=roles,
            maximum_distance_m=float(transfer["maximum_nearest_reference_distance_m"]),
            minimum_rows=int(transfer["minimum_raw_rows_per_region"]),
            minimum_fraction=float(transfer["minimum_transferred_fraction_per_region"]),
        )
        routes[source_key] = transferred
        diagnostics.insert(0, "source_key", source_key)
        transfer_frames.append(diagnostics)
    region_diagnostics = pd.concat(transfer_frames, ignore_index=True)

    duration = float(config["temporal_aggregation"]["duration_seconds"])
    units: dict[tuple[str, str], pd.DataFrame] = {}
    unit_inventory_rows: list[dict[str, object]] = []
    for source_key, route in routes.items():
        unit_config = config["upv_units"][source_key]
        source_units, inventory = _aggregate_units(
            route,
            source_key=source_key,
            source_role=str(unit_config["role"]),
            device=str(unit_config["device"]),
            roles=roles,
            duration=duration,
            minimum_aggregated_rows=int(
                transfer["minimum_aggregated_rows_per_unit"]
            ),
        )
        units.update(source_units)
        unit_inventory_rows.extend(inventory)
    if ("primary", "calibration") not in units:
        raise ValueError("the primary UPV calibration unit is unavailable")
    unit_inventory = pd.DataFrame(unit_inventory_rows)

    selected = _selection_rows(
        _read_json(paths["selection_manifest"]), _read_json(paths["campaign_state"])
    )
    simulation, execution_inventory = _load_rfsim(
        selected, paths["executions_root"], config, config["targets"]
    )
    for frame in simulation.values():
        frame["relative_rsrp_db"] = (
            frame["ss_rsrp_dbm"] - float(frame["ss_rsrp_dbm"].median())
        )

    metric_specs = [
        ("relative_RSRP", "relative_rsrp_db", "relative_rsrp_db"),
        ("SINR", "sinr_db", "ss_sinr_db"),
    ]
    execution_tables = []
    candidate_tables = []
    repeatability_tables = []
    scaling_rows = []
    balanced_tables = []
    for metric, upv_column, rfsim_column in metric_specs:
        executions, candidates, repeatability, summary, balanced = _metric_support(
            metric=metric,
            upv_column=upv_column,
            rfsim_column=rfsim_column,
            units=units,
            unit_metadata=unit_inventory,
            simulation=simulation,
            selected=selected,
            config=config,
        )
        execution_tables.append(executions)
        candidate_tables.append(candidates)
        repeatability_tables.append(repeatability)
        scaling_rows.append(summary)
        balanced_tables.append(balanced)
    execution_support = pd.concat(execution_tables, ignore_index=True)
    candidate_support = pd.concat(candidate_tables, ignore_index=True)
    repeatability = pd.concat(repeatability_tables, ignore_index=True)
    scaling = pd.DataFrame(scaling_rows)
    balanced_reference = pd.concat(balanced_tables, ignore_index=True)
    decision, reservation = _phase3b_decision(candidate_support, selected, config)
    distribution_diagnostics = _distribution_diagnostics(
        units, unit_inventory, simulation, selected
    )
    sensitivity_summary = _sensitivity_summary(candidate_support)

    primary_rankings = candidate_support.loc[
        candidate_support["source_key"].eq("primary")
        & candidate_support["locked_role"].eq("calibration")
    ].sort_values(["metric", "rank_within_unit"])
    best_by_metric = primary_rankings.loc[primary_rankings["rank_within_unit"].eq(1), [
        "metric", "candidate_id"
    ]]
    validation_rows = []
    for item in best_by_metric.itertuples(index=False):
        validation_rows.append(candidate_support.loc[
            candidate_support["metric"].eq(item.metric)
            & candidate_support["source_key"].eq("primary")
            & candidate_support["locked_role"].str.startswith("spatial_validation_")
            & candidate_support["candidate_id"].eq(item.candidate_id)
        ])
    validation_support = pd.concat(validation_rows, ignore_index=True)

    software = _git_revision()
    staging = paths["output"].parent / f".{paths['output'].name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        preprocessing = {
            "schema_version": 1,
            "analysis_implementation_revision": software["revision"],
            "config_sha256": _sha256(paths["config"]),
            "selected_measurement_branch": config["selected_measurement_branch"],
            "frozen_before_candidate_comparison": True,
            "targets": config["targets"],
            "temporal_aggregation": config["temporal_aggregation"],
            "missing_value_policy": config["missing_value_policy"],
            "region_transfer": config["region_transfer"],
            "upv_units": config["upv_units"],
            "rfsim_selection": config["rfsim_selection"],
            "balanced_reference": config["balanced_reference"],
            "kernel": config["kernel"],
            "bootstrap": config["bootstrap"],
            "support_rules": config["support_rules"],
            "realized_target_preprocessing": scaling.to_dict("records"),
            "included_rfsim_executions": selected.to_dict("records"),
            "claim_limits": config["claim_limits"],
        }
        _write_json(staging / "preprocessing_specification.json", preprocessing)
        input_inventory = pd.DataFrame([
            {
                "input_kind": name,
                "source_id": path.name,
                "sha256": _sha256(path),
                "rows_read": (
                    len(primary_route) if name == "route_observations"
                    else len(split) if name == "locked_split"
                    else math.nan
                ),
            }
            for name, path in paths.items()
            if name not in {"executions_root", "output"}
        ])
        input_inventory = pd.concat(
            [input_inventory, execution_inventory], ignore_index=True, sort=False
        )
        table_outputs = {
            "input_inventory.csv": input_inventory,
            "region_transfer_diagnostics.csv": region_diagnostics,
            "upv_unit_inventory.csv": unit_inventory,
            "distribution_diagnostics.csv": distribution_diagnostics,
            "scaling_parameters.csv": scaling,
            "balanced_reference.csv": balanced_reference,
            "repeatability_pairs.csv": repeatability,
            "execution_support.csv": execution_support,
            "candidate_support.csv": candidate_support,
            "primary_rankings.csv": primary_rankings,
            "locked_validation_support.csv": validation_support,
        }
        for name, frame in table_outputs.items():
            _write_csv(staging / name, frame)
        _write_json(staging / "phase3b_decision.json", decision)
        _write_json(staging / "sensitivity_summary.json", sensitivity_summary)
        _write_json(staging / "reservation_gate_v3.json", reservation)
        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "analysis_implementation_revision": software["revision"],
            "tracked_worktree_dirty_at_start": software["tracked_worktree_dirty"],
            "selected_measurement_branch": "insufficient_metadata",
            "input_sha256": {
                name: _sha256(path)
                for name, path in paths.items()
                if name not in {"executions_root", "output"}
            },
            "upv_valid_unit_count": len(unit_inventory),
            "rfsim_execution_count": len(selected),
            "candidate_state_count": int(
                selected[["ploss", "noise_power_dB"]].drop_duplicates().shape[0]
            ),
            "decision_code": decision["decision_code"],
            "abc_performed": False,
            "reservation_should_be_requested_now": False,
            "output_sha256_before_manifest": output_hashes,
        }
        _write_json(staging / "analysis_manifest.json", manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(staging / "SHA256SUMS.json", {
            "schema_version": 1,
            "files": checksums,
        })
        staging.replace(paths["output"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(paths["output"]),
        "decision_code": decision["decision_code"],
        "upv_units": len(unit_inventory),
        "executions": len(selected),
        "candidate_states": int(
            selected[["ploss", "noise_power_dB"]].drop_duplicates().shape[0]
        ),
        "abc_performed": False,
        "reservation_should_be_requested_now": False,
    }
