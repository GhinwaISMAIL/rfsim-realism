from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .mmd_abc import biased_rbf_mmd2, median_heuristic_bandwidth
from .upv_phase3c13 import _attached
from .upv_support import _circular_block_sample


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


def validate_phase3c15_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3C15 schema_version must be 1")
    if config.get("stage") != "phase_3c15_offline_awgn_support_diagnostic":
        raise ValueError("unexpected Phase 3C15 stage")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("Phase 3C15 cannot authorize execution or ABC")

    measurement = config.get("measurement") or {}
    if measurement.get("calibration_role") != "calibration":
        raise ValueError("the calibration trace must remain explicit")
    if measurement.get("holdout_roles") != [
        "spatial_validation_1",
        "spatial_validation_2",
        "spatial_validation_3",
        "spatial_validation_4",
    ]:
        raise ValueError("the four locked holdouts must remain unchanged")
    if int(measurement.get("route_bin_size_m", 0)) != 15:
        raise ValueError("Phase 3C15 requires the frozen 15 m split")
    if measurement.get("relative_rsrp_transform") != (
        "subtract_within_trace_median_after_aggregation"
    ):
        raise ValueError("relative RSRP must be trace-median centred")

    simulation = config.get("rfsim") or {}
    if simulation.get("channel_family") != "AWGN":
        raise ValueError("Phase 3C15 evaluates the accepted AWGN control")
    if simulation.get("expected_replays") != [f"awgn-{index}" for index in range(1, 6)]:
        raise ValueError("five frozen AWGN replays are required")
    if float(simulation.get("expected_noise_power_db", math.nan)) != -30.0:
        raise ValueError("the control noise must remain fixed at -30 dB")
    if simulation.get("interpretation") != (
        "validated_control_envelope_not_a_calibrated_UPV_trace"
    ):
        raise ValueError("the control-envelope interpretation must remain explicit")

    preprocessing = config.get("preprocessing") or {}
    if preprocessing.get("fitted_from") != "UPV_calibration_trace_only":
        raise ValueError("preprocessing must be fitted from calibration only")
    if not bool(preprocessing.get("holdouts_excluded_from_all_fit_and_threshold_steps")):
        raise ValueError("holdouts cannot affect preprocessing or thresholds")
    if preprocessing.get("mmd_estimator") != "biased_mmd_squared_v_statistic":
        raise ValueError("Phase 3C15 requires nonnegative biased MMD squared")
    if preprocessing.get("posthoc_clipping") != "prohibited":
        raise ValueError("MMD clipping remains prohibited")

    reference = config.get("conditional_reference") or {}
    if reference.get("method") != ("paired_circular_moving_block_bootstrap_of_calibration_trace"):
        raise ValueError("unexpected conditional-reference method")
    if int(reference.get("block_length_rows", 0)) < 2:
        raise ValueError("the moving block must contain at least two rows")
    if int(reference.get("repetitions", 0)) < 100:
        raise ValueError("the conditional reference requires at least 100 repetitions")
    quantile = float(reference.get("support_quantile", 0))
    if not 0.5 < quantile < 1.0:
        raise ValueError("the support quantile must be between 0.5 and 1")

    rules = config.get("support_rules") or {}
    if not bool(
        rules.get("control_envelope_failure_does_not_prove_AWGN_model_class_impossibility")
    ):
        raise ValueError("the control-envelope claim boundary is required")
    if rules.get("holdout_evaluation") != (
        "apply_calibration_preprocessing_bandwidths_and_thresholds_without_refitting"
    ):
        raise ValueError("holdout evaluation cannot refit")

    claims = config.get("claim_limits") or {}
    required_prohibitions = {
        "current_control_envelope_is_final_model",
        "absolute_rsrp_calibration",
        "physical_ploss_inference",
        "absolute_noise_power_calibration",
        "joint_identifiability",
        "abc",
    }
    if any(claims.get(key) != "prohibited" for key in required_prohibitions):
        raise ValueError("Phase 3C15 claim prohibitions are incomplete")

    reservation = config.get("reservation") or {}
    if bool(reservation.get("request_now")):
        raise ValueError("the offline phase cannot request a reservation")
    if reservation.get("gate_state") != "closed_for_offline_analysis":
        raise ValueError("the reservation gate must remain closed")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def _verify_inputs(paths: dict[str, Path], config: dict[str, Any]) -> None:
    frozen = config["frozen_inputs"]
    checksum_keys = {
        "route_observations": "route_observations_sha256",
        "locked_spatial_split": "locked_spatial_split_sha256",
        "phase3c14_telemetry": "phase3c14_telemetry_sha256",
        "phase3c14_evaluation": "phase3c14_evaluation_sha256",
        "phase3c14_result": "phase3c14_result_sha256",
        "phase3b_decision": "phase3b_decision_sha256",
        "phase3b_distribution_diagnostics": "phase3b_distribution_diagnostics_sha256",
        "phase3b_locked_validation_support": "phase3b_locked_validation_support_sha256",
    }
    for input_name, checksum_key in checksum_keys.items():
        if _sha256(paths[input_name]) != frozen[checksum_key]:
            raise ValueError(f"frozen input checksum mismatch: {input_name}")

    phase3c14_evaluation = _read_json(paths["phase3c14_evaluation"])
    phase3c14_result = _read_json(paths["phase3c14_result"])
    expected = frozen["required_phase3c14_decision"]
    if phase3c14_evaluation.get("decision_code") != expected:
        raise ValueError("Phase 3C14 evaluation prerequisite is not satisfied")
    if phase3c14_result.get("decision") != expected:
        raise ValueError("Phase 3C14 recorded result prerequisite is not satisfied")
    if phase3c14_evaluation.get("control_gate_pass") is not True:
        raise ValueError("Phase 3C14 control gate did not pass")

    phase3b = _read_json(paths["phase3b_decision"])
    if phase3b.get("decision_code") != frozen["required_phase3b_decision"]:
        raise ValueError("Phase 3B decision prerequisite is not satisfied")
    if phase3b.get("abc_performed") is not False:
        raise ValueError("Phase 3B must not have performed ABC")


def _aggregate_upv(
    route_path: Path,
    split_path: Path,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    measurement = config["measurement"]
    route = pd.read_parquet(route_path)
    split = pd.read_csv(split_path)
    required = set(measurement["required_columns"])
    missing = sorted(required - set(route.columns))
    if missing:
        raise ValueError(f"UPV route table is missing columns: {missing}")
    route = route.loc[route["source_path"].eq(measurement["source_path"])].copy()
    if route.empty:
        raise ValueError("the frozen UPV source is absent")
    bin_size = int(measurement["route_bin_size_m"])
    selected = split.loc[split["bin_size_m"].eq(bin_size)].copy()
    role_by_bin = selected.set_index("route_bin_id")["locked_role"].astype(str).to_dict()
    route["trace_role"] = route[f"route_bin_{bin_size}m"].map(role_by_bin)
    roles = [measurement["calibration_role"], *measurement["holdout_roles"]]
    duration = float(measurement["temporal_aggregation_seconds"])
    traces: dict[str, pd.DataFrame] = {}
    for role in roles:
        raw = (
            route.loc[route["trace_role"].eq(role)]
            .dropna(subset=["seconds_of_day", "rsrp_dbm", "sinr_db"])
            .sort_values("seconds_of_day")
            .copy()
        )
        if raw.empty:
            raise ValueError(f"locked UPV trace is empty: {role}")
        origin = float(raw["seconds_of_day"].min())
        raw["time_bin"] = np.floor((raw["seconds_of_day"] - origin) / duration).astype(int)
        aggregated = (
            raw.groupby("time_bin", sort=True)[["rsrp_dbm", "sinr_db"]].median().reset_index()
        )
        if len(aggregated) < 10:
            raise ValueError(f"locked UPV trace has fewer than ten aggregated rows: {role}")
        aggregated["trace_role"] = role
        aggregated["split_role"] = "calibration" if role == roles[0] else "holdout"
        aggregated["t_s"] = aggregated["time_bin"].astype(float) * duration
        aggregated["relative_rsrp_db"] = aggregated["rsrp_dbm"] - float(
            aggregated["rsrp_dbm"].median()
        )
        traces[role] = aggregated[
            [
                "trace_role",
                "split_role",
                "time_bin",
                "t_s",
                "rsrp_dbm",
                "relative_rsrp_db",
                "sinr_db",
            ]
        ].copy()
    return traces


def _load_awgn(
    telemetry_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    simulation = config["rfsim"]
    telemetry = pd.read_csv(telemetry_path)
    required = {
        "replay_id",
        "t_s",
        "commanded_gain_db",
        "applied_gain_db",
        "channel_family",
        "noise_power_db",
        simulation["distribution_rsrp_column"],
        simulation["sinr_column"],
        "attached",
    }
    missing = sorted(required - set(telemetry.columns))
    if missing:
        raise ValueError(f"AWGN telemetry is missing columns: {missing}")
    numeric = sorted(required - {"replay_id", "channel_family", "attached"})
    for column in numeric:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    if not np.isfinite(telemetry[numeric].to_numpy(float)).all():
        raise ValueError("AWGN telemetry contains non-finite required values")
    telemetry["attached"] = _attached(telemetry["attached"])
    if not telemetry["attached"].all():
        raise ValueError("AWGN telemetry contains detached rows")
    if set(telemetry["channel_family"].astype(str)) != {simulation["channel_family"]}:
        raise ValueError("unexpected channel family in AWGN telemetry")
    if (
        float((telemetry["noise_power_db"] - simulation["expected_noise_power_db"]).abs().max())
        > 1e-9
    ):
        raise ValueError("AWGN telemetry does not retain the frozen noise setting")
    if float((telemetry["commanded_gain_db"] - telemetry["applied_gain_db"]).abs().max()) > 1e-9:
        raise ValueError("commanded and applied gain differ")
    observed = sorted(telemetry["replay_id"].astype(str).unique())
    if observed != simulation["expected_replays"]:
        raise ValueError("unexpected AWGN replay identities")

    duration = float(simulation["segment_duration_seconds"])
    settling = float(simulation["distribution_settling_exclusion_seconds"])
    rsrp_column = simulation["distribution_rsrp_column"]
    sinr_column = simulation["sinr_column"]
    distribution: dict[str, pd.DataFrame] = {}
    temporal: dict[str, pd.DataFrame] = {}
    for replay_id in observed:
        replay = telemetry.loc[telemetry["replay_id"].astype(str).eq(replay_id)].copy()
        replay = replay.sort_values("t_s").reset_index(drop=True)
        replay["trace_role"] = replay_id
        replay["split_role"] = "candidate"
        replay["relative_rsrp_db"] = replay[rsrp_column] - float(replay[rsrp_column].median())
        replay["sinr_db"] = replay[sinr_column]
        temporal[replay_id] = replay[
            ["trace_role", "split_role", "t_s", "relative_rsrp_db", "sinr_db"]
        ].copy()

        replay["segment_index"] = np.floor(replay["t_s"] / duration).astype(int)
        replay["within_segment_s"] = replay["t_s"] - replay["segment_index"] * duration
        retained = replay.loc[
            replay["segment_index"].between(0, 4)
            & replay["within_segment_s"].ge(settling)
            & replay["within_segment_s"].lt(duration)
        ].copy()
        retained["relative_rsrp_db"] = retained[rsrp_column] - float(retained[rsrp_column].median())
        if len(retained) != 35:
            raise ValueError(f"expected 35 post-settling rows for {replay_id}")
        retained["sinr_db"] = retained[sinr_column]
        distribution[replay_id] = retained[
            [
                "trace_role",
                "split_role",
                "t_s",
                "segment_index",
                "commanded_gain_db",
                "relative_rsrp_db",
                "sinr_db",
            ]
        ].copy()
    return distribution, temporal


def _robust_location_scale(values: np.ndarray, factor: float) -> tuple[float, float, str]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) * factor)
    method = "normalized_mad"
    if scale <= np.finfo(float).eps:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        method = "iqr_divided_by_1.349"
    if scale <= np.finfo(float).eps:
        scale = float(np.std(values, ddof=0))
        method = "population_standard_deviation"
    if scale <= np.finfo(float).eps:
        raise ValueError("calibration feature has zero robust scale")
    return center, scale, method


def _fit_preprocessing(calibration: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    factor = float(config["preprocessing"]["normalized_mad_factor"])
    features: dict[str, dict[str, object]] = {}
    for name, column in (("relative_RSRP", "relative_rsrp_db"), ("SINR", "sinr_db")):
        center, scale, method = _robust_location_scale(calibration[column].to_numpy(float), factor)
        features[name] = {
            "column": column,
            "center": center,
            "scale": scale,
            "scale_method": method,
        }
    transformed = _transform(calibration, features, "joint")
    bandwidths = {
        "relative_RSRP": median_heuristic_bandwidth(
            _transform(calibration, features, "relative_RSRP")
        ),
        "SINR": median_heuristic_bandwidth(_transform(calibration, features, "SINR")),
        "joint": median_heuristic_bandwidth(transformed),
    }
    return {"features": features, "bandwidths": bandwidths}


def _transform(
    frame: pd.DataFrame,
    features: dict[str, dict[str, object]],
    target: str,
) -> np.ndarray:
    names = [target] if target != "joint" else ["relative_RSRP", "SINR"]
    columns: list[np.ndarray] = []
    for name in names:
        spec = features[name]
        values = frame[str(spec["column"])].to_numpy(float)
        columns.append((values - float(spec["center"])) / float(spec["scale"]))
    return np.column_stack(columns)


def _acf(values: np.ndarray, lags: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result: list[float] = []
    for lag in lags:
        left = values[:-lag]
        right = values[lag:]
        if (
            len(left) < 3
            or np.std(left) <= np.finfo(float).eps
            or np.std(right) <= np.finfo(float).eps
        ):
            result.append(math.nan)
        else:
            result.append(float(np.corrcoef(left, right)[0, 1]))
    return np.asarray(result, dtype=float)


def _relationship(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    x = values[:, 0]
    y = values[:, 1]
    if np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return {
            "spearman_correlation": math.nan,
            "standardized_ols_slope": math.nan,
            "standardized_residual_sd": math.nan,
        }
    ranks_x = pd.Series(x).rank(method="average").to_numpy(float)
    ranks_y = pd.Series(y).rank(method="average").to_numpy(float)
    spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "spearman_correlation": spearman,
        "standardized_ols_slope": float(slope),
        "standardized_residual_sd": float(np.std(residual, ddof=1)),
    }


def _finite_quantile(values: list[float], quantile: float, label: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < max(20, len(values) // 2):
        raise ValueError(f"insufficient finite bootstrap statistics: {label}")
    return float(np.quantile(finite, quantile))


def _conditional_thresholds(
    calibration: pd.DataFrame,
    preprocessing: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = config["conditional_reference"]
    repetitions = int(reference["repetitions"])
    block_length = int(reference["block_length_rows"])
    quantile = float(reference["support_quantile"])
    rng = np.random.default_rng(int(reference["seed"]))
    lags = [int(value) for value in config["temporal"]["lags_seconds"]]
    indices = np.arange(len(calibration))
    distributions = {name: [] for name in ("relative_RSRP", "SINR", "joint")}
    temporal = {name: [] for name in ("relative_RSRP", "SINR")}
    relationship = {
        name: []
        for name in (
            "spearman_correlation",
            "standardized_ols_slope",
            "standardized_residual_sd",
        )
    }
    features = preprocessing["features"]
    bandwidths = preprocessing["bandwidths"]
    for _ in range(repetitions):
        left_indices = _circular_block_sample(
            indices, target_count=len(indices), block_length=block_length, rng=rng
        ).astype(int)
        right_indices = _circular_block_sample(
            indices, target_count=len(indices), block_length=block_length, rng=rng
        ).astype(int)
        left = calibration.iloc[left_indices]
        right = calibration.iloc[right_indices]
        for target in distributions:
            distributions[target].append(
                biased_rbf_mmd2(
                    _transform(left, features, target),
                    _transform(right, features, target),
                    bandwidth=float(bandwidths[target]),
                )
            )
        for name, column in (("relative_RSRP", "relative_rsrp_db"), ("SINR", "sinr_db")):
            left_acf = _acf(left[column].to_numpy(float), lags)
            right_acf = _acf(right[column].to_numpy(float), lags)
            temporal[name].append(float(np.nanmax(np.abs(left_acf - right_acf))))
        left_relationship = _relationship(_transform(left, features, "joint"))
        right_relationship = _relationship(_transform(right, features, "joint"))
        for name in relationship:
            relationship[name].append(abs(left_relationship[name] - right_relationship[name]))

    return {
        "method": reference["method"],
        "repetitions": repetitions,
        "block_length_rows": block_length,
        "support_quantile": quantile,
        "distribution_mmd_squared": {
            name: _finite_quantile(values, quantile, f"{name} MMD")
            for name, values in distributions.items()
        },
        "temporal_max_abs_acf_difference": {
            name: _finite_quantile(values, quantile, f"{name} ACF")
            for name, values in temporal.items()
        },
        "relationship_abs_difference": {
            name: _finite_quantile(values, quantile, name) for name, values in relationship.items()
        },
        "interpretation": reference["interpretation"],
    }


def _summary_row(
    source_kind: str,
    trace_id: str,
    split_role: str,
    frame: pd.DataFrame,
    temporal_frame: pd.DataFrame | None,
    lags: list[int],
) -> dict[str, object]:
    temporal_values = temporal_frame if temporal_frame is not None else frame
    rsrp = frame["relative_rsrp_db"].to_numpy(float)
    sinr = frame["sinr_db"].to_numpy(float)
    rsrp_acf = _acf(temporal_values["relative_rsrp_db"].to_numpy(float), lags)
    sinr_acf = _acf(temporal_values["sinr_db"].to_numpy(float), lags)
    row: dict[str, object] = {
        "source_kind": source_kind,
        "trace_id": trace_id,
        "split_role": split_role,
        "distribution_rows": len(frame),
        "temporal_rows": len(temporal_values),
        "relative_rsrp_min_db": float(np.min(rsrp)),
        "relative_rsrp_q25_db": float(np.quantile(rsrp, 0.25)),
        "relative_rsrp_median_db": float(np.median(rsrp)),
        "relative_rsrp_q75_db": float(np.quantile(rsrp, 0.75)),
        "relative_rsrp_max_db": float(np.max(rsrp)),
        "relative_rsrp_iqr_db": float(np.quantile(rsrp, 0.75) - np.quantile(rsrp, 0.25)),
        "sinr_min_db": float(np.min(sinr)),
        "sinr_q25_db": float(np.quantile(sinr, 0.25)),
        "sinr_median_db": float(np.median(sinr)),
        "sinr_q75_db": float(np.quantile(sinr, 0.75)),
        "sinr_max_db": float(np.max(sinr)),
        "sinr_iqr_db": float(np.quantile(sinr, 0.75) - np.quantile(sinr, 0.25)),
    }
    for index, lag in enumerate(lags):
        row[f"relative_rsrp_acf_lag_{lag}s"] = float(rsrp_acf[index])
        row[f"sinr_acf_lag_{lag}s"] = float(sinr_acf[index])
    return row


def _evaluate_role(
    role: str,
    upv: pd.DataFrame,
    awgn_distribution: dict[str, pd.DataFrame],
    awgn_temporal: dict[str, pd.DataFrame],
    preprocessing: dict[str, Any],
    thresholds: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    features = preprocessing["features"]
    bandwidths = preprocessing["bandwidths"]
    lags = [int(value) for value in config["temporal"]["lags_seconds"]]
    replay_rows: list[dict[str, object]] = []
    for replay_id, candidate in awgn_distribution.items():
        row: dict[str, object] = {"trace_role": role, "replay_id": replay_id}
        for target in ("relative_RSRP", "SINR", "joint"):
            row[f"{target}_mmd_squared"] = biased_rbf_mmd2(
                _transform(upv, features, target),
                _transform(candidate, features, target),
                bandwidth=float(bandwidths[target]),
            )
        replay_rows.append(row)
    replay_frame = pd.DataFrame(replay_rows)

    result: dict[str, object] = {
        "trace_role": role,
        "split_role": str(upv["split_role"].iloc[0]),
        "upv_rows": len(upv),
        "candidate_replays": len(awgn_distribution),
    }
    for target, label in (
        ("relative_RSRP", "relative_rsrp"),
        ("SINR", "sinr"),
        ("joint", "joint"),
    ):
        values = replay_frame[f"{target}_mmd_squared"].to_numpy(float)
        threshold = float(thresholds["distribution_mmd_squared"][target])
        result[f"{label}_mean_mmd_squared"] = float(np.mean(values))
        result[f"{label}_minimum_mmd_squared"] = float(np.min(values))
        result[f"{label}_maximum_mmd_squared"] = float(np.max(values))
        result[f"{label}_mmd_threshold"] = threshold
        result[f"{label}_distribution_pass"] = bool(np.mean(values) <= threshold)

    for name, column, label in (
        ("relative_RSRP", "relative_rsrp_db", "relative_rsrp"),
        ("SINR", "sinr_db", "sinr"),
    ):
        upv_acf = _acf(upv[column].to_numpy(float), lags)
        candidate_acfs = np.vstack(
            [_acf(frame[column].to_numpy(float), lags) for frame in awgn_temporal.values()]
        )
        candidate_acf = np.nanmean(candidate_acfs, axis=0)
        difference = float(np.nanmax(np.abs(upv_acf - candidate_acf)))
        threshold = float(thresholds["temporal_max_abs_acf_difference"][name])
        result[f"{label}_temporal_max_abs_acf_difference"] = difference
        result[f"{label}_temporal_threshold"] = threshold
        result[f"{label}_temporal_pass"] = bool(np.isfinite(difference) and difference <= threshold)
        for index, lag in enumerate(lags):
            result[f"{label}_upv_acf_lag_{lag}s"] = float(upv_acf[index])
            result[f"{label}_candidate_acf_lag_{lag}s"] = float(candidate_acf[index])

    upv_relationship = _relationship(_transform(upv, features, "joint"))
    candidate_relationships = [
        _relationship(_transform(frame, features, "joint")) for frame in awgn_distribution.values()
    ]
    relationship_passes: list[bool] = []
    for name in config["relationship"]["metrics"]:
        candidate_value = float(np.nanmean([row[name] for row in candidate_relationships]))
        difference = abs(upv_relationship[name] - candidate_value)
        threshold = float(thresholds["relationship_abs_difference"][name])
        passed = bool(np.isfinite(difference) and difference <= threshold)
        relationship_passes.append(passed)
        result[f"relationship_{name}_upv"] = upv_relationship[name]
        result[f"relationship_{name}_candidate"] = candidate_value
        result[f"relationship_{name}_abs_difference"] = difference
        result[f"relationship_{name}_threshold"] = threshold
        result[f"relationship_{name}_pass"] = passed
    result["relationship_pass"] = all(relationship_passes)

    pooled = pd.concat(awgn_distribution.values(), ignore_index=True)
    rsrp_range = (
        float(pooled["relative_rsrp_db"].min()),
        float(pooled["relative_rsrp_db"].max()),
    )
    sinr_range = (float(pooled["sinr_db"].min()), float(pooled["sinr_db"].max()))
    rsrp_covered = upv["relative_rsrp_db"].between(*rsrp_range)
    sinr_covered = upv["sinr_db"].between(*sinr_range)
    result["relative_rsrp_interval_coverage_fraction"] = float(rsrp_covered.mean())
    result["sinr_interval_coverage_fraction"] = float(sinr_covered.mean())
    result["joint_rectangle_coverage_fraction"] = float((rsrp_covered & sinr_covered).mean())
    result["candidate_relative_rsrp_min_db"] = rsrp_range[0]
    result["candidate_relative_rsrp_max_db"] = rsrp_range[1]
    result["candidate_sinr_min_db"] = sinr_range[0]
    result["candidate_sinr_max_db"] = sinr_range[1]

    result["trace_gate_pass"] = all(
        bool(result[key])
        for key in (
            "relative_rsrp_distribution_pass",
            "relative_rsrp_temporal_pass",
            "sinr_distribution_pass",
            "sinr_temporal_pass",
            "joint_distribution_pass",
            "relationship_pass",
        )
    )
    return result, replay_rows


def _decision(
    support: pd.DataFrame,
    phase3b: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    measurement = config["measurement"]
    calibration = support.loc[support["trace_role"].eq(measurement["calibration_role"])].iloc[0]
    holdout = support.loc[support["trace_role"].isin(measurement["holdout_roles"])]
    calibration_pass = bool(calibration["trace_gate_pass"])
    holdout_all_pass = bool(holdout["trace_gate_pass"].all())
    sinr_or_joint_failure = not all(
        bool(calibration[key])
        for key in (
            "sinr_distribution_pass",
            "sinr_temporal_pass",
            "joint_distribution_pass",
            "relationship_pass",
        )
    )
    relative_or_temporal_failure = not all(
        bool(calibration[key])
        for key in ("relative_rsrp_distribution_pass", "relative_rsrp_temporal_pass")
    )
    rules = config["decision_rules"]
    if calibration_pass and holdout_all_pass:
        branch = rules["all_calibration_and_holdout_gates_pass"]
    elif sinr_or_joint_failure:
        branch = rules["SINR_or_joint_gate_fails"]
    elif relative_or_temporal_failure:
        branch = rules["relative_RSRP_or_temporal_gate_fails_without_SINR_failure"]
    else:
        branch = rules["calibration_passes_but_any_holdout_fails"]

    previous_sinr = phase3b["metric_support"]["SINR"]
    return {
        "schema_version": 1,
        "stage": "phase_3c15_offline_awgn_support_decision",
        "decision_code": branch["code"],
        "next_action": branch["action"],
        "gain_only_control_envelope_supported": bool(calibration_pass and holdout_all_pass),
        "calibration_trace_gate_pass": calibration_pass,
        "locked_holdout_traces_passed": int(holdout["trace_gate_pass"].sum()),
        "locked_holdout_traces_evaluated": len(holdout),
        "all_locked_holdouts_pass": holdout_all_pass,
        "gain_process_revision_also_required": relative_or_temporal_failure,
        "noise_control_dimension_required": sinr_or_joint_failure,
        "time_varying_noise_sufficiency": "not_established",
        "phase3b_fixed_noise_evidence": {
            "candidate_id": previous_sinr["best_candidate_id"],
            "calibration_supported": previous_sinr["primary_calibration_supported"],
            "locked_holdouts_supported": previous_sinr["locked_validation_regions_supported"],
            "locked_holdouts_evaluated": previous_sinr["locked_validation_regions_evaluated"],
            "interpretation": config["noise_dimension_evidence"]["inference"],
        },
        "control_envelope_interpretation": config["rfsim"]["interpretation"],
        "control_envelope_failure_proves_AWGN_impossibility": False,
        "abc_authorized": False,
        "additional_reservation_should_be_requested_now": False,
        "claim_limits": config["claim_limits"],
    }


def analyze_phase3c15_support(
    *,
    route_observations: str | Path,
    locked_spatial_split: str | Path,
    phase3c14_telemetry: str | Path,
    phase3c14_evaluation: str | Path,
    phase3c14_result: str | Path,
    phase3b_decision: str | Path,
    phase3b_distribution_diagnostics: str | Path,
    phase3b_locked_validation_support: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "route_observations": _require_file(route_observations, "UPV route observations"),
        "locked_spatial_split": _require_file(locked_spatial_split, "locked spatial split"),
        "phase3c14_telemetry": _require_file(phase3c14_telemetry, "Phase 3C14 telemetry"),
        "phase3c14_evaluation": _require_file(phase3c14_evaluation, "Phase 3C14 evaluation"),
        "phase3c14_result": _require_file(phase3c14_result, "Phase 3C14 result"),
        "phase3b_decision": _require_file(phase3b_decision, "Phase 3B decision"),
        "phase3b_distribution_diagnostics": _require_file(
            phase3b_distribution_diagnostics, "Phase 3B distribution diagnostics"
        ),
        "phase3b_locked_validation_support": _require_file(
            phase3b_locked_validation_support, "Phase 3B locked validation support"
        ),
        "config": _require_file(config_path, "Phase 3C15 configuration"),
    }
    config = _read_yaml(paths["config"])
    validate_phase3c15_config(config)
    _verify_inputs(paths, config)

    upv_traces = _aggregate_upv(paths["route_observations"], paths["locked_spatial_split"], config)
    awgn_distribution, awgn_temporal = _load_awgn(paths["phase3c14_telemetry"], config)
    calibration = upv_traces[config["measurement"]["calibration_role"]]
    preprocessing = _fit_preprocessing(calibration, config)
    thresholds = _conditional_thresholds(calibration, preprocessing, config)
    lags = [int(value) for value in config["temporal"]["lags_seconds"]]

    trace_summaries = [
        _summary_row("UPV", role, str(frame["split_role"].iloc[0]), frame, None, lags)
        for role, frame in upv_traces.items()
    ]
    replay_summaries = [
        _summary_row(
            "RFsim",
            replay_id,
            "candidate",
            awgn_distribution[replay_id],
            awgn_temporal[replay_id],
            lags,
        )
        for replay_id in config["rfsim"]["expected_replays"]
    ]
    support_rows: list[dict[str, object]] = []
    replay_distance_rows: list[dict[str, object]] = []
    for role, trace in upv_traces.items():
        support, replay_rows = _evaluate_role(
            role,
            trace,
            awgn_distribution,
            awgn_temporal,
            preprocessing,
            thresholds,
            config,
        )
        support_rows.append(support)
        replay_distance_rows.extend(replay_rows)
    support_frame = pd.DataFrame(support_rows)
    phase3b = _read_json(paths["phase3b_decision"])
    decision = _decision(support_frame, phase3b, config)

    upv_aggregated = pd.concat(upv_traces.values(), ignore_index=True)
    awgn_aggregated = pd.concat(awgn_distribution.values(), ignore_index=True)
    return {
        "config": config,
        "config_path": paths["config"],
        "input_paths": paths,
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "preprocessing": preprocessing,
        "thresholds": thresholds,
        "trace_summaries": pd.DataFrame(trace_summaries),
        "replay_summaries": pd.DataFrame(replay_summaries),
        "support": support_frame,
        "replay_distances": pd.DataFrame(replay_distance_rows),
        "upv_aggregated": upv_aggregated,
        "awgn_aggregated": awgn_aggregated,
        "decision": decision,
    }


def write_phase3c15_support_analysis(
    *,
    route_observations: str | Path,
    locked_spatial_split: str | Path,
    phase3c14_telemetry: str | Path,
    phase3c14_evaluation: str | Path,
    phase3c14_result: str | Path,
    phase3b_decision: str | Path,
    phase3b_distribution_diagnostics: str | Path,
    phase3b_locked_validation_support: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 3C15 output: {output}")
    analysis = analyze_phase3c15_support(
        route_observations=route_observations,
        locked_spatial_split=locked_spatial_split,
        phase3c14_telemetry=phase3c14_telemetry,
        phase3c14_evaluation=phase3c14_evaluation,
        phase3c14_result=phase3c14_result,
        phase3b_decision=phase3b_decision,
        phase3b_distribution_diagnostics=phase3b_distribution_diagnostics,
        phase3b_locked_validation_support=phase3b_locked_validation_support,
        config_path=config_path,
    )
    output.mkdir(parents=True)
    _write_csv(output / "upv_aggregated_traces.csv", analysis["upv_aggregated"])
    _write_csv(output / "awgn_control_analysis_rows.csv", analysis["awgn_aggregated"])
    _write_csv(output / "upv_trace_summaries.csv", analysis["trace_summaries"])
    _write_csv(output / "awgn_replay_summaries.csv", analysis["replay_summaries"])
    _write_csv(output / "support_results.csv", analysis["support"])
    _write_csv(output / "replay_distances.csv", analysis["replay_distances"])
    _write_json(output / "preprocessing_specification.json", analysis["preprocessing"])
    _write_json(output / "conditional_thresholds.json", analysis["thresholds"])
    _write_json(output / "phase3c15_decision.json", analysis["decision"])

    output_files = sorted(path for path in output.iterdir() if path.is_file())
    output_hashes = {path.name: _sha256(path) for path in output_files}
    manifest = {
        "schema_version": 1,
        "stage": "phase_3c15_offline_awgn_support_analysis",
        "git": _git_revision(),
        "input_sha256": analysis["input_sha256"],
        "output_sha256_before_manifest": output_hashes,
        "decision_code": analysis["decision"]["decision_code"],
        "abc_performed": False,
        "reservation_requested": False,
    }
    _write_json(output / "analysis_manifest.json", manifest)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision_code": analysis["decision"]["decision_code"],
        "abc_authorized": False,
        "reservation_requested": False,
    }
