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

from .mmd_abc import median_heuristic_bandwidth, unbiased_rbf_mmd2
from .upv_protocol import (
    _load_radio_csv,
    _normal_member_path,
    build_locked_split,
    build_route_table,
)


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


def validate_upv_support_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("UPV support schema_version must be 1")
    if config.get("stage") != "existing_bank_support_diagnostic_only":
        raise ValueError("UPV support analysis must remain diagnostic")
    phase1 = config.get("phase1_protocol") or {}
    revision = str(phase1.get("revision") or "")
    if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
        raise ValueError("phase1_protocol.revision must be a full Git SHA")
    if phase1.get("tag") != "upv-protocol-v1":
        raise ValueError("phase1_protocol.tag must be upv-protocol-v1")
    if str((config.get("model") or {}).get("family")) != "TDL_B":
        raise ValueError("the existing-bank support analysis requires TDL_B")
    features = sorted(config.get("features") or [], key=lambda item: int(item["order"]))
    if [str(item.get("name")) for item in features] != ["RSRP", "SINR"]:
        raise ValueError("feature order must be RSRP, SINR")
    if any(float(item.get("weight", 0)) <= 0 for item in features):
        raise ValueError("feature weights must be positive")
    if config.get("missing_value_policy") != "complete_case_drop_before_aggregation":
        raise ValueError("missing-value policy must be complete-case deletion")
    aggregation = config.get("temporal_aggregation") or {}
    if float(aggregation.get("duration_seconds", 0)) <= 0:
        raise ValueError("temporal aggregation duration must be positive")
    if aggregation.get("statistic") != "median":
        raise ValueError("the frozen temporal aggregation statistic is median")
    reference = config.get("balanced_reference") or {}
    if reference.get("robust_center") != "median":
        raise ValueError("the balanced reference must use the median center")
    if reference.get("robust_scale") != "normalized_mad":
        raise ValueError("the balanced reference must use normalized MAD")
    kernel = config.get("kernel") or {}
    if kernel.get("name") != "rbf" or kernel.get("estimator") != "unbiased_mmd_squared":
        raise ValueError("the kernel must be RBF with unbiased MMD squared")
    bootstrap = config.get("bootstrap") or {}
    if int(bootstrap.get("repetitions", 0)) < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if int(bootstrap.get("block_length_aggregated_rows", 0)) < 1:
        raise ValueError("bootstrap block length must be positive")
    sensitivity = config.get("sensitivity") or {}
    if sensitivity.get("fit_unit") != "execution_level_aggregated_mean":
        raise ValueError("sensitivity must be fitted at execution level")
    rules = config.get("support_rules") or {}
    quantiles = [float(value) for value in rules.get("marginal_interval_quantiles") or []]
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("marginal support requires two ordered quantiles")
    if not bool((config.get("decision_rules") or {}).get("abc_is_prohibited_for_existing_bank")):
        raise ValueError("ABC must remain prohibited for this two-repetition bank")


def _aggregate(
    frame: pd.DataFrame,
    *,
    time_column: str,
    origin: float,
    duration: float,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected = frame[[time_column, *feature_columns]].copy()
    selected[time_column] = pd.to_numeric(selected[time_column], errors="coerce")
    for column in feature_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=[time_column, *feature_columns])
    selected["aggregate_index"] = np.floor(
        (selected[time_column] - origin) / duration + 1e-12
    ).astype(int)
    result = selected.groupby("aggregate_index", sort=True).agg(
        time_seconds=(time_column, "median"),
        raw_rows=(time_column, "size"),
        **{column: (column, "median") for column in feature_columns},
    )
    return result.reset_index()


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if lag <= 0 or lag >= len(values):
        return math.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def initial_positive_sequence_ess(values: np.ndarray) -> tuple[float, int, float]:
    values = np.asarray(values, dtype=float)
    count = len(values)
    if count < 3 or np.std(values) == 0:
        return float(count), 0, 1.0
    autocorrelations = [_autocorrelation(values, lag) for lag in range(1, count)]
    included: list[float] = []
    last_lag = 0
    for start in range(0, len(autocorrelations) - 1, 2):
        pair = autocorrelations[start : start + 2]
        if not all(math.isfinite(value) for value in pair) or sum(pair) <= 0:
            break
        included.extend(pair)
        last_lag = start + 2
    integrated_time = max(1.0, 1.0 + 2.0 * sum(included))
    ess = min(float(count), max(1.0, count / integrated_time))
    return ess, last_lag, integrated_time


def _calibration_diagnostics(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    duration = float(frame["seconds_of_day"].max() - frame["seconds_of_day"].min())
    intervals = np.diff(frame.sort_values("seconds_of_day")["seconds_of_day"].to_numpy(float))
    rows: list[dict[str, object]] = []
    autocorrelation_rows: list[dict[str, object]] = []
    ess_values: dict[str, float] = {}
    for column in feature_columns:
        values = frame.sort_values("seconds_of_day")[column].to_numpy(float)
        ess, last_lag, integrated_time = initial_positive_sequence_ess(values)
        ess_values[column] = ess
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        median = float(quantiles[2])
        rows.append({
            "metric": column,
            "sample_count": len(values),
            "duration_seconds": duration,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "median": median,
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "maximum": float(np.max(values)),
            "mad": float(np.median(np.abs(values - median))),
            "effective_sample_size_ips": ess,
            "ips_last_included_lag": last_lag,
            "integrated_autocorrelation_time": integrated_time,
        })
        for lag in range(1, min(10, len(values) - 1) + 1):
            autocorrelation_rows.append({
                "metric": column,
                "lag_samples": lag,
                "approximately_lag_seconds": lag * float(np.median(intervals)),
                "autocorrelation": _autocorrelation(values, lag),
            })
    summary = {
        "sample_count": len(frame),
        "duration_seconds": duration,
        "start_seconds_of_day": float(frame["seconds_of_day"].min()),
        "end_seconds_of_day": float(frame["seconds_of_day"].max()),
        "median_sample_interval_seconds": float(np.median(intervals)),
        "minimum_effective_sample_size": min(ess_values.values()),
        "route_start_m": float(frame["route_distance_m"].min()),
        "route_end_m": float(frame["route_distance_m"].max()),
        "centroid_longitude_deg": float(frame["longitude_deg"].mean()),
        "centroid_latitude_deg": float(frame["latitude_deg"].mean()),
    }
    return pd.DataFrame(rows), pd.DataFrame(autocorrelation_rows), summary


def _distance_matrix_m(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    latitude_origin = math.radians(float(pd.concat([
        left["latitude_deg"], right["latitude_deg"]
    ]).mean()))
    scale_x = 111320.0 * math.cos(latitude_origin)
    left_x = left["longitude_deg"].to_numpy(float) * scale_x
    right_x = right["longitude_deg"].to_numpy(float) * scale_x
    left_y = left["latitude_deg"].to_numpy(float) * 110540.0
    right_y = right["latitude_deg"].to_numpy(float) * 110540.0
    return np.hypot(left_x[:, None] - right_x[None, :], left_y[:, None] - right_y[None, :])


def validation_separation(
    route: pd.DataFrame,
    split: pd.DataFrame,
    *,
    primary_bin_size_m: int,
    calibration_role: str,
    validation_prefix: str,
    adjacency_threshold_m: float,
) -> pd.DataFrame:
    primary = split.loc[split["bin_size_m"].eq(primary_bin_size_m)].copy()
    calibration_row = primary.loc[primary["locked_role"].eq(calibration_role)]
    if len(calibration_row) != 1:
        raise ValueError("locked split must contain exactly one calibration bin")
    calibration_bin = int(calibration_row.iloc[0]["route_bin_id"])
    column = f"route_bin_{primary_bin_size_m}m"
    calibration = route.loc[route[column].eq(calibration_bin)]
    rows = []
    for item in primary.loc[primary["locked_role"].str.startswith(validation_prefix)].itertuples():
        validation = route.loc[route[column].eq(int(item.route_bin_id))]
        distances = _distance_matrix_m(calibration, validation)
        route_gap = max(
            0.0,
            float(calibration["route_distance_m"].min() - validation["route_distance_m"].max()),
            float(validation["route_distance_m"].min() - calibration["route_distance_m"].max()),
        )
        centroid_distance = float(_distance_matrix_m(
            calibration[["longitude_deg", "latitude_deg"]].mean().to_frame().T,
            validation[["longitude_deg", "latitude_deg"]].mean().to_frame().T,
        )[0, 0])
        rows.append({
            "validation_role": item.locked_role,
            "validation_bin": int(item.route_bin_id),
            "calibration_bin": calibration_bin,
            "minimum_along_route_gap_m": route_gap,
            "route_center_separation_m": abs(
                float(item.route_center_m) - float(calibration_row.iloc[0]["route_center_m"])
            ),
            "minimum_euclidean_separation_m": float(np.min(distances)),
            "centroid_euclidean_separation_m": centroid_distance,
            "geographically_adjacent_under_threshold": bool(
                float(np.min(distances)) < adjacency_threshold_m
            ),
        })
    return pd.DataFrame(rows).sort_values("validation_role").reset_index(drop=True)


def _selection_rows(
    selection_manifest: dict[str, Any], campaign_state: dict[str, Any]
) -> pd.DataFrame:
    completed = campaign_state.get("completed") or {}
    rows = []
    for point in selection_manifest.get("points") or []:
        point_id = str(point["point_id"])
        execution = completed.get(point_id)
        if not isinstance(execution, dict):
            raise ValueError(f"campaign state is missing {point_id}")
        planned = point.get("controls") or {}
        observed = execution.get("controls") or {}
        for parameter in ("ploss", "noise_power_dB"):
            if not math.isclose(
                float(planned[parameter]), float(observed[parameter]), rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(f"campaign controls disagree for {point_id}: {parameter}")
        rows.append({
            "point_id": point_id,
            "execution_id": str(execution["execution_id"]),
            "repetition": int(point["repetition"]),
            "ploss": float(planned["ploss"]),
            "noise_power_dB": float(planned["noise_power_dB"]),
        })
    result = pd.DataFrame(rows)
    if result.empty or result["execution_id"].duplicated().any():
        raise ValueError("selection manifest must resolve to unique executions")
    return result.sort_values(["ploss", "noise_power_dB", "repetition"]).reset_index(drop=True)


def _load_rfsim(
    selected: pd.DataFrame,
    executions_root: Path,
    config: dict[str, Any],
    features: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    selection = config["rfsim_selection"]
    filename = str(selection["observation_filename"])
    feature_columns = [str(item["rfsim_column"]) for item in features]
    required = {
        str(selection["execution_id_column"]),
        str(selection["time_column"]),
        str(selection["model_column"]),
        str(selection["ploss_column"]),
        str(selection["noise_column"]),
        *feature_columns,
        *(str(value) for value in selection.get("require_true") or []),
        *(str(value) for value in selection.get("require_false") or []),
    }
    start = float(selection["analysis_window_start_seconds"])
    end = float(selection["analysis_window_end_seconds"])
    duration = float(config["temporal_aggregation"]["duration_seconds"])
    minimum = int(selection["minimum_complete_aggregated_rows"])
    frames: dict[str, pd.DataFrame] = {}
    inventory = []
    for item in selected.itertuples(index=False):
        path = executions_root / item.execution_id / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe RFsim observation file: {path}")
        raw = pd.read_parquet(path)
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"{item.execution_id} is missing fields: " + ", ".join(missing))
        frame = raw.copy()
        numeric_columns = {
            str(selection["time_column"]),
            str(selection["ploss_column"]),
            str(selection["noise_column"]),
            *feature_columns,
        }
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in selection.get("require_true") or []:
            frame = frame.loc[frame[column].fillna(False).astype(bool)]
        for column in selection.get("require_false") or []:
            frame = frame.loc[~frame[column].fillna(True).astype(bool)]
        frame = frame.loc[frame[str(selection["model_column"])].astype("string").eq(
            str(config["model"]["family"])
        )]
        time_column = str(selection["time_column"])
        frame[time_column] = pd.to_numeric(frame[time_column], errors="coerce")
        frame = frame.loc[frame[time_column].ge(start) & frame[time_column].lt(end)]
        for column in [
            str(selection["ploss_column"]),
            str(selection["noise_column"]),
            *feature_columns,
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[time_column, *feature_columns])
        if not np.isclose(
            frame[str(selection["ploss_column"])].to_numpy(float), item.ploss, atol=1e-9
        ).all():
            raise ValueError(f"ploss mismatch in {item.execution_id}")
        if not np.isclose(
            frame[str(selection["noise_column"])].to_numpy(float),
            item.noise_power_dB,
            atol=1e-9,
        ).all():
            raise ValueError(f"noise mismatch in {item.execution_id}")
        aggregated = _aggregate(
            frame,
            time_column=time_column,
            origin=start,
            duration=duration,
            feature_columns=feature_columns,
        )
        if len(aggregated) < minimum:
            raise ValueError(f"{item.execution_id} has too few complete aggregated rows")
        aggregated["execution_id"] = item.execution_id
        aggregated["point_id"] = item.point_id
        aggregated["repetition"] = item.repetition
        aggregated["ploss"] = item.ploss
        aggregated["noise_power_dB"] = item.noise_power_dB
        frames[item.execution_id] = aggregated
        inventory.append({
            "input_kind": "rfsim_execution",
            "source_id": item.execution_id,
            "sha256": _sha256(path),
            "rows_read": len(raw),
            "rows_after_frozen_filter": len(frame),
            "aggregated_rows": len(aggregated),
            "ploss": item.ploss,
            "noise_power_dB": item.noise_power_dB,
            "repetition": item.repetition,
        })
    repetitions = selected.groupby(["ploss", "noise_power_dB"])["execution_id"].nunique()
    required_repetitions = int(selection["required_repetitions_per_state"])
    if repetitions.lt(required_repetitions).any():
        raise ValueError("every retained state requires the frozen number of executions")
    return frames, pd.DataFrame(inventory)


def _alternative_route(
    archive: Path,
    phase1_config: dict[str, Any],
    support_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = support_config["scenario_interpretation_sensitivity"]
    source_path = str(sensitivity["source_path"])
    payload = None
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if _normal_member_path(member.filename) == source_path:
                payload = source.read(member)
                break
    if payload is None:
        raise ValueError(f"UPV archive does not contain {source_path}")
    radio, _ = _load_radio_csv(payload)
    route_config = phase1_config["route"]
    route = build_route_table(
        radio,
        source_path=source_path,
        corrected_test_id=int(sensitivity["filename_test_id"]),
        bin_sizes_m=[int(value) for value in route_config["bin_sizes_m"]],
        minimum_step_m_for_heading=float(route_config["minimum_step_m_for_heading"]),
        direction_sectors=int(route_config["direction_sectors"]),
    )
    split = build_locked_split(route, phase1_config)
    return route, split


def _evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 2 or len(frame) < count:
        raise ValueError("balanced selection requires at least two rows from every source")
    ordered = frame.sort_values("time_seconds").reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    if len(np.unique(indices)) != count:
        raise AssertionError("evenly-spaced selection produced duplicate indices")
    return ordered.iloc[indices].reset_index(drop=True)


def _robust_transform(
    balanced: pd.DataFrame,
    feature_columns: list[str],
    features: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = balanced[feature_columns].to_numpy(float)
    center = np.median(matrix, axis=0)
    factor = float(config["balanced_reference"]["normalized_mad_factor"])
    scale = np.median(np.abs(matrix - center), axis=0) * factor
    for index, value in enumerate(scale):
        if value <= np.finfo(float).eps:
            q25, q75 = np.quantile(matrix[:, index], [0.25, 0.75])
            scale[index] = (q75 - q25) / 1.349
        if scale[index] <= np.finfo(float).eps:
            raise ValueError(
                f"balanced reference has zero robust scale for {feature_columns[index]}"
            )
    weights = np.sqrt(np.array([float(item["weight"]) for item in features]))
    transformed = (matrix - center) / scale * weights
    return transformed, center, scale


def _transform(
    frame: pd.DataFrame,
    feature_columns: list[str],
    features: list[dict[str, Any]],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    weights = np.sqrt(np.array([float(item["weight"]) for item in features]))
    return (frame[feature_columns].to_numpy(float) - center) / scale * weights


def _circular_block_sample(
    values: np.ndarray,
    *,
    target_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(values) < 2:
        raise ValueError("block bootstrap requires at least two observations")
    blocks = math.ceil(target_count / block_length)
    starts = rng.integers(0, len(values), size=blocks)
    indices = np.concatenate([
        (start + np.arange(block_length)) % len(values) for start in starts
    ])[:target_count]
    return values[indices]


def _interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _candidate_label(ploss: float, noise: float) -> str:
    return f"ploss={ploss:g}|noise={noise:g}"


def _support_tables(
    upv: pd.DataFrame,
    simulation: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    *,
    balanced_count: int,
    feature_columns: list[str],
    features: list[dict[str, Any]],
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    upv_balanced = _evenly_spaced(upv, balanced_count)
    upv_values = _transform(upv_balanced, feature_columns, features, center, scale)
    upv_full = _transform(upv, feature_columns, features, center, scale)
    quantile_low, quantile_high = [
        float(value) for value in config["support_rules"]["marginal_interval_quantiles"]
    ]
    execution_rows = []
    balanced_simulation: dict[str, pd.DataFrame] = {}
    for item in selected.itertuples(index=False):
        balanced = _evenly_spaced(simulation[item.execution_id], balanced_count)
        balanced_simulation[item.execution_id] = balanced
        values = _transform(balanced, feature_columns, features, center, scale)
        row: dict[str, object] = {
            "candidate_id": _candidate_label(item.ploss, item.noise_power_dB),
            "execution_id": item.execution_id,
            "point_id": item.point_id,
            "repetition": item.repetition,
            "ploss": item.ploss,
            "noise_power_dB": item.noise_power_dB,
            "balanced_rows": balanced_count,
            "mmd_squared": unbiased_rbf_mmd2(
                upv_values, values, bandwidth=bandwidth, maximum_samples=balanced_count
            ),
        }
        for feature, column in zip(features, feature_columns, strict=True):
            name = str(feature["name"])
            real_interval = tuple(np.quantile(upv_balanced[column], [quantile_low, quantile_high]))
            simulated_interval = tuple(np.quantile(balanced[column], [quantile_low, quantile_high]))
            row[f"{name}_upv_q_low"] = float(real_interval[0])
            row[f"{name}_upv_q_high"] = float(real_interval[1])
            row[f"{name}_rfsim_q_low"] = float(simulated_interval[0])
            row[f"{name}_rfsim_q_high"] = float(simulated_interval[1])
            row[f"{name}_marginal_supported"] = _interval_overlap(
                real_interval, simulated_interval
            )
            row[f"{name}_median_gap"] = float(
                np.median(balanced[column]) - np.median(upv_balanced[column])
            )
        execution_rows.append(row)
    execution_support = pd.DataFrame(execution_rows)

    repeatability_rows = []
    for (ploss, noise), group in selected.groupby(["ploss", "noise_power_dB"], sort=True):
        for left_id, right_id in itertools.combinations(group["execution_id"].tolist(), 2):
            left = _transform(
                balanced_simulation[left_id], feature_columns, features, center, scale
            )
            right = _transform(
                balanced_simulation[right_id], feature_columns, features, center, scale
            )
            repeatability_rows.append({
                "candidate_id": _candidate_label(ploss, noise),
                "ploss": ploss,
                "noise_power_dB": noise,
                "left_execution_id": left_id,
                "right_execution_id": right_id,
                "mmd_squared": unbiased_rbf_mmd2(
                    left, right, bandwidth=bandwidth, maximum_samples=balanced_count
                ),
            })
    repeatability = pd.DataFrame(repeatability_rows)
    threshold_quantile = float(config["support_rules"]["repeatability_threshold_quantile"])
    repeatability_threshold = float(np.quantile(repeatability["mmd_squared"], threshold_quantile))

    bootstrap = config["bootstrap"]
    repetitions = int(bootstrap["repetitions"])
    block_length = int(bootstrap["block_length_aggregated_rows"])
    confidence = float(bootstrap["confidence_level"])
    lower_probability = (1.0 - confidence) / 2.0
    upper_probability = 1.0 - lower_probability
    rng = np.random.default_rng(int(bootstrap["seed"]))
    candidate_rows = []
    for (ploss, noise), group in selected.groupby(["ploss", "noise_power_dB"], sort=True):
        candidate_id = _candidate_label(ploss, noise)
        execution_ids = group["execution_id"].tolist()
        bootstrap_values = np.empty(repetitions)
        simulation_full = {
            execution_id: _transform(
                simulation[execution_id], feature_columns, features, center, scale
            )
            for execution_id in execution_ids
        }
        for index in range(repetitions):
            real_sample = _circular_block_sample(
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
                distances.append(unbiased_rbf_mmd2(
                    real_sample,
                    simulated_sample,
                    bandwidth=bandwidth,
                    maximum_samples=balanced_count,
                ))
            bootstrap_values[index] = float(np.mean(distances))
        execution_group = execution_support.loc[
            execution_support["candidate_id"].eq(candidate_id)
        ]
        pooled = pd.concat(
            [balanced_simulation[execution_id] for execution_id in execution_ids],
            ignore_index=True,
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "ploss": ploss,
            "noise_power_dB": noise,
            "execution_count": len(execution_ids),
            "balanced_rows_per_execution": balanced_count,
            "mean_execution_mmd_squared": float(execution_group["mmd_squared"].mean()),
            "minimum_execution_mmd_squared": float(execution_group["mmd_squared"].min()),
            "maximum_execution_mmd_squared": float(execution_group["mmd_squared"].max()),
            "conditional_bootstrap_ci_low": float(np.quantile(bootstrap_values, lower_probability)),
            "conditional_bootstrap_ci_high": float(
                np.quantile(bootstrap_values, upper_probability)
            ),
            "repeatability_threshold_mmd_squared": repeatability_threshold,
            "joint_supported": bool(
                float(execution_group["mmd_squared"].mean()) <= repeatability_threshold
            ),
        }
        for feature, column in zip(features, feature_columns, strict=True):
            name = str(feature["name"])
            real_interval = tuple(np.quantile(upv_balanced[column], [quantile_low, quantile_high]))
            simulated_interval = tuple(np.quantile(pooled[column], [quantile_low, quantile_high]))
            row[f"{name}_upv_q_low"] = float(real_interval[0])
            row[f"{name}_upv_q_high"] = float(real_interval[1])
            row[f"{name}_rfsim_q_low"] = float(simulated_interval[0])
            row[f"{name}_rfsim_q_high"] = float(simulated_interval[1])
            row[f"{name}_marginal_supported"] = _interval_overlap(
                real_interval, simulated_interval
            )
            row[f"{name}_median_gap"] = float(
                np.median(pooled[column]) - np.median(upv_balanced[column])
            )
        candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["mean_execution_mmd_squared", "ploss", "noise_power_dB"]
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return execution_support, candidates, repeatability, repeatability_threshold


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    matrix = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(matrix, y, rcond=None)[0]
    fitted = intercept + slope * x
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else math.nan
    return float(intercept), float(slope), r_squared


def _bootstrap_slope(
    frame: pd.DataFrame,
    parameter: str,
    response: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [group for _, group in frame.groupby(parameter, sort=True)]
    values = np.empty(repetitions)
    for index in range(repetitions):
        sampled = pd.concat([
            group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups
        ], ignore_index=True)
        values[index] = _fit_line(
            sampled[parameter].to_numpy(float), sampled[response].to_numpy(float)
        )[1]
    return values


def _sensitivity_tables(
    simulation: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    features: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    feature_columns = [str(item["rfsim_column"]) for item in features]
    rows = []
    for item in selected.itertuples(index=False):
        row = {
            "execution_id": item.execution_id,
            "ploss": item.ploss,
            "noise_power_dB": item.noise_power_dB,
        }
        for feature, column in zip(features, feature_columns, strict=True):
            row[str(feature["name"])] = float(simulation[item.execution_id][column].mean())
        rows.append(row)
    execution_means = pd.DataFrame(rows)
    sensitivity = config["sensitivity"]
    sweep_specs = [
        (
            "ploss",
            execution_means.loc[execution_means["noise_power_dB"].eq(
                float(sensitivity["ploss_sweep_fixed_noise_power_dB"])
            )],
            "noise_power_dB",
            float(sensitivity["ploss_sweep_fixed_noise_power_dB"]),
        ),
        (
            "noise_power_dB",
            execution_means.loc[execution_means["ploss"].eq(
                float(sensitivity["noise_sweep_fixed_ploss"])
            )],
            "ploss",
            float(sensitivity["noise_sweep_fixed_ploss"]),
        ),
    ]
    minimum_levels = int(sensitivity["minimum_unique_levels"])
    repetitions = int(sensitivity["bootstrap_repetitions"])
    rng = np.random.default_rng(int(sensitivity["seed"]))
    global_rows = []
    local_rows = []
    slope_bootstraps: dict[tuple[str, str], np.ndarray] = {}
    for parameter, sweep, fixed_parameter, fixed_value in sweep_specs:
        if sweep[parameter].nunique() < minimum_levels:
            raise ValueError(f"{parameter} sweep has too few supported levels")
        for feature in features:
            response = str(feature["name"])
            x = sweep[parameter].to_numpy(float)
            y = sweep[response].to_numpy(float)
            intercept, slope, r_squared = _fit_line(x, y)
            bootstrap = _bootstrap_slope(
                sweep, parameter, response, repetitions=repetitions, rng=rng
            )
            slope_bootstraps[(response, parameter)] = bootstrap
            global_rows.append({
                "parameter": parameter,
                "response": response,
                "fixed_parameter": fixed_parameter,
                "fixed_value": fixed_value,
                "supported_minimum": float(sweep[parameter].min()),
                "supported_maximum": float(sweep[parameter].max()),
                "unique_control_levels": int(sweep[parameter].nunique()),
                "execution_count": len(sweep),
                "intercept": intercept,
                "slope_per_db": slope,
                "slope_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "slope_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "r_squared": r_squared,
                "fit_unit": sensitivity["fit_unit"],
                "extrapolation_permitted": False,
            })
            point_means = sweep.groupby(parameter, sort=True)[response].mean().reset_index()
            pairs = zip(
                point_means.iloc[:-1].itertuples(),
                point_means.iloc[1:].itertuples(),
                strict=True,
            )
            for left, right in pairs:
                left_x = float(getattr(left, parameter))
                right_x = float(getattr(right, parameter))
                left_y = float(getattr(left, response))
                right_y = float(getattr(right, response))
                local_rows.append({
                    "parameter": parameter,
                    "response": response,
                    "interval_start": left_x,
                    "interval_end": right_x,
                    "interval_midpoint": (left_x + right_x) / 2.0,
                    "finite_difference_slope_per_db": (right_y - left_y) / (right_x - left_x),
                    "fixed_parameter": fixed_parameter,
                    "fixed_value": fixed_value,
                    "execution_means_per_level": int(
                        sweep.groupby(parameter)["execution_id"].nunique().min()
                    ),
                })
    global_frame = pd.DataFrame(global_rows)
    local_frame = pd.DataFrame(local_rows)
    matrix = np.array([
        [
            global_frame.loc[
                global_frame["response"].eq("RSRP")
                & global_frame["parameter"].eq("ploss"),
                "slope_per_db",
            ].iloc[0],
            global_frame.loc[
                global_frame["response"].eq("RSRP")
                & global_frame["parameter"].eq("noise_power_dB"),
                "slope_per_db",
            ].iloc[0],
        ],
        [
            global_frame.loc[
                global_frame["response"].eq("SINR")
                & global_frame["parameter"].eq("ploss"),
                "slope_per_db",
            ].iloc[0],
            global_frame.loc[
                global_frame["response"].eq("SINR")
                & global_frame["parameter"].eq("noise_power_dB"),
                "slope_per_db",
            ].iloc[0],
        ],
    ], dtype=float)
    condition = float(np.linalg.cond(matrix))
    condition_bootstrap = np.array([
        np.linalg.cond(np.array([
            [
                slope_bootstraps[("RSRP", "ploss")][index],
                slope_bootstraps[("RSRP", "noise_power_dB")][index],
            ],
            [
                slope_bootstraps[("SINR", "ploss")][index],
                slope_bootstraps[("SINR", "noise_power_dB")][index],
            ],
        ]))
        for index in range(repetitions)
    ])
    matrix_frame = pd.DataFrame([
        {
            "response": response,
            "d_response_d_ploss": matrix[row_index, 0],
            "d_response_d_noise_power_dB": matrix[row_index, 1],
        }
        for row_index, response in enumerate(["RSRP", "SINR"])
    ])
    summary = {
        "matrix": matrix.tolist(),
        "row_order": ["RSRP", "SINR"],
        "column_order": ["ploss", "noise_power_dB"],
        "condition_number": condition,
        "condition_number_bootstrap_ci_low": float(np.quantile(condition_bootstrap, 0.025)),
        "condition_number_bootstrap_ci_high": float(np.quantile(condition_bootstrap, 0.975)),
        "interpretation": "composite of two marginal axis sweeps, not a local joint Jacobian",
        "full_interaction_surface_identified": False,
    }
    return global_frame, local_frame, matrix_frame, summary


def _scenario_sensitivity(
    primary: pd.DataFrame,
    alternative: pd.DataFrame,
    simulation: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    *,
    balanced_count: int,
    feature_columns: list[str],
    features: list[dict[str, Any]],
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
) -> pd.DataFrame:
    if len(alternative) < balanced_count:
        raise ValueError("original-filename sensitivity region is too short for comparison")
    rows = []
    for interpretation, real in [
        ("timestamp_gps_corrected", primary),
        ("original_filenames", alternative),
    ]:
        real_balanced = _evenly_spaced(real, balanced_count)
        real_values = _transform(real_balanced, feature_columns, features, center, scale)
        for (ploss, noise), group in selected.groupby(["ploss", "noise_power_dB"], sort=True):
            distances = []
            for execution_id in group["execution_id"]:
                simulated = _evenly_spaced(simulation[execution_id], balanced_count)
                simulated_values = _transform(
                    simulated, feature_columns, features, center, scale
                )
                distances.append(unbiased_rbf_mmd2(
                    real_values,
                    simulated_values,
                    bandwidth=bandwidth,
                    maximum_samples=balanced_count,
                ))
            rows.append({
                "interpretation": interpretation,
                "candidate_id": _candidate_label(ploss, noise),
                "ploss": ploss,
                "noise_power_dB": noise,
                "mean_execution_mmd_squared": float(np.mean(distances)),
                "upv_rsrp_median": float(np.median(real_balanced[feature_columns[0]])),
                "upv_sinr_median": float(np.median(real_balanced[feature_columns[1]])),
                "balanced_rows": balanced_count,
            })
    result = pd.DataFrame(rows)
    result["rank_within_interpretation"] = result.groupby("interpretation")[
        "mean_execution_mmd_squared"
    ].rank(method="first")
    return result.sort_values(
        ["interpretation", "rank_within_interpretation"]
    ).reset_index(drop=True)


def _decision(
    candidates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scenario_sensitivity: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    best = candidates.sort_values("rank").iloc[0]
    ploss_min, ploss_max = candidates["ploss"].min(), candidates["ploss"].max()
    noise_min, noise_max = candidates["noise_power_dB"].min(), candidates["noise_power_dB"].max()
    ploss_boundary = bool(best["ploss"] in {ploss_min, ploss_max})
    noise_boundary = bool(best["noise_power_dB"] in {noise_min, noise_max})
    rsrp_support = bool(candidates["RSRP_marginal_supported"].any())
    sinr_support = bool(candidates["SINR_marginal_supported"].any())
    rsrp_gaps = candidates["RSRP_median_gap"].to_numpy(float)
    systematic_rsrp_offset = bool(
        not rsrp_support
        and sinr_support
        and (np.all(rsrp_gaps > 0) or np.all(rsrp_gaps < 0))
    )
    best_joint = bool(
        best["joint_supported"]
        and best["RSRP_marginal_supported"]
        and best["SINR_marginal_supported"]
    )
    crossed = bool(
        config["decision_rules"]["current_design_is_crossed_two_dimensional"]
    )
    if systematic_rsrp_offset:
        code = "systematic_rsrp_offset_but_sinr_support"
        number = 4
        action = (
            "Investigate UPV/RFsim reference-power and measurement equivalence before "
            "extending ploss; retain the SINR-supported state as a conditional anchor."
        )
    elif ploss_boundary or noise_boundary:
        code = "closest_candidate_at_boundary"
        number = 3
        action = "Design a small boundary-extension experiment with independent repetitions."
    elif best_joint and not crossed:
        code = "sequential_support_only"
        number = 2
        action = "Calibrate ploss first, then noise conditionally; do not claim a joint posterior."
    elif best_joint and crossed:
        code = "joint_support_and_local_identifiability"
        number = 1
        action = "Add independent repetitions before a pilot ABC design."
    else:
        code = "gross_joint_mismatch"
        number = 5
        action = "Revise the simulator discrepancy model rather than forcing calibration."
    interpretation_best = scenario_sensitivity.loc[
        scenario_sensitivity["rank_within_interpretation"].eq(1)
    ][["interpretation", "candidate_id"]]
    interpretation_map = dict(zip(
        interpretation_best["interpretation"],
        interpretation_best["candidate_id"],
        strict=True,
    ))
    decision = {
        "schema_version": 1,
        "phase1_revision": config["phase1_protocol"]["revision"],
        "phase1_tag": config["phase1_protocol"]["tag"],
        "decision_number": number,
        "decision_code": code,
        "action": action,
        "best_candidate": {
            "candidate_id": best["candidate_id"],
            "ploss": float(best["ploss"]),
            "noise_power_dB": float(best["noise_power_dB"]),
            "mean_execution_mmd_squared": float(best["mean_execution_mmd_squared"]),
            "joint_supported": bool(best["joint_supported"]),
            "rsrp_marginal_supported": bool(best["RSRP_marginal_supported"]),
            "sinr_marginal_supported": bool(best["SINR_marginal_supported"]),
            "ploss_boundary": ploss_boundary,
            "noise_boundary": noise_boundary,
        },
        "bank_support": {
            "any_rsrp_marginal_support": rsrp_support,
            "any_sinr_marginal_support": sinr_support,
            "systematic_rsrp_offset": systematic_rsrp_offset,
            "best_candidate_joint_support": best_joint,
        },
        "identifiability": {
            "crossed_two_dimensional_design": crossed,
            "full_interaction_surface_identified": False,
            "composite_axis_condition_number": float(
                sensitivity.attrs["condition_number"]
            ),
        },
        "scenario_interpretation_sensitivity": {
            "primary_best_candidate": interpretation_map.get("timestamp_gps_corrected"),
            "original_filename_best_candidate": interpretation_map.get("original_filenames"),
            "ranking_is_robust": interpretation_map.get("timestamp_gps_corrected")
            == interpretation_map.get("original_filenames"),
        },
        "claim_limits": [
            "diagnostic existing-bank support analysis, not ABC",
            "bootstrap intervals are traversal-conditional, not between-run uncertainty",
            "two executions per state are insufficient for final calibration",
            "the marginal-axis sensitivity matrix is not a full joint Jacobian",
        ],
    }
    preferred = int(config["support_rules"]["preferred_independent_repetitions_for_abc"])
    minimum = int(config["support_rules"]["minimum_independent_repetitions_for_abc"])
    reservation = {
        "schema_version": 1,
        "phase1_revision": config["phase1_protocol"]["revision"],
        "decision_code": code,
        "minimum_independent_executions_per_state": minimum,
        "preferred_independent_executions_per_state": preferred,
        "best_existing_state": {
            "ploss": float(best["ploss"]),
            "noise_power_dB": float(best["noise_power_dB"]),
        },
        "design_action": action,
        "required_controls": [
            "repeat every selected state as an independent execution",
            "retain the same TDL-B family, topology, duration, warm-up, and analysis window",
            "include a local crossed design before any joint-identifiability claim",
            "hold out at least one spatial UPV validation region from all tuning",
        ],
        "automatic_execution_authorized": False,
    }
    return decision, reservation


def analyze_upv_support(
    *,
    route_observations: str | Path,
    locked_split: str | Path,
    upv_archive: str | Path,
    phase1_config: str | Path,
    selection_manifest: str | Path,
    campaign_state: str | Path,
    executions_root: str | Path,
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
        "config": Path(config_path).resolve(),
        "output": Path(output_dir).resolve(),
    }
    for name in [
        "route_observations",
        "locked_split",
        "upv_archive",
        "phase1_config",
        "selection_manifest",
        "campaign_state",
        "config",
    ]:
        if not paths[name].is_file() or paths[name].is_symlink():
            raise ValueError(f"missing or unsafe input: {paths[name]}")
    if not paths["executions_root"].is_dir() or paths["executions_root"].is_symlink():
        raise ValueError("executions root must be a real directory")
    if paths["output"].exists():
        raise FileExistsError(f"UPV support output already exists: {paths['output']}")

    config = _read_yaml(paths["config"])
    validate_upv_support_config(config)
    phase1 = _read_yaml(paths["phase1_config"])
    software = _git_revision()
    features = sorted(config["features"], key=lambda item: int(item["order"]))
    upv_columns = [str(item["upv_column"]) for item in features]
    rfsim_columns = [str(item["rfsim_column"]) for item in features]
    primary_route = pd.read_csv(paths["route_observations"])
    split = pd.read_csv(paths["locked_split"])
    upv_selection = config["upv_selection"]
    bin_size = int(upv_selection["primary_bin_size_m"])
    bin_column = f"route_bin_{bin_size}m"
    calibration_row = split.loc[
        split["bin_size_m"].eq(bin_size)
        & split["locked_role"].eq(str(upv_selection["calibration_role"]))
    ]
    if len(calibration_row) != 1:
        raise ValueError("Phase 1 split does not contain one configured calibration bin")
    calibration_bin = int(calibration_row.iloc[0]["route_bin_id"])
    required_upv = [str(value) for value in upv_selection["required_complete_columns"]]
    primary_raw = primary_route.loc[
        primary_route[bin_column].eq(calibration_bin)
        & primary_route["source_path"].eq(str(upv_selection["source_path"]))
    ].dropna(subset=required_upv).sort_values("seconds_of_day").reset_index(drop=True)
    if primary_raw.empty:
        raise ValueError("the locked UPV calibration bin is empty")
    duration = float(config["temporal_aggregation"]["duration_seconds"])
    primary_aggregated = _aggregate(
        primary_raw,
        time_column="seconds_of_day",
        origin=float(primary_raw["seconds_of_day"].min()),
        duration=duration,
        feature_columns=upv_columns,
    )
    primary_aggregated = primary_aggregated.rename(columns=dict(zip(
        upv_columns, rfsim_columns, strict=True
    )))

    diagnostics, autocorrelation, diagnostic_summary = _calibration_diagnostics(
        primary_raw, upv_columns
    )
    separation = validation_separation(
        primary_route,
        split,
        primary_bin_size_m=bin_size,
        calibration_role=str(upv_selection["calibration_role"]),
        validation_prefix=str(upv_selection["validation_role_prefix"]),
        adjacency_threshold_m=float(
            config["support_rules"]["euclidean_adjacency_threshold_m"]
        ),
    )

    selected = _selection_rows(
        _read_json(paths["selection_manifest"]), _read_json(paths["campaign_state"])
    )
    simulation, execution_inventory = _load_rfsim(
        selected, paths["executions_root"], config, features
    )
    alternative_route, alternative_split = _alternative_route(
        paths["upv_archive"], phase1, config
    )
    alternative_calibration = alternative_split.loc[
        alternative_split["bin_size_m"].eq(bin_size)
        & alternative_split["locked_role"].eq(str(upv_selection["calibration_role"]))
    ]
    alternative_bin = int(alternative_calibration.iloc[0]["route_bin_id"])
    alternative_raw = alternative_route.loc[
        alternative_route[bin_column].eq(alternative_bin)
    ].dropna(subset=required_upv).sort_values("seconds_of_day").reset_index(drop=True)
    alternative_aggregated = _aggregate(
        alternative_raw,
        time_column="seconds_of_day",
        origin=float(alternative_raw["seconds_of_day"].min()),
        duration=duration,
        feature_columns=upv_columns,
    ).rename(columns=dict(zip(upv_columns, rfsim_columns, strict=True)))

    balanced_count = min(
        len(primary_aggregated), *(len(frame) for frame in simulation.values())
    )
    minimum = int(config["rfsim_selection"]["minimum_complete_aggregated_rows"])
    if balanced_count < minimum:
        raise ValueError("balanced reference contains too few rows per source")
    balanced_frames = []
    primary_balanced = _evenly_spaced(primary_aggregated, balanced_count)
    primary_balanced["reference_source"] = "upv_calibration"
    balanced_frames.append(primary_balanced)
    for execution_id in selected["execution_id"]:
        balanced = _evenly_spaced(simulation[execution_id], balanced_count)
        balanced["reference_source"] = execution_id
        balanced_frames.append(balanced)
    balanced_reference = pd.concat(balanced_frames, ignore_index=True)
    transformed_reference, center, scale = _robust_transform(
        balanced_reference, rfsim_columns, features, config
    )
    bandwidth = median_heuristic_bandwidth(
        transformed_reference, maximum_samples=len(transformed_reference)
    )

    staging = paths["output"].parent / f".{paths['output'].name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        preprocessing = {
            "schema_version": 1,
            "phase1_revision": config["phase1_protocol"]["revision"],
            "phase1_tag": config["phase1_protocol"]["tag"],
            "analysis_implementation_revision": software["revision"],
            "config_sha256": _sha256(paths["config"]),
            "frozen_before_candidate_comparison": True,
            "included_upv_rows": {
                "source_path": upv_selection["source_path"],
                "calibration_bin_size_m": bin_size,
                "calibration_bin": calibration_bin,
                "raw_complete_rows": len(primary_raw),
                "aggregated_rows": len(primary_aggregated),
            },
            "included_rfsim_executions": selected.to_dict("records"),
            "feature_order": [str(item["name"]) for item in features],
            "feature_columns": {
                str(item["name"]): {
                    "upv": item["upv_column"],
                    "rfsim": item["rfsim_column"],
                }
                for item in features
            },
            "missing_value_policy": config["missing_value_policy"],
            "temporal_aggregation": config["temporal_aggregation"],
            "rfsim_selection": config["rfsim_selection"],
            "balanced_reference": {
                **config["balanced_reference"],
                "realized_rows_per_source": balanced_count,
                "realized_source_count": len(balanced_frames),
            },
            "robust_transform": {
                "center": dict(zip([item["name"] for item in features], center, strict=True)),
                "scale": dict(zip([item["name"] for item in features], scale, strict=True)),
                "feature_weights": {
                    item["name"]: float(item["weight"]) for item in features
                },
            },
            "kernel": {**config["kernel"], "realized_bandwidth": bandwidth},
            "bootstrap": config["bootstrap"],
            "sensitivity": config["sensitivity"],
            "support_rules": config["support_rules"],
            "decision_rules": config["decision_rules"],
            "scenario_interpretation_sensitivity": config[
                "scenario_interpretation_sensitivity"
            ],
        }
        _write_json(staging / "preprocessing_specification.json", preprocessing)

        execution_support, candidate_support, repeatability, threshold = _support_tables(
            primary_aggregated,
            simulation,
            selected,
            balanced_count=balanced_count,
            feature_columns=rfsim_columns,
            features=features,
            center=center,
            scale=scale,
            bandwidth=bandwidth,
            config=config,
        )
        global_sensitivity, local_sensitivity, matrix, sensitivity_summary = (
            _sensitivity_tables(simulation, selected, features, config)
        )
        global_sensitivity.attrs["condition_number"] = sensitivity_summary[
            "condition_number"
        ]
        scenario_sensitivity = _scenario_sensitivity(
            primary_aggregated,
            alternative_aggregated,
            simulation,
            selected,
            balanced_count=balanced_count,
            feature_columns=rfsim_columns,
            features=features,
            center=center,
            scale=scale,
            bandwidth=bandwidth,
        )
        decision, reservation = _decision(
            candidate_support, global_sensitivity, scenario_sensitivity, config
        )

        input_rows = [
            {
                "input_kind": "upv_route_observations",
                "source_id": paths["route_observations"].name,
                "sha256": _sha256(paths["route_observations"]),
                "rows_read": len(primary_route),
                "rows_after_frozen_filter": len(primary_raw),
                "aggregated_rows": len(primary_aggregated),
                "ploss": math.nan,
                "noise_power_dB": math.nan,
                "repetition": math.nan,
            },
            {
                "input_kind": "upv_locked_split",
                "source_id": paths["locked_split"].name,
                "sha256": _sha256(paths["locked_split"]),
                "rows_read": len(split),
                "rows_after_frozen_filter": len(calibration_row),
                "aggregated_rows": math.nan,
                "ploss": math.nan,
                "noise_power_dB": math.nan,
                "repetition": math.nan,
            },
            {
                "input_kind": "upv_source_archive",
                "source_id": paths["upv_archive"].name,
                "sha256": _sha256(paths["upv_archive"]),
                "rows_read": math.nan,
                "rows_after_frozen_filter": len(alternative_raw),
                "aggregated_rows": len(alternative_aggregated),
                "ploss": math.nan,
                "noise_power_dB": math.nan,
                "repetition": math.nan,
            },
            *execution_inventory.to_dict("records"),
        ]
        input_inventory = pd.DataFrame(input_rows)
        scaling = pd.DataFrame([
            {
                "feature": item["name"],
                "center": center[index],
                "scale": scale[index],
                "weight": float(item["weight"]),
                "balanced_rows_per_source": balanced_count,
                "balanced_source_count": len(balanced_frames),
                "kernel_bandwidth": bandwidth,
            }
            for index, item in enumerate(features)
        ])
        repeatability_summary = pd.DataFrame([{
            "pair_count": len(repeatability),
            "threshold_quantile": float(
                config["support_rules"]["repeatability_threshold_quantile"]
            ),
            "threshold_mmd_squared": threshold,
            "minimum_pair_mmd_squared": float(repeatability["mmd_squared"].min()),
            "median_pair_mmd_squared": float(repeatability["mmd_squared"].median()),
            "maximum_pair_mmd_squared": float(repeatability["mmd_squared"].max()),
        }])
        tables = {
            "input_inventory.csv": input_inventory,
            "calibration_bin_diagnostics.csv": diagnostics,
            "calibration_bin_autocorrelation.csv": autocorrelation,
            "validation_separation.csv": separation,
            "scaling_parameters.csv": scaling,
            "balanced_reference.csv": balanced_reference[[
                "reference_source", "aggregate_index", "time_seconds", *rfsim_columns
            ]],
            "execution_support.csv": execution_support,
            "candidate_support.csv": candidate_support,
            "repeatability_pairs.csv": repeatability,
            "repeatability_threshold.csv": repeatability_summary,
            "sensitivity_global.csv": global_sensitivity,
            "sensitivity_local.csv": local_sensitivity,
            "sensitivity_matrix.csv": matrix,
            "scenario_interpretation_sensitivity.csv": scenario_sensitivity,
        }
        for name, frame in tables.items():
            _write_csv(staging / name, frame)
        _write_json(staging / "sensitivity_summary.json", {
            "schema_version": 1,
            "phase1_revision": config["phase1_protocol"]["revision"],
            **sensitivity_summary,
        })
        _write_json(staging / "phase2_decision.json", decision)
        _write_json(staging / "next_reservation_plan.json", reservation)

        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "phase1_revision": config["phase1_protocol"]["revision"],
            "phase1_tag": config["phase1_protocol"]["tag"],
            "analysis_implementation_revision": software["revision"],
            "tracked_worktree_dirty_at_start": software["tracked_worktree_dirty"],
            "preprocessing_specification_sha256": output_hashes[
                "preprocessing_specification.json"
            ],
            "input_sha256": {
                "route_observations": _sha256(paths["route_observations"]),
                "locked_split": _sha256(paths["locked_split"]),
                "upv_archive": _sha256(paths["upv_archive"]),
                "phase1_config": _sha256(paths["phase1_config"]),
                "selection_manifest": _sha256(paths["selection_manifest"]),
                "campaign_state": _sha256(paths["campaign_state"]),
                "analysis_config": _sha256(paths["config"]),
            },
            "calibration_bin": diagnostic_summary,
            "validation_bins_geographically_adjacent": int(
                separation["geographically_adjacent_under_threshold"].sum()
            ),
            "balanced_reference_rows_per_source": balanced_count,
            "balanced_reference_source_count": len(balanced_frames),
            "repeatability_threshold_mmd_squared": threshold,
            "decision_code": decision["decision_code"],
            "abc_performed": False,
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
            "phase1_revision": config["phase1_protocol"]["revision"],
            "files": checksums,
        })
        staging.replace(paths["output"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(paths["output"]),
        "decision_code": decision["decision_code"],
        "best_candidate": decision["best_candidate"]["candidate_id"],
        "executions": len(selected),
        "candidates": len(candidate_support),
        "files": len(list(paths["output"].iterdir())),
        "abc_performed": False,
    }
