from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _read_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _read_yaml(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def _software_provenance() -> dict[str, Any]:
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
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"repository_revision": "unavailable", "tracked_worktree_dirty": None}
    return {
        "repository_revision": revision,
        "tracked_worktree_dirty": bool(status.strip()),
    }


def validate_distribution_calibration_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("distribution calibration schema_version must be 1")
    if not str(config.get("name") or "").strip():
        raise ValueError("distribution calibration requires a name")
    if config.get("candidate_policy") != "observed_safe_states_only":
        raise ValueError("the first distribution calibration cannot extrapolate")
    if config.get("holdout_unit") != "complete_execution":
        raise ValueError("repeatability must hold out complete executions")
    if int(config.get("minimum_samples_per_scenario", 0)) < 2:
        raise ValueError("minimum_samples_per_scenario must be at least 2")
    if int(config.get("minimum_samples_per_execution", 0)) < 2:
        raise ValueError("minimum_samples_per_execution must be at least 2")
    metrics = config.get("selection_metrics") or []
    if not metrics:
        raise ValueError("at least one selection metric is required")
    names: set[str] = set()
    for metric in metrics:
        name = str(metric.get("name") or "").strip().lower()
        if not name or name in names:
            raise ValueError("selection metric names must be unique")
        names.add(name)
        if not metric.get("real_column") or not metric.get("simulated_column"):
            raise ValueError("selection metrics require real and simulated columns")
        if float(metric.get("scale", 0)) <= 0:
            raise ValueError("selection metric scales must be positive")
    distance = config.get("distance") or {}
    if distance.get("primary") != "rbf_mmd":
        raise ValueError("the first distribution calibration uses rbf_mmd")
    if float(distance.get("kernel_bandwidth", 0)) <= 0:
        raise ValueError("kernel_bandwidth must be positive")
    if int(distance.get("maximum_samples", 0)) < 2:
        raise ValueError("maximum_samples must be at least 2")
    if int(distance.get("wasserstein_quantiles", 0)) < 3:
        raise ValueError("wasserstein_quantiles must be at least 3")
    if not str(config.get("real_scenario_column") or "").strip():
        raise ValueError("real_scenario_column is required")
    if not (config.get("expected_model_types") or []):
        raise ValueError("at least one expected RFsim model type is required")


def _selected_execution_rows(
    selection_manifest: dict[str, Any],
    campaign_state: dict[str, Any],
) -> list[dict[str, Any]]:
    points = selection_manifest.get("points") or []
    completed = campaign_state.get("completed") or {}
    if not points:
        raise ValueError("selection manifest contains no points")
    rows = []
    for point in points:
        point_id = str(point["point_id"])
        result = completed.get(point_id)
        if not isinstance(result, dict):
            raise ValueError(f"selected point is not complete: {point_id}")
        planned = point.get("controls") or {}
        observed = result.get("controls") or {}
        for control in ("ploss", "noise_power_dB"):
            if not math.isclose(
                float(planned[control]),
                float(observed[control]),
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"campaign state disagrees for {point_id}: {control}")
        rows.append({
            "point_id": point_id,
            "execution_id": str(result["execution_id"]),
            "repetition": int(point["repetition"]),
            "ploss": float(planned["ploss"]),
            "noise_power_dB": float(planned["noise_power_dB"]),
        })
    frame = pd.DataFrame(rows)
    if frame["point_id"].duplicated().any() or frame["execution_id"].duplicated().any():
        raise ValueError("selection manifest contains duplicate points or executions")
    return frame.sort_values(["ploss", "noise_power_dB", "repetition"]).to_dict("records")


def _load_real_observations(
    path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    scenario = str(config["real_scenario_column"])
    metrics = [*(config.get("selection_metrics") or []), *(config.get("diagnostic_metrics") or [])]
    required = {scenario, *(str(metric["real_column"]) for metric in metrics)}
    required.update(str(value) for value in config.get("real_context_columns") or [])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("real observations are missing fields: " + ", ".join(missing))
    selection_columns = []
    for metric in metrics:
        column = str(metric["real_column"])
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if metric in (config.get("selection_metrics") or []):
            selection_columns.append(column)
    frame = frame.dropna(subset=[scenario, *selection_columns]).copy()
    minimum = int(config["minimum_samples_per_scenario"])
    counts = frame.groupby(scenario).size()
    too_small = sorted(str(value) for value in counts[counts < minimum].index)
    if too_small:
        raise ValueError("real scenarios have too few observations: " + ", ".join(too_small))
    return frame.sort_values([scenario]).reset_index(drop=True)


def _load_simulated_observations(
    executions_root: Path,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    metrics = [*(config.get("selection_metrics") or []), *(config.get("diagnostic_metrics") or [])]
    metric_columns = [str(metric["simulated_column"]) for metric in metrics]
    require_true = [str(value) for value in config.get("require_true") or []]
    require_false = [str(value) for value in config.get("require_false") or []]
    required = {
        "execution_id",
        "dl_model_type",
        "dl_ploss",
        "dl_noise_power_dB",
        *metric_columns,
        *require_true,
        *require_false,
    }
    expected_models = {str(value) for value in config["expected_model_types"]}
    frames = []
    source_hashes: dict[str, str] = {}
    minimum = int(config["minimum_samples_per_execution"])
    for item in selected:
        path = executions_root / item["execution_id"] / "ue_second_features.parquet"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe RFsim observation file: {path}")
        frame = pd.read_parquet(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{item['execution_id']} is missing fields: " + ", ".join(missing))
        for field in require_true:
            frame = frame.loc[frame[field].fillna(False).astype(bool)]
        for field in require_false:
            frame = frame.loc[~frame[field].fillna(True).astype(bool)]
        frame = frame.loc[frame["dl_model_type"].astype("string").isin(expected_models)].copy()
        for column in ["dl_ploss", "dl_noise_power_dB", *metric_columns]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        selection_columns = [
            str(metric["simulated_column"]) for metric in config["selection_metrics"]
        ]
        frame = frame.dropna(subset=[
            "dl_ploss", "dl_noise_power_dB", *selection_columns
        ])
        if len(frame) < minimum:
            raise ValueError(f"{item['execution_id']} has too few verified observations")
        if frame["execution_id"].astype("string").nunique() != 1:
            raise ValueError(f"{item['execution_id']} contains mixed execution identifiers")
        if not frame["execution_id"].astype("string").eq(item["execution_id"]).all():
            raise ValueError(f"execution identifier mismatch in {path}")
        for observed, expected in (
            ("dl_ploss", item["ploss"]),
            ("dl_noise_power_dB", item["noise_power_dB"]),
        ):
            if not np.isclose(frame[observed].to_numpy(), expected, atol=1e-9).all():
                raise ValueError(
                    f"selected controls disagree in {item['execution_id']}: {observed}"
                )
        frame["point_id"] = item["point_id"]
        frame["repetition"] = item["repetition"]
        frame["model_type"] = frame["dl_model_type"].astype("string")
        frame["ploss"] = frame["dl_ploss"]
        frame["noise_power_dB"] = frame["dl_noise_power_dB"]
        frames.append(frame[[
            "execution_id",
            "point_id",
            "repetition",
            "model_type",
            "ploss",
            "noise_power_dB",
            *metric_columns,
        ]])
        source_hashes[item["execution_id"]] = _sha256(path)
    result = pd.concat(frames, ignore_index=True)
    state_columns = ["model_type", "ploss", "noise_power_dB"]
    repetitions = result.groupby(state_columns)["execution_id"].nunique()
    if repetitions.lt(2).any():
        raise ValueError("every retained RFsim state requires at least two executions")
    return result, source_hashes


def _downsample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
    return values[indices]


def rbf_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bandwidth: float = 1.0,
    maximum_samples: int = 512,
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be two matrices with the same column count")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("MMD requires at least two observations per distribution")
    if bandwidth <= 0 or maximum_samples < 2:
        raise ValueError("MMD bandwidth and sample limit must be positive")
    left = _downsample(left, maximum_samples)
    right = _downsample(right, maximum_samples)

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(first * first, axis=1)[:, None]
            + np.sum(second * second, axis=1)[None, :]
            - 2 * first @ second.T
        )
        return np.exp(-np.maximum(squared, 0) / (2 * bandwidth**2))

    value = kernel(left, left).mean() + kernel(right, right).mean()
    value -= 2 * kernel(left, right).mean()
    return max(float(value), 0.0)


def quantile_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    *,
    quantiles: int = 201,
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) < 2 or len(right) < 2:
        raise ValueError("Wasserstein inputs must contain at least two scalar observations")
    probabilities = np.linspace(0, 1, quantiles)
    return float(np.mean(np.abs(
        np.quantile(left, probabilities) - np.quantile(right, probabilities)
    )))


def _metric_matrix(
    frame: pd.DataFrame,
    metrics: list[dict[str, Any]],
    side: str,
) -> np.ndarray:
    column_key = "real_column" if side == "real" else "simulated_column"
    return np.column_stack([
        pd.to_numeric(frame[str(metric[column_key])], errors="raise").to_numpy()
        / float(metric["scale"])
        for metric in metrics
    ])


def _summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: list[dict[str, Any]],
    side: str,
    context_columns: list[str] | None = None,
) -> pd.DataFrame:
    column_key = "real_column" if side == "real" else "simulated_column"
    rows = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else key
        row = dict(zip(group_columns, keys, strict=True))
        row["observation_count"] = len(group)
        if side == "simulated":
            row["execution_count"] = int(group["execution_id"].nunique())
        for context in context_columns or []:
            values = group[context].dropna().astype("string").unique()
            row[context] = str(values[0]) if len(values) == 1 else "mixed"
        for metric in metrics:
            name = str(metric["name"]).lower()
            values = pd.to_numeric(group[str(metric[column_key])], errors="coerce").dropna()
            row[f"{name}_count"] = len(values)
            if values.empty:
                for statistic in ("mean", "std", "p05", "p25", "p50", "p75", "p95"):
                    row[f"{name}_{statistic}"] = math.nan
                continue
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
            for label, probability in (
                ("p05", 0.05),
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p95", 0.95),
            ):
                row[f"{name}_{label}"] = float(values.quantile(probability))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _candidate_rankings(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    scenario_column = str(config["real_scenario_column"])
    state_columns = ["model_type", "ploss", "noise_power_dB"]
    metrics = list(config["selection_metrics"])
    distance = config["distance"]
    rows = []
    for scenario_id, real_group in real.groupby(scenario_column, sort=True):
        candidates = []
        real_matrix = _metric_matrix(real_group, metrics, "real")
        for state, simulated_group in simulated.groupby(state_columns, sort=True):
            row = {
                scenario_column: scenario_id,
                **dict(zip(state_columns, state, strict=True)),
                "real_observation_count": len(real_group),
                "simulated_observation_count": len(simulated_group),
                "execution_count": int(simulated_group["execution_id"].nunique()),
            }
            simulated_matrix = _metric_matrix(simulated_group, metrics, "simulated")
            row["joint_mmd2"] = rbf_mmd2(
                real_matrix,
                simulated_matrix,
                bandwidth=float(distance["kernel_bandwidth"]),
                maximum_samples=int(distance["maximum_samples"]),
            )
            normalized_wasserstein = []
            for metric in metrics:
                name = str(metric["name"]).lower()
                value = quantile_wasserstein(
                    pd.to_numeric(real_group[str(metric["real_column"])], errors="raise")
                    .to_numpy(),
                    pd.to_numeric(
                        simulated_group[str(metric["simulated_column"])], errors="raise"
                    ).to_numpy(),
                    quantiles=int(distance["wasserstein_quantiles"]),
                )
                row[f"{name}_wasserstein"] = value
                normalized_wasserstein.append(value / float(metric["scale"]))
            row["mean_normalized_wasserstein"] = float(np.mean(normalized_wasserstein))
            for context in config.get("real_context_columns") or []:
                values = real_group[str(context)].dropna().astype("string").unique()
                row[str(context)] = str(values[0]) if len(values) == 1 else "mixed"
            candidates.append(row)
        candidates.sort(key=lambda item: (
            item["joint_mmd2"],
            item["mean_normalized_wasserstein"],
            item["model_type"],
            item["ploss"],
            item["noise_power_dB"],
        ))
        for rank, row in enumerate(candidates, start=1):
            row["candidate_rank"] = rank
            row["candidate_role"] = "nearest_distribution" if rank == 1 else "alternative"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [scenario_column, "candidate_rank"]
    ).reset_index(drop=True)


def _repeatability(
    simulated: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    state_columns = ["model_type", "ploss", "noise_power_dB"]
    metrics = list(config["selection_metrics"])
    distance = config["distance"]
    rows = []
    for state, state_group in simulated.groupby(state_columns, sort=True):
        executions = {
            str(execution_id): group
            for execution_id, group in state_group.groupby("execution_id", sort=True)
        }
        for (left_id, left), (right_id, right) in combinations(executions.items(), 2):
            row = {
                **dict(zip(state_columns, state, strict=True)),
                "left_execution_id": left_id,
                "right_execution_id": right_id,
                "left_observation_count": len(left),
                "right_observation_count": len(right),
                "joint_mmd2": rbf_mmd2(
                    _metric_matrix(left, metrics, "simulated"),
                    _metric_matrix(right, metrics, "simulated"),
                    bandwidth=float(distance["kernel_bandwidth"]),
                    maximum_samples=int(distance["maximum_samples"]),
                ),
            }
            for metric in metrics:
                name = str(metric["name"]).lower()
                column = str(metric["simulated_column"])
                row[f"{name}_wasserstein"] = quantile_wasserstein(
                    pd.to_numeric(left[column], errors="raise").to_numpy(),
                    pd.to_numeric(right[column], errors="raise").to_numpy(),
                    quantiles=int(distance["wasserstein_quantiles"]),
                )
            rows.append(row)
    if not rows:
        raise ValueError("no repeated RFsim executions were available")
    return pd.DataFrame(rows).sort_values(
        [*state_columns, "left_execution_id", "right_execution_id"]
    ).reset_index(drop=True)


def run_distribution_calibration(
    *,
    real_observations: str | Path,
    executions_root: str | Path,
    selection_manifest: str | Path,
    campaign_state: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    real_path = Path(real_observations).resolve()
    executions_root = Path(executions_root).resolve()
    selection_path = Path(selection_manifest).resolve()
    campaign_path = Path(campaign_state).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"distribution calibration output already exists: {output_dir}")
    config = _read_yaml(config_path)
    validate_distribution_calibration_config(config)
    selection = _read_json(selection_path)
    campaign = _read_json(campaign_path)
    selected = _selected_execution_rows(selection, campaign)
    real = _load_real_observations(real_path, config)
    simulated, execution_hashes = _load_simulated_observations(
        executions_root, selected, config
    )
    all_metrics = [
        *config["selection_metrics"],
        *(config.get("diagnostic_metrics") or []),
    ]
    scenario_column = str(config["real_scenario_column"])
    real_summary = _summary(
        real,
        group_columns=[scenario_column],
        metrics=all_metrics,
        side="real",
        context_columns=[str(value) for value in config.get("real_context_columns") or []],
    )
    simulated_summary = _summary(
        simulated,
        group_columns=["model_type", "ploss", "noise_power_dB"],
        metrics=all_metrics,
        side="simulated",
    )
    rankings = _candidate_rankings(real, simulated, config)
    repeatability = _repeatability(simulated, config)
    selected_rankings = rankings.loc[rankings["candidate_rank"].eq(1)]
    manifest = {
        "schema_version": 1,
        "calibration_id": config["name"],
        "method": "likelihood_free_distribution_calibration",
        "implementation_scope": "discrete_executed_state_mmd_screen",
        "primary_distance": "biased_rbf_mmd_squared",
        "candidate_policy": config["candidate_policy"],
        "holdout_unit": config["holdout_unit"],
        "selection_metrics": config["selection_metrics"],
        "diagnostic_metrics": config.get("diagnostic_metrics") or [],
        "real_scenarios": int(real[scenario_column].nunique()),
        "real_observations": len(real),
        "rfsim_executions": int(simulated["execution_id"].nunique()),
        "rfsim_states": len(simulated_summary),
        "rfsim_observations": len(simulated),
        "software_provenance": _software_provenance(),
        "ranking_rows": len(rankings),
        "repeatability_pairs": len(repeatability),
        "best_candidate_joint_mmd2": {
            "minimum": float(selected_rankings["joint_mmd2"].min()),
            "median": float(selected_rankings["joint_mmd2"].median()),
            "maximum": float(selected_rankings["joint_mmd2"].max()),
        },
        "source_sha256": {
            "real_observations": _sha256(real_path),
            "selection_manifest": _sha256(selection_path),
            "campaign_state": _sha256(campaign_path),
            "config": _sha256(config_path),
            "execution_observations": execution_hashes,
        },
        "limitations": [
            "candidate ranking is restricted to executed safe RFsim states",
            "RFsim controls proposed by interpolation require a new held-out execution",
            "real SNR and OAI SS-SINR remain diagnostic because their definitions differ",
            "distribution similarity does not prove that the underlying propagation physics match",
            (
                "candidate rankings pool complete repetitions and remain exploratory "
                "until repeatability is accepted"
            ),
            "time dependence and interval uncertainty require block-aware resampling",
            "traffic and topology generalization require separate validation",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ))
    try:
        _write_csv(real_summary, staging / "real_scenario_summaries.csv")
        _write_csv(simulated_summary, staging / "rfsim_state_summaries.csv")
        _write_csv(rankings, staging / "candidate_rankings.csv")
        _write_csv(repeatability, staging / "state_repeatability.csv")
        _write_json(staging / "calibration_manifest.json", manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(staging / "SHA256SUMS.json", checksums)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(output_dir),
        "scenarios": manifest["real_scenarios"],
        "states": manifest["rfsim_states"],
        "executions": manifest["rfsim_executions"],
        "ranking_rows": manifest["ranking_rows"],
        "files": len(checksums) + 1,
    }
