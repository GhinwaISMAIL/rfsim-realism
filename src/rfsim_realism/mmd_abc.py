from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .distribution_calibration import quantile_wasserstein


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


def _inferred_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config.get("inferred_parameters") or [])


def _fixed_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config.get("fixed_parameters") or [])


def validate_mmd_abc_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("MMD-ABC schema_version must be 1")
    if not str(config.get("name") or "").strip():
        raise ValueError("MMD-ABC configuration requires a name")
    if config.get("implementation") != "execution_bank_rejection_abc":
        raise ValueError("implementation must be execution_bank_rejection_abc")
    if str((config.get("model") or {}).get("family")) != "TDL_B":
        raise ValueError("the first MMD-ABC experiment must keep TDL_B fixed")
    if config.get("holdout_unit") != "complete_execution":
        raise ValueError("MMD-ABC must hold out complete executions")

    inferred = _inferred_parameters(config)
    fixed = _fixed_parameters(config)
    if not inferred:
        raise ValueError("at least one inferred parameter is required")
    names: set[str] = set()
    for parameter in [*inferred, *fixed]:
        name = str(parameter.get("name") or "").strip()
        if not name or name in names:
            raise ValueError("parameter names must be present and unique")
        names.add(name)
        if parameter in inferred:
            prior = parameter.get("prior") or {}
            if prior.get("distribution") != "uniform":
                raise ValueError("the first implementation supports uniform priors")
            lower = float(prior.get("lower"))
            upper = float(prior.get("upper"))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"invalid uniform prior for {name}")
            values = parameter.get("proposal_values") or []
            if len(values) < 2:
                raise ValueError(f"{name} requires at least two proposal_values")
            numeric = [float(value) for value in values]
            if numeric != sorted(set(numeric)):
                raise ValueError(f"{name} proposal_values must be sorted and unique")
            if numeric[0] < lower or numeric[-1] > upper:
                raise ValueError(f"{name} proposal_values fall outside the prior")
        elif "value" not in parameter:
            raise ValueError(f"fixed parameter {name} requires a value")

    metrics = config.get("selection_metrics") or []
    if len(metrics) != 2:
        raise ValueError("the first MMD-ABC experiment requires joint RSRP and RSRQ")
    metric_names: set[str] = set()
    for metric in metrics:
        name = str(metric.get("name") or "").strip()
        if not name or name in metric_names:
            raise ValueError("selection metric names must be present and unique")
        metric_names.add(name)
        if not metric.get("real_column") or not metric.get("simulated_column"):
            raise ValueError("selection metrics require real and simulated columns")

    transform = config.get("transform") or {}
    if transform.get("method") != "pooled_real_covariance_whitening":
        raise ValueError("transform must be pooled_real_covariance_whitening")
    if float(transform.get("relative_eigenvalue_floor", 0)) <= 0:
        raise ValueError("relative_eigenvalue_floor must be positive")

    kernel = config.get("kernel") or {}
    estimators = {"unbiased_mmd_squared", "biased_mmd_squared_v_statistic"}
    if kernel.get("name") != "rbf" or kernel.get("estimator") not in estimators:
        raise ValueError(
            "kernel must use RBF with unbiased_mmd_squared or "
            "biased_mmd_squared_v_statistic"
        )
    multipliers = [float(value) for value in kernel.get("bandwidth_multipliers") or []]
    if 1.0 not in multipliers or any(value <= 0 for value in multipliers):
        raise ValueError("bandwidth multipliers must be positive and include 1.0")
    if int(kernel.get("maximum_reference_samples", 0)) < 2:
        raise ValueError("maximum_reference_samples must be at least 2")
    if int(kernel.get("maximum_samples_per_distribution", 0)) < 2:
        raise ValueError("maximum_samples_per_distribution must be at least 2")

    execution = config.get("execution") or {}
    if int(execution.get("independent_repetitions", 0)) < 2:
        raise ValueError("at least two independent repetitions are required")
    if int(execution.get("minimum_samples_per_scenario", 0)) < 2:
        raise ValueError("minimum_samples_per_scenario must be at least 2")
    if int(execution.get("minimum_samples_per_execution", 0)) < 2:
        raise ValueError("minimum_samples_per_execution must be at least 2")
    if float(execution.get("run_seconds", 0)) <= 0:
        raise ValueError("run_seconds must be positive")

    abc = config.get("abc") or {}
    fraction = float(abc.get("acceptance_fraction", 0))
    if not 0 < fraction <= 1:
        raise ValueError("acceptance_fraction must be in (0, 1]")
    for field in (
        "minimum_total_simulations",
        "minimum_accepted_samples",
        "minimum_unique_parameter_values",
    ):
        if int(abc.get(field, 0)) < 1:
            raise ValueError(f"{field} must be positive")
    if float(abc.get("minimum_effective_sample_size", 0)) <= 0:
        raise ValueError("minimum_effective_sample_size must be positive")

    if not str(config.get("real_scenario_column") or "").strip():
        raise ValueError("real_scenario_column is required")
    if not str(execution.get("observation_filename") or "").strip():
        raise ValueError("execution observation_filename is required")


def build_mmd_abc_plan(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _read_yaml(path)
    validate_mmd_abc_config(config)
    inferred = _inferred_parameters(config)
    repetitions = int(config["execution"]["independent_repetitions"])
    values = [parameter["proposal_values"] for parameter in inferred]
    proposals = []
    points = []
    for proposal_index, combination in enumerate(product(*values), start=1):
        proposal_id = f"proposal-{proposal_index:03d}"
        theta = {
            parameter["name"]: float(value)
            for parameter, value in zip(inferred, combination, strict=True)
        }
        controls = {
            **theta,
            **{
                parameter["name"]: parameter["value"]
                for parameter in _fixed_parameters(config)
                if bool(parameter.get("include_in_execution_controls", False))
            },
        }
        proposals.append({"proposal_id": proposal_id, "theta": theta})
        for repetition in range(1, repetitions + 1):
            points.append({
                "point_id": f"{proposal_id}-r{repetition:02d}",
                "proposal_id": proposal_id,
                "repetition": repetition,
                "theta": theta,
                "controls": controls,
                "model_family": config["model"]["family"],
                "run_seconds": float(config["execution"]["run_seconds"]),
                "stage": config.get("stage"),
            })
    return {
        "schema_version": 1,
        "plan_id": config["name"],
        "implementation": config["implementation"],
        "stage": config.get("stage"),
        "model": config["model"],
        "holdout_unit": config["holdout_unit"],
        "seed": int(config["seed"]),
        "config_sha256": _sha256(path),
        "proposal_count": len(proposals),
        "execution_count": len(points),
        "proposals": proposals,
        "points": points,
        "limitations": [
            "planned controls are proposals and are not evidence until executed",
            "the pilot is underpowered for a resolved continuous ABC posterior",
            "every accepted row requires a complete quality-gated execution",
        ],
    }


def write_mmd_abc_plan(config_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"MMD-ABC plan already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(output, build_mmd_abc_plan(config_path))


def _downsample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
    return values[indices]


def reference_whitener(
    values: np.ndarray,
    *,
    relative_eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("whitening requires a matrix with at least two rows")
    if relative_eigenvalue_floor <= 0:
        raise ValueError("relative_eigenvalue_floor must be positive")
    center = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(float(eigenvalues.max()), np.finfo(float).eps)
    floored = np.maximum(eigenvalues, maximum * relative_eigenvalue_floor)
    whitener = eigenvectors @ np.diag(1.0 / np.sqrt(floored)) @ eigenvectors.T
    transformed = (values - center) @ whitener
    return transformed, center, covariance, whitener


def median_heuristic_bandwidth(values: np.ndarray, *, maximum_samples: int = 512) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("bandwidth estimation requires at least two observations")
    values = _downsample(values, maximum_samples)
    squared = (
        np.sum(values * values, axis=1)[:, None]
        + np.sum(values * values, axis=1)[None, :]
        - 2 * values @ values.T
    )
    distances = np.maximum(squared[np.triu_indices(len(values), k=1)], 0)
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("reference observations have zero pairwise distance")
    median = float(np.median(distances))
    if median <= 0:
        median = float(np.median(positive))
    return math.sqrt(median / 2.0)


def unbiased_rbf_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bandwidth: float,
    maximum_samples: int = 512,
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be matrices with equal column count")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("unbiased MMD requires at least two observations per sample")
    if bandwidth <= 0 or maximum_samples < 2:
        raise ValueError("bandwidth and maximum_samples must be positive")
    left = _downsample(left, maximum_samples)
    right = _downsample(right, maximum_samples)

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(first * first, axis=1)[:, None]
            + np.sum(second * second, axis=1)[None, :]
            - 2 * first @ second.T
        )
        return np.exp(-np.maximum(squared, 0) / (2 * bandwidth**2))

    left_kernel = kernel(left, left)
    right_kernel = kernel(right, right)
    cross_kernel = kernel(left, right)
    left_term = (left_kernel.sum() - np.trace(left_kernel)) / (len(left) * (len(left) - 1))
    right_term = (right_kernel.sum() - np.trace(right_kernel)) / (
        len(right) * (len(right) - 1)
    )
    return float(left_term + right_term - 2 * cross_kernel.mean())


def biased_rbf_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bandwidth: float,
    maximum_samples: int = 512,
) -> float:
    """Return the RBF MMD-squared V-statistic without post-hoc clipping."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be matrices with equal column count")
    if len(left) < 1 or len(right) < 1:
        raise ValueError("biased MMD requires at least one observation per sample")
    if bandwidth <= 0 or maximum_samples < 1:
        raise ValueError("bandwidth and maximum_samples must be positive")
    left = _downsample(left, maximum_samples)
    right = _downsample(right, maximum_samples)

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(first * first, axis=1)[:, None]
            + np.sum(second * second, axis=1)[None, :]
            - 2 * first @ second.T
        )
        return np.exp(-np.maximum(squared, 0) / (2 * bandwidth**2))

    value = kernel(left, left).mean()
    value += kernel(right, right).mean()
    value -= 2 * kernel(left, right).mean()
    return float(value)


def _configured_rbf_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    *,
    estimator: str,
    bandwidth: float,
    maximum_samples: int,
) -> tuple[float, float]:
    if estimator == "biased_mmd_squared_v_statistic":
        value = biased_rbf_mmd2(
            left,
            right,
            bandwidth=bandwidth,
            maximum_samples=maximum_samples,
        )
        return value, value
    if estimator == "unbiased_mmd_squared":
        raw = unbiased_rbf_mmd2(
            left,
            right,
            bandwidth=bandwidth,
            maximum_samples=maximum_samples,
        )
        return raw, max(raw, 0.0)
    raise ValueError(f"unsupported MMD estimator: {estimator}")


def _load_real_observations(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    scenario = str(config["real_scenario_column"])
    metrics = [*(config["selection_metrics"]), *(config.get("diagnostic_metrics") or [])]
    required = {scenario, *(str(metric["real_column"]) for metric in metrics)}
    required.update(str(value) for value in config.get("real_context_columns") or [])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("real observations are missing fields: " + ", ".join(missing))
    selected_columns = []
    for metric in metrics:
        column = str(metric["real_column"])
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if metric in config["selection_metrics"]:
            selected_columns.append(column)
    frame = frame.dropna(subset=[scenario, *selected_columns]).copy()
    minimum = int(config["execution"]["minimum_samples_per_scenario"])
    counts = frame.groupby(scenario).size()
    too_small = sorted(str(value) for value in counts[counts < minimum].index)
    if too_small:
        raise ValueError("real scenarios have too few observations: " + ", ".join(too_small))
    return frame.sort_values([scenario]).reset_index(drop=True)


def _completed_plan_points(
    plan: dict[str, Any],
    campaign: dict[str, Any],
    *,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    points = plan.get("points") or []
    completed = campaign.get("completed") or {}
    if not points:
        raise ValueError("MMD-ABC plan contains no points")
    rows = []
    missing = []
    for point in points:
        point_id = str(point["point_id"])
        result = completed.get(point_id)
        if not isinstance(result, dict):
            missing.append(point_id)
            continue
        planned_controls = point.get("controls") or {}
        observed_controls = result.get("controls") or {}
        for name, planned in planned_controls.items():
            if name not in observed_controls:
                raise ValueError(f"campaign state is missing {name} for {point_id}")
            observed = observed_controls[name]
            if isinstance(planned, (int, float)):
                if not math.isclose(float(planned), float(observed), rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"campaign state disagrees for {point_id}: {name}")
            elif planned != observed:
                raise ValueError(f"campaign state disagrees for {point_id}: {name}")
        rows.append({
            **point,
            "execution_id": str(result["execution_id"]),
        })
    if missing and not allow_partial:
        raise ValueError("planned MMD-ABC points are incomplete: " + ", ".join(missing))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("campaign state contains no completed MMD-ABC points")
    if frame["point_id"].duplicated().any() or frame["execution_id"].duplicated().any():
        raise ValueError("campaign contains duplicate points or execution identifiers")
    return frame.sort_values(["proposal_id", "repetition"]).to_dict("records")


def _load_simulated_observations(
    executions_root: Path,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    execution_config = config["execution"]
    metrics = [*(config["selection_metrics"]), *(config.get("diagnostic_metrics") or [])]
    metric_columns = [str(metric["simulated_column"]) for metric in metrics]
    require_true = [str(value) for value in execution_config.get("require_true") or []]
    require_false = [str(value) for value in execution_config.get("require_false") or []]
    model_column = str(execution_config["model_column"])
    execution_column = str(execution_config["execution_id_column"])
    verified_parameters = [
        parameter
        for parameter in [*_inferred_parameters(config), *_fixed_parameters(config)]
        if parameter.get("simulated_column")
    ]
    required = {
        execution_column,
        model_column,
        *metric_columns,
        *require_true,
        *require_false,
        *(str(parameter["simulated_column"]) for parameter in verified_parameters),
    }
    frames = []
    hashes: dict[str, str] = {}
    minimum = int(execution_config["minimum_samples_per_execution"])
    family = str(config["model"]["family"])
    filename = str(execution_config["observation_filename"])
    for item in selected:
        path = executions_root / item["execution_id"] / filename
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
        frame = frame.loc[frame[model_column].astype("string").eq(family)].copy()
        numeric_columns = [
            *metric_columns,
            *(str(parameter["simulated_column"]) for parameter in verified_parameters),
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        selection_columns = [
            str(metric["simulated_column"]) for metric in config["selection_metrics"]
        ]
        frame = frame.dropna(subset=[*selection_columns])
        if len(frame) < minimum:
            raise ValueError(f"{item['execution_id']} has too few verified observations")
        if frame[execution_column].astype("string").nunique() != 1:
            raise ValueError(f"{item['execution_id']} contains mixed execution identifiers")
        if not frame[execution_column].astype("string").eq(item["execution_id"]).all():
            raise ValueError(f"execution identifier mismatch in {path}")
        theta = item.get("theta") or {}
        fixed_values = {
            parameter["name"]: parameter["value"] for parameter in _fixed_parameters(config)
        }
        for parameter in verified_parameters:
            name = str(parameter["name"])
            expected = theta[name] if name in theta else fixed_values[name]
            observed_column = str(parameter["simulated_column"])
            if not np.isclose(
                frame[observed_column].to_numpy(),
                float(expected),
                atol=float(parameter.get("verification_tolerance", 1e-9)),
                rtol=0,
            ).all():
                raise ValueError(
                    f"selected control disagrees in {item['execution_id']}: {observed_column}"
                )
        frame["execution_id"] = item["execution_id"]
        frame["point_id"] = item["point_id"]
        frame["proposal_id"] = item["proposal_id"]
        frame["repetition"] = int(item["repetition"])
        for parameter in _inferred_parameters(config):
            frame[str(parameter["name"])] = float(theta[str(parameter["name"])])
        frames.append(frame[[
            "execution_id",
            "point_id",
            "proposal_id",
            "repetition",
            *(str(parameter["name"]) for parameter in _inferred_parameters(config)),
            *metric_columns,
        ]])
        hashes[item["execution_id"]] = _sha256(path)
    return pd.concat(frames, ignore_index=True), hashes


def _metric_matrix(
    frame: pd.DataFrame,
    metrics: list[dict[str, Any]],
    side: str,
) -> np.ndarray:
    column_key = "real_column" if side == "real" else "simulated_column"
    return np.column_stack([
        pd.to_numeric(frame[str(metric[column_key])], errors="raise").to_numpy()
        for metric in metrics
    ])


def _bandwidth_suffix(multiplier: float) -> str:
    return str(multiplier).replace("-", "m").replace(".", "p")


def _proposal_discrepancies(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
    config: dict[str, Any],
    *,
    center: np.ndarray,
    whitener: np.ndarray,
    bandwidth: float,
) -> pd.DataFrame:
    scenario_column = str(config["real_scenario_column"])
    metrics = list(config["selection_metrics"])
    parameters = [str(parameter["name"]) for parameter in _inferred_parameters(config)]
    kernel = config["kernel"]
    maximum = int(kernel["maximum_samples_per_distribution"])
    multipliers = [float(value) for value in kernel["bandwidth_multipliers"]]
    estimator = str(kernel["estimator"])
    rows = []
    for scenario_id, real_group in real.groupby(scenario_column, sort=True):
        real_matrix = (_metric_matrix(real_group, metrics, "real") - center) @ whitener
        for execution_id, execution in simulated.groupby("execution_id", sort=True):
            simulated_matrix = (
                _metric_matrix(execution, metrics, "simulated") - center
            ) @ whitener
            first = execution.iloc[0]
            row = {
                scenario_column: scenario_id,
                "execution_id": execution_id,
                "point_id": str(first["point_id"]),
                "proposal_id": str(first["proposal_id"]),
                "repetition": int(first["repetition"]),
                "real_observation_count": len(real_group),
                "simulated_observation_count": len(execution),
                **{parameter: float(first[parameter]) for parameter in parameters},
            }
            for multiplier in multipliers:
                raw, discrepancy = _configured_rbf_mmd2(
                    real_matrix,
                    simulated_matrix,
                    estimator=estimator,
                    bandwidth=bandwidth * multiplier,
                    maximum_samples=maximum,
                )
                suffix = _bandwidth_suffix(multiplier)
                row[f"joint_mmd2_raw_bw_{suffix}"] = raw
                row[f"joint_mmd2_bw_{suffix}"] = discrepancy
                if multiplier == 1.0:
                    row["joint_mmd2_raw"] = raw
                    row["joint_mmd2"] = discrepancy
            for metric in metrics:
                name = str(metric["name"]).lower()
                row[f"{name}_wasserstein"] = quantile_wasserstein(
                    pd.to_numeric(real_group[str(metric["real_column"])], errors="raise")
                    .to_numpy(),
                    pd.to_numeric(
                        execution[str(metric["simulated_column"])], errors="raise"
                    ).to_numpy(),
                    quantiles=int(config["diagnostics"]["wasserstein_quantiles"]),
                )
            for context in config.get("real_context_columns") or []:
                values = real_group[str(context)].dropna().astype("string").unique()
                row[str(context)] = str(values[0]) if len(values) == 1 else "mixed"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [scenario_column, "joint_mmd2", "execution_id"]
    ).reset_index(drop=True)


def _abc_weights(discrepancies: np.ndarray) -> tuple[np.ndarray, float]:
    discrepancies = np.asarray(discrepancies, dtype=float)
    if discrepancies.ndim != 1 or not len(discrepancies):
        raise ValueError("ABC weighting requires a non-empty discrepancy vector")
    epsilon = float(discrepancies.max())
    if epsilon <= np.finfo(float).eps:
        weights = np.ones(len(discrepancies), dtype=float)
    else:
        weights = np.maximum(1.0 - (discrepancies / epsilon) ** 2, 0.0)
        if float(weights.sum()) <= np.finfo(float).eps:
            weights = np.ones(len(discrepancies), dtype=float)
    weights /= weights.sum()
    return weights, epsilon


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("weighted quantile inputs must be non-empty equal-length vectors")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, probability, side="left")])


def _posterior_tables(
    discrepancies: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_column = str(config["real_scenario_column"])
    parameters = _inferred_parameters(config)
    parameter_names = [str(parameter["name"]) for parameter in parameters]
    abc = config["abc"]
    acceptance_fraction = float(abc["acceptance_fraction"])
    samples = []
    summaries = []
    for scenario_id, group in discrepancies.groupby(scenario_column, sort=True):
        ranked = group.sort_values(["joint_mmd2", "execution_id"]).reset_index(drop=True)
        accepted_count = max(1, math.ceil(len(ranked) * acceptance_fraction))
        accepted = ranked.iloc[:accepted_count].copy()
        weights, epsilon = _abc_weights(accepted["joint_mmd2"].to_numpy())
        accepted["abc_rank"] = np.arange(1, len(accepted) + 1)
        accepted["abc_weight"] = weights
        accepted["abc_epsilon"] = epsilon
        accepted["acceptance_fraction"] = acceptance_fraction
        effective_sample_size = float(1.0 / np.sum(weights**2))
        unique_values = int(accepted[parameter_names].drop_duplicates().shape[0])
        failures = []
        if len(group) < int(abc["minimum_total_simulations"]):
            failures.append("minimum_total_simulations")
        if accepted_count < int(abc["minimum_accepted_samples"]):
            failures.append("minimum_accepted_samples")
        if unique_values < int(abc["minimum_unique_parameter_values"]):
            failures.append("minimum_unique_parameter_values")
        if effective_sample_size < float(abc["minimum_effective_sample_size"]):
            failures.append("minimum_effective_sample_size")
        if bool(abc.get("pilot_only", False)):
            status = "pilot_only_not_established"
        elif failures:
            status = "underpowered_not_established"
        else:
            status = "abc_posterior_established"
        accepted["posterior_status"] = status
        samples.append(accepted)

        summary: dict[str, Any] = {
            scenario_column: scenario_id,
            "posterior_status": status,
            "failed_gates": ";".join(failures),
            "total_simulations": len(group),
            "accepted_samples": accepted_count,
            "accepted_unique_parameter_values": unique_values,
            "acceptance_fraction": acceptance_fraction,
            "abc_epsilon": epsilon,
            "effective_sample_size": effective_sample_size,
            "minimum_joint_mmd2": float(group["joint_mmd2"].min()),
            "median_joint_mmd2": float(group["joint_mmd2"].median()),
        }
        for parameter in parameters:
            name = str(parameter["name"])
            values = accepted[name].to_numpy(dtype=float)
            mean = float(np.sum(weights * values))
            variance = float(np.sum(weights * (values - mean) ** 2))
            lower = float(parameter["prior"]["lower"])
            upper = float(parameter["prior"]["upper"])
            width = upper - lower
            boundary_fraction = float(abc.get("boundary_fraction", 0.1))
            summary.update({
                f"{name}_weighted_mean": mean,
                f"{name}_weighted_std": math.sqrt(max(variance, 0.0)),
                f"{name}_q05": _weighted_quantile(values, weights, 0.05),
                f"{name}_q50": _weighted_quantile(values, weights, 0.50),
                f"{name}_q95": _weighted_quantile(values, weights, 0.95),
                f"{name}_prior_std": width / math.sqrt(12),
                f"{name}_posterior_to_prior_std": (
                    math.sqrt(max(variance, 0.0)) / (width / math.sqrt(12))
                ),
                f"{name}_lower_boundary_weight": float(
                    weights[values <= lower + boundary_fraction * width].sum()
                ),
                f"{name}_upper_boundary_weight": float(
                    weights[values >= upper - boundary_fraction * width].sum()
                ),
            })
        if len(parameter_names) > 1:
            matrix = accepted[parameter_names].to_numpy(dtype=float)
            mean = np.sum(matrix * weights[:, None], axis=0)
            centered = matrix - mean
            covariance = (centered * weights[:, None]).T @ centered
            eigenvalues = np.linalg.eigvalsh(covariance)
            positive = eigenvalues[eigenvalues > np.finfo(float).eps]
            summary["parameter_covariance_condition_number"] = (
                float(positive.max() / positive.min()) if len(positive) else math.inf
            )
        summaries.append(summary)
    return (
        pd.concat(samples, ignore_index=True).sort_values(
            [scenario_column, "abc_rank"]
        ).reset_index(drop=True),
        pd.DataFrame(summaries).sort_values(scenario_column).reset_index(drop=True),
    )


def _replicate_summary(discrepancies: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scenario = str(config["real_scenario_column"])
    parameters = [str(parameter["name"]) for parameter in _inferred_parameters(config)]
    grouped = discrepancies.groupby([scenario, *parameters], sort=True, dropna=False)
    result = grouped["joint_mmd2"].agg(
        execution_count="count",
        joint_mmd2_mean="mean",
        joint_mmd2_std="std",
        joint_mmd2_minimum="min",
        joint_mmd2_maximum="max",
    ).reset_index()
    result["joint_mmd2_std"] = result["joint_mmd2_std"].fillna(0.0)
    return result


def _summary_support(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    scenario = str(config["real_scenario_column"])
    metrics = list(config["selection_metrics"])
    rows = []
    simulated_statistics: dict[str, dict[str, np.ndarray]] = {}
    for metric in metrics:
        name = str(metric["name"])
        column = str(metric["simulated_column"])
        execution_values = []
        for _, execution in simulated.groupby("execution_id", sort=True):
            values = pd.to_numeric(execution[column], errors="raise").to_numpy()
            execution_values.append({
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            })
        simulated_statistics[name] = {
            statistic: np.array([item[statistic] for item in execution_values])
            for statistic in ("mean", "std", "p05", "p50", "p95")
        }
    for scenario_id, real_group in real.groupby(scenario, sort=True):
        for metric in metrics:
            name = str(metric["name"])
            values = pd.to_numeric(
                real_group[str(metric["real_column"])], errors="raise"
            ).to_numpy()
            real_statistics = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for statistic, real_value in real_statistics.items():
                simulated_values = simulated_statistics[name][statistic]
                lower = float(simulated_values.min())
                upper = float(simulated_values.max())
                rows.append({
                    scenario: scenario_id,
                    "metric": name,
                    "statistic": statistic,
                    "real_value": real_value,
                    "simulated_minimum": lower,
                    "simulated_maximum": upper,
                    "inside_simulated_range": lower <= real_value <= upper,
                })
    return pd.DataFrame(rows).sort_values([scenario, "metric", "statistic"]).reset_index(
        drop=True
    )


def _bandwidth_sensitivity(
    discrepancies: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    scenario = str(config["real_scenario_column"])
    fraction = float(config["abc"]["acceptance_fraction"])
    primary = {
        scenario_id: set(group["execution_id"].astype(str))
        for scenario_id, group in posterior_samples.groupby(scenario, sort=True)
    }
    rows = []
    for scenario_id, group in discrepancies.groupby(scenario, sort=True):
        accepted_count = max(1, math.ceil(len(group) * fraction))
        for multiplier in config["kernel"]["bandwidth_multipliers"]:
            multiplier = float(multiplier)
            column = f"joint_mmd2_bw_{_bandwidth_suffix(multiplier)}"
            accepted = set(
                group.sort_values([column, "execution_id"])
                .iloc[:accepted_count]["execution_id"]
                .astype(str)
            )
            union = primary[scenario_id] | accepted
            rows.append({
                scenario: scenario_id,
                "bandwidth_multiplier": multiplier,
                "accepted_count": accepted_count,
                "accepted_execution_jaccard": (
                    len(primary[scenario_id] & accepted) / len(union) if union else 1.0
                ),
                "best_execution_id": str(
                    group.sort_values([column, "execution_id"]).iloc[0]["execution_id"]
                ),
            })
    return pd.DataFrame(rows).sort_values([scenario, "bandwidth_multiplier"]).reset_index(
        drop=True
    )


def run_mmd_abc(
    *,
    real_observations: str | Path,
    executions_root: str | Path,
    proposal_plan: str | Path,
    campaign_state: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    real_path = Path(real_observations).resolve()
    executions_root = Path(executions_root).resolve()
    plan_path = Path(proposal_plan).resolve()
    campaign_path = Path(campaign_state).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"MMD-ABC output already exists: {output_dir}")
    config = _read_yaml(config_path)
    validate_mmd_abc_config(config)
    plan = _read_json(plan_path)
    campaign = _read_json(campaign_path)
    if plan.get("plan_id") != config["name"]:
        raise ValueError("proposal plan does not match the MMD-ABC configuration")
    if plan.get("config_sha256") != _sha256(config_path):
        raise ValueError("proposal plan configuration hash does not match")
    selected = _completed_plan_points(
        plan,
        campaign,
        allow_partial=bool(config["execution"].get("allow_partial_bank", False)),
    )
    real = _load_real_observations(real_path, config)
    simulated, execution_hashes = _load_simulated_observations(
        executions_root,
        selected,
        config,
    )
    metrics = list(config["selection_metrics"])
    pooled_real = _metric_matrix(real, metrics, "real")
    transformed, center, covariance, whitener = reference_whitener(
        pooled_real,
        relative_eigenvalue_floor=float(config["transform"]["relative_eigenvalue_floor"]),
    )
    bandwidth = median_heuristic_bandwidth(
        transformed,
        maximum_samples=int(config["kernel"]["maximum_reference_samples"]),
    )
    discrepancies = _proposal_discrepancies(
        real,
        simulated,
        config,
        center=center,
        whitener=whitener,
        bandwidth=bandwidth,
    )
    posterior_samples, posterior_summaries = _posterior_tables(discrepancies, config)
    replicate_summary = _replicate_summary(discrepancies, config)
    summary_support = _summary_support(real, simulated, config)
    sensitivity = _bandwidth_sensitivity(
        discrepancies,
        posterior_samples,
        config,
    )
    scenario = str(config["real_scenario_column"])
    status_counts = {
        str(key): int(value)
        for key, value in posterior_summaries["posterior_status"].value_counts().items()
    }
    estimator = str(config["kernel"]["estimator"])
    if estimator == "biased_mmd_squared_v_statistic":
        estimator_label = "biased_rbf_mmd_squared_v_statistic"
        discrepancy_label = "biased_mmd_squared_without_posthoc_clipping"
    else:
        estimator_label = "unbiased_rbf_mmd_squared"
        discrepancy_label = "maximum_of_unbiased_mmd_squared_and_zero"
    manifest = {
        "schema_version": 1,
        "calibration_id": config["name"],
        "method": "execution_bank_mmd_rejection_abc",
        "implementation_scope": config["implementation"],
        "stage": config.get("stage"),
        "posterior_claim": (
            "established"
            if set(posterior_summaries["posterior_status"]) == {"abc_posterior_established"}
            else "not_established"
        ),
        "posterior_status_counts": status_counts,
        "model": config["model"],
        "inferred_parameters": config["inferred_parameters"],
        "fixed_parameters": config.get("fixed_parameters") or [],
        "selection_metrics": config["selection_metrics"],
        "diagnostic_metrics": config.get("diagnostic_metrics") or [],
        "holdout_unit": config["holdout_unit"],
        "mmd_estimator": estimator_label,
        "abc_discrepancy": discrepancy_label,
        "abc_weighting": "epanechnikov",
        "abc_settings": config["abc"],
        "reference_transform": {
            "method": config["transform"]["method"],
            "metric_order": [str(metric["name"]) for metric in metrics],
            "center": center.tolist(),
            "covariance": covariance.tolist(),
            "whitener": whitener.tolist(),
            "relative_eigenvalue_floor": float(
                config["transform"]["relative_eigenvalue_floor"]
            ),
        },
        "kernel": {
            **config["kernel"],
            "reference_median_heuristic_bandwidth": bandwidth,
        },
        "real_scenarios": int(real[scenario].nunique()),
        "real_observations": len(real),
        "simulator_executions": int(simulated["execution_id"].nunique()),
        "simulator_observations": len(simulated),
        "unique_parameter_values": int(
            simulated[
                [str(parameter["name"]) for parameter in _inferred_parameters(config)]
            ].drop_duplicates().shape[0]
        ),
        "proposal_discrepancy_rows": len(discrepancies),
        "posterior_sample_rows": len(posterior_samples),
        "software_provenance": _software_provenance(),
        "source_sha256": {
            "real_observations": _sha256(real_path),
            "proposal_plan": _sha256(plan_path),
            "campaign_state": _sha256(campaign_path),
            "config": _sha256(config_path),
            "execution_observations": execution_hashes,
        },
        "quality_gates": {
            "require_true": config["execution"].get("require_true") or [],
            "require_false": config["execution"].get("require_false") or [],
            "minimum_samples_per_scenario": int(
                config["execution"]["minimum_samples_per_scenario"]
            ),
            "minimum_samples_per_execution": int(
                config["execution"]["minimum_samples_per_execution"]
            ),
        },
        "limitations": [
            "a pilot-only or failed inference gate does not establish an ABC posterior",
            "each complete execution is one stochastic simulator draw",
            "consecutive one-second observations are not treated as independent runs",
            "the first parameter vector keeps TDL-B and all non-ploss controls fixed",
            "real SNR and OAI SS-SINR remain definition-distinct diagnostics",
            "distributional similarity does not prove identical propagation physics",
            "posterior-predictive validation requires new held-out executions",
            "traffic, topology, temporal, and channel-family generalization are separate stages",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_csv(discrepancies, staging / "proposal_discrepancies.csv")
        _write_csv(replicate_summary, staging / "replicate_discrepancy_summary.csv")
        _write_csv(posterior_samples, staging / "posterior_samples.csv")
        _write_csv(posterior_summaries, staging / "posterior_summaries.csv")
        _write_csv(summary_support, staging / "model_support_diagnostics.csv")
        _write_csv(sensitivity, staging / "bandwidth_sensitivity.csv")
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
        "posterior_claim": manifest["posterior_claim"],
        "scenarios": manifest["real_scenarios"],
        "executions": manifest["simulator_executions"],
        "unique_parameter_values": manifest["unique_parameter_values"],
        "files": len(checksums) + 1,
    }


def build_posterior_predictive_plan(
    *,
    calibration_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    calibration_dir = Path(calibration_dir).resolve()
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    validate_mmd_abc_config(config)
    manifest = _read_json(calibration_dir / "calibration_manifest.json")
    if manifest.get("calibration_id") != config["name"]:
        raise ValueError("calibration output does not match configuration")
    if manifest.get("posterior_claim") != "established":
        raise ValueError("posterior-predictive planning requires an established posterior")
    samples = pd.read_csv(calibration_dir / "posterior_samples.csv")
    scenario = str(config["real_scenario_column"])
    probabilities = [
        float(value) for value in config["validation"]["posterior_quantiles"]
    ]
    repetitions = int(config["validation"]["independent_repetitions_per_candidate"])
    parameters = [str(parameter["name"]) for parameter in _inferred_parameters(config)]
    fixed_controls = {
        parameter["name"]: parameter["value"]
        for parameter in _fixed_parameters(config)
        if bool(parameter.get("include_in_execution_controls", False))
    }
    candidates = []
    for scenario_id, group in samples.groupby(scenario, sort=True):
        group = group.sort_values("abc_rank").reset_index(drop=True)
        weights = group["abc_weight"].to_numpy(dtype=float)
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        for probability in probabilities:
            index = min(int(np.searchsorted(cumulative, probability, side="left")), len(group) - 1)
            row = group.iloc[index]
            theta = {parameter: float(row[parameter]) for parameter in parameters}
            candidates.append({
                "target_scenario": str(scenario_id),
                "posterior_quantile": probability,
                "theta": theta,
            })
    points = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_id = f"validation-candidate-{candidate_index:03d}"
        for repetition in range(1, repetitions + 1):
            points.append({
                "point_id": f"{candidate_id}-r{repetition:02d}",
                "candidate_id": candidate_id,
                "target_scenario": candidate["target_scenario"],
                "posterior_quantile": candidate["posterior_quantile"],
                "theta": candidate["theta"],
                "controls": {**candidate["theta"], **fixed_controls},
                "model_family": config["model"]["family"],
                "run_seconds": float(config["execution"]["run_seconds"]),
                "holdout_role": "posterior_predictive_validation",
            })
    return {
        "schema_version": 1,
        "plan_id": f"{config['name']}_posterior_predictive_validation",
        "source_calibration_id": manifest["calibration_id"],
        "source_calibration_manifest_sha256": _sha256(
            calibration_dir / "calibration_manifest.json"
        ),
        "source_posterior_samples_sha256": _sha256(
            calibration_dir / "posterior_samples.csv"
        ),
        "model": config["model"],
        "holdout_unit": "complete_execution",
        "candidate_count": len(candidates),
        "execution_count": len(points),
        "candidates": candidates,
        "points": points,
        "rule": {
            "posterior_quantiles": probabilities,
            "independent_repetitions_per_candidate": repetitions,
            "reuse_calibration_transform_and_bandwidth": True,
            "calibration_executions_may_not_be_reclassified_as_validation": True,
        },
    }


def write_posterior_predictive_plan(
    *,
    calibration_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"posterior-predictive plan already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(
        output,
        build_posterior_predictive_plan(
            calibration_dir=calibration_dir,
            config_path=config_path,
        ),
    )
