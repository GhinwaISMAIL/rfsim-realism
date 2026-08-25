from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

STAGES = (
    "cirdb_update_us",
    "preparation_us",
    "convolution_us",
    "shared_write_us",
    "history_copy_us",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": _quantile(values, 0.95),
        "maximum": float(np.max(values)),
    }


def _require_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label} rows must be JSON objects")
    return value


def _numeric_column(rows: list[dict[str, Any]], field: str, label: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in {label}") from error
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite {field} in {label}")
    return values


def _bootstrap_intervals(
    matrix: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    base_seed: int,
) -> dict[str, Any]:
    n = len(matrix)
    if not 0 < block_length <= n:
        raise ValueError("bootstrap block length must be between one and n")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, block_length]))
    offsets = np.arange(block_length)
    block_count = math.ceil(n / block_length)
    results = np.empty((repetitions, 2 + 2 * len(STAGES)), dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = matrix[indices]
        total_median = float(np.median(sample[:, 0]))
        residual_median = float(np.median(np.abs(sample[:, 1])))
        stage_medians = np.median(sample[:, 2:], axis=0)
        results[repetition, 0] = total_median
        results[repetition, 1] = residual_median
        results[repetition, 2 : 2 + len(STAGES)] = stage_medians
        results[repetition, 2 + len(STAGES) :] = stage_medians / total_median

    def interval(column: int) -> list[float]:
        return [
            _quantile(results[:, column], 0.025),
            _quantile(results[:, column], 0.975),
        ]

    return {
        "block_length_rows": block_length,
        "repetitions": repetitions,
        "seed_sequence": [base_seed, block_length],
        "median_split_total_us_95_interval": interval(0),
        "median_absolute_residual_us_95_interval": interval(1),
        "stages": {
            stage: {
                "median_us_95_interval": interval(2 + index),
                "median_fraction_95_interval": interval(2 + len(STAGES) + index),
            }
            for index, stage in enumerate(STAGES)
        },
    }


def analyze_split_timing(
    *,
    runtime_path: str | Path,
    split_path: str | Path,
    specification_path: str | Path,
) -> dict[str, Any]:
    runtime_file = Path(runtime_path).resolve()
    split_file = Path(split_path).resolve()
    specification_file = Path(specification_path).resolve()
    for path in (runtime_file, split_file, specification_file):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe input: {path}")

    runtime_rows = _require_rows(_read_json(runtime_file), "runtime")
    split_rows = _require_rows(_read_json(split_file), "split")
    specification = _read_json(specification_file)
    if not isinstance(specification, dict):
        raise ValueError("analysis specification must be a JSON object")
    if specification.get("stage") != "phase_3c12_split_timing_analysis_specification":
        raise ValueError("unexpected analysis specification stage")

    validation_spec = specification["validation"]
    minimum_rows = int(validation_spec["minimum_rows"])
    if len(runtime_rows) < minimum_rows or len(split_rows) < minimum_rows:
        raise ValueError("insufficient runtime or split rows")
    if len(runtime_rows) != len(split_rows):
        raise ValueError("runtime and split row counts differ")

    def key(row: dict[str, Any]) -> tuple[str, int]:
        try:
            return str(row["role"]), int(row["elapsed_second"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid runtime/split key") from error

    runtime_keys = [key(row) for row in runtime_rows]
    split_keys = [key(row) for row in split_rows]
    if runtime_keys != split_keys:
        raise ValueError("runtime and split row keys are not identical in order")
    if len(set(runtime_keys)) != len(runtime_keys):
        raise ValueError("runtime/split row keys are not unique")
    expected_role = str(validation_spec["required_role"])
    if any(role != expected_role for role, _ in runtime_keys):
        raise ValueError("unexpected runtime/split role")
    elapsed = np.asarray([second for _, second in runtime_keys], dtype=int)
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("elapsed seconds are not strictly increasing")

    runtime_total = _numeric_column(runtime_rows, "channel_processing_us", "runtime")
    split_total = _numeric_column(split_rows, "total_us", "split")
    accounted = _numeric_column(split_rows, "accounted_us", "split")
    residual = _numeric_column(split_rows, "residual_us", "split")
    stage_values = {stage: _numeric_column(split_rows, stage, "split") for stage in STAGES}
    stage_matrix = np.column_stack([stage_values[stage] for stage in STAGES])
    if np.any(stage_matrix < 0):
        raise ValueError("a split stage duration is negative")

    runtime_total_error = np.abs(runtime_total - split_total)
    component_error = np.abs(np.sum(stage_matrix, axis=1) - accounted)
    accounting_error = np.abs(accounted + residual - split_total)
    runtime_total_pass = bool(
        np.max(runtime_total_error)
        <= float(validation_spec["maximum_absolute_runtime_total_difference_us"])
    )
    component_pass = bool(
        np.max(component_error)
        <= float(validation_spec["maximum_absolute_component_accounting_difference_us"])
    )
    accounting_pass = bool(
        np.max(accounting_error)
        <= float(validation_spec["maximum_absolute_total_accounting_difference_us"])
    )

    total_summary = _summary(split_total)
    residual_summary = _summary(residual)
    absolute_residual_summary = _summary(np.abs(residual))
    stage_summaries = {stage: _summary(values) for stage, values in stage_values.items()}
    total_median = total_summary["median"]
    stage_fractions = {
        stage: summary["median"] / total_median for stage, summary in stage_summaries.items()
    }
    dominant_stage = max(stage_fractions, key=stage_fractions.__getitem__)

    residual_gate = bool(
        absolute_residual_summary["median"]
        <= float(validation_spec["maximum_absolute_residual_median_us"])
        and absolute_residual_summary["p95"]
        <= float(validation_spec["maximum_absolute_residual_p95_us"])
    )
    reference = float(specification["inputs"]["phase3c10_reference_total_median_us"])
    total_ratio = total_median / reference
    total_ratio_pass = 0.8 <= total_ratio <= 1.2
    dominant_fraction_pass = bool(
        stage_fractions[dominant_stage]
        >= float(specification["point_statistics"]["dominant_stage_threshold"])
    )
    telemetry_validation_pass = bool(
        runtime_total_pass and component_pass and accounting_pass and residual_gate
    )

    bootstrap_spec = specification["bootstrap"]
    bootstrap_matrix = np.column_stack(
        [split_total, residual, *(stage_values[stage] for stage in STAGES)]
    )
    bootstrap = [
        _bootstrap_intervals(
            bootstrap_matrix,
            block_length=int(block_length),
            repetitions=int(bootstrap_spec["repetitions"]),
            base_seed=int(bootstrap_spec["base_seed"]),
        )
        for block_length in bootstrap_spec["block_lengths_rows"]
    ]

    if not telemetry_validation_pass:
        branch = "telemetry_validation_failure"
    elif not total_ratio_pass or not dominant_fraction_pass:
        branch = "no_dominant_stage_or_total_mismatch"
    else:
        branch = {
            "shared_write_us": "shared_write_dominant",
            "cirdb_update_us": "cirdb_update_dominant",
            "convolution_us": "convolution_dominant_only_live",
            "preparation_us": "preparation_or_history_dominant",
            "history_copy_us": "preparation_or_history_dominant",
        }[dominant_stage]

    return {
        "schema_version": 1,
        "rows": len(split_rows),
        "first_elapsed_second": int(elapsed[0]),
        "last_elapsed_second": int(elapsed[-1]),
        "input_sha256": {
            "runtime": _sha256(runtime_file),
            "split": _sha256(split_file),
            "analysis_specification": _sha256(specification_file),
        },
        "validation": {
            "row_count_pass": True,
            "keys_unique_identical_and_ordered": True,
            "required_role_only": True,
            "elapsed_second_strictly_increasing": True,
            "all_numeric_values_finite": True,
            "all_stage_durations_nonnegative": True,
            "maximum_absolute_runtime_total_difference_us": float(np.max(runtime_total_error)),
            "runtime_total_identity_pass": runtime_total_pass,
            "maximum_absolute_component_accounting_difference_us": float(np.max(component_error)),
            "component_accounting_pass": component_pass,
            "maximum_absolute_total_accounting_difference_us": float(np.max(accounting_error)),
            "total_accounting_pass": accounting_pass,
            "absolute_residual_gate_pass": residual_gate,
            "telemetry_validation_pass": telemetry_validation_pass,
        },
        "timing_us": {
            "split_total": total_summary,
            "residual": residual_summary,
            "absolute_residual": absolute_residual_summary,
            "stages": stage_summaries,
        },
        "stage_median_fractions": stage_fractions,
        "dominant_stage": dominant_stage,
        "dominant_stage_fraction": stage_fractions[dominant_stage],
        "dominant_stage_fraction_pass": dominant_fraction_pass,
        "split_total_median_ratio_to_phase3c10": total_ratio,
        "split_total_median_ratio_pass": total_ratio_pass,
        "bootstrap": bootstrap,
        "decision_branch": branch,
        "uncertainty_interpretation": (
            "single-traversal conditional only; not between-run uncertainty"
        ),
    }


def write_split_timing_analysis(
    *,
    runtime_path: str | Path,
    split_path: str | Path,
    specification_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis result: {output}")
    result = analyze_split_timing(
        runtime_path=runtime_path,
        split_path=split_path,
        specification_path=specification_path,
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
