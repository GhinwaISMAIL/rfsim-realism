from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfsim_realism.upv_phase3c12 import analyze_split_timing


def _write_inputs(tmp_path: Path, *, split_delta: float = 0) -> tuple[Path, Path, Path]:
    runtime = []
    split = []
    for second in range(120):
        stages = [5.0, 10.0, 20.0, 300.0, 5.0]
        accounted = sum(stages)
        total = accounted + 1.0
        runtime.append(
            {
                "role": "server",
                "elapsed_second": second,
                "channel_processing_us": total,
            }
        )
        split.append(
            {
                "role": "server",
                "elapsed_second": second,
                "total_us": total + split_delta,
                "cirdb_update_us": stages[0],
                "preparation_us": stages[1],
                "convolution_us": stages[2],
                "shared_write_us": stages[3],
                "history_copy_us": stages[4],
                "accounted_us": accounted,
                "residual_us": 1.0 + split_delta,
            }
        )
    runtime_path = tmp_path / "runtime.json"
    split_path = tmp_path / "split.json"
    spec_path = tmp_path / "spec.json"
    runtime_path.write_text(json.dumps(runtime))
    split_path.write_text(json.dumps(split))
    spec_path.write_text(
        json.dumps(
            {
                "stage": "phase_3c12_split_timing_analysis_specification",
                "inputs": {"phase3c10_reference_total_median_us": 341.0},
                "validation": {
                    "minimum_rows": 120,
                    "required_role": "server",
                    "maximum_absolute_runtime_total_difference_us": 0.001,
                    "maximum_absolute_component_accounting_difference_us": 0.01,
                    "maximum_absolute_total_accounting_difference_us": 0.01,
                    "maximum_absolute_residual_median_us": 5,
                    "maximum_absolute_residual_p95_us": 10,
                },
                "point_statistics": {"dominant_stage_threshold": 0.5},
                "bootstrap": {
                    "block_lengths_rows": [5, 10, 20],
                    "repetitions": 50,
                    "base_seed": 20260825,
                },
            }
        )
    )
    return runtime_path, split_path, spec_path


def test_split_analyzer_accepts_additive_shared_write_dominance(tmp_path: Path) -> None:
    runtime, split, spec = _write_inputs(tmp_path)

    result = analyze_split_timing(
        runtime_path=runtime,
        split_path=split,
        specification_path=spec,
    )

    assert result["validation"]["telemetry_validation_pass"] is True
    assert result["dominant_stage"] == "shared_write_us"
    assert result["dominant_stage_fraction"] == pytest.approx(300 / 341)
    assert result["decision_branch"] == "shared_write_dominant"
    assert len(result["bootstrap"]) == 3


def test_split_analyzer_rejects_runtime_total_mismatch(tmp_path: Path) -> None:
    runtime, split, spec = _write_inputs(tmp_path, split_delta=0.01)

    result = analyze_split_timing(
        runtime_path=runtime,
        split_path=split,
        specification_path=spec,
    )

    assert result["validation"]["runtime_total_identity_pass"] is False
    assert result["decision_branch"] == "telemetry_validation_failure"


def test_split_analyzer_rejects_misaligned_keys(tmp_path: Path) -> None:
    runtime, split, spec = _write_inputs(tmp_path)
    rows = json.loads(split.read_text())
    rows[2]["elapsed_second"] = 999
    split.write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="keys are not identical"):
        analyze_split_timing(
            runtime_path=runtime,
            split_path=split,
            specification_path=spec,
        )
