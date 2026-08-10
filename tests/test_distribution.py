import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rfsim_realism.distribution import (
    _joint_rf_distribution,
    run_distribution_analysis,
    validate_distribution_config,
)
from rfsim_realism.ucc_static import build_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_trace(dataset: Path, *, skip_second: int | None = None) -> Path:
    path = dataset / "Amazon_Prime" / "Static" / "test-show" / "trace.csv"
    path.parent.mkdir(parents=True)
    start = datetime(2026, 1, 1, 12, 0, 0)
    states = [
        (-96.0, -11.0, 2.0),
        (-96.0, -11.0, 2.0),
        (-94.0, -10.0, 3.0),
        (-94.0, -10.0, 3.0),
        (-85.0, -5.0, 8.0),
        (-85.0, -5.0, 8.0),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Timestamp",
                "NetworkMode",
                "CellID",
                "RSRP",
                "RSRQ",
                "SNR",
                "CQI",
                "RSSI",
                "Speed",
                "State",
            ],
        )
        writer.writeheader()
        for second, (rsrp, rsrq, snr) in enumerate(states):
            if second == skip_second:
                continue
            writer.writerow(
                {
                    "Timestamp": (start + timedelta(seconds=second)).strftime("%Y.%m.%d_%H.%M.%S"),
                    "NetworkMode": "5G",
                    "CellID": "cell-1",
                    "RSRP": rsrp,
                    "RSRQ": rsrq,
                    "SNR": snr,
                    "CQI": 10,
                    "RSSI": -70,
                    "Speed": 0,
                    "State": "CONNECTED",
                }
            )
    return path


def _write_mapping(mapping_dir: Path) -> None:
    mapping_dir.mkdir()
    pd.DataFrame(
        [
            {
                "applied_ploss": -2.0,
                "applied_noise_power_dB": -2.0,
                "execution_count": 2,
                "ss_rsrp_dbm_segment_mean_mean": -96.0,
                "ss_rsrq_db_segment_mean_mean": -11.0,
                "ss_sinr_db_segment_mean_mean": 2.5,
            },
            {
                "applied_ploss": -1.0,
                "applied_noise_power_dB": -1.0,
                "execution_count": 2,
                "ss_rsrp_dbm_segment_mean_mean": -94.0,
                "ss_rsrq_db_segment_mean_mean": -10.0,
                "ss_sinr_db_segment_mean_mean": 3.5,
            },
        ]
    ).to_csv(mapping_dir / "state_mapping.csv", index=False)
    _write_json(
        mapping_dir / "mapping_manifest.json",
        {
            "schema_version": 1,
            "mapping_id": "test-safe-mapping",
            "model_kind": "empirical_safe_state_lookup",
            "candidate_policy": "observed_safe_states_only",
        },
    )
    _write_json(
        mapping_dir / "SHA256SUMS.json",
        {
            name: _sha256(mapping_dir / name)
            for name in ("mapping_manifest.json", "state_mapping.csv")
        },
    )


def _config() -> dict:
    return {
        "schema_version": 1,
        "name": "test_rf_distribution",
        "selection": {
            "require_dynamic_replay_eligible": True,
            "applications": [],
            "trace_ids": [],
        },
        "catalog": {
            "scenario_unit": "selected_trace_window",
            "trace_weighting": "equal_trace",
            "joint_metrics": ["RSRP", "RSRQ", "SNR"],
        },
        "mapping": {
            "mode": "optional_annotation",
            "policy": "nearest_observed_safe_state",
            "allow_extrapolation": False,
            "rsrp_absolute_tolerance_db": 3.0,
            "rsrq_absolute_tolerance_db": 2.0,
            "minimum_representable_fraction": 0.8,
        },
        "temporal": {
            "transition_max_gap_seconds": 1,
            "missing_seconds": "preserve",
        },
    }


def _fixture(tmp_path: Path, *, skip_second: int | None = None) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    _write_trace(dataset, skip_second=skip_second)
    manifest = build_manifest(dataset, policy={"window_seconds": 6})
    manifest_path = tmp_path / "ucc.json"
    _write_json(manifest_path, manifest)
    mapping_dir = tmp_path / "mapping"
    _write_mapping(mapping_dir)
    config_path = tmp_path / "distribution.yaml"
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False))
    return {
        "dataset": dataset,
        "manifest": manifest_path,
        "mapping": mapping_dir,
        "config": config_path,
    }


def _run(paths: dict[str, Path], output: Path, *, with_mapping: bool = False) -> dict:
    return run_distribution_analysis(
        dataset=paths["dataset"],
        manifest_path=paths["manifest"],
        config_path=paths["config"],
        output_dir=output,
        mapping_dir=paths["mapping"] if with_mapping else None,
    )


def test_distribution_builds_replay_ready_joint_scenarios(tmp_path):
    paths = _fixture(tmp_path)

    result = _run(paths, tmp_path / "output")

    assert result["status"] == "catalog_ready"
    assert result["scenarios"] == 1
    assert result["observations"] == 6
    assert result["joint_states"] == 3
    assert result["rfsim_mapping_applied"] is False
    assert result["rfsim_representable_fraction"] is None
    manifest = json.loads((tmp_path / "output" / "distribution_manifest.json").read_text())
    assert manifest["probabilities"]["metrics_are_sampled_jointly"] is True
    assert manifest["temporal"]["transitions"] == 5
    assert manifest["temporal"]["sequence_runs"] == 3
    joint = pd.read_csv(tmp_path / "output" / "joint_rf_distribution.csv")
    assert joint["pooled_time_probability"].sum() == pytest.approx(1)
    assert joint["equal_trace_probability"].sum() == pytest.approx(1)
    assert set(joint["target_snr_db"]) == {2.0, 3.0, 8.0}
    sequences = pd.read_csv(tmp_path / "output" / "scenario_sequences.csv")
    assert sequences["duration_seconds"].tolist() == [2, 2, 2]
    catalog = json.loads((tmp_path / "output" / "scenario_catalog.json").read_text())
    assert catalog["measurement_contract"]["independent_metric_sampling"] is False
    assert catalog["replay_contract"]["reuse_across_ue_counts_and_traffic_profiles"]


def test_optional_mapping_reports_support_without_changing_catalog(tmp_path):
    paths = _fixture(tmp_path)

    result = _run(paths, tmp_path / "output", with_mapping=True)

    assert result["status"] == "catalog_ready"
    assert result["rfsim_mapping_applied"] is True
    assert result["rfsim_representable_fraction"] == pytest.approx(4 / 6)
    manifest = json.loads((tmp_path / "output" / "distribution_manifest.json").read_text())
    assert manifest["rfsim_support"]["status"] == "insufficient"
    assert manifest["rfsim_support"]["unrepresentable_observations"] == 2
    uncovered = pd.read_csv(tmp_path / "output" / "uncovered_real_rf_states.csv")
    assert uncovered["observations"].sum() == 2
    transitions = pd.read_csv(tmp_path / "output" / "transition_distribution.csv")
    assert "from_mapped_control_state_id" not in transitions


def test_distribution_is_deterministic_and_does_not_cross_missing_seconds(tmp_path):
    paths = _fixture(tmp_path, skip_second=2)

    _run(paths, tmp_path / "output-a")
    _run(paths, tmp_path / "output-b")

    assert json.loads((tmp_path / "output-a" / "SHA256SUMS.json").read_text()) == (
        json.loads((tmp_path / "output-b" / "SHA256SUMS.json").read_text())
    )
    manifest = json.loads((tmp_path / "output-a" / "distribution_manifest.json").read_text())
    assert manifest["selection"]["complete_rf_observations"] == 5
    assert manifest["temporal"]["transitions"] == 3
    sequences = pd.read_csv(tmp_path / "output-a" / "scenario_sequences.csv")
    assert sequences["duration_seconds"].tolist() == [2, 1, 2]
    assert sequences["gap_before_seconds"].tolist() == [0, 1, 0]


def test_distribution_rejects_extrapolation():
    config = _config()
    config["mapping"]["allow_extrapolation"] = True

    with pytest.raises(ValueError, match="reject extrapolation"):
        validate_distribution_config(config)


def test_equal_trace_probability_does_not_overweight_longer_traces():
    observations = pd.DataFrame(
        [
            {
                "trace_id": "long",
                "real_rf_state_id": "state-a",
                "target_rsrp_dbm": -96.0,
                "target_rsrq_db": -11.0,
                "target_snr_db": 2.0,
            },
            {
                "trace_id": "long",
                "real_rf_state_id": "state-a",
                "target_rsrp_dbm": -96.0,
                "target_rsrq_db": -11.0,
                "target_snr_db": 2.0,
            },
            {
                "trace_id": "long",
                "real_rf_state_id": "state-a",
                "target_rsrp_dbm": -96.0,
                "target_rsrq_db": -11.0,
                "target_snr_db": 2.0,
            },
            {
                "trace_id": "short",
                "real_rf_state_id": "state-b",
                "target_rsrp_dbm": -85.0,
                "target_rsrq_db": -5.0,
                "target_snr_db": 8.0,
            },
        ]
    )

    distribution = _joint_rf_distribution(observations, trace_count=2).set_index("real_rf_state_id")

    assert distribution.loc["state-a", "pooled_time_probability"] == pytest.approx(0.75)
    assert distribution.loc["state-b", "pooled_time_probability"] == pytest.approx(0.25)
    assert distribution.loc["state-a", "equal_trace_probability"] == pytest.approx(0.5)
    assert distribution.loc["state-b", "equal_trace_probability"] == pytest.approx(0.5)
