import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rfsim_realism.mapping import run_static_mapping, validate_mapping_config


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_config() -> dict:
    return {
        "schema_version": 1,
        "name": "test_static_mapping",
        "model_type": "AWGN",
        "direction": "dl",
        "conditional_pre_run_inputs": [
            "applied_ploss",
            "applied_noise_power_dB",
        ],
        "pre_run_context": ["app_mix", "designed_offered_mbps"],
        "minimum_repetitions_per_state": 2,
        "candidate_policy": "observed_safe_states_only",
        "evaluation_policy": "leave_one_execution_out_within_control_state",
        "inverse_matching": {
            "selection_metrics": [
                {
                    "reference_metric": "RSRP",
                    "observed_metric": "ss_rsrp_dbm_segment_mean",
                    "tolerance": 3.0,
                },
                {
                    "reference_metric": "RSRQ",
                    "observed_metric": "ss_rsrq_db_segment_mean",
                    "tolerance": 2.0,
                },
            ],
            "diagnostic_metrics": [
                {
                    "reference_metric": "SNR",
                    "observed_metric": "ss_sinr_db_segment_mean",
                }
            ],
        },
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = [(0.0, -2.0), (-1.0, -1.0)]
    segment_rows = []
    packet_rows = []
    selection_points = []
    completed = {}
    for state_index, (ploss, noise) in enumerate(states):
        for repetition in (1, 2):
            execution_id = f"mgen-state{state_index}-r{repetition}"
            point_id = f"state{state_index}-r{repetition}"
            start = 1000.0 + state_index * 1000 + repetition * 200
            latency_values = [10.0 + state_index + repetition, 20.0 + state_index]
            latency_p95 = pd.Series(latency_values).quantile(0.95)
            received_mbps = 2000 * 8 / 180 / 1e6
            segment_rows.append({
                "execution_id": execution_id,
                "segment_id": f"{execution_id}:ue1:dl:0",
                "ue": "ue1",
                "direction": "dl",
                "parameter": "joint",
                "model_type": "AWGN",
                "model_name": "rfsimu_channel_enB0",
                "model_index": 0,
                "segment_start_utc": start,
                "segment_end_utc": start + 180,
                "duration_s": 180.0,
                "app_mix": '["youtube"]',
                "designed_offered_mbps": 0.024,
                "sent_packets": 3,
                "received_packets": 2,
                "lost_packets": 1,
                "loss_rate": 1 / 3,
                "received_mbps": received_mbps,
                "latency_samples": 2,
                "latency_ms_p95": latency_p95,
                "valid_clock_fraction": 1.0,
                "radio_samples": 180,
                "ue_radio_samples": 180,
                "radio_join_clock": "core_receipt_utc",
                "radio_clock_lag_warning": False,
                "split": "test" if execution_id == "mgen-state0-r1" else "train",
                "applied_ploss": ploss,
                "applied_noise_power_dB": noise,
                "ss_rsrp_dbm_segment_mean": -95.0 + ploss + repetition / 10,
                "ss_rsrq_db_segment_mean": -11.0 + noise / 10 + repetition / 100,
                "ss_sinr_db_segment_mean": 4.0 - state_index + repetition / 10,
                "controlled": True,
                "verified": True,
                "channel_agreement": True,
                "training_eligible": True,
                "model_mapping_valid": True,
                "packet_evidence": True,
                "ploss_verified": True,
                "ploss_agreement": True,
                "noise_power_dB_verified": True,
                "noise_power_dB_agreement": True,
                "ue_radio_clock_valid": True,
            })
            for packet_index, latency in enumerate([*latency_values, None]):
                received = latency is not None
                packet_rows.append({
                    "execution_id": execution_id,
                    "ue": "ue1",
                    "direction": "dl",
                    "sent_time_utc": start + 10 + packet_index,
                    "received": received,
                    "lost": not received,
                    "size_bytes": 1000,
                    "latency_ms": latency,
                    "packet_clock_valid": True,
                    "negative_latency": False,
                })
            packet_rows.append({
                "execution_id": execution_id,
                "ue": "ue1",
                "direction": "ul",
                "sent_time_utc": start + 10,
                "received": True,
                "lost": False,
                "size_bytes": 9000,
                "latency_ms": 9999.0,
                "packet_clock_valid": True,
                "negative_latency": False,
            })
            controls = {"ploss": ploss, "noise_power_dB": noise}
            selection_points.append({
                "point_id": point_id,
                "controls": controls,
                "repetition": repetition,
                "run_seconds": 180.0,
            })
            completed[point_id] = {
                "execution_id": execution_id,
                "controls": controls,
            }
    _write_json(dataset / "dataset_manifest.json", {
        "schema_version": 2,
        "split_unit": "execution_id",
    })
    pd.DataFrame(segment_rows).to_parquet(
        dataset / "segment_training_table.parquet", index=False
    )
    pd.DataFrame(packet_rows).to_parquet(dataset / "packet_outcomes.parquet", index=False)
    _write_json(dataset / "SHA256SUMS.json", {
        name: _sha256(dataset / name)
        for name in (
            "dataset_manifest.json",
            "packet_outcomes.parquet",
            "segment_training_table.parquet",
        )
    })
    selection = tmp_path / "selection.json"
    _write_json(selection, {
        "schema_version": 1,
        "campaign": "test_safe_selection",
        "points": selection_points,
    })
    campaign = tmp_path / "campaign.json"
    _write_json(campaign, {
        "schema_version": 1,
        "campaign": "test_campaign",
        "completed": completed,
        "failures": [],
    })
    ucc = tmp_path / "ucc.json"
    _write_json(ucc, {
        "schema_version": 1,
        "traces": [{
            "trace_id": "steady-anchor",
            "classification": "steady_anchor",
            "app": "Amazon_Prime",
            "content": "test-content",
            "selected_window": {
                "metrics": {
                    "RSRP": {"p50": -96.0},
                    "RSRQ": {"p50": -11.0},
                    "SNR": {"p50": 3.0},
                }
            },
        }],
    })
    contract = tmp_path / "contract.json"
    _write_json(contract, {
        "schema_version": 1,
        "split_unit": "source_trace_and_execution",
    })
    config = tmp_path / "mapping.json"
    _write_json(config, _mapping_config())
    return {
        "dataset": dataset,
        "selection": selection,
        "campaign": campaign,
        "ucc": ucc,
        "contract": contract,
        "config": config,
        "output": tmp_path / "mapping-output",
    }


def test_mapping_rejects_post_run_radio_as_input():
    config = _mapping_config()
    config["conditional_pre_run_inputs"].append("ss_rsrp_dbm_segment_mean")

    with pytest.raises(ValueError, match="verified RFsim controls"):
        validate_mapping_config(config)


def test_static_mapping_accepts_configured_tdl_family(tmp_path):
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text())
    config["model_type"] = "TDL_B"
    _write_json(paths["config"], config)
    segments = pd.read_parquet(paths["dataset"] / "segment_training_table.parquet")
    segments["model_type"] = "TDL_B"
    segments.to_parquet(paths["dataset"] / "segment_training_table.parquet", index=False)
    checksums = json.loads((paths["dataset"] / "SHA256SUMS.json").read_text())
    checksums["segment_training_table.parquet"] = _sha256(
        paths["dataset"] / "segment_training_table.parquet"
    )
    _write_json(paths["dataset"] / "SHA256SUMS.json", checksums)

    result = run_static_mapping(
        dataset_dir=paths["dataset"],
        selection_manifest=paths["selection"],
        campaign_state=paths["campaign"],
        ucc_manifest=paths["ucc"],
        comparison_contract=paths["contract"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    assert result["states"] == 2
    manifest = json.loads((paths["output"] / "mapping_manifest.json").read_text())
    assert manifest["model_type"] == "TDL_B"


def test_static_mapping_recomputes_packets_and_holds_out_executions(tmp_path):
    paths = _fixture(tmp_path)

    result = run_static_mapping(
        dataset_dir=paths["dataset"],
        selection_manifest=paths["selection"],
        campaign_state=paths["campaign"],
        ucc_manifest=paths["ucc"],
        comparison_contract=paths["contract"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    assert result == {
        "output": str(paths["output"].resolve()),
        "executions": 4,
        "states": 2,
        "anchors": 1,
        "files": 7,
    }
    executions = pd.read_csv(paths["output"] / "execution_metrics.csv")
    assert len(executions) == 4
    assert executions["direction"].eq("dl").all()
    assert executions["sent_packets"].eq(3).all()
    assert executions["received_packets"].eq(2).all()
    assert executions["lost_packets"].eq(1).all()
    assert executions["stored_latency_ms_p95_delta"].eq(0).all()
    predictions = pd.read_csv(
        paths["output"] / "cross_execution_predictions.csv"
    )
    assert len(predictions) == 4
    assert predictions["training_execution_ids"].str.contains("mgen-").all()
    assert not predictions["training_execution_ids"].str.contains(
        predictions.iloc[0]["execution_id"], regex=False
    ).iloc[0]


def test_static_mapping_ranks_only_observed_states_and_checksums_outputs(tmp_path):
    paths = _fixture(tmp_path)
    run_static_mapping(
        dataset_dir=paths["dataset"],
        selection_manifest=paths["selection"],
        campaign_state=paths["campaign"],
        ucc_manifest=paths["ucc"],
        comparison_contract=paths["contract"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    candidates = pd.read_csv(paths["output"] / "anchor_candidates.csv")
    assert len(candidates) == 2
    assert set(candidates["primary_rank"]) == {1, 2}
    assert set(zip(
        candidates["candidate_ploss"],
        candidates["candidate_noise_power_dB"],
        strict=True,
    )) == {(0.0, -2.0), (-1.0, -1.0)}
    manifest = json.loads((paths["output"] / "mapping_manifest.json").read_text())
    assert manifest["post_run_radio_in_input_matrix"] is False
    assert manifest["selected_executions"] == 4
    assert manifest["retained_control_states"] == 2
    checksums = json.loads((paths["output"] / "SHA256SUMS.json").read_text())
    assert len(checksums) == 6
    for name, expected in checksums.items():
        assert _sha256(paths["output"] / name) == expected
