import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rfsim_realism.mmd_abc import (
    build_mmd_abc_plan,
    build_posterior_predictive_plan,
    median_heuristic_bandwidth,
    reference_whitener,
    run_mmd_abc,
    unbiased_rbf_mmd2,
    validate_mmd_abc_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _config(*, pilot_only: bool = False) -> dict:
    return {
        "schema_version": 1,
        "name": "synthetic_tdl_b_mmd_abc",
        "stage": "test",
        "implementation": "execution_bank_rejection_abc",
        "seed": 17,
        "holdout_unit": "complete_execution",
        "model": {
            "family": "TDL_B",
            "topology": {"gnb_count": 1, "ue_count": 1, "ue_mobility": "stationary"},
        },
        "inferred_parameters": [{
            "name": "ploss",
            "simulated_column": "dl_ploss",
            "verification_tolerance": 1e-9,
            "prior": {"distribution": "uniform", "lower": -2.0, "upper": 0.0},
            "proposal_values": [-2.0, -1.0, 0.0],
        }],
        "fixed_parameters": [{
            "name": "noise_power_dB",
            "value": -30.0,
            "simulated_column": "dl_noise_power_dB",
            "verification_tolerance": 1e-9,
            "include_in_execution_controls": True,
        }],
        "real_scenario_column": "trace_id",
        "real_context_columns": ["app", "content"],
        "selection_metrics": [
            {
                "name": "RSRP",
                "real_column": "target_rsrp_dbm",
                "simulated_column": "ss_rsrp_dbm",
            },
            {
                "name": "RSRQ",
                "real_column": "target_rsrq_db",
                "simulated_column": "ss_rsrq_db",
            },
        ],
        "diagnostic_metrics": [{
            "name": "SNR_PROXY",
            "real_column": "target_snr_db",
            "simulated_column": "ss_sinr_db",
        }],
        "transform": {
            "method": "pooled_real_covariance_whitening",
            "relative_eigenvalue_floor": 1e-8,
        },
        "kernel": {
            "name": "rbf",
            "estimator": "unbiased_mmd_squared",
            "bandwidth_source": "pooled_real_reference_median_heuristic",
            "bandwidth_multipliers": [0.5, 1.0, 2.0],
            "maximum_reference_samples": 64,
            "maximum_samples_per_distribution": 64,
        },
        "execution": {
            "independent_repetitions": 2,
            "run_seconds": 180,
            "observation_filename": "ue_second_features.parquet",
            "execution_id_column": "execution_id",
            "model_column": "dl_model_type",
            "minimum_samples_per_scenario": 4,
            "minimum_samples_per_execution": 4,
            "allow_partial_bank": False,
            "require_true": [
                "channel_schedule_enabled",
                "channel_state_success",
                "channel_verified",
            ],
            "require_false": ["channel_transition_partial"],
        },
        "abc": {
            "acceptance_fraction": 0.5,
            "minimum_total_simulations": 6,
            "minimum_accepted_samples": 3,
            "minimum_unique_parameter_values": 2,
            "minimum_effective_sample_size": 1.5,
            "boundary_fraction": 0.1,
            "pilot_only": pilot_only,
        },
        "diagnostics": {"wasserstein_quantiles": 21},
        "validation": {
            "posterior_quantiles": [0.1, 0.5, 0.9],
            "independent_repetitions_per_candidate": 2,
        },
    }


def _fixture(tmp_path: Path, *, pilot_only: bool = False) -> dict[str, Path]:
    config = _config(pilot_only=pilot_only)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = build_mmd_abc_plan(config_path)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    executions = tmp_path / "executions"
    completed = {}
    centers = {-2.0: (-102.0, -12.0), -1.0: (-98.5, -11.0), 0.0: (-95.0, -10.0)}
    offsets = np.array([-0.45, -0.2, -0.05, 0.1, 0.25, 0.4])
    for point in plan["points"]:
        execution_id = f"execution-{point['point_id']}"
        directory = executions / execution_id
        directory.mkdir(parents=True)
        ploss = float(point["theta"]["ploss"])
        rsrp, rsrq = centers[ploss]
        repetition_shift = 0.04 * int(point["repetition"])
        frame = pd.DataFrame({
            "execution_id": execution_id,
            "dl_model_type": "TDL_B",
            "dl_ploss": ploss,
            "dl_noise_power_dB": -30.0,
            "ss_rsrp_dbm": rsrp + offsets + repetition_shift,
            "ss_rsrq_db": rsrq + offsets[::-1] / 3 - repetition_shift / 2,
            "ss_sinr_db": 6.0 + offsets,
            "channel_schedule_enabled": True,
            "channel_state_success": True,
            "channel_verified": True,
            "channel_transition_partial": False,
        })
        frame.to_parquet(directory / "ue_second_features.parquet", index=False)
        completed[point["point_id"]] = {
            "execution_id": execution_id,
            "controls": point["controls"],
        }
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, {"schema_version": 1, "completed": completed})
    real_rows = []
    for trace_id, app, (rsrp, rsrq) in [
        ("trace-weak", "video", centers[-2.0]),
        ("trace-strong", "download", centers[0.0]),
    ]:
        for index, offset in enumerate(offsets):
            real_rows.append({
                "trace_id": trace_id,
                "app": app,
                "content": f"content-{trace_id}",
                "observation_index": index,
                "target_rsrp_dbm": rsrp + offset,
                "target_rsrq_db": rsrq + offsets[::-1][index] / 3,
                "target_snr_db": 5.0 + offset,
            })
    real_path = tmp_path / "real.csv"
    pd.DataFrame(real_rows).to_csv(real_path, index=False)
    return {
        "config": config_path,
        "plan": plan_path,
        "campaign": campaign_path,
        "executions": executions,
        "real": real_path,
        "output": tmp_path / "output",
    }


def test_reference_whitening_and_median_bandwidth_are_data_derived():
    values = np.array([
        [-103.0, -15.0],
        [-100.0, -11.0],
        [-96.0, -14.0],
        [-92.0, -10.0],
    ])

    transformed, center, _, _ = reference_whitener(values)

    assert center.tolist() == pytest.approx(values.mean(axis=0).tolist())
    assert np.cov(transformed, rowvar=False) == pytest.approx(np.eye(2))
    assert median_heuristic_bandwidth(transformed) > 0


def test_unbiased_mmd_is_small_for_identical_empirical_samples():
    values = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 0.5]])

    value = unbiased_rbf_mmd2(values, values, bandwidth=1.0)

    assert value <= 0


def test_predeclared_pilot_plan_has_24_complete_execution_points():
    config = REPOSITORY / "configs" / "mmd_abc_tdl_b_ploss_pilot_v1.yaml"

    plan = build_mmd_abc_plan(config)

    assert plan["proposal_count"] == 8
    assert plan["execution_count"] == 24
    assert plan["points"][0]["controls"] == {"ploss": -10.0, "noise_power_dB": -30.0}
    assert plan["points"][-1]["controls"] == {"ploss": 0.0, "noise_power_dB": -30.0}
    assert plan["limitations"][1].startswith("the pilot is underpowered")


def test_config_rejects_family_inference_in_first_stage():
    config = _config()
    config["model"]["family"] = "TDL_A"

    with pytest.raises(ValueError, match="keep TDL_B fixed"):
        validate_mmd_abc_config(config)


def test_mmd_abc_builds_weighted_posterior_and_checksummed_outputs(tmp_path):
    paths = _fixture(tmp_path)

    result = run_mmd_abc(
        real_observations=paths["real"],
        executions_root=paths["executions"],
        proposal_plan=paths["plan"],
        campaign_state=paths["campaign"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    assert result == {
        "output": str(paths["output"].resolve()),
        "posterior_claim": "established",
        "scenarios": 2,
        "executions": 6,
        "unique_parameter_values": 3,
        "files": 8,
    }
    summaries = pd.read_csv(paths["output"] / "posterior_summaries.csv").set_index(
        "trace_id"
    )
    assert summaries.loc["trace-weak", "ploss_weighted_mean"] < -1.0
    assert summaries.loc["trace-strong", "ploss_weighted_mean"] > -1.0
    assert set(summaries["posterior_status"]) == {"abc_posterior_established"}
    manifest = json.loads((paths["output"] / "calibration_manifest.json").read_text())
    assert manifest["reference_transform"]["method"] == (
        "pooled_real_covariance_whitening"
    )
    assert manifest["kernel"]["reference_median_heuristic_bandwidth"] > 0
    assert manifest["posterior_claim"] == "established"
    checksums = json.loads((paths["output"] / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((paths["output"] / name).read_bytes()).hexdigest() == expected

    validation = build_posterior_predictive_plan(
        calibration_dir=paths["output"],
        config_path=paths["config"],
    )
    assert validation["candidate_count"] == 6
    assert validation["execution_count"] == 12
    assert all(
        point["holdout_role"] == "posterior_predictive_validation"
        for point in validation["points"]
    )


def test_pilot_only_bank_cannot_claim_an_established_posterior(tmp_path):
    paths = _fixture(tmp_path, pilot_only=True)

    result = run_mmd_abc(
        real_observations=paths["real"],
        executions_root=paths["executions"],
        proposal_plan=paths["plan"],
        campaign_state=paths["campaign"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    assert result["posterior_claim"] == "not_established"
    with pytest.raises(ValueError, match="established posterior"):
        build_posterior_predictive_plan(
            calibration_dir=paths["output"],
            config_path=paths["config"],
        )
