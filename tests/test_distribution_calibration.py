import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rfsim_realism.distribution_calibration import (
    quantile_wasserstein,
    rbf_mmd2,
    run_distribution_calibration,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    executions_root = tmp_path / "executions"
    points = []
    completed = {}
    for state_index, (ploss, noise, rsrp, rsrq) in enumerate([
        (-5.0, -30.0, -101.0, -10.5),
        (0.0, -10.0, -111.0, -14.0),
    ]):
        for repetition in (1, 2):
            execution_id = f"execution-{state_index}-{repetition}"
            point_id = f"state-{state_index}-r{repetition}"
            directory = executions_root / execution_id
            directory.mkdir(parents=True)
            offsets = np.array([-0.3, -0.1, 0.1, 0.3]) + repetition * 0.01
            pd.DataFrame({
                "execution_id": execution_id,
                "dl_model_type": "TDL_B",
                "dl_ploss": ploss,
                "dl_noise_power_dB": noise,
                "ss_rsrp_dbm": rsrp + offsets,
                "ss_rsrq_db": rsrq + offsets / 2,
                "ss_sinr_db": 8.0 - state_index + offsets,
                "channel_schedule_enabled": True,
                "channel_state_success": True,
                "channel_verified": True,
                "channel_transition_partial": False,
            }).to_parquet(directory / "ue_second_features.parquet", index=False)
            controls = {"ploss": ploss, "noise_power_dB": noise}
            points.append({
                "point_id": point_id,
                "controls": controls,
                "repetition": repetition,
            })
            completed[point_id] = {
                "execution_id": execution_id,
                "controls": controls,
            }
    real = tmp_path / "real.csv"
    rows = []
    for trace_id, app, rsrp, rsrq in [
        ("trace-a", "video", -101.0, -10.5),
        ("trace-b", "download", -111.0, -14.0),
    ]:
        for index, offset in enumerate([-0.3, -0.1, 0.1, 0.3]):
            rows.append({
                "trace_id": trace_id,
                "app": app,
                "content": f"content-{trace_id}",
                "observation_index": index,
                "target_rsrp_dbm": rsrp + offset,
                "target_rsrq_db": rsrq + offset / 2,
                "target_snr_db": 7.0 + offset,
            })
    pd.DataFrame(rows).to_csv(real, index=False)
    selection = tmp_path / "selection.json"
    campaign = tmp_path / "campaign.json"
    _write_json(selection, {"schema_version": 1, "points": points})
    _write_json(campaign, {"schema_version": 1, "completed": completed})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1,
        "name": "test_distribution_calibration",
        "candidate_policy": "observed_safe_states_only",
        "holdout_unit": "complete_execution",
        "expected_model_types": ["TDL_B"],
        "real_scenario_column": "trace_id",
        "real_context_columns": ["app", "content"],
        "selection_metrics": [
            {
                "name": "RSRP",
                "real_column": "target_rsrp_dbm",
                "simulated_column": "ss_rsrp_dbm",
                "scale": 3.0,
            },
            {
                "name": "RSRQ",
                "real_column": "target_rsrq_db",
                "simulated_column": "ss_rsrq_db",
                "scale": 2.0,
            },
        ],
        "diagnostic_metrics": [{
            "name": "SNR_PROXY",
            "real_column": "target_snr_db",
            "simulated_column": "ss_sinr_db",
        }],
        "distance": {
            "primary": "rbf_mmd",
            "kernel_bandwidth": 1.0,
            "maximum_samples": 32,
            "wasserstein_quantiles": 21,
        },
        "minimum_samples_per_scenario": 4,
        "minimum_samples_per_execution": 4,
        "require_true": [
            "channel_schedule_enabled",
            "channel_state_success",
            "channel_verified",
        ],
        "require_false": ["channel_transition_partial"],
    }, sort_keys=False))
    return {
        "real": real,
        "executions": executions_root,
        "selection": selection,
        "campaign": campaign,
        "config": config,
        "output": tmp_path / "output",
    }


def test_distribution_distances_are_zero_for_identical_samples():
    values = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])

    assert rbf_mmd2(values, values) == 0.0
    assert quantile_wasserstein(values[:, 0], values[:, 0]) == 0.0


def test_distribution_calibration_ranks_matching_states_and_checksums_output(tmp_path):
    paths = _fixture(tmp_path)

    result = run_distribution_calibration(
        real_observations=paths["real"],
        executions_root=paths["executions"],
        selection_manifest=paths["selection"],
        campaign_state=paths["campaign"],
        config_path=paths["config"],
        output_dir=paths["output"],
    )

    assert result == {
        "output": str(paths["output"].resolve()),
        "scenarios": 2,
        "states": 2,
        "executions": 4,
        "ranking_rows": 4,
        "files": 6,
    }
    candidates = pd.read_csv(paths["output"] / "candidate_rankings.csv")
    best = candidates.loc[candidates["candidate_rank"].eq(1)].set_index("trace_id")
    assert best.loc["trace-a", "ploss"] == -5.0
    assert best.loc["trace-b", "ploss"] == 0.0
    repeatability = pd.read_csv(paths["output"] / "state_repeatability.csv")
    assert len(repeatability) == 2
    manifest = json.loads((paths["output"] / "calibration_manifest.json").read_text())
    assert manifest["implementation_scope"] == "discrete_executed_state_mmd_screen"
    assert len(manifest["software_provenance"]["repository_revision"]) == 40

    checksums = json.loads((paths["output"] / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((paths["output"] / name).read_bytes()).hexdigest() == expected
