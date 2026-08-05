import hashlib
import json

from rfsim_realism.campaign_export import export_campaign, verify_bundle
from rfsim_realism.sweep import plan_sha256


def test_completed_campaign_exports_portable_verified_bundle(tmp_path):
    archive = tmp_path / "source" / "mgen-test"
    archive.mkdir(parents=True)
    (archive / "payload.txt").write_text("measurement\n")
    checksums = {"payload.txt": hashlib.sha256((archive / "payload.txt").read_bytes()).hexdigest()}
    (archive / "SHA256SUMS.json").write_text(json.dumps(checksums))

    point = {
        "point_id": "ploss-p0-r1",
        "parameter": "ploss",
        "value": 0,
        "baseline": 0,
        "repetition": 1,
        "treatment_at_s": 45,
        "return_at_s": 135,
        "measurement_start_s": 90,
        "measurement_end_s": 135,
    }
    plan = {
        "schema_version": 1,
        "campaign": "test",
        "model_type": "AWGN",
        "direction": "dl",
        "target": "ue1",
        "run_seconds": 180.0,
        "required_observations": [],
        "quality_requirements": {},
        "points": [point],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n")
    state = {
        "campaign": "test",
        "plan_sha256": plan_sha256(plan),
        "run_dir": "/local/run",
        "dashboard_repo": "/local/dashboard",
        "repositories": {
            "dashboard": {"path": "/local/dashboard", "commit": "abc123"}
        },
        "completed": {
            "ploss-p0-r1": {
                "archive": str(archive),
                "execution_id": "mgen-test",
                "measurement_start_utc": 100.0,
                "measurement_end_utc": 145.0,
                "measurement_duration_s": 45.0,
                "ss_rsrp_dbm_segment_mean": -41.0,
                "ss_rsrq_db_segment_mean": -10.4,
                "ss_sinr_db_segment_mean": 47.0,
                "latency_ms_p95": 2.0,
                "loss_rate": 0.0,
                "valid_clock_fraction": 1.0,
                "radio_clock_lag_s_p95": 0.01,
                "ue_radio_emit_lag_s_p95": 0.02,
                "verified_files": 1,
            }
        },
        "failures": [],
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    output = tmp_path / "export"

    summary = export_campaign(state_path, config_path, plan_path, output)

    portable = json.loads((output / "campaign_state.json").read_text())
    assert summary == {
        "output": str(output),
        "points": 1,
        "archive_files": 1,
        "bundle_files": 9,
    }
    assert "run_dir" not in portable
    assert "dashboard_repo" not in portable
    assert "path" not in portable["repositories"]["dashboard"]
    assert portable["completed"]["ploss-p0-r1"]["archive"] == "executions/mgen-test"
    assert (output / "executions" / "mgen-test" / "payload.txt").read_text() == "measurement\n"
    assert verify_bundle(output) == 9
