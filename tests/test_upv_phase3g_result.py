from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "manifests/upv_phase3g_direct_trace_v1"
RESULT = RESULT_DIR / "preparation_result.json"


def test_recorded_phase3g_preparation_remains_fail_closed() -> None:
    result = json.loads(RESULT.read_text())
    assert result["decision_code"] == "bounded_response_experiment_ready_for_runner_freeze"
    assert result["target_trace"]["session_id"] == "corrected_test_1_ASUS"
    assert result["target_trace"]["rows"] == 305
    assert result["target_trace"]["interpolation_used"] is False
    assert result["bounded_response_experiment"]["total_execution_units"] == 45
    assert result["bounded_response_experiment"]["stage_counts"] == {
        "gain_safety": 4,
        "noise_safety": 2,
        "factorial": 27,
        "boundary": 12,
    }
    assert result["bounded_response_experiment"]["execution_authorized"] is False
    assert not any(result["authorizations"].values())
    assert result["reservation"]["request_now"] is False
    assert result["reservation"]["preparation_lead_time_minutes"] >= 30


def test_recorded_phase3g_preparation_checksum() -> None:
    checksums = json.loads((RESULT_DIR / "SHA256SUMS.json").read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == checksums[RESULT.name]
