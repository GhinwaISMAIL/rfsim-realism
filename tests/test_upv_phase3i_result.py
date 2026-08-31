from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "manifests/upv_phase3i_short_trace_result_v1"


def test_phase3i_result_passes_frozen_runtime_and_fidelity_gates() -> None:
    decision = json.loads((RESULT / "phase3i_short_trace_decision.json").read_text())
    assert decision["decision_code"] == "representative_short_trace_replay_passed"
    assert decision["campaign"]["paired_rows"] == 60
    assert decision["campaign"]["clipped_rows"] == 0
    assert decision["campaign"]["primary_alignment_seconds"] == 0
    assert decision["campaign"]["lag_search_used_for_gate_selection"] is False
    assert decision["gates"]["runtime_gate_passed"] is True
    assert decision["gates"]["fidelity_gate_passed"] is True
    assert decision["gates"]["all_gates_passed"] is True
    assert decision["full_trace_protocol_freeze_authorized"] is True
    assert decision["full_trace_replay_currently_authorized"] is False
    assert decision["final_test6_accessed"] is False
    assert decision["abc_authorized"] is False


def test_phase3i_recovery_is_complete_deterministic_and_hardware_free() -> None:
    recovery = json.loads((RESULT / "recovery_report.json").read_text())
    assert recovery["command_events"] == 60
    assert recovery["paired_trace_rows"] == 60
    assert recovery["same_second_channel_matches"] == 0
    assert recovery["next_second_channel_matches"] == 60
    assert recovery["channel_verification_alignment_seconds"] == 1
    assert recovery["primary_kpi_alignment_seconds"] == 0
    assert recovery["critical_failure_count"] == 0
    assert recovery["ping_successes"] == recovery["ping_checks"] == 24
    assert recovery["rollback_passed"] is True
    assert recovery["hardware_rerun_used"] is False
    assert recovery["scientific_target_or_gate_changed"] is False


def test_phase3i_result_contains_all_sixty_paired_rows() -> None:
    paired = pd.read_csv(RESULT / "paired_short_trace_fidelity.csv")
    assert len(paired) == 60
    assert paired["command_index"].tolist() == list(range(60))
    assert paired["attached"].astype(bool).all()


def test_phase3i_result_checksums() -> None:
    checksums = json.loads((RESULT / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((RESULT / name).read_bytes()).hexdigest() == expected
