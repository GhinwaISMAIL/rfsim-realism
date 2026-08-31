from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rfsim_realism.noise_response import (
    evaluate_corrected_noise_response,
    validate_noise_response_analysis_spec,
    write_corrected_noise_response_evaluation,
)

STATES = [-60.0, -40.0, -30.0, -25.0, -20.0]
ORDER = [
    [-60.0, -30.0, -20.0, -40.0, -25.0],
    [-25.0, -60.0, -40.0, -20.0, -30.0],
    [-20.0, -40.0, -25.0, -30.0, -60.0],
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    archive = tmp_path / "raw.tar.gz"
    archive.write_bytes(b"frozen raw archive")
    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {
            "stage": "corrected_rfsim_noise_response_validation_protocol",
            "design": {
                "noise_power_db_states": STATES,
                "independent_executions_per_state": 3,
                "state_order_by_repetition": ORDER,
            },
            "analysis_unit": "execution",
            "valid_execution_rules": {
                "paired_radio_samples_minimum": 10,
                "ping_success_fraction_minimum": 0.9,
                "critical_pbch_failure_count_maximum": 0,
                "critical_pusch_failure_count_maximum": 0,
                "unintended_ue_restart_count_maximum": 0,
                "gnb_restart_count_change_maximum": 0,
                "gnb_health": "healthy",
            },
            "not_authorized": ["final UPV validation"],
        },
    )

    plan = []
    seed = 41001
    for repetition, states in enumerate(ORDER, start=1):
        for position, noise in enumerate(states, start=1):
            plan.append(
                {
                    "repetition": repetition,
                    "position": position,
                    "noise_power_db": noise,
                    "oai_rng_seed": seed,
                }
            )
            seed += 1
    freeze = tmp_path / "freeze.json"
    _write_json(
        freeze,
        {
            "stage": "corrected_rfsim_noise_hardware_execution_freeze",
            "revision_identity": {
                "oai_revision": "oai",
                "execution_profile_revision": "profile",
                "runner_sha256": "runner",
            },
            "configuration_identity": {
                "compose_sha256": "compose",
                "channel_config_sha256": "channel",
                "ue_config_sha256": "ue",
                "derived_attach_minus60_config_sha256": "attach",
            },
            "image_identity": {
                "corrected_tag": "corrected:image",
                "corrected_image_id": "sha256:corrected",
                "rollback_image_id": "sha256:rollback",
            },
            "execution_plan": plan,
            "authorization": {"final_upv_test6_access": False},
        },
    )

    telemetry_rows = []
    execution_summaries = []
    for expected in plan:
        noise = expected["noise_power_db"]
        execution_id = f"r{expected['repetition']}-p{expected['position']}-n{abs(int(noise))}"
        for sample in range(12):
            telemetry_rows.append(
                {
                    "execution_id": execution_id,
                    "repetition": expected["repetition"],
                    "position": expected["position"],
                    "oai_rng_seed": expected["oai_rng_seed"],
                    "commanded_noise_power_db": noise,
                    "applied_noise_power_db": noise,
                    "commanded_gain_db": 0.0,
                    "applied_gain_db": 0.0,
                    "channel_family": "AWGN",
                    "channel_model_name": "rfsimu_channel_enB0",
                    "channel_snapshot_id": "static-0",
                    "tap_fingerprint_fnv1a64": "0123456789abcdef",
                    "tap_energy_linear": 1.0,
                    "channel_length": 1,
                    "nb_taps": 1,
                    "nb_tx": 1,
                    "nb_rx": 1,
                    "rsrp_db_per_re_unquantized": 51.0 + sample / 10000,
                    "ss_rsrp_dbm_integer": -97,
                    "ss_sinr_db": -noise + expected["repetition"] / 100 + sample / 1000,
                    "attached": True,
                }
            )
        execution_summaries.append(
            {
                "execution_id": execution_id,
                "repetition": expected["repetition"],
                "position": expected["position"],
                "oai_rng_seed": expected["oai_rng_seed"],
                "commanded_noise_power_db": noise,
                "applied_command_result": {
                    "verified": True,
                    "parameter": "noise_power_dB",
                    "model_type": "AWGN",
                    "observed": noise,
                    "requested": noise,
                },
                "channel_identity_at_attach": {
                    "reachable": True,
                    "model_type": "AWGN",
                    "observed": -60.0,
                },
                "continuous_attachment": True,
                "paired_radio_samples": 12,
                "ping_success_fraction": 1.0,
                "critical_failure_count": 0,
                "critical_pbch_failure_count": 0,
                "critical_pusch_failure_count": 0,
                "failure_marker_counts": {
                    "ue": {"lost_sync": 0},
                    "gnb": {"radio_link_failure": 0},
                },
                "ue_restart_count": 0,
                "gnb_restart_count": 0,
                "gnb_health": "healthy",
            }
        )
    telemetry = tmp_path / "telemetry.csv"
    pd.DataFrame(telemetry_rows).to_csv(telemetry, index=False)
    state = tmp_path / "state.json"
    _write_json(
        state,
        {
            "stage": "corrected_rfsim_noise_response_validation",
            "execution_completed": True,
            "error": None,
            "oai_revision": "oai",
            "profile_revision": "profile",
            "runner_sha256": "runner",
            "compose_sha256": "compose",
            "channel_config_sha256": "channel",
            "ue_config_sha256": "ue",
            "attach_config_sha256": "attach",
            "debug_image": "corrected:image",
            "debug_image_id": "sha256:corrected",
            "debug_image_revision_label": "oai",
            "gNB_untouched": True,
            "rollback": {
                "passed": True,
                "attached": True,
                "restored_image_id": "sha256:rollback",
                "gnb_restart_count_before": 0,
                "gnb_restart_count_after": 0,
            },
            "execution_plan": plan,
            "executions": execution_summaries,
        },
    )
    route_means = tmp_path / "route.csv"
    pd.DataFrame(
        {
            "route_relative_rsrp_db": [-3.0, 0.0, 6.0],
            "route_sinr_db": [11.0, 15.0, 22.0],
            "supported": [True, True, True],
            "fold_id": ["fold-1", "fold-1", "fold-1"],
        }
    ).to_csv(route_means, index=False)
    phase3d = tmp_path / "phase3d.json"
    _write_json(phase3d, {"final_evaluation": {"payload_opened": False}})

    spec = tmp_path / "spec.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "stage": "corrected_rfsim_noise_response_analysis_specification",
            "status": "frozen_before_confirmatory_evaluation",
            "timing_disclosure": {
                "bootstrap_intervals_inspected_before_this_freeze": False,
                "decision_gate_changed_after_execution": False,
            },
            "frozen_inputs": {
                "raw_archive_sha256": _sha256(archive),
                "telemetry_sha256": _sha256(telemetry),
                "execution_state_sha256": _sha256(state),
                "protocol_sha256": _sha256(protocol),
                "hardware_freeze_sha256": _sha256(freeze),
                "development_route_means_sha256": _sha256(route_means),
                "phase3d_decision_sha256": _sha256(phase3d),
            },
            "analysis": {
                "unit": "execution",
                "within_execution_summary": "median",
                "state_summary": "mean_of_three_execution_medians",
                "metrics": [
                    "rsrp_db_per_re_unquantized",
                    "ss_rsrp_dbm_integer",
                    "ss_sinr_db",
                ],
                "monotonic_metric": "ss_sinr_db",
                "maximum_allowed_upward_sinr_step_db": 0.5,
                "bootstrap": {
                    "resampling_unit": "execution_median",
                    "groups_resampled_independently": True,
                    "repetitions": 1000,
                    "seed": 7,
                    "confidence_level": 0.95,
                    "interval": "percentile",
                },
            },
            "development_comparison": {
                "final_test6_access": False,
                "fixed_noise_selection_authorized": False,
            },
            "claim_limits": {
                "absolute_environmental_noise_calibration": "prohibited",
                "absolute_rsrp_calibration": "prohibited",
            },
        },
    )
    return {
        "raw_archive_path": archive,
        "telemetry_path": telemetry,
        "execution_state_path": state,
        "protocol_path": protocol,
        "hardware_freeze_path": freeze,
        "analysis_spec_path": spec,
        "development_route_means_path": route_means,
        "phase3d_decision_path": phase3d,
    }


def test_noise_response_accepts_monotonic_execution_level_control(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result, executions, states, pairwise = evaluate_corrected_noise_response(**inputs)
    assert result["decision_code"] == "corrected_control_valid"
    assert result["control_gate_pass"] is True
    assert result["fixed_noise_selection_authorized"] is False
    assert result["development_comparison"]["final_test6_accessed"] is False
    assert len(executions) == 15
    assert len(states) == 5
    assert len(pairwise) == 30


def test_noise_response_rejects_nonmonotonic_sinr(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    telemetry = pd.read_csv(inputs["telemetry_path"])
    telemetry.loc[telemetry["commanded_noise_power_db"].eq(-20.0), "ss_sinr_db"] = 70.0
    telemetry.to_csv(inputs["telemetry_path"], index=False)
    spec = json.loads(inputs["analysis_spec_path"].read_text())
    spec["frozen_inputs"]["telemetry_sha256"] = _sha256(inputs["telemetry_path"])
    _write_json(inputs["analysis_spec_path"], spec)

    result, *_ = evaluate_corrected_noise_response(**inputs)
    assert result["control_gate_pass"] is False
    assert result["decision_code"] == "implementation_or_transport_invalid"


def test_noise_response_fails_closed_on_input_checksum_change(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["telemetry_path"].write_text(inputs["telemetry_path"].read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_corrected_noise_response(**inputs)


def test_noise_response_writer_is_immutable(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "evaluation"
    paths = write_corrected_noise_response_evaluation(**inputs, output_dir=output)
    assert set(paths) == {
        "evaluation",
        "execution_medians",
        "state_summary",
        "pairwise_differences",
        "checksums",
    }
    with pytest.raises(FileExistsError):
        write_corrected_noise_response_evaluation(**inputs, output_dir=output)


def test_repository_noise_response_spec_is_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "manifests/upv_rfsim_validity_audit_v1/hardware_analysis_spec.json").read_text()
    )
    validate_noise_response_analysis_spec(spec)
