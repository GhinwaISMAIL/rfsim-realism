from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_phase3c3_replay_gate_v1.yaml").read_text()
    )


def test_phase3c3_is_fail_closed_and_not_a_calibration() -> None:
    config = _config()

    assert config["execution_authorized"] is False
    assert config["powder_action_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["reservation"]["gate_state"] == "closed"
    assert config["reservation"]["reservation_should_be_requested_now"] is False
    assert config["claim_limits"]["transport_and_attachment_safety_only"] is True
    assert config["claim_limits"]["absolute_rsrp_calibration"] == "prohibited"


def test_phase3c3_freezes_executable_failure_limits() -> None:
    config = _config()
    acceptance = config["acceptance_rules"]

    assert acceptance["valid_repetitions_required"] == 2
    assert acceptance["attachment_fraction_required"] == 1.0
    assert acceptance["pbch_decode_error_count_maximum"] == 0
    assert acceptance["critical_pusch_failure_count_maximum"] == 0
    assert acceptance["maximum_consecutive_skipped_cirdb_snapshots"] == 10
    assert acceptance["trace_cycle_coverage_fraction_minimum"] == 0.99


def test_phase3c3_rejects_existing_debug_images_for_vrtsim() -> None:
    config = _config()

    assert config["existing_image_audit"]["decision"] == (
        "existing_phase3c_debug_images_are_not_accepted_as_vrtsim_replay_images"
    )
    assert "nr-softmodem" in config["required_vrtsim_build"]["runtime_artifacts"][
        "gnb"
    ]
    assert "libvrtsim.so" in config["required_vrtsim_build"]["runtime_artifacts"][
        "ue"
    ]


def test_phase3c3_freezes_one_gnb_one_ue_virtual_replay() -> None:
    config = _config()
    replay = config["small_replay"]

    assert replay["topology"] == {
        "gnb_count": 1,
        "ue_count": 1,
        "channel": "virtual_VRTSIM_CIRDB",
        "physical_RF_or_SDR": False,
    }
    assert replay["trace"]["discrete_taps"] == 8
    assert replay["per_repetition"]["post_attachment_observation_s"] == 330.0
