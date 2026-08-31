from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rfsim_realism.upv_phase3d import (
    analyze_phase3d_radio_process,
    validate_phase3d_config,
    write_phase3d_protocol_freeze,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/upv_phase3d_radio_process_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _radio_payload(test_id: int, rows: int = 160) -> bytes:
    sample = np.arange(rows, dtype=float)
    seconds = sample / 2.0
    state = ((seconds // 12).astype(int) % 2).astype(float)
    rsrp = -88.0 + 2.5 * state + 1.8 * np.sin(seconds / 8.0 + test_id * 0.2)
    sinr = 15.0 + 3.0 * state + 1.3 * np.cos(seconds / 9.0 + test_id * 0.15)
    start = pd.Timestamp("2026-01-01 12:00:00")
    times = [
        (start + pd.to_timedelta(value, unit="s")).strftime("%H:%M:%S.%f")[:-3] for value in seconds
    ]
    frame = pd.DataFrame(
        {
            "Time": times,
            "RSRP (NR SpCell)": rsrp,
            "RSRQ (NR SpCell)": np.full(rows, -10.5),
            "SINR (NR SpCell)": sinr,
            "Physical cell identity (NR SpCell)": np.full(rows, 41),
            "RSRP (NR neighbor)": np.full(rows, -104.0),
            "Physical cell identity (NR neighbor)": np.full(rows, 61),
            "Longitude": -0.35 + sample * 0.000002,
            "Latitude": np.full(rows, 39.48),
        }
    )
    return frame.to_csv(index=False, sep=";").encode()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = yaml.safe_load(CONFIG.read_text())
    config["models"]["hmm"]["initializations"] = 2
    config["models"]["hmm"]["maximum_iterations"] = 40
    config["evaluation"]["generation_repetitions"] = 20
    config["evaluation"]["joint_distribution"]["maximum_rows_per_trace"] = 80
    config["preprocessing"]["spatial_conditioning"]["minimum_training_sessions_per_bin"] = 3
    archive = tmp_path / "upv.zip"
    payloads: dict[str, bytes] = {}
    for session in config["development"]["sessions"]:
        session["trim_last_seconds"] = 0
        payload = _radio_payload(int(session["corrected_test_id"]))
        payloads[session["source_path"]] = payload
        session["source_sha256"] = hashlib.sha256(payload).hexdigest()
    final_path = config["final_evaluation"]["source_path"]
    final_payload = _radio_payload(6)
    payloads[final_path] = final_payload
    config["final_evaluation"]["source_sha256"] = hashlib.sha256(final_payload).hexdigest()
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in payloads.items():
            output.writestr(name, payload)
    config["source"]["expected_sha256"] = _sha256(archive)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    phase3c15 = tmp_path / "phase3c15.json"
    phase3c15.write_text(
        json.dumps({"decision": "gain_only_rejected_noise_control_required"}) + "\n"
    )
    return archive, config_path, phase3c15


def test_phase3d_config_is_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_phase3d_config(config)
    assert config["execution_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["final_evaluation_authorized"] is False
    assert config["reservation"]["request_now"] is False
    assert config["reservation"]["preparation_lead_time_minutes"] == 30


def test_phase3d_rejects_final_session_in_development() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["final_evaluation"]["source_path"] = config["development"]["sessions"][0]["source_path"]
    try:
        validate_phase3d_config(config)
    except ValueError as error:
        assert "final session" in str(error)
    else:
        raise AssertionError("the final-session overlap must be rejected")


def test_phase3d_freeze_and_analysis_never_read_final_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive, config, phase3c15 = _fixture(tmp_path)
    final_path = yaml.safe_load(config.read_text())["final_evaluation"]["source_path"]
    original_read = zipfile.ZipFile.read
    opened: list[str] = []

    def guarded_read(self, name, *args, **kwargs):
        value = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        normalized = value.split("Remote Driving Dataset in UPV's 5G Private network/")[-1]
        opened.append(normalized)
        if normalized == final_path:
            raise AssertionError("final evaluation payload was opened")
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    protocol = tmp_path / "protocol"
    freeze = write_phase3d_protocol_freeze(
        archive_path=archive,
        phase3c15_result_path=phase3c15,
        config_path=config,
        output_dir=protocol,
    )
    assert freeze["final_evaluation_locked"] is True
    lock = json.loads((protocol / "final_evaluation_lock.json").read_text())
    assert lock["payload_opened_by_freeze"] is False
    output = tmp_path / "analysis"
    result = analyze_phase3d_radio_process(
        archive_path=archive,
        phase3c15_result_path=phase3c15,
        protocol_dir=protocol,
        config_path=config,
        output_dir=output,
    )
    assert final_path not in opened
    assert result["final_evaluation"]["payload_opened"] is False
    assert result["powder_reservation_authorized"] is False
    metrics = pd.read_csv(output / "fold_model_metrics.csv")
    assert metrics["fold_id"].nunique() == 5
    assert set(metrics["candidate_id"]) == {
        "paired_moving_block_bootstrap",
        "gaussian_1state",
        "gaussian_2state",
        "student_t_1state",
        "student_t_2state",
        "gamma_gaussian_1state",
        "gamma_gaussian_2state",
    }
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        assert _sha256(output / name) == expected
