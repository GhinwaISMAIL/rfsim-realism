from __future__ import annotations

import hashlib
import itertools
import json
import math
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .sweep import (
    _git_snapshot,
    _run,
    _testbed_config,
    dashboard_command,
    dashboard_preflight,
    load_config,
    ntp_preflight,
    verify_checksums,
    write_json,
)

CONTROL_PARAMETERS = ("ploss", "noise_power_dB")


@dataclass(frozen=True)
class StaticGridPoint:
    point_id: str
    controls: dict[str, float]
    repetition: int
    run_seconds: float


def _number_token(value: float) -> str:
    numeric = float(value)
    sign = "m" if numeric < 0 else "p"
    magnitude = f"{abs(numeric):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    topology = config.get("topology") or {}
    if topology.get("cells") != 1 or topology.get("ues") != 1:
        raise ValueError("the static AWGN grid requires one cell and one generated UE")
    channel = config.get("channel") or {}
    if channel.get("model_type") != "AWGN" or channel.get("direction") != "dl":
        raise ValueError("the static grid requires downlink AWGN")
    baseline = channel.get("baseline") or {}
    grid = channel.get("grid") or {}
    for parameter in CONTROL_PARAMETERS:
        if parameter not in baseline:
            raise ValueError(f"missing baseline value for {parameter}")
        values = grid.get(parameter) or []
        if not values:
            raise ValueError(f"the grid has no values for {parameter}")
        if len({float(value) for value in values}) != len(values):
            raise ValueError(f"the grid contains duplicate {parameter} values")
    if any(float(value) > 0 for value in grid["ploss"]):
        raise ValueError("RFsim ploss is path gain; attenuation values must be non-positive")
    if int(config.get("repetitions", 0)) < 1:
        raise ValueError("repetitions must be positive")
    run_seconds = float((config.get("timing") or {}).get("run_seconds", 0))
    if run_seconds <= 0:
        raise ValueError("run_seconds must be positive")
    provenance = config.get("provenance") or {}
    required = (
        "rsrp_offset_db",
        "ue_image",
        "ue_image_digest",
        "oai_source_commit",
        "reference_trace_id",
    )
    missing = [name for name in required if provenance.get(name) is None]
    if missing:
        raise ValueError("missing provenance fields: " + ", ".join(missing))


def build_plan(config: dict[str, Any]) -> list[StaticGridPoint]:
    validate_config(config)
    channel = config["channel"]
    run_seconds = float(config["timing"]["run_seconds"])
    points = []
    for repetition in range(1, int(config["repetitions"]) + 1):
        values = itertools.product(
            channel["grid"]["ploss"],
            channel["grid"]["noise_power_dB"],
        )
        for ploss, noise in values:
            controls = {
                "ploss": float(ploss),
                "noise_power_dB": float(noise),
            }
            point_id = (
                f"ploss-{_number_token(controls['ploss'])}_"
                f"noise-{_number_token(controls['noise_power_dB'])}-r{repetition}"
            )
            points.append(StaticGridPoint(
                point_id=point_id,
                controls=controls,
                repetition=repetition,
                run_seconds=run_seconds,
            ))
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("static grid point identifiers are not unique")
    return points


def plan_document(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign": config["name"],
        "experiment_type": "static_joint_awgn_grid",
        "model_type": config["channel"]["model_type"],
        "direction": config["channel"]["direction"],
        "target": "ue1",
        "run_seconds": float(config["timing"]["run_seconds"]),
        "provenance": dict(config["provenance"]),
        "required_observations": list(config.get("required_observations") or []),
        "quality_requirements": dict(config.get("quality_requirements") or {}),
        "points": [asdict(point) for point in build_plan(config)],
    }


def plan_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def schedule_for(
    point: StaticGridPoint, *, model_type: str = "AWGN"
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": True,
        "expected_model_type": model_type,
        "events": [
            {
                "at_s": 0.0,
                "target": "ue1",
                "direction": "dl",
                "parameter": parameter,
                "value": point.controls[parameter],
            }
            for parameter in CONTROL_PARAMETERS
        ],
    }


def idle_schedule(config: dict[str, Any]) -> dict[str, Any]:
    point = StaticGridPoint(
        point_id="baseline",
        controls={
            parameter: float(config["channel"]["baseline"][parameter])
            for parameter in CONTROL_PARAMETERS
        },
        repetition=0,
        run_seconds=float(config["timing"]["run_seconds"]),
    )
    return schedule_for(point, model_type=config["channel"]["model_type"])


def _json_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("channel command produced no JSON result")


def apply_controls(
    run_dir: str | Path, controls: dict[str, float]
) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    _, testbed = _testbed_config(run_dir)
    box = testbed["ues"]["boxes"]["ue1"]
    host = str(box["ssh_host"])
    cell = int(box["cell"])
    ue = int(box["ue_index"])
    remote_bin = str((testbed.get("mgen") or {}).get(
        "remote_bin", "/local/repository/bin"))
    common = [
        "sudo",
        "python3",
        f"{remote_bin}/channel-cell.py",
    ]

    for parameter in CONTROL_PARAMETERS:
        value = float(controls[parameter])
        remote = shlex.join([
            *common,
            "set",
            "--cell",
            str(cell),
            "--direction",
            "dl",
            "--ue",
            str(ue),
            "--parameter",
            parameter,
            "--value",
            str(value),
        ])
        result = _run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            host, remote,
        ], capture=True)
        payload = _json_output(result.stdout)
        if not payload.get("verified"):
            raise ValueError(f"{parameter} set was not verified")

    readbacks = {}
    for parameter in CONTROL_PARAMETERS:
        remote = shlex.join([
            *common,
            "show",
            "--cell",
            str(cell),
            "--direction",
            "dl",
            "--ue",
            str(ue),
            "--parameter",
            parameter,
        ])
        result = _run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            host, remote,
        ], capture=True)
        payload = _json_output(result.stdout)
        observed = float(payload["observed"])
        expected = float(controls[parameter])
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(
                f"{parameter} readback is {observed:g}; expected {expected:g}")
        if (
            payload.get("model_name") != "rfsimu_channel_enB0"
            or int(payload.get("model_index", -1)) != 0
        ):
            raise ValueError(f"{parameter} readback used the wrong RFsim model")
        readbacks[parameter] = payload
    return readbacks


def prepare_run_config(
    run_dir: str | Path,
    point: StaticGridPoint,
    config: dict[str, Any],
) -> Path:
    target = Path(run_dir) / "config.json"
    document = json.loads(target.read_text())
    duration = float(
        document.get("simulation_duration")
        or document.get("duration_s")
        or document.get("duration")
        or 0
    )
    if not math.isclose(duration, point.run_seconds, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"run duration is {duration:g}s; grid requires {point.run_seconds:g}s")
    document["rf_calibration"] = {
        "schema_version": 1,
        "method": "rfsimulator_rsrp_reporting_offset",
        "changes_iq_samples": False,
        "changes_ss_sinr": False,
        **dict(config["provenance"]),
        "experiment_role": "ucc_static_grid",
        "campaign": config["name"],
        "grid_point": {
            "point_id": point.point_id,
            "repetition": point.repetition,
            "controls": point.controls,
        },
    }
    return write_json(target, document)


def _quality_gate(segment: pd.Series, config: dict[str, Any]) -> None:
    required = config.get("quality_requirements") or {}
    min_clock = float(required.get("min_packet_clock_valid_fraction", 0.95))
    if float(segment["valid_clock_fraction"]) < min_clock:
        raise ValueError("packet clock validity is below the configured minimum")
    for field in (
        "controlled",
        "verified",
        "channel_agreement",
        "training_eligible",
        "ploss_verified",
        "ploss_agreement",
        "noise_power_dB_verified",
        "noise_power_dB_agreement",
    ):
        if not bool(segment[field]):
            raise ValueError(f"joint segment has {field}=false")
    if int(segment.get("radio_samples", 0)) <= 0:
        raise ValueError("joint segment has no radio samples")
    if int(segment.get("ue_radio_samples", 0)) <= 0:
        raise ValueError("joint segment has no UE serving-cell radio samples")
    if not bool(segment.get("ue_radio_clock_valid", False)):
        raise ValueError("joint segment has an invalid UE radio clock")
    max_ue_lag = float(required.get("max_ue_radio_emit_lag_p95_s", 0.5))
    ue_lag = float(segment["ue_radio_emit_lag_s_p95"])
    if ue_lag > max_ue_lag:
        raise ValueError("UE radio emission lag exceeds the configured limit")
    if bool(segment.get("radio_clock_lag_warning", False)):
        raise ValueError("joint segment has a radio clock lag warning")
    max_lag = float(required.get("max_radio_clock_lag_p95_s", 0.1))
    lag = float(segment["radio_clock_lag_s_segment_p95"])
    if lag > max_lag:
        raise ValueError("radio clock lag exceeds the configured limit")


def validate_archive(
    archive: str | Path,
    point: StaticGridPoint,
    config: dict[str, Any],
) -> dict[str, Any]:
    archive = Path(archive)
    verified_files = verify_checksums(archive)
    metadata = json.loads((archive / "metadata.json").read_text())
    quality = metadata.get("quality") or {}
    if not quality.get("channel_state_verified"):
        raise ValueError("channel state was not verified")
    xapp = quality.get("xapp") or {}
    if not xapp.get("clean_shutdown") or xapp.get("errors"):
        raise ValueError("xApp did not shut down cleanly")

    archived_config = json.loads((archive / "config.json").read_text())
    archived_point = (
        (archived_config.get("rf_calibration") or {}).get("grid_point") or {})
    if archived_point.get("point_id") != point.point_id:
        raise ValueError("archived config does not identify the current grid point")

    training = pd.read_parquet(archive / "segment_training_table.parquet")
    candidates = training[
        training["direction"].eq("dl")
        & training["parameter"].astype("string").eq("joint")
    ].copy()
    for parameter in CONTROL_PARAMETERS:
        candidates = candidates[
            pd.to_numeric(
                candidates[f"requested_{parameter}"], errors="coerce")
            .sub(point.controls[parameter])
            .abs()
            .le(1e-9)
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one complete joint segment for {point.point_id}; "
            f"found {len(candidates)}")
    segment = candidates.iloc[0]
    if int(segment["control_count"]) != len(CONTROL_PARAMETERS):
        raise ValueError("joint segment does not contain both AWGN controls")
    if not math.isclose(
        float(segment["duration_s"]), point.run_seconds,
        rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError("joint segment does not cover the complete execution")
    for parameter in CONTROL_PARAMETERS:
        applied = float(segment[f"applied_{parameter}"])
        if not math.isclose(
            applied, point.controls[parameter], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"joint segment has the wrong applied {parameter}")
    _quality_gate(segment, config)

    packets = pd.read_parquet(archive / "packet_outcomes.parquet")
    packet_rows = packets[
        packets["ue"].astype(str).eq(str(segment["ue"]))
        & packets["direction"].eq("dl")
        & packets["sent_time_utc"].ge(float(segment["segment_start_utc"]))
        & packets["sent_time_utc"].lt(float(segment["segment_end_utc"]))
    ]
    latency = packet_rows.loc[
        packet_rows["packet_clock_valid"] & packet_rows["received"],
        "latency_ms",
    ]
    expected_p95 = float(latency.quantile(.95))
    if not math.isclose(
        float(segment["latency_ms_p95"]), expected_p95,
        rel_tol=0, abs_tol=1e-9,
    ):
        raise ValueError("segment latency p95 was not reconstructed from packet rows")

    return {
        "execution_id": metadata["execution_id"],
        "archive": str(archive),
        "verified_files": verified_files,
        "segment_id": str(segment["segment_id"]),
        "controls": point.controls,
        "valid_clock_fraction": float(segment["valid_clock_fraction"]),
        "sent_packets": int(segment["sent_packets"]),
        "received_packets": int(segment["received_packets"]),
        "loss_rate": float(segment["loss_rate"]),
        "latency_ms_p95": expected_p95,
        "received_mbps": float(segment["received_mbps"]),
        "ss_rsrp_dbm_segment_mean": float(
            segment["ss_rsrp_dbm_segment_mean"]),
        "ss_rsrq_db_segment_mean": float(segment["ss_rsrq_db_segment_mean"]),
        "ss_sinr_db_segment_mean": float(segment["ss_sinr_db_segment_mean"]),
        "radio_clock_lag_s_p95": float(
            segment["radio_clock_lag_s_segment_p95"]),
        "ue_radio_emit_lag_s_p95": float(
            segment["ue_radio_emit_lag_s_p95"]),
    }


def _state(
    path: Path,
    document: dict[str, Any],
    run_dir: Path,
    dashboard_repo: Path,
) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("plan_sha256") != plan_sha256(document):
            raise ValueError("campaign state does not match the current grid plan")
        return value
    mgen_repo = run_dir.parent.parent
    realism_repo = Path(__file__).resolve().parents[2]
    value = {
        "schema_version": 1,
        "campaign": document["campaign"],
        "plan_sha256": plan_sha256(document),
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "dashboard_repo": str(dashboard_repo),
        "repositories": {
            "dashboard": _git_snapshot(dashboard_repo),
            "mgen": _git_snapshot(mgen_repo),
            "realism": _git_snapshot(realism_repo),
        },
        "completed": {},
        "failures": [],
    }
    write_json(path, value)
    return value


def run_campaign(
    config_path: str | Path,
    run_dir: str | Path,
    dashboard_repo: str | Path,
    state_path: str | Path,
    *,
    point_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    document = plan_document(config)
    points = build_plan(config)
    run_dir = Path(run_dir).resolve()
    dashboard_repo = Path(dashboard_repo).resolve()
    state_path = Path(state_path).resolve()
    state = _state(state_path, document, run_dir, dashboard_repo)
    pending = [
        point for point in points
        if point.point_id not in state["completed"]
        and (point_ids is None or point.point_id in point_ids)
    ]
    if point_ids is not None:
        unknown = point_ids - {point.point_id for point in points}
        if unknown:
            raise ValueError("unknown grid points: " + ", ".join(sorted(unknown)))
    if limit is not None:
        pending = pending[:limit]

    schedule_path = run_dir / "channel_schedule.json"
    baseline = {
        parameter: float(config["channel"]["baseline"][parameter])
        for parameter in CONTROL_PARAMETERS
    }
    for point in pending:
        before = (
            json.loads((run_dir / "logs" / "run_timing.json").read_text())
            .get("run_id")
            if (run_dir / "logs" / "run_timing.json").exists()
            else None
        )
        try:
            peers = ntp_preflight(run_dir, config)
            prepare_run_config(run_dir, point, config)
            write_json(
                schedule_path,
                schedule_for(point, model_type=config["channel"]["model_type"]),
            )
            dashboard_preflight(run_dir, dashboard_repo)
            readbacks = apply_controls(run_dir, point.controls)
            command, env = dashboard_command(
                dashboard_repo, "deploy", run_dir.name)
            _run(command, cwd=dashboard_repo, env=env)
            timing = json.loads(
                (run_dir / "logs" / "run_timing.json").read_text())
            execution_id = str(timing.get("run_id") or "")
            if not execution_id or execution_id == before:
                raise ValueError("deployment did not produce a new execution identifier")
            archive = run_dir / "executions" / execution_id
            result = validate_archive(archive, point, config)
            result["pre_run_readback"] = readbacks
            result["ntp"] = [asdict(peer) for peer in peers]
            result["completed_at"] = datetime.now(UTC).isoformat()
            state["completed"][point.point_id] = result
            write_json(state_path, state)
        except Exception as exc:
            state["failures"].append({
                "point_id": point.point_id,
                "failed_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            })
            write_json(state_path, state)
            raise
        finally:
            apply_controls(run_dir, baseline)
            write_json(schedule_path, idle_schedule(config))
    return state
