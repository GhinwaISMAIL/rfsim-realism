from __future__ import annotations

import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    parameter: str
    value: float
    baseline: float
    repetition: int
    treatment_at_s: float
    return_at_s: float
    measurement_start_s: float
    measurement_end_s: float


@dataclass(frozen=True)
class NtpPeer:
    host: str
    server: str
    reach: int
    offset_ms: float
    jitter_ms: float


def _number_token(value: float) -> str:
    value = float(value)
    sign = "m" if value < 0 else "p"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("calibration configuration must be a mapping")
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    topology = config.get("topology") or {}
    if topology.get("cells") != 1 or topology.get("ues") != 1:
        raise ValueError("the AWGN calibration requires one cell and one generated UE")
    channel = config.get("channel") or {}
    if channel.get("model_type") != "AWGN" or channel.get("direction") != "dl":
        raise ValueError("the first calibration campaign must use downlink AWGN")
    baseline = channel.get("baseline") or {}
    timing = config.get("timing") or {}
    required_timing = (
        "run_seconds",
        "baseline_lead_seconds",
        "segment_seconds",
        "settle_seconds",
        "measurement_seconds",
        "baseline_tail_seconds",
    )
    missing = [name for name in required_timing if name not in timing]
    if missing:
        raise ValueError("missing timing values: " + ", ".join(missing))
    lead = float(timing["baseline_lead_seconds"])
    segment = float(timing["segment_seconds"])
    settle = float(timing["settle_seconds"])
    measurement = float(timing["measurement_seconds"])
    tail = float(timing["baseline_tail_seconds"])
    run_seconds = float(timing["run_seconds"])
    if not math.isclose(settle + measurement, segment):
        raise ValueError("settle_seconds + measurement_seconds must equal segment_seconds")
    if not math.isclose(lead + segment + tail, run_seconds):
        raise ValueError("lead + segment + tail must equal run_seconds")
    if not timing.get("return_to_baseline_between_values"):
        raise ValueError("return_to_baseline_between_values must be enabled")
    repetitions = int(config.get("repetitions", 0))
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    experiments = config.get("experiments") or []
    if not experiments:
        raise ValueError("at least one experiment is required")
    for experiment in experiments:
        parameter = experiment.get("parameter")
        if parameter not in baseline:
            raise ValueError(f"no baseline is configured for {parameter}")
        values = experiment.get("values") or []
        if not values:
            raise ValueError(f"no values are configured for {parameter}")
        if len({float(value) for value in values}) != len(values):
            raise ValueError(f"duplicate values are configured for {parameter}")
        if parameter == "ploss" and any(float(value) > 0 for value in values):
            raise ValueError("RFsim ploss is path gain; attenuation values must be non-positive")


def build_plan(config: dict[str, Any]) -> list[SweepPoint]:
    validate_config(config)
    baseline = config["channel"]["baseline"]
    timing = config["timing"]
    treatment_at = float(timing["baseline_lead_seconds"])
    return_at = treatment_at + float(timing["segment_seconds"])
    measurement_start = treatment_at + float(timing["settle_seconds"])
    measurement_end = measurement_start + float(timing["measurement_seconds"])
    points = []
    for experiment in config["experiments"]:
        parameter = str(experiment["parameter"])
        for repetition in range(1, int(config["repetitions"]) + 1):
            for value in experiment["values"]:
                numeric = float(value)
                point_id = f"{parameter}-{_number_token(numeric)}-r{repetition}"
                points.append(
                    SweepPoint(
                        point_id=point_id,
                        parameter=parameter,
                        value=numeric,
                        baseline=float(baseline[parameter]),
                        repetition=repetition,
                        treatment_at_s=treatment_at,
                        return_at_s=return_at,
                        measurement_start_s=measurement_start,
                        measurement_end_s=measurement_end,
                    )
                )
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("sweep point identifiers are not unique")
    return points


def plan_document(config: dict[str, Any]) -> dict[str, Any]:
    points = [asdict(point) for point in build_plan(config)]
    return {
        "schema_version": 1,
        "campaign": config["name"],
        "model_type": config["channel"]["model_type"],
        "direction": config["channel"]["direction"],
        "target": "ue1",
        "run_seconds": float(config["timing"]["run_seconds"]),
        "required_observations": list(config.get("required_observations") or []),
        "quality_requirements": dict(config.get("quality_requirements") or {}),
        "points": points,
    }


def plan_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def schedule_for(point: SweepPoint, *, model_type: str = "AWGN") -> dict[str, Any]:
    events = [
        {
            "at_s": 0.0,
            "target": "ue1",
            "direction": "dl",
            "parameter": point.parameter,
            "value": point.baseline,
        },
        {
            "at_s": point.treatment_at_s,
            "target": "ue1",
            "direction": "dl",
            "parameter": point.parameter,
            "value": point.value,
        },
        {
            "at_s": point.return_at_s,
            "target": "ue1",
            "direction": "dl",
            "parameter": point.parameter,
            "value": point.baseline,
        },
    ]
    return {
        "schema_version": 1,
        "enabled": True,
        "expected_model_type": model_type,
        "events": events,
    }


def idle_schedule(config: dict[str, Any]) -> dict[str, Any]:
    baseline = float(config["channel"]["baseline"]["noise_power_dB"])
    return {
        "schema_version": 1,
        "enabled": True,
        "expected_model_type": config["channel"]["model_type"],
        "events": [
            {
                "at_s": 0.0,
                "target": "ue1",
                "direction": config["channel"]["direction"],
                "parameter": "noise_power_dB",
                "value": baseline,
            }
        ],
    }


def parse_ntpq(output: str, host: str) -> NtpPeer:
    selected = next((line for line in output.splitlines() if line.startswith("*")), None)
    if selected is None:
        raise ValueError(f"{host} has no selected NTP peer")
    fields = selected.split()
    if len(fields) < 10:
        raise ValueError(f"cannot parse selected NTP peer on {host}: {selected}")
    try:
        return NtpPeer(
            host=host,
            server=fields[0].removeprefix("*"),
            reach=int(fields[6], 8),
            offset_ms=float(fields[8]),
            jitter_ms=float(fields[9]),
        )
    except ValueError as exc:
        raise ValueError(f"cannot parse selected NTP peer on {host}: {selected}") from exc


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        env=env,
    )


def _testbed_config(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    repository = run_dir.parent.parent
    path = repository / "testbed_config.yaml"
    value = yaml.safe_load(path.read_text()) or {}
    if value.get("testbed") != "powder_ric5g_distributed":
        raise ValueError("testbed_config.yaml is not the distributed RIC5G profile")
    return path, value


def ntp_preflight(run_dir: str | Path, config: dict[str, Any]) -> list[NtpPeer]:
    run_dir = Path(run_dir).resolve()
    _, testbed = _testbed_config(run_dir)
    nodes = testbed.get("nodes") or {}
    hosts = [(nodes.get("core") or {}).get("ssh_host")]
    hosts.extend(cell.get("ssh_host") for cell in nodes.get("cells") or [])
    hosts = [str(host) for host in hosts if host]
    if len(hosts) != 2:
        raise ValueError("the calibration preflight requires one core and one cell host")
    peers = []
    for host in hosts:
        result = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, "ntpq -pn"],
            capture=True,
        )
        peers.append(parse_ntpq(result.stdout, host))
    quality = config.get("quality_requirements") or {}
    max_spread = float(quality.get("max_ntp_offset_spread_ms", 0.5))
    max_jitter = float(quality.get("max_ntp_jitter_ms", 1.0))
    spread = max(peer.offset_ms for peer in peers) - min(peer.offset_ms for peer in peers)
    if spread > max_spread:
        raise ValueError(f"NTP offset spread is {spread:.3f} ms; limit is {max_spread:.3f} ms")
    excessive = [peer for peer in peers if peer.jitter_ms > max_jitter]
    if excessive:
        details = ", ".join(f"{peer.host}={peer.jitter_ms:.3f}" for peer in excessive)
        raise ValueError(f"NTP jitter exceeds {max_jitter:.3f} ms: {details}")
    if any(peer.reach == 0 for peer in peers):
        raise ValueError("an NTP peer is not reachable")
    return peers


def dashboard_command(
    dashboard_repo: str | Path, command: str, run_name: str
) -> tuple[list[str], dict[str, str]]:
    dashboard_repo = Path(dashboard_repo).resolve()
    python = dashboard_repo / ".venv" / "bin" / "python"
    executable = str(python if python.exists() else Path(sys.executable))
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(dashboard_repo)
    return [executable, "-m", "twindash.cli", command, run_name], env


def dashboard_preflight(run_dir: str | Path, dashboard_repo: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    command, env = dashboard_command(dashboard_repo, "preflight", run_dir.name)
    _run(command, cwd=Path(dashboard_repo).resolve(), env=env)


def restore_baselines(run_dir: str | Path, config: dict[str, Any]) -> None:
    run_dir = Path(run_dir).resolve()
    _, testbed = _testbed_config(run_dir)
    box = testbed["ues"]["boxes"]["ue1"]
    host = str(box["ssh_host"])
    cell = int(box["cell"])
    ue = int(box["ue_index"])
    remote_bin = str((testbed.get("mgen") or {}).get("remote_bin", "/local/repository/bin"))
    for parameter, value in config["channel"]["baseline"].items():
        remote = shlex.join(
            [
                "sudo",
                "python3",
                f"{remote_bin}/channel-cell.py",
                "set",
                "--cell",
                str(cell),
                "--direction",
                "dl",
                "--ue",
                str(ue),
                "--parameter",
                str(parameter),
                "--value",
                str(float(value)),
            ]
        )
        _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, remote])


def verify_checksums(archive: str | Path) -> int:
    archive = Path(archive)
    manifest = json.loads((archive / "SHA256SUMS.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("archive checksum manifest is missing or empty")
    for relative, expected in manifest.items():
        path = archive / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"archive file is missing or unsafe: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"archive checksum mismatch: {relative}")
    return len(manifest)


def validate_archive(
    archive: str | Path, point: SweepPoint, config: dict[str, Any]
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
    training = pd.read_parquet(archive / "segment_training_table.parquet")
    candidates = training[
        training["direction"].eq("dl")
        & training["parameter"].astype("string").eq(point.parameter)
        & pd.to_numeric(training["requested_value"], errors="coerce")
        .sub(point.value)
        .abs()
        .le(1e-9)
    ].copy()
    if candidates.empty:
        raise ValueError(f"no treatment segment was archived for {point.point_id}")
    target_duration = point.return_at_s - point.treatment_at_s
    candidates["duration_error"] = (
        pd.to_numeric(candidates["duration_s"], errors="coerce") - target_duration
    ).abs()
    segment = candidates.sort_values(["duration_error", "segment_start_utc"]).iloc[0]
    required = config.get("quality_requirements") or {}
    min_clock = float(required.get("min_packet_clock_valid_fraction", 0.95))
    if float(segment["valid_clock_fraction"]) < min_clock:
        raise ValueError(
            f"packet clock validity is {float(segment['valid_clock_fraction']):.3f}; "
            f"minimum is {min_clock:.3f}"
        )
    for field in ("controlled", "verified", "channel_agreement", "training_eligible"):
        if not bool(segment[field]):
            raise ValueError(f"treatment segment has {field}=false")
    if int(segment.get("radio_samples", 0)) <= 0:
        raise ValueError("treatment segment has no radio samples")
    if int(segment.get("ue_radio_samples", 0)) <= 0:
        raise ValueError("treatment segment has no UE serving-cell radio samples")
    for field in (
        "ss_rsrp_dbm_segment_mean", "ss_rsrq_db_segment_mean",
        "ss_sinr_db_segment_mean",
    ):
        if pd.isna(segment.get(field)):
            raise ValueError(f"treatment segment has no {field}")
    if not bool(segment.get("ue_radio_clock_valid", False)):
        raise ValueError("treatment segment has an invalid UE radio clock")
    ue_lag_p95 = segment.get("ue_radio_emit_lag_s_p95")
    max_ue_lag = float(required.get("max_ue_radio_emit_lag_p95_s", 0.5))
    if pd.isna(ue_lag_p95) or float(ue_lag_p95) > max_ue_lag:
        raise ValueError(
            f"UE radio emission lag p95 is missing or exceeds {max_ue_lag:.3f}s"
        )
    lag_warning = segment.get("radio_clock_lag_warning")
    if pd.notna(lag_warning) and bool(lag_warning):
        raise ValueError("treatment segment has a radio clock lag warning")
    lag_p95 = segment.get("radio_clock_lag_s_segment_p95")
    max_lag = float(required.get("max_radio_clock_lag_p95_s", 0.1))
    if pd.isna(lag_p95) or float(lag_p95) > max_lag:
        raise ValueError(f"radio clock lag p95 is missing or exceeds {max_lag:.3f}s")

    measurement_start = float(segment["segment_start_utc"]) + float(
        config["timing"]["settle_seconds"])
    measurement_end = float(segment["segment_end_utc"])
    ue_name = str(segment["ue"])
    serving = pd.read_csv(archive / "logs" / "ue_radio_by_second.csv")
    serving = serving[
        serving["ue"].astype(str).eq(ue_name)
        & pd.to_numeric(serving["utc_second"], errors="coerce").ge(measurement_start)
        & pd.to_numeric(serving["utc_second"], errors="coerce").lt(measurement_end)
    ].copy()
    expected_radio = max(int(math.floor(measurement_end) - math.ceil(measurement_start)), 1)
    if len(serving) < max(expected_radio - 2, 1):
        raise ValueError(
            f"settled window has only {len(serving)}/{expected_radio} UE radio seconds"
        )
    settled_radio = {}
    for source, target in (
        ("ss_rsrp_dbm", "ss_rsrp_dbm_segment_mean"),
        ("ss_rsrq_db", "ss_rsrq_db_segment_mean"),
        ("ss_sinr_db", "ss_sinr_db_segment_mean"),
    ):
        values = pd.to_numeric(serving[source], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"settled window has no {source}")
        settled_radio[target] = float(values.mean())
    emission_lag = (
        pd.to_numeric(serving["emitted_epoch_us"], errors="coerce") / 1e6
        - (pd.to_numeric(serving["utc_second"], errors="coerce") + 1)
    ).dropna().abs()
    settled_ue_lag_p95 = float(emission_lag.quantile(.95))
    if settled_ue_lag_p95 > max_ue_lag:
        raise ValueError(
            f"settled UE radio emission lag p95 exceeds {max_ue_lag:.3f}s"
        )

    packets = pd.read_parquet(archive / "packet_outcomes.parquet")
    packets = packets[
        packets["ue"].astype(str).eq(ue_name)
        & packets["direction"].eq("dl")
        & packets["sent_time_utc"].ge(measurement_start)
        & packets["sent_time_utc"].lt(measurement_end)
    ].copy()
    if packets.empty:
        raise ValueError("settled window has no packet evidence")
    settled_valid_fraction = float(packets["packet_clock_valid"].mean())
    if settled_valid_fraction < min_clock:
        raise ValueError(
            f"settled packet clock validity is {settled_valid_fraction:.3f}; "
            f"minimum is {min_clock:.3f}"
        )
    latency = packets.loc[
        packets["packet_clock_valid"] & packets["received"], "latency_ms"]
    return {
        "execution_id": metadata["execution_id"],
        "archive": str(archive),
        "verified_files": verified_files,
        "segment_id": str(segment["segment_id"]),
        "segment_duration_s": float(segment["duration_s"]),
        "measurement_start_utc": measurement_start,
        "measurement_end_utc": measurement_end,
        "measurement_duration_s": measurement_end - measurement_start,
        "valid_clock_fraction": settled_valid_fraction,
        "radio_samples": int(segment["radio_samples"]),
        "ue_radio_samples": len(serving),
        "ue_radio_emit_lag_s_p95": settled_ue_lag_p95,
        **settled_radio,
        "radio_clock_lag_s_p95": float(lag_p95),
        "sent_packets": len(packets),
        "received_packets": int(packets["received"].sum()),
        "loss_rate": float(packets["lost"].mean()),
        "latency_ms_p95": (
            None if latency.empty else float(latency.quantile(.95))
        ),
    }


def _git_snapshot(repository: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repository, capture=True).stdout.strip()
    status = _run(["git", "status", "--short"], cwd=repository, capture=True).stdout
    return {
        "path": str(repository),
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _state(
    path: Path, document: dict[str, Any], run_dir: Path, dashboard_repo: Path
) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("plan_sha256") != plan_sha256(document):
            raise ValueError("campaign state does not match the current sweep plan")
        return value
    mgen_repo = run_dir.parent.parent
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
        point
        for point in points
        if point.point_id not in state["completed"]
        and (point_ids is None or point.point_id in point_ids)
    ]
    if point_ids is not None:
        unknown = point_ids - {point.point_id for point in points}
        if unknown:
            raise ValueError("unknown sweep points: " + ", ".join(sorted(unknown)))
    if limit is not None:
        pending = pending[:limit]
    schedule_path = run_dir / "channel_schedule.json"
    for point in pending:
        before = (
            json.loads((run_dir / "logs" / "run_timing.json").read_text()).get("run_id")
            if (run_dir / "logs" / "run_timing.json").exists()
            else None
        )
        try:
            peers = ntp_preflight(run_dir, config)
            dashboard_preflight(run_dir, dashboard_repo)
            restore_baselines(run_dir, config)
            write_json(
                schedule_path, schedule_for(point, model_type=config["channel"]["model_type"])
            )
            command, env = dashboard_command(dashboard_repo, "deploy", run_dir.name)
            _run(command, cwd=dashboard_repo, env=env)
            timing = json.loads((run_dir / "logs" / "run_timing.json").read_text())
            execution_id = str(timing.get("run_id") or "")
            if not execution_id or execution_id == before:
                raise ValueError("deployment did not produce a new execution identifier")
            archive = run_dir / "executions" / execution_id
            result = validate_archive(archive, point, config)
            result["ntp"] = [asdict(peer) for peer in peers]
            result["completed_at"] = datetime.now(UTC).isoformat()
            state["completed"][point.point_id] = result
            write_json(state_path, state)
        except Exception as exc:
            state["failures"].append(
                {
                    "point_id": point.point_id,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                }
            )
            write_json(state_path, state)
            raise
        finally:
            restore_baselines(run_dir, config)
            write_json(schedule_path, idle_schedule(config))
    return state
