from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _git_state(repository: Path) -> dict[str, object]:
    if not (repository / ".git").exists():
        raise ValueError(f"not a Git checkout: {repository}")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git checkout {repository}: {error}") from error
    return {"revision": revision, "tracked_worktree_dirty": bool(dirty)}


def _implementation_state() -> dict[str, object]:
    try:
        return _git_state(Path(__file__).resolve().parents[2])
    except ValueError:
        return {"revision": "unavailable", "tracked_worktree_dirty": None}


def validate_phase3c_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3C0 schema_version must be 1")
    if config.get("stage") != (
        "phase_3c0_static_channel_causal_audit_and_replay_preparation"
    ):
        raise ValueError("unexpected Phase 3C0 stage")
    if bool(config.get("execution_authorized")) or bool(config.get("abc_authorized")):
        raise ValueError("Phase 3C0 cannot authorize RF execution or ABC")

    frozen = config.get("frozen_inputs") or {}
    if frozen.get("phase3b_decision_code") != "relative_rsrp_shape_mismatch":
        raise ValueError("Phase 3C0 requires the frozen Phase 3B model-mismatch decision")
    if not bool(frozen.get("phase2_and_phase3a_snapshots_must_remain_unchanged")):
        raise ValueError("Phase 2 and Phase 3A must remain immutable")

    scope = config.get("model_scope") or {}
    if scope.get("prohibited_conclusion") != (
        "all_RFsim_or_VRTSIM_configurations_are_inadequate"
    ):
        raise ValueError("Phase 3C0 must not generalize beyond the tested RFsim path")
    if scope.get("primary_mechanism") != (
        "frozen_terrestrial_channel_realization_plus_constant_path_gain"
    ):
        raise ValueError("the frozen-channel mechanism must be explicit")
    if scope.get("sinr_mechanism_status") != (
        "requires_instrumented_confirmation_not_noise_only_assertion"
    ):
        raise ValueError("the SINR mechanism must remain an instrumented hypothesis")

    assertions = config.get("source_assertions") or []
    paths = {str(item.get("path")) for item in assertions}
    required_paths = {
        "radio/rfsimulator/simulator.cpp",
        "radio/rfsimulator/apply_channelmod.c",
        "openair1/PHY/NR_UE_ESTIMATION/nr_ue_measurements.c",
        "radio/vrtsim/CMakeLists.txt",
        "radio/vrtsim/README.md",
        "radio/vrtsim/cirdb_provider.c",
    }
    if not assertions or not required_paths.issubset(paths):
        raise ValueError("Phase 3C0 source assertions are incomplete")

    envelope = config.get("deterministic_scalar_envelope") or {}
    if envelope.get("gain_db_sequence") != [0.0, -2.0, -4.0, -2.0, 0.0]:
        raise ValueError("unexpected deterministic gain sequence")
    if float(envelope.get("segment_duration_seconds", 0)) <= 0:
        raise ValueError("the envelope segment duration must be positive")
    settle = float(envelope.get("settling_seconds_per_segment", -1))
    duration = float(envelope["segment_duration_seconds"])
    if settle < 0 or settle >= duration:
        raise ValueError("the envelope settling interval must fit inside each segment")
    if not math.isclose(
        float(envelope.get("analysis_seconds_per_segment", -1)),
        duration - settle,
    ):
        raise ValueError("the frozen envelope analysis duration is inconsistent")
    if int(envelope.get("independent_local_replays_required", 0)) < 2:
        raise ValueError("at least two independent local replays are required")

    acceptance = config.get("deterministic_replay_acceptance") or {}
    if set(acceptance.get("required_telemetry_fields") or []) != set(
        config["instrumentation_contract"]["required_fields"]
    ):
        raise ValueError("replay telemetry fields must match the instrumentation contract")
    if int(acceptance.get("independent_replays_required", 0)) != int(
        envelope["independent_local_replays_required"]
    ):
        raise ValueError("the replay-count gate must match the envelope contract")
    if int(acceptance.get("minimum_analysis_rows_per_segment", 0)) < 1:
        raise ValueError("each deterministic segment requires analysis rows")
    if float(acceptance.get("attachment_fraction_required", 0)) != 1.0:
        raise ValueError("the deterministic replay requires continuous attachment")
    slope = acceptance.get("float_rsrp_transfer_slope_range") or []
    if len(slope) != 2 or not (float(slope[0]) <= 1.0 <= float(slope[1])):
        raise ValueError("the float-RSRP transfer slope range must contain one")
    if int(acceptance.get("required_unique_channel_snapshots_per_replay", 0)) != 1:
        raise ValueError("the scalar-envelope test must retain one raw channel snapshot")

    temporal = config.get("time_varying_channel_contract") or {}
    if temporal.get("small_scale_normalization") != (
        "E[sum_l(abs(alpha_tilde_l(t))^2)]=1"
    ):
        raise ValueError("the two-timescale channel requires normalized small-scale power")
    if temporal.get("prohibited_generator_behavior") != (
        "independent_snapshot_regeneration_at_receive_cadence"
    ):
        raise ValueError("unphysical receive-cadence regeneration must be prohibited")

    generator = config.get("external_generator_status") or {}
    if generator.get("state") != (
        "provisional_pending_pinned_revision_source_audit_and_temporal_tests"
    ):
        raise ValueError("the external generator must remain provisional")
    if not bool(generator.get("transport_and_generator_validity_are_separate_gates")):
        raise ValueError("transport and generator validity must remain separate gates")

    claims = config.get("claim_limits") or {}
    if claims.get("absolute_rsrp_calibration") != "prohibited":
        raise ValueError("absolute RSRP calibration remains prohibited")
    if claims.get("abc") != "prohibited":
        raise ValueError("ABC remains prohibited")
    reservation = config.get("reservation") or {}
    if bool(reservation.get("request_now")) or reservation.get("gate_state") != "closed":
        raise ValueError("the Phase 3C0 reservation gate must remain closed")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def deterministic_envelope(config: dict[str, Any]) -> pd.DataFrame:
    validate_phase3c_config(config)
    envelope = config["deterministic_scalar_envelope"]
    duration = float(envelope["segment_duration_seconds"])
    settling = float(envelope["settling_seconds_per_segment"])
    rows: list[dict[str, object]] = []
    for index, (gain_db, label) in enumerate(
        zip(envelope["gain_db_sequence"], envelope["labels"], strict=True)
    ):
        start = index * duration
        gain = float(gain_db)
        rows.append({
            "segment_index": index,
            "segment_label": str(label),
            "start_s": start,
            "analysis_start_s": start + settling,
            "end_s": start + duration,
            "commanded_gain_db": gain,
            "expected_amplitude_multiplier": 10 ** (gain / 20.0),
            "expected_power_ratio": 10 ** (gain / 10.0),
            "expected_relative_float_rsrp_db": gain,
        })
    return pd.DataFrame(rows)


def _source_trace(
    config: dict[str, Any], oai_source: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    state = _git_state(oai_source)
    expected = str(config["pinned_sources"]["oai"]["expected_revision"])
    if state["revision"] != expected:
        raise ValueError(f"OAI revision {state['revision']} does not match {expected}")
    if bool(state["tracked_worktree_dirty"]):
        raise ValueError("tracked files are dirty in the OAI source checkout")

    rows: list[dict[str, object]] = []
    for assertion_id, assertion in enumerate(config["source_assertions"], start=1):
        relative = str(assertion["path"])
        path = oai_source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe source evidence: {path}")
        source_text = path.read_text(errors="replace")
        lines = source_text.splitlines()
        for snippet_id, raw_snippet in enumerate(assertion["snippets"], start=1):
            snippet = str(raw_snippet)
            occurrences = source_text.count(snippet)
            if occurrences < 1:
                raise ValueError(f"source assertion not found in {relative}: {snippet}")
            offset = source_text.index(snippet)
            line_number = source_text[:offset].count("\n") + 1
            rows.append({
                "assertion_id": assertion_id,
                "snippet_id": snippet_id,
                "oai_revision": state["revision"],
                "relative_path": relative,
                "file_sha256": _sha256(path),
                "fact": assertion["fact"],
                "snippet": snippet,
                "occurrence_count": occurrences,
                "first_line": line_number,
                "line_text": lines[line_number - 1].strip(),
                "verified": True,
            })
    return pd.DataFrame(rows), state


def _validate_phase3b_inputs(
    config: dict[str, Any], decision_path: Path, gate_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _read_json(decision_path)
    gate = _read_json(gate_path)
    frozen = config["frozen_inputs"]
    if _sha256(decision_path) != frozen["phase3b_decision_sha256"]:
        raise ValueError("Phase 3B decision checksum does not match the frozen input")
    if _sha256(gate_path) != frozen["phase3b_gate_sha256"]:
        raise ValueError("Phase 3B reservation-gate checksum does not match the frozen input")
    if decision.get("decision_code") != frozen["phase3b_decision_code"]:
        raise ValueError("Phase 3B decision code does not match the Phase 3C0 prerequisite")
    if gate.get("gate_state") != "closed" or bool(
        gate.get("reservation_should_be_requested_now")
    ):
        raise ValueError("the supplied Phase 3B reservation gate is not closed")
    if bool(decision.get("new_execution_authorized")) or bool(
        decision.get("abc_performed")
    ):
        raise ValueError("the supplied Phase 3B decision exceeds diagnostic scope")
    return decision, gate


def build_phase3c_plan(
    *,
    phase3b_decision: str | Path,
    phase3b_gate: str | Path,
    oai_source: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    decision_path = Path(phase3b_decision).resolve()
    gate_path = Path(phase3b_gate).resolve()
    source_path = Path(oai_source).resolve()
    protocol_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    for path in [decision_path, gate_path, protocol_path]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe Phase 3C0 input: {path}")
    if not source_path.is_dir() or source_path.is_symlink():
        raise ValueError(f"missing or unsafe OAI checkout: {source_path}")
    if output.exists():
        raise FileExistsError(f"Phase 3C0 output already exists: {output}")

    config = _read_yaml(protocol_path)
    validate_phase3c_config(config)
    decision, _ = _validate_phase3b_inputs(config, decision_path, gate_path)
    source_trace, source_state = _source_trace(config, source_path)
    envelope = deterministic_envelope(config)
    implementation = _implementation_state()

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_csv(staging / "source_code_trace.csv", source_trace)
        _write_csv(staging / "deterministic_scalar_envelope.csv", envelope)
        _write_json(
            staging / "instrumentation_contract.json",
            config["instrumentation_contract"],
        )
        _write_json(
            staging / "deterministic_replay_acceptance.json",
            config["deterministic_replay_acceptance"],
        )
        _write_json(
            staging / "time_varying_channel_contract.json",
            config["time_varying_channel_contract"],
        )
        phase3c_decision = {
            "schema_version": 1,
            "decision_code": (
                "current_static_terrestrial_tdl_b_path_inadequate_for_"
                "route_scale_relative_rsrp"
            ),
            "model_scope": config["model_scope"],
            "phase3b_metric_support": decision["metric_support"],
            "source_assertions_verified": len(source_trace),
            "selected_transport_candidate": "VRTSIM_external_taps",
            "transport_build_validation_complete": False,
            "external_generator_temporal_validity_established": False,
            "absolute_rsrp_calibration_authorized": False,
            "new_powder_execution_authorized": False,
            "abc_authorized": False,
            "next_action": (
                "add parallel debug telemetry and execute the deterministic scalar "
                "envelope locally before evaluating any time-varying generator"
            ),
        }
        _write_json(staging / "phase3c0_decision.json", phase3c_decision)
        reservation = {
            "schema_version": 1,
            "decision_code": phase3c_decision["decision_code"],
            "gate_state": "closed",
            "reservation_should_be_requested_now": False,
            "preparation_lead_time_minutes": config["reservation"][
                "preparation_lead_time_minutes"
            ],
            "notification_rule": config["reservation"]["notification_rule"],
            "blocking_conditions": [
                "debug float and channel-state instrumentation has not passed",
                "deterministic scalar replay has not passed",
                "time-varying trace temporal validity has not passed",
                "attachment and PBCH/PUSCH parser validation has not passed",
                "the small replay experiment has not been frozen",
            ],
            "opening_conditions": config["reservation"]["opening_conditions"],
            "next_action": "continue Phase 3C offline",
        }
        _write_json(staging / "reservation_gate_v4.json", reservation)
        input_inventory = pd.DataFrame([
            {
                "input": "phase3b_decision",
                "source_id": decision_path.name,
                "sha256": _sha256(decision_path),
                "git_revision": None,
                "tracked_worktree_dirty": None,
            },
            {
                "input": "phase3b_gate",
                "source_id": gate_path.name,
                "sha256": _sha256(gate_path),
                "git_revision": None,
                "tracked_worktree_dirty": None,
            },
            {
                "input": "protocol_config",
                "source_id": protocol_path.name,
                "sha256": _sha256(protocol_path),
                "git_revision": None,
                "tracked_worktree_dirty": None,
            },
            {
                "input": "oai_source_checkout",
                "source_id": source_path.name,
                "sha256": None,
                "git_revision": source_state["revision"],
                "tracked_worktree_dirty": source_state["tracked_worktree_dirty"],
            },
        ])
        _write_csv(staging / "input_inventory.csv", input_inventory)

        before_manifest = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "name": config["name"],
            "stage": config["stage"],
            "analysis_implementation_revision": implementation["revision"],
            "tracked_worktree_dirty_at_start": implementation[
                "tracked_worktree_dirty"
            ],
            "frozen_inputs": config["frozen_inputs"],
            "oai_source_state": source_state,
            "source_assertions_verified": len(source_trace),
            "offline_gate_order": config["offline_gate_order"],
            "claim_limits": config["claim_limits"],
            "external_generator_status": config["external_generator_status"],
            "reservation_should_be_requested_now": False,
            "abc_performed": False,
            "output_sha256_before_manifest": before_manifest,
        }
        _write_json(staging / "analysis_manifest.json", manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.json"
        }
        _write_json(staging / "SHA256SUMS.json", checksums)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "decision_code": phase3c_decision["decision_code"],
        "source_assertions_verified": len(source_trace),
        "deterministic_segments": len(envelope),
        "execution_authorized": False,
        "abc_authorized": False,
        "reservation_should_be_requested_now": False,
    }


def evaluate_deterministic_replay(
    *, telemetry_path: str | Path, plan_dir: str | Path
) -> dict[str, object]:
    telemetry_file = Path(telemetry_path).resolve()
    plan = Path(plan_dir).resolve()
    if not telemetry_file.is_file() or telemetry_file.is_symlink():
        raise ValueError(f"missing or unsafe deterministic telemetry: {telemetry_file}")
    if not plan.is_dir() or plan.is_symlink():
        raise ValueError(f"missing or unsafe Phase 3C0 plan directory: {plan}")
    envelope = pd.read_csv(plan / "deterministic_scalar_envelope.csv")
    acceptance = _read_json(plan / "deterministic_replay_acceptance.json")
    telemetry = pd.read_csv(telemetry_file)
    required = set(acceptance["required_telemetry_fields"])
    missing = sorted(required - set(telemetry.columns))
    if missing:
        raise ValueError(f"deterministic telemetry is missing columns: {missing}")
    numeric = sorted(required - {"replay_id", "channel_snapshot_id", "attached"})
    for column in numeric:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    finite = np.isfinite(telemetry[numeric].to_numpy(float)).all()
    if not finite:
        raise ValueError("deterministic telemetry contains non-finite required values")
    telemetry["attached"] = telemetry["attached"].astype(str).str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })
    if telemetry["attached"].isna().any():
        raise ValueError("attached must contain only true/false or 1/0")

    rows: list[dict[str, object]] = []
    replay_summaries: list[dict[str, object]] = []
    for replay_id, replay in telemetry.groupby("replay_id", sort=True):
        segment_rows: list[dict[str, object]] = []
        for segment in envelope.to_dict("records"):
            selected = replay.loc[
                replay["t_s"].ge(float(segment["analysis_start_s"]))
                & replay["t_s"].lt(float(segment["end_s"]))
            ]
            expected = float(segment["commanded_gain_db"])
            coverage_pass = len(selected) >= int(
                acceptance["minimum_analysis_rows_per_segment"]
            )
            applied_error = (
                float((selected["applied_gain_db"] - expected).abs().max())
                if coverage_pass
                else math.inf
            )
            commanded_error = (
                float((selected["commanded_gain_db"] - expected).abs().max())
                if coverage_pass
                else math.inf
            )
            row = {
                "replay_id": str(replay_id),
                "segment_index": int(segment["segment_index"]),
                "segment_label": str(segment["segment_label"]),
                "expected_gain_db": expected,
                "analysis_rows": len(selected),
                "coverage_pass": coverage_pass,
                "commanded_gain_max_abs_error_db": commanded_error,
                "applied_gain_max_abs_error_db": applied_error,
                "float_rsrp_median_db": (
                    float(selected["rsrp_db_per_re_unquantized"].median())
                    if coverage_pass
                    else math.nan
                ),
                "integer_rsrp_median_dbm": (
                    float(selected["ss_rsrp_dbm_integer"].median())
                    if coverage_pass
                    else math.nan
                ),
            }
            rows.append(row)
            segment_rows.append(row)

        segments = pd.DataFrame(segment_rows)
        baseline_float = float(
            segments.loc[segments["expected_gain_db"].eq(0), "float_rsrp_median_db"].mean()
        )
        baseline_integer = float(
            segments.loc[
                segments["expected_gain_db"].eq(0), "integer_rsrp_median_dbm"
            ].mean()
        )
        x = segments["expected_gain_db"].to_numpy(float)
        y = segments["float_rsrp_median_db"].to_numpy(float) - baseline_float
        slope, intercept = np.polyfit(x, y, deg=1)
        predicted = slope * x + intercept
        residual = float(np.sum((y - predicted) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
        float_delta_error = float(np.max(np.abs(y - x)))
        integer_relative = (
            segments["integer_rsrp_median_dbm"].to_numpy(float) - baseline_integer
        )
        integer_delta_error = float(np.max(np.abs(integer_relative - x)))
        by_level = segments.groupby("expected_gain_db", sort=True)
        hysteresis = float(
            max(
                (float(group["float_rsrp_median_db"].max())
                 - float(group["float_rsrp_median_db"].min()))
                for _, group in by_level
            )
        )
        tap_mean = float(replay["tap_energy_linear"].mean())
        tap_cv = (
            float(replay["tap_energy_linear"].std(ddof=0) / tap_mean)
            if tap_mean != 0
            else math.inf
        )
        snapshot_count = int(replay["channel_snapshot_id"].astype(str).nunique())
        attachment_fraction = float(replay["attached"].mean())
        gain_tolerance = float(
            acceptance["commanded_applied_gain_max_abs_error_db"]
        )
        slope_bounds = acceptance["float_rsrp_transfer_slope_range"]
        passes = {
            "coverage": bool(segments["coverage_pass"].all()),
            "commanded_gain": bool(
                segments["commanded_gain_max_abs_error_db"].max() <= gain_tolerance
            ),
            "applied_gain": bool(
                segments["applied_gain_max_abs_error_db"].max() <= gain_tolerance
            ),
            "attachment": attachment_fraction
            >= float(acceptance["attachment_fraction_required"]),
            "float_slope": float(slope_bounds[0]) <= slope <= float(slope_bounds[1]),
            "float_r_squared": r_squared
            >= float(acceptance["float_rsrp_transfer_r_squared_minimum"]),
            "float_delta": float_delta_error
            <= float(acceptance["float_rsrp_delta_max_abs_error_db"]),
            "integer_delta": integer_delta_error
            <= float(acceptance["integer_rsrp_delta_max_abs_error_db"]),
            "hysteresis": hysteresis
            <= float(acceptance["repeated_level_hysteresis_max_abs_db"]),
            "tap_energy": tap_cv
            <= float(acceptance["raw_tap_energy_coefficient_of_variation_maximum"]),
            "snapshot_count": snapshot_count
            == int(acceptance["required_unique_channel_snapshots_per_replay"]),
        }
        passes = {name: bool(value) for name, value in passes.items()}
        replay_summaries.append({
            "replay_id": str(replay_id),
            "float_rsrp_transfer_slope": float(slope),
            "float_rsrp_transfer_intercept_db": float(intercept),
            "float_rsrp_transfer_r_squared": r_squared,
            "float_rsrp_delta_max_abs_error_db": float_delta_error,
            "integer_rsrp_delta_max_abs_error_db": integer_delta_error,
            "repeated_level_hysteresis_max_abs_db": hysteresis,
            "raw_tap_energy_coefficient_of_variation": tap_cv,
            "unique_channel_snapshots": snapshot_count,
            "attachment_fraction": attachment_fraction,
            "gate_results": passes,
            "replay_pass": all(passes.values()),
        })

    required_replays = int(acceptance["independent_replays_required"])
    overall = len(replay_summaries) >= required_replays and all(
        bool(item["replay_pass"]) for item in replay_summaries
    )
    return {
        "schema_version": 1,
        "telemetry_sha256": _sha256(telemetry_file),
        "replays_evaluated": len(replay_summaries),
        "minimum_replays_required": required_replays,
        "segment_results": rows,
        "replay_results": replay_summaries,
        "deterministic_replay_gate_pass": overall,
        "reservation_should_be_requested_now": False,
        "next_action": (
            "proceed to pinned time-varying trace validation"
            if overall
            else "keep the temporal replay gate closed and debug the measurement path"
        ),
    }


def write_deterministic_replay_evaluation(
    *, telemetry_path: str | Path, plan_dir: str | Path, output_path: str | Path
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"deterministic replay evaluation already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_deterministic_replay(
        telemetry_path=telemetry_path,
        plan_dir=plan_dir,
    )
    _write_json(output, result)
    return output
