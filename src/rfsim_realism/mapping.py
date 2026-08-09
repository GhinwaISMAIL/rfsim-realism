from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

CONTROL_COLUMNS = ("applied_ploss", "applied_noise_power_dB")
APPLICATION_OUTPUTS = ("latency_ms_p95", "loss_rate", "received_mbps")
RADIO_OUTPUTS = (
    "ss_rsrp_dbm_segment_mean",
    "ss_rsrq_db_segment_mean",
    "ss_sinr_db_segment_mean",
)
QUALITY_TRUE_FIELDS = (
    "controlled",
    "verified",
    "channel_agreement",
    "training_eligible",
    "model_mapping_valid",
    "packet_evidence",
    "ploss_verified",
    "ploss_agreement",
    "noise_power_dB_verified",
    "noise_power_dB_agreement",
    "ue_radio_clock_valid",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_mapping_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("mapping schema_version must be 1")
    if config.get("direction") != "dl":
        raise ValueError("the first static mapping must keep downlink separate")
    if tuple(config.get("conditional_pre_run_inputs") or ()) != CONTROL_COLUMNS:
        raise ValueError("mapping inputs must be the two verified AWGN controls")
    if int(config.get("minimum_repetitions_per_state", 0)) < 2:
        raise ValueError("at least two executions per state are required")
    if config.get("candidate_policy") != "observed_safe_states_only":
        raise ValueError("candidate selection must not extrapolate beyond observed states")
    if config.get("evaluation_policy") != "leave_one_execution_out_within_control_state":
        raise ValueError("evaluation must hold out a complete execution")
    inverse = config.get("inverse_matching") or {}
    selection_metrics = inverse.get("selection_metrics") or []
    if not selection_metrics:
        raise ValueError("inverse matching requires at least one selection metric")
    observed = set(RADIO_OUTPUTS)
    for metric in selection_metrics:
        if metric.get("observed_metric") not in observed:
            raise ValueError("selection metric is not an approved radio output")
        if float(metric.get("tolerance", 0)) <= 0:
            raise ValueError("selection tolerances must be positive")


def verify_dataset_checksums(dataset_dir: str | Path) -> int:
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "SHA256SUMS.json"
    manifest = _read_json(manifest_path)
    for relative, expected in sorted(manifest.items()):
        path = (dataset_dir / relative).resolve()
        if dataset_dir not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe dataset file: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"dataset checksum mismatch: {relative}")
    return len(manifest)


def selected_execution_metadata(
    selection_manifest: dict[str, Any],
    campaign_state: dict[str, Any],
) -> pd.DataFrame:
    points = selection_manifest.get("points") or []
    completed = campaign_state.get("completed") or {}
    if not points:
        raise ValueError("safe selection contains no points")
    rows = []
    for point in points:
        point_id = str(point["point_id"])
        result = completed.get(point_id)
        if not isinstance(result, dict):
            raise ValueError(f"selected point is not completed: {point_id}")
        controls = point.get("controls") or {}
        completed_controls = result.get("controls") or {}
        for control in ("ploss", "noise_power_dB"):
            if not math.isclose(
                float(controls[control]),
                float(completed_controls[control]),
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"campaign state disagrees for {point_id}: {control}")
        rows.append({
            "execution_id": str(result["execution_id"]),
            "point_id": point_id,
            "repetition": int(point["repetition"]),
            "selected_ploss": float(controls["ploss"]),
            "selected_noise_power_dB": float(controls["noise_power_dB"]),
        })
    frame = pd.DataFrame(rows)
    if frame["point_id"].duplicated().any():
        raise ValueError("safe selection contains duplicate point identifiers")
    if frame["execution_id"].duplicated().any():
        raise ValueError("safe selection maps more than one point to an execution")
    return frame.sort_values(["repetition", "point_id"]).reset_index(drop=True)


def select_training_segments(
    segments: pd.DataFrame,
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "execution_id",
        "segment_id",
        "ue",
        "direction",
        "parameter",
        "model_type",
        "model_name",
        "model_index",
        "segment_start_utc",
        "segment_end_utc",
        "duration_s",
        "app_mix",
        "designed_offered_mbps",
        "sent_packets",
        "received_packets",
        "lost_packets",
        "loss_rate",
        "received_mbps",
        "latency_samples",
        "latency_ms_p95",
        "valid_clock_fraction",
        "radio_samples",
        "ue_radio_samples",
        "radio_join_clock",
        "radio_clock_lag_warning",
        "split",
        *CONTROL_COLUMNS,
        *RADIO_OUTPUTS,
        *QUALITY_TRUE_FIELDS,
    }
    missing = sorted(required.difference(segments.columns))
    if missing:
        raise ValueError("segment table is missing fields: " + ", ".join(missing))
    execution_ids = set(selected["execution_id"])
    frame = segments.loc[
        segments["execution_id"].isin(execution_ids)
        & segments["direction"].eq(config["direction"])
    ].copy()
    if set(frame["execution_id"]) != execution_ids:
        raise ValueError("not every selected execution has a downlink segment")
    if len(frame) != len(selected) or frame["execution_id"].duplicated().any():
        raise ValueError("expected exactly one downlink segment per selected execution")
    if not frame["parameter"].astype("string").eq("joint").all():
        raise ValueError("selected segments do not preserve the joint control state")
    if not frame["model_type"].astype("string").eq("AWGN").all():
        raise ValueError("selected segments are not all AWGN")
    if not frame["model_name"].astype("string").eq("rfsimu_channel_enB0").all():
        raise ValueError("selected segments use the wrong RFsim model")
    if not pd.to_numeric(frame["model_index"], errors="coerce").eq(0).all():
        raise ValueError("selected segments use the wrong RFsim model index")
    for field in QUALITY_TRUE_FIELDS:
        if not frame[field].fillna(False).astype(bool).all():
            raise ValueError(f"selected segment has {field}=false")
    if frame["radio_clock_lag_warning"].fillna(True).astype(bool).any():
        raise ValueError("selected segment has a radio clock warning")
    if not frame["radio_join_clock"].astype("string").eq("core_receipt_utc").all():
        raise ValueError("selected segment uses the wrong radio join clock")
    if pd.to_numeric(frame["radio_samples"], errors="coerce").le(0).any():
        raise ValueError("selected segment has no RIC radio samples")
    if pd.to_numeric(frame["ue_radio_samples"], errors="coerce").le(0).any():
        raise ValueError("selected segment has no UE radio samples")
    frame = frame.merge(selected, on="execution_id", validate="one_to_one")
    for observed, expected in (
        ("applied_ploss", "selected_ploss"),
        ("applied_noise_power_dB", "selected_noise_power_dB"),
    ):
        disagreement = (
            pd.to_numeric(frame[observed], errors="coerce")
            .sub(pd.to_numeric(frame[expected], errors="coerce"))
            .abs()
            .gt(1e-9)
        )
        if disagreement.any():
            raise ValueError(f"selected manifest disagrees with {observed}")
    state_counts = frame.groupby(list(CONTROL_COLUMNS)).size()
    minimum = int(config["minimum_repetitions_per_state"])
    if state_counts.lt(minimum).any():
        raise ValueError("a retained control state has too few executions")
    return frame.sort_values([*CONTROL_COLUMNS, "repetition"]).reset_index(drop=True)


def recompute_execution_metrics(
    segments: pd.DataFrame,
    packets: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    required_packets = {
        "execution_id",
        "ue",
        "direction",
        "sent_time_utc",
        "received",
        "lost",
        "size_bytes",
        "latency_ms",
        "packet_clock_valid",
        "negative_latency",
    }
    missing = sorted(required_packets.difference(packets.columns))
    if missing:
        raise ValueError("packet table is missing fields: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    for segment in segments.to_dict("records"):
        packet_rows = packets.loc[
            packets["execution_id"].eq(segment["execution_id"])
            & packets["ue"].astype("string").eq(str(segment["ue"]))
            & packets["direction"].eq(segment["direction"])
            & packets["sent_time_utc"].ge(float(segment["segment_start_utc"]))
            & packets["sent_time_utc"].lt(float(segment["segment_end_utc"]))
        ].copy()
        if packet_rows.empty:
            raise ValueError(f"selected segment has no packets: {segment['execution_id']}")
        received_mask = packet_rows["received"].fillna(False).astype(bool)
        valid_latency = (
            received_mask
            & packet_rows["packet_clock_valid"].fillna(False).astype(bool)
            & ~packet_rows["negative_latency"].fillna(True).astype(bool)
            & packet_rows["latency_ms"].notna()
        )
        latencies = pd.to_numeric(
            packet_rows.loc[valid_latency, "latency_ms"], errors="coerce"
        ).dropna()
        if latencies.empty:
            raise ValueError(f"selected segment has no valid latency: {segment['execution_id']}")
        sent_packets = len(packet_rows)
        received_packets = int(received_mask.sum())
        lost_packets = int(packet_rows["lost"].fillna(False).astype(bool).sum())
        loss_rate = lost_packets / sent_packets
        received_bytes = float(
            pd.to_numeric(
                packet_rows.loc[received_mask, "size_bytes"], errors="coerce"
            ).sum()
        )
        received_mbps = received_bytes * 8 / float(segment["duration_s"]) / 1e6
        latency_ms_p95 = float(latencies.quantile(0.95))
        comparisons = {
            "sent_packets": (sent_packets, int(segment["sent_packets"])),
            "received_packets": (received_packets, int(segment["received_packets"])),
            "lost_packets": (lost_packets, int(segment["lost_packets"])),
            "loss_rate": (loss_rate, float(segment["loss_rate"])),
            "received_mbps": (received_mbps, float(segment["received_mbps"])),
            "latency_ms_p95": (latency_ms_p95, float(segment["latency_ms_p95"])),
        }
        for field, (observed, stored) in comparisons.items():
            if not math.isclose(float(observed), float(stored), rel_tol=0, abs_tol=tolerance):
                raise ValueError(
                    f"packet reconstruction disagrees for {segment['execution_id']}: {field}"
                )
        rows.append({
            "execution_id": str(segment["execution_id"]),
            "point_id": str(segment["point_id"]),
            "repetition": int(segment["repetition"]),
            "split": str(segment["split"]),
            "segment_id": str(segment["segment_id"]),
            "ue": str(segment["ue"]),
            "direction": str(segment["direction"]),
            "app_mix": _json_text(segment["app_mix"]),
            "designed_offered_mbps": float(segment["designed_offered_mbps"]),
            "applied_ploss": float(segment["applied_ploss"]),
            "applied_noise_power_dB": float(segment["applied_noise_power_dB"]),
            "sent_packets": sent_packets,
            "received_packets": received_packets,
            "lost_packets": lost_packets,
            "latency_samples": len(latencies),
            "latency_ms_p95": latency_ms_p95,
            "loss_rate": loss_rate,
            "received_mbps": received_mbps,
            "ss_rsrp_dbm_segment_mean": float(segment["ss_rsrp_dbm_segment_mean"]),
            "ss_rsrq_db_segment_mean": float(segment["ss_rsrq_db_segment_mean"]),
            "ss_sinr_db_segment_mean": float(segment["ss_sinr_db_segment_mean"]),
            "stored_latency_ms_p95_delta": latency_ms_p95
            - float(segment["latency_ms_p95"]),
            "stored_loss_rate_delta": loss_rate - float(segment["loss_rate"]),
            "stored_received_mbps_delta": received_mbps
            - float(segment["received_mbps"]),
        })
    return pd.DataFrame(rows).sort_values(
        [*CONTROL_COLUMNS, "repetition"]
    ).reset_index(drop=True)


def build_state_mapping(executions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outputs = (*RADIO_OUTPUTS, *APPLICATION_OUTPUTS)
    for controls, group in executions.groupby(list(CONTROL_COLUMNS), sort=True):
        row: dict[str, Any] = {
            "applied_ploss": float(controls[0]),
            "applied_noise_power_dB": float(controls[1]),
            "execution_count": len(group),
            "execution_ids": _json_text(sorted(group["execution_id"].astype(str))),
        }
        for output in outputs:
            values = pd.to_numeric(group[output], errors="raise")
            row[f"{output}_mean"] = float(values.mean())
            row[f"{output}_std"] = float(values.std(ddof=1))
            row[f"{output}_min"] = float(values.min())
            row[f"{output}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(CONTROL_COLUMNS)).reset_index(drop=True)


def cross_execution_validation(
    executions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = (*RADIO_OUTPUTS, *APPLICATION_OUTPUTS)
    rows: list[dict[str, Any]] = []
    for held_out in executions.to_dict("records"):
        peers = executions.loc[
            executions["applied_ploss"].eq(held_out["applied_ploss"])
            & executions["applied_noise_power_dB"].eq(
                held_out["applied_noise_power_dB"]
            )
            & ~executions["execution_id"].eq(held_out["execution_id"])
        ]
        if peers.empty:
            raise ValueError("cross-execution validation found a state with no peer")
        row: dict[str, Any] = {
            "execution_id": held_out["execution_id"],
            "point_id": held_out["point_id"],
            "repetition": held_out["repetition"],
            "split": held_out["split"],
            "applied_ploss": held_out["applied_ploss"],
            "applied_noise_power_dB": held_out["applied_noise_power_dB"],
            "training_execution_ids": _json_text(
                sorted(peers["execution_id"].astype(str))
            ),
        }
        for target in targets:
            actual = float(held_out[target])
            predicted = float(pd.to_numeric(peers[target], errors="raise").mean())
            error = predicted - actual
            row[f"{target}_actual"] = actual
            row[f"{target}_predicted"] = predicted
            row[f"{target}_error"] = error
            row[f"{target}_absolute_error"] = abs(error)
        rows.append(row)
    predictions = pd.DataFrame(rows).sort_values(
        [*CONTROL_COLUMNS, "repetition"]
    ).reset_index(drop=True)
    metrics: dict[str, Any] = {
        "evaluation_policy": "leave_one_execution_out_within_control_state",
        "split_unit": "execution_id",
        "held_out_executions": len(predictions),
        "targets": {},
    }
    for target in targets:
        errors = pd.to_numeric(predictions[f"{target}_error"], errors="raise")
        absolute = errors.abs()
        metrics["targets"][target] = {
            "mae": float(absolute.mean()),
            "rmse": math.sqrt(float((errors.pow(2)).mean())),
            "max_absolute_error": float(absolute.max()),
        }
    return predictions, metrics


def _steady_anchors(ucc_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = []
    for trace in ucc_manifest.get("traces") or []:
        if trace.get("classification") != "steady_anchor":
            continue
        metrics = ((trace.get("selected_window") or {}).get("metrics") or {})
        anchors.append({
            "trace_id": str(trace["trace_id"]),
            "app": str(trace.get("app") or ""),
            "content": str(trace.get("content") or ""),
            "RSRP": float(metrics["RSRP"]["p50"]),
            "RSRQ": float(metrics["RSRQ"]["p50"]),
            "SNR": float(metrics["SNR"]["p50"]),
        })
    if not anchors:
        raise ValueError("UCC manifest contains no steady anchors")
    return sorted(anchors, key=lambda item: item["trace_id"])


def rank_anchor_candidates(
    state_mapping: pd.DataFrame,
    ucc_manifest: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    inverse = config["inverse_matching"]
    selection_metrics = inverse["selection_metrics"]
    diagnostic_metrics = inverse.get("diagnostic_metrics") or []
    observed_ranges = {}
    for metric in selection_metrics:
        mean_column = f"{metric['observed_metric']}_mean"
        observed_ranges[metric["reference_metric"]] = (
            float(state_mapping[mean_column].min()),
            float(state_mapping[mean_column].max()),
        )
    rows: list[dict[str, Any]] = []
    for anchor in _steady_anchors(ucc_manifest):
        anchor_rows = []
        for state in state_mapping.to_dict("records"):
            row: dict[str, Any] = {
                "trace_id": anchor["trace_id"],
                "app": anchor["app"],
                "content": anchor["content"],
                "candidate_ploss": float(state["applied_ploss"]),
                "candidate_noise_power_dB": float(
                    state["applied_noise_power_dB"]
                ),
                "execution_count": int(state["execution_count"]),
            }
            squared = []
            within_tolerance = []
            target_in_range = []
            for metric in selection_metrics:
                reference = str(metric["reference_metric"])
                observed = str(metric["observed_metric"])
                tolerance = float(metric["tolerance"])
                target = float(anchor[reference])
                candidate = float(state[f"{observed}_mean"])
                error = candidate - target
                row[f"target_{reference.lower()}"] = target
                row[f"candidate_{observed}"] = candidate
                row[f"candidate_{observed}_std"] = float(
                    state[f"{observed}_std"]
                )
                row[f"{reference.lower()}_error"] = error
                row[f"{reference.lower()}_absolute_error"] = abs(error)
                squared.append((error / tolerance) ** 2)
                within_tolerance.append(abs(error) <= tolerance)
                lower, upper = observed_ranges[reference]
                target_in_range.append(lower <= target <= upper)
            row["primary_distance"] = math.sqrt(sum(squared) / len(squared))
            row["within_declared_tolerance"] = all(within_tolerance)
            row["target_within_observed_primary_range"] = all(target_in_range)
            for metric in diagnostic_metrics:
                reference = str(metric["reference_metric"])
                observed = str(metric["observed_metric"])
                target = float(anchor[reference])
                candidate = float(state[f"{observed}_mean"])
                row[f"target_{reference.lower()}_diagnostic"] = target
                row[f"candidate_{observed}_diagnostic"] = candidate
                row[f"{reference.lower()}_proxy_error"] = candidate - target
            for output in APPLICATION_OUTPUTS:
                row[f"candidate_{output}_mean"] = float(state[f"{output}_mean"])
                row[f"candidate_{output}_std"] = float(state[f"{output}_std"])
            anchor_rows.append(row)
        anchor_rows.sort(key=lambda item: (
            item["primary_distance"],
            item["candidate_ploss"],
            item["candidate_noise_power_dB"],
        ))
        for rank, row in enumerate(anchor_rows, start=1):
            row["primary_rank"] = rank
            row["candidate_role"] = (
                "nearest_observed_state" if rank == 1 else "alternative_observed_state"
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["trace_id", "primary_rank"]
    ).reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def run_static_mapping(
    *,
    dataset_dir: str | Path,
    selection_manifest: str | Path,
    campaign_state: str | Path,
    ucc_manifest: str | Path,
    comparison_contract: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    selection_path = Path(selection_manifest).resolve()
    campaign_path = Path(campaign_state).resolve()
    ucc_path = Path(ucc_manifest).resolve()
    contract_path = Path(comparison_contract).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"mapping output already exists: {output_dir}")
    config = _read_json(config_path)
    validate_mapping_config(config)
    dataset_files_verified = verify_dataset_checksums(dataset_dir)
    dataset_manifest = _read_json(dataset_dir / "dataset_manifest.json")
    if dataset_manifest.get("schema_version") != 2:
        raise ValueError("static mapping requires Dataset Contract V2")
    if dataset_manifest.get("split_unit") != "execution_id":
        raise ValueError("dataset split unit must be execution_id")
    contract = _read_json(contract_path)
    if contract.get("split_unit") != "source_trace_and_execution":
        raise ValueError("comparison contract has the wrong split policy")
    selection = _read_json(selection_path)
    campaign = _read_json(campaign_path)
    ucc = _read_json(ucc_path)
    selected = selected_execution_metadata(selection, campaign)
    segments = pd.read_parquet(dataset_dir / "segment_training_table.parquet")
    packets = pd.read_parquet(dataset_dir / "packet_outcomes.parquet")
    training = select_training_segments(segments, selected, config)
    executions = recompute_execution_metrics(training, packets)
    state_mapping = build_state_mapping(executions)
    predictions, validation = cross_execution_validation(executions)
    candidates = rank_anchor_candidates(state_mapping, ucc, config)
    delta_columns = [
        "stored_latency_ms_p95_delta",
        "stored_loss_rate_delta",
        "stored_received_mbps_delta",
    ]
    max_deltas = {
        column: float(pd.to_numeric(executions[column], errors="raise").abs().max())
        for column in delta_columns
    }
    split_counts = {
        str(name): int(count)
        for name, count in executions.groupby("split")["execution_id"].nunique().items()
    }
    mapping_manifest = {
        "schema_version": 1,
        "mapping_id": config["name"],
        "model_kind": "empirical_safe_state_lookup",
        "candidate_policy": config["candidate_policy"],
        "direction": config["direction"],
        "conditional_pre_run_inputs": list(config["conditional_pre_run_inputs"]),
        "pre_run_context": list(config.get("pre_run_context") or []),
        "radio_outputs": list(RADIO_OUTPUTS),
        "application_outputs": list(APPLICATION_OUTPUTS),
        "packet_percentile_rule": "quantile(packet latency_ms, 0.95)",
        "packet_percentile_source": "packet_outcomes rows inside the half-open segment",
        "post_run_radio_in_input_matrix": False,
        "dataset_files_verified": dataset_files_verified,
        "selected_executions": len(executions),
        "retained_control_states": len(state_mapping),
        "steady_reference_anchors": candidates["trace_id"].nunique(),
        "split_unit": "execution_id",
        "stored_split_execution_counts": split_counts,
        "cross_execution_validation": validation,
        "stored_metric_max_absolute_delta": max_deltas,
        "source_sha256": {
            "dataset_SHA256SUMS.json": _sha256(dataset_dir / "SHA256SUMS.json"),
            "selection_manifest": _sha256(selection_path),
            "campaign_state": _sha256(campaign_path),
            "ucc_manifest": _sha256(ucc_path),
            "comparison_contract": _sha256(contract_path),
            "mapping_config": _sha256(config_path),
        },
        "limitations": [
            "two executions per retained control state",
            "candidate ranking is restricted to observed safe states",
            "cross-execution validation measures repeatability within a known state",
            "UCC SNR and OAI SS-SINR are compared only as a diagnostic proxy",
            "the stored 15/1 train-test split has no validation execution",
            "new POWDER executions are required for held-out inverse-map validation",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ))
    try:
        _write_csv(executions, staging / "execution_metrics.csv")
        _write_csv(state_mapping, staging / "state_mapping.csv")
        _write_csv(predictions, staging / "cross_execution_predictions.csv")
        _write_csv(candidates, staging / "anchor_candidates.csv")
        _write_json(staging / "validation_metrics.json", validation)
        _write_json(staging / "mapping_manifest.json", mapping_manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(staging / "SHA256SUMS.json", checksums)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(output_dir),
        "executions": len(executions),
        "states": len(state_mapping),
        "anchors": int(candidates["trace_id"].nunique()),
        "files": len(checksums) + 1,
    }
