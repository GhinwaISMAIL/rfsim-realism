from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import ucc_static

REAL_RF_METRICS = ("RSRP", "RSRQ", "SNR")
MAPPING_COLUMNS = {
    "applied_ploss",
    "applied_noise_power_dB",
    "execution_count",
    "ss_rsrp_dbm_segment_mean_mean",
    "ss_rsrq_db_segment_mean_mean",
    "ss_sinr_db_segment_mean_mean",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return document


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def _percentile(values: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(values, errors="raise").quantile(quantile))


def _real_state_label(rsrp: float, rsrq: float, snr: float) -> str:
    return f"rsrp={rsrp:g}|rsrq={rsrq:g}|snr={snr:g}"


def _primary_state_label(rsrp: float, rsrq: float) -> str:
    return f"rsrp={rsrp:g}|rsrq={rsrq:g}"


def _control_label(ploss: float, noise: float) -> str:
    return f"ploss={ploss:g}|noise_power_dB={noise:g}"


def validate_distribution_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("distribution schema_version must be 1")
    if not str(config.get("name") or "").strip():
        raise ValueError("distribution name is required")
    selection = config.get("selection") or {}
    if selection.get("require_dynamic_replay_eligible") is not True:
        raise ValueError("the first catalog must use dynamic replay-eligible traces")
    for field in ("applications", "trace_ids"):
        values = selection.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"selection {field} must be a list of non-empty strings")
    catalog = config.get("catalog") or {}
    if catalog.get("scenario_unit") != "selected_trace_window":
        raise ValueError("catalog scenario_unit must be selected_trace_window")
    if catalog.get("trace_weighting") != "equal_trace":
        raise ValueError("catalog trace_weighting must be equal_trace")
    if tuple(catalog.get("joint_metrics") or ()) != REAL_RF_METRICS:
        raise ValueError("catalog joint_metrics must preserve RSRP, RSRQ, and SNR")
    mapping = config.get("mapping")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("mapping must be a YAML mapping")
        if mapping.get("mode") != "optional_annotation":
            raise ValueError("mapping mode must be optional_annotation")
        if mapping.get("policy") != "nearest_observed_safe_state":
            raise ValueError("mapping must use nearest observed safe states")
        if mapping.get("allow_extrapolation") is not False:
            raise ValueError("mapping must reject extrapolation")
        for field in ("rsrp_absolute_tolerance_db", "rsrq_absolute_tolerance_db"):
            if float(mapping.get(field, 0)) <= 0:
                raise ValueError(f"mapping {field} must be positive")
        minimum = float(mapping.get("minimum_representable_fraction", -1))
        if not 0 <= minimum <= 1:
            raise ValueError("mapping minimum representable fraction must be between zero and one")
    temporal = config.get("temporal") or {}
    if int(temporal.get("transition_max_gap_seconds", 0)) != 1:
        raise ValueError("the first catalog must use consecutive-second transitions")
    if temporal.get("missing_seconds") != "preserve":
        raise ValueError("the first catalog must preserve missing seconds")


def _verify_mapping_bundle(mapping_dir: Path) -> tuple[int, dict[str, Any]]:
    checksums = _read_json(mapping_dir / "SHA256SUMS.json")
    for relative, expected in sorted(checksums.items()):
        path = (mapping_dir / relative).resolve()
        if mapping_dir not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe mapping file: {relative}")
        if _sha256_file(path) != expected:
            raise ValueError(f"mapping checksum mismatch: {relative}")
    manifest = _read_json(mapping_dir / "mapping_manifest.json")
    if manifest.get("model_kind") != "empirical_safe_state_lookup":
        raise ValueError("RFsim annotation requires an empirical safe-state mapping")
    if manifest.get("candidate_policy") != "observed_safe_states_only":
        raise ValueError("RFsim mapping bundle permits unsupported candidates")
    return len(checksums), manifest


def _load_state_mapping(mapping_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(mapping_dir / "state_mapping.csv")
    missing = sorted(MAPPING_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError("state mapping is missing fields: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("state mapping contains no safe states")
    controls = ["applied_ploss", "applied_noise_power_dB"]
    if frame.duplicated(controls).any():
        raise ValueError("state mapping contains duplicate control states")
    if pd.to_numeric(frame["execution_count"], errors="raise").lt(2).any():
        raise ValueError("state mapping contains a state with fewer than two executions")
    return frame.sort_values(controls).reset_index(drop=True)


def _select_traces(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict]:
    selection = config["selection"]
    applications = set(selection.get("applications") or [])
    trace_ids = set(selection.get("trace_ids") or [])
    traces = []
    for trace in manifest.get("traces") or []:
        if not trace.get("dynamic_replay_eligible"):
            continue
        window = trace.get("selected_window") or {}
        if not window.get("quality_eligible"):
            continue
        if applications and trace.get("app") not in applications:
            continue
        if trace_ids and trace.get("trace_id") not in trace_ids:
            continue
        traces.append(trace)
    if not traces:
        raise ValueError("distribution selection contains no eligible traces")
    return sorted(traces, key=lambda trace: str(trace["trace_id"]))


def _trace_observations(trace: dict, payload: bytes) -> list[dict[str, Any]]:
    if _sha256_bytes(payload) != trace["source_sha256"]:
        raise ValueError(f"UCC trace checksum mismatch: {trace['trace_id']}")
    raw_rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    observations, _, _ = ucc_static._deduplicate(raw_rows)
    window = trace["selected_window"]
    start = datetime.fromisoformat(window["start"])
    end = datetime.fromisoformat(window["end"])
    selected = [(timestamp, row) for timestamp, row in observations if start <= timestamp <= end]
    if len(selected) != int(window["observed_seconds"]):
        raise ValueError(f"selected UCC window disagrees for trace {trace['trace_id']}")
    allowed_cells = {str(value) for value in window["cell_ids"]}
    complete = []
    previous_timestamp = None
    segment_index = -1
    for timestamp, row in selected:
        if row.get("NetworkMode") != "5G" or str(row.get("CellID")) not in allowed_cells:
            raise ValueError(f"selected UCC window changed radio context: {trace['trace_id']}")
        rsrp = ucc_static._number(row.get("RSRP"))
        rsrq = ucc_static._number(row.get("RSRQ"))
        snr = ucc_static._number(row.get("SNR"))
        if rsrp is None or rsrq is None or snr is None:
            continue
        delta = (
            None
            if previous_timestamp is None
            else int((timestamp - previous_timestamp).total_seconds())
        )
        if delta != 1:
            segment_index += 1
        gap_before = 0 if delta is None else max(delta - 1, 0)
        trace_id = str(trace["trace_id"])
        complete.append(
            {
                "scenario_id": f"ucc-static-{trace_id}",
                "trace_id": trace_id,
                "app": str(trace.get("app") or ""),
                "content": str(trace.get("content") or ""),
                "observation_index": len(complete),
                "scenario_second": int((timestamp - start).total_seconds()),
                "observed_segment_index": segment_index,
                "gap_before_seconds": gap_before,
                "source_timestamp": timestamp,
                "real_rf_state_id": _real_state_label(rsrp, rsrq, snr),
                "primary_rf_state_id": _primary_state_label(rsrp, rsrq),
                "target_rsrp_dbm": rsrp,
                "target_rsrq_db": rsrq,
                "target_snr_db": snr,
            }
        )
        previous_timestamp = timestamp
    window_seconds = int((end - start).total_seconds()) + 1
    expected_complete = round(float(window["primary_radio_coverage"]) * window_seconds)
    if len(complete) != expected_complete:
        raise ValueError(f"primary UCC coverage disagrees for trace {trace['trace_id']}")
    return complete


def _load_observations(
    dataset: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict], dict]:
    traces = _select_traces(manifest, config)
    source_rows, source_metadata = ucc_static._source_csvs(dataset)
    payloads = dict(source_rows)
    rows = []
    for trace in traces:
        source_path = str(trace["source_path"])
        payload = payloads.get(source_path)
        if payload is None:
            raise ValueError(f"selected UCC trace is absent from dataset: {source_path}")
        rows.extend(_trace_observations(trace, payload))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("distribution contains no complete RF observations")
    ordered = frame.sort_values(["trace_id", "source_timestamp"]).reset_index(drop=True)
    return ordered, traces, source_metadata


def _joint_rf_distribution(observations: pd.DataFrame, trace_count: int) -> pd.DataFrame:
    state_columns = [
        "real_rf_state_id",
        "target_rsrp_dbm",
        "target_rsrq_db",
        "target_snr_db",
    ]
    frame = (
        observations.groupby(state_columns, sort=False)
        .agg(
            observations=("trace_id", "size"),
            traces=("trace_id", "nunique"),
        )
        .reset_index()
    )
    frame["pooled_time_probability"] = frame["observations"] / len(observations)
    per_trace = observations.groupby(["trace_id", *state_columns], sort=False).size()
    per_trace = per_trace.rename("observations").reset_index()
    totals = per_trace.groupby("trace_id")["observations"].transform("sum")
    per_trace["within_trace_probability"] = per_trace["observations"] / totals
    equal_trace = (
        per_trace.groupby(state_columns, sort=False)["within_trace_probability"].sum() / trace_count
    )
    equal_trace = equal_trace.rename("equal_trace_probability").reset_index()
    frame = frame.merge(equal_trace, on=state_columns, validate="one_to_one")
    return frame.sort_values(
        ["equal_trace_probability", "target_rsrp_dbm", "target_rsrq_db", "target_snr_db"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _application_joint_distribution(observations: pd.DataFrame) -> pd.DataFrame:
    state_columns = [
        "app",
        "real_rf_state_id",
        "target_rsrp_dbm",
        "target_rsrq_db",
        "target_snr_db",
    ]
    frame = (
        observations.groupby(state_columns, sort=False)
        .agg(
            observations=("trace_id", "size"),
            traces=("trace_id", "nunique"),
        )
        .reset_index()
    )
    app_totals = frame.groupby("app")["observations"].transform("sum")
    frame["within_application_time_probability"] = frame["observations"] / app_totals
    per_trace = observations.groupby(["trace_id", *state_columns], sort=False).size()
    per_trace = per_trace.rename("observations").reset_index()
    trace_totals = per_trace.groupby("trace_id")["observations"].transform("sum")
    per_trace["within_trace_probability"] = per_trace["observations"] / trace_totals
    app_trace_counts = observations.groupby("app")["trace_id"].nunique()
    equal_trace = per_trace.groupby(state_columns, sort=False)["within_trace_probability"].sum()
    equal_trace = (
        (equal_trace / app_trace_counts)
        .rename("equal_trace_within_application_probability")
        .reset_index()
    )
    frame = frame.merge(equal_trace, on=state_columns, validate="one_to_one")
    return frame.sort_values(
        ["app", "equal_trace_within_application_probability", "real_rf_state_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _scenario_sequences(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trace_id, group in observations.groupby("trace_id", sort=True):
        records = group.sort_values("source_timestamp").to_dict("records")
        if not records:
            continue
        start = records[0]
        previous = records[0]
        run_index = 0
        for current in records[1:]:
            delta = int(
                (current["source_timestamp"] - previous["source_timestamp"]).total_seconds()
            )
            if delta == 1 and current["real_rf_state_id"] == previous["real_rf_state_id"]:
                previous = current
                continue
            rows.append(_sequence_row(trace_id, run_index, start, previous))
            run_index += 1
            start = current
            previous = current
        rows.append(_sequence_row(trace_id, run_index, start, previous))
    return pd.DataFrame(rows).sort_values(["trace_id", "run_index"]).reset_index(drop=True)


def _sequence_row(
    trace_id: str,
    run_index: int,
    start: dict[str, Any],
    end: dict[str, Any],
) -> dict[str, Any]:
    duration = int((end["source_timestamp"] - start["source_timestamp"]).total_seconds()) + 1
    return {
        "scenario_id": start["scenario_id"],
        "trace_id": trace_id,
        "app": start["app"],
        "content": start["content"],
        "run_index": run_index,
        "observed_segment_index": int(start["observed_segment_index"]),
        "gap_before_seconds": int(start["gap_before_seconds"]),
        "start_second": int(start["scenario_second"]),
        "end_second_exclusive": int(end["scenario_second"]) + 1,
        "duration_seconds": duration,
        "real_rf_state_id": start["real_rf_state_id"],
        "target_rsrp_dbm": float(start["target_rsrp_dbm"]),
        "target_rsrq_db": float(start["target_rsrq_db"]),
        "target_snr_db": float(start["target_snr_db"]),
    }


def _scenario_index(
    observations: pd.DataFrame,
    sequences: pd.DataFrame,
    traces: list[dict],
) -> pd.DataFrame:
    sequence_counts = sequences.groupby("trace_id").size()
    rows = []
    trace_count = len(traces)
    for trace in traces:
        trace_id = str(trace["trace_id"])
        group = observations.loc[observations["trace_id"] == trace_id]
        window = trace["selected_window"]
        start = datetime.fromisoformat(window["start"])
        end = datetime.fromisoformat(window["end"])
        nominal_seconds = int((end - start).total_seconds()) + 1
        rows.append(
            {
                "scenario_id": f"ucc-static-{trace_id}",
                "trace_id": trace_id,
                "app": str(trace.get("app") or ""),
                "content": str(trace.get("content") or ""),
                "selected_window_start": window["start"],
                "selected_window_end": window["end"],
                "nominal_duration_seconds": nominal_seconds,
                "observed_rf_seconds": len(group),
                "missing_rf_seconds": nominal_seconds - len(group),
                "rf_observation_coverage": len(group) / nominal_seconds,
                "joint_rf_states": int(group["real_rf_state_id"].nunique()),
                "observed_segments": int(group["observed_segment_index"].nunique()),
                "sequence_runs": int(sequence_counts.loc[trace_id]),
                "selection_weight": 1 / trace_count,
                "source_path": str(trace["source_path"]),
                "source_sha256": str(trace["source_sha256"]),
            }
        )
    return pd.DataFrame(rows).sort_values("trace_id").reset_index(drop=True)


def _transition_distribution(observations: pd.DataFrame, max_gap_seconds: int) -> pd.DataFrame:
    transitions = []
    for trace_id, group in observations.groupby("trace_id", sort=True):
        records = group.sort_values("source_timestamp").to_dict("records")
        for previous, current in pairwise(records):
            delta = int(
                (current["source_timestamp"] - previous["source_timestamp"]).total_seconds()
            )
            if delta != max_gap_seconds:
                continue
            transitions.append(
                {
                    "trace_id": trace_id,
                    "from_state_id": previous["real_rf_state_id"],
                    "to_state_id": current["real_rf_state_id"],
                }
            )
    columns = [
        "from_state_id",
        "to_state_id",
        "transitions",
        "traces",
        "conditional_probability",
    ]
    if not transitions:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(transitions)
    grouped = (
        frame.groupby(["from_state_id", "to_state_id"], sort=False)
        .agg(
            transitions=("trace_id", "size"),
            traces=("trace_id", "nunique"),
        )
        .reset_index()
    )
    totals = grouped.groupby("from_state_id")["transitions"].transform("sum")
    grouped["conditional_probability"] = grouped["transitions"] / totals
    return grouped.sort_values(
        ["transitions", "from_state_id", "to_state_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _dwell_distribution(sequences: pd.DataFrame) -> pd.DataFrame:
    grouped = sequences.groupby("real_rf_state_id", sort=False)
    frame = grouped.agg(
        dwell_runs=("duration_seconds", "size"),
        total_seconds=("duration_seconds", "sum"),
        duration_seconds_mean=("duration_seconds", "mean"),
        duration_seconds_p50=("duration_seconds", lambda values: _percentile(values, 0.50)),
        duration_seconds_p90=("duration_seconds", lambda values: _percentile(values, 0.90)),
        duration_seconds_max=("duration_seconds", "max"),
    ).reset_index()
    return frame.sort_values(
        ["total_seconds", "real_rf_state_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _match_observations(
    observations: pd.DataFrame,
    mapping: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rsrp_tolerance = float(config["mapping"]["rsrp_absolute_tolerance_db"])
    rsrq_tolerance = float(config["mapping"]["rsrq_absolute_tolerance_db"])
    rsrp_column = "ss_rsrp_dbm_segment_mean_mean"
    rsrq_column = "ss_rsrq_db_segment_mean_mean"
    sinr_column = "ss_sinr_db_segment_mean_mean"
    rsrp_range = (float(mapping[rsrp_column].min()), float(mapping[rsrp_column].max()))
    rsrq_range = (float(mapping[rsrq_column].min()), float(mapping[rsrq_column].max()))
    rows = []
    states = mapping.to_dict("records")
    for observation in observations.to_dict("records"):
        candidates = []
        for state in states:
            rsrp_error = float(state[rsrp_column]) - float(observation["target_rsrp_dbm"])
            rsrq_error = float(state[rsrq_column]) - float(observation["target_rsrq_db"])
            distance = math.sqrt(
                ((rsrp_error / rsrp_tolerance) ** 2 + (rsrq_error / rsrq_tolerance) ** 2) / 2
            )
            candidates.append(
                (
                    distance,
                    float(state["applied_ploss"]),
                    float(state["applied_noise_power_dB"]),
                    state,
                    rsrp_error,
                    rsrq_error,
                )
            )
        distance, ploss, noise, state, rsrp_error, rsrq_error = min(candidates)
        target_rsrp = float(observation["target_rsrp_dbm"])
        target_rsrq = float(observation["target_rsrq_db"])
        mapped_sinr = float(state[sinr_column])
        rows.append(
            {
                **observation,
                "mapped_control_state_id": _control_label(ploss, noise),
                "mapped_ploss": ploss,
                "mapped_noise_power_dB": noise,
                "mapped_execution_count": int(state["execution_count"]),
                "mapped_ss_rsrp_dbm": float(state[rsrp_column]),
                "mapped_ss_rsrq_db": float(state[rsrq_column]),
                "mapped_ss_sinr_db_diagnostic": mapped_sinr,
                "rsrp_error_db": rsrp_error,
                "rsrp_absolute_error_db": abs(rsrp_error),
                "rsrp_within_declared_tolerance": abs(rsrp_error) <= rsrp_tolerance,
                "rsrq_error_db": rsrq_error,
                "rsrq_absolute_error_db": abs(rsrq_error),
                "rsrq_within_declared_tolerance": abs(rsrq_error) <= rsrq_tolerance,
                "snr_proxy_error_db_diagnostic": mapped_sinr - float(observation["target_snr_db"]),
                "primary_distance": distance,
                "within_declared_tolerance": abs(rsrp_error) <= rsrp_tolerance
                and abs(rsrq_error) <= rsrq_tolerance,
                "target_within_observed_primary_range": rsrp_range[0]
                <= target_rsrp
                <= rsrp_range[1]
                and rsrq_range[0] <= target_rsrq <= rsrq_range[1],
            }
        )
    return pd.DataFrame(rows).sort_values(["trace_id", "source_timestamp"]).reset_index(drop=True)


def _mapping_support_distribution(matched: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "real_rf_state_id",
        "primary_rf_state_id",
        "target_rsrp_dbm",
        "target_rsrq_db",
        "target_snr_db",
        "mapped_control_state_id",
        "mapped_ploss",
        "mapped_noise_power_dB",
        "mapped_ss_rsrp_dbm",
        "mapped_ss_rsrq_db",
        "mapped_ss_sinr_db_diagnostic",
        "rsrp_absolute_error_db",
        "rsrp_within_declared_tolerance",
        "rsrq_absolute_error_db",
        "rsrq_within_declared_tolerance",
        "primary_distance",
        "within_declared_tolerance",
        "target_within_observed_primary_range",
    ]
    frame = (
        matched.groupby(group_columns, dropna=False, sort=False)
        .agg(
            observations=("trace_id", "size"),
            traces=("trace_id", "nunique"),
        )
        .reset_index()
    )
    frame["pooled_time_probability"] = frame["observations"] / len(matched)
    return frame.sort_values(
        ["observations", "target_rsrp_dbm", "target_rsrq_db", "target_snr_db"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _coverage_summary(matched: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in matched.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "observations": len(group),
                "unique_real_rf_states": group["real_rf_state_id"].nunique(),
                "representable_observations": int(group["within_declared_tolerance"].sum()),
                "representable_fraction": float(group["within_declared_tolerance"].mean()),
                "rsrp_within_tolerance_fraction": float(
                    group["rsrp_within_declared_tolerance"].mean()
                ),
                "rsrq_within_tolerance_fraction": float(
                    group["rsrq_within_declared_tolerance"].mean()
                ),
                "in_observed_primary_range_fraction": float(
                    group["target_within_observed_primary_range"].mean()
                ),
                "rsrp_absolute_error_db_mean": float(group["rsrp_absolute_error_db"].mean()),
                "rsrp_absolute_error_db_p95": _percentile(group["rsrp_absolute_error_db"], 0.95),
                "rsrq_absolute_error_db_mean": float(group["rsrq_absolute_error_db"].mean()),
                "rsrq_absolute_error_db_p95": _percentile(group["rsrq_absolute_error_db"], 0.95),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _scenario_documents(index: pd.DataFrame) -> list[dict[str, Any]]:
    documents = []
    for row in index.to_dict("records"):
        documents.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "trace_id": str(row["trace_id"]),
                "application": str(row["app"]),
                "content": str(row["content"]),
                "selected_window": {
                    "start": str(row["selected_window_start"]),
                    "end": str(row["selected_window_end"]),
                    "nominal_duration_seconds": int(row["nominal_duration_seconds"]),
                    "observed_rf_seconds": int(row["observed_rf_seconds"]),
                    "missing_rf_seconds": int(row["missing_rf_seconds"]),
                },
                "joint_rf_states": int(row["joint_rf_states"]),
                "observed_segments": int(row["observed_segments"]),
                "sequence_runs": int(row["sequence_runs"]),
                "selection_weight": float(row["selection_weight"]),
                "source": {
                    "path": str(row["source_path"]),
                    "sha256": str(row["source_sha256"]),
                },
            }
        )
    return documents


def run_distribution_analysis(
    *,
    dataset: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    mapping_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    resolved_mapping_dir = Path(mapping_dir).resolve() if mapping_dir is not None else None
    if output_dir.exists():
        raise FileExistsError(f"distribution output already exists: {output_dir}")
    config = _read_yaml(config_path)
    validate_distribution_config(config)
    manifest = _read_json(manifest_path)
    observations, traces, source_metadata = _load_observations(dataset, manifest, config)
    joint = _joint_rf_distribution(observations, len(traces))
    application_joint = _application_joint_distribution(observations)
    sequences = _scenario_sequences(observations)
    scenario_index = _scenario_index(observations, sequences, traces)
    max_gap = int(config["temporal"]["transition_max_gap_seconds"])
    transitions = _transition_distribution(observations, max_gap)
    dwell_distribution = _dwell_distribution(sequences)

    source_document = {
        "dataset": {
            "archive_name": source_metadata.get("archive_name"),
            "archive_sha256": source_metadata.get("archive_sha256"),
            "official_archive_verified": source_metadata.get("official_archive_verified"),
        },
        "manifest_sha256": _sha256_file(manifest_path),
        "config_sha256": _sha256_file(config_path),
    }
    output_files = [
        "real_rf_observations.csv",
        "joint_rf_distribution.csv",
        "application_joint_rf_distribution.csv",
        "scenario_index.csv",
        "scenario_sequences.csv",
        "transition_distribution.csv",
        "dwell_distribution.csv",
    ]
    mapping_outputs: dict[str, pd.DataFrame] = {}
    mapping_summary = None
    representable_fraction = None
    if resolved_mapping_dir is not None:
        if config.get("mapping") is None:
            raise ValueError("mapping configuration is required when --mapping-dir is used")
        mapping_files_verified, mapping_manifest = _verify_mapping_bundle(resolved_mapping_dir)
        state_mapping = _load_state_mapping(resolved_mapping_dir)
        matched = _match_observations(observations, state_mapping, config)
        support = _mapping_support_distribution(matched)
        trace_coverage = _coverage_summary(matched, ["trace_id", "app", "content"])
        app_coverage = _coverage_summary(matched, ["app"])
        uncovered = support.loc[~support["within_declared_tolerance"]].copy()
        representable_fraction = float(matched["within_declared_tolerance"].mean())
        minimum_fraction = float(config["mapping"]["minimum_representable_fraction"])
        mapping_summary = {
            "role": "optional_replay_capability_annotation",
            "status": (
                "sufficient" if representable_fraction >= minimum_fraction else "insufficient"
            ),
            "mapping_id": mapping_manifest["mapping_id"],
            "bundle_files_verified": mapping_files_verified,
            "bundle_sha256": _sha256_file(resolved_mapping_dir / "SHA256SUMS.json"),
            "policy": config["mapping"]["policy"],
            "safe_states": len(state_mapping),
            "allow_extrapolation": False,
            "rsrp_absolute_tolerance_db": float(config["mapping"]["rsrp_absolute_tolerance_db"]),
            "rsrq_absolute_tolerance_db": float(config["mapping"]["rsrq_absolute_tolerance_db"]),
            "minimum_representable_fraction": minimum_fraction,
            "representable_observations": int(matched["within_declared_tolerance"].sum()),
            "unrepresentable_observations": int((~matched["within_declared_tolerance"]).sum()),
            "representable_fraction": representable_fraction,
            "rsrp_within_tolerance_fraction": float(
                matched["rsrp_within_declared_tolerance"].mean()
            ),
            "rsrq_within_tolerance_fraction": float(
                matched["rsrq_within_declared_tolerance"].mean()
            ),
            "in_observed_primary_range_fraction": float(
                matched["target_within_observed_primary_range"].mean()
            ),
            "uncovered_real_rf_states": len(uncovered),
        }
        mapping_outputs = {
            "mapped_observations.csv": matched,
            "rfsim_support_by_real_state.csv": support,
            "uncovered_real_rf_states.csv": uncovered,
            "trace_rfsim_support.csv": trace_coverage,
            "application_rfsim_support.csv": app_coverage,
        }
        output_files.extend(mapping_outputs)

    catalog = {
        "schema_version": 1,
        "catalog_id": config["name"],
        "status": "ready",
        "source": source_document,
        "measurement_contract": {
            "joint_metrics": ["RSRP_dBm", "RSRQ_dB", "SNR_dB"],
            "state_definition": "observed_joint_triplet",
            "independent_metric_sampling": False,
            "interpolation": False,
        },
        "selection_contract": {
            "scenario_unit": config["catalog"]["scenario_unit"],
            "trace_weighting": config["catalog"]["trace_weighting"],
            "scenario_count": len(traces),
            "selection_weight_sum": float(scenario_index["selection_weight"].sum()),
        },
        "replay_contract": {
            "sequence_file": "scenario_sequences.csv",
            "index_file": "scenario_index.csv",
            "time_origin": "selected_window_start",
            "start_second_is_inclusive": True,
            "end_second_exclusive": True,
            "missing_seconds": "preserved_as_gaps",
            "gap_filling_policy": None,
            "reuse_across_ue_counts_and_traffic_profiles": True,
        },
        "scenarios": _scenario_documents(scenario_index),
        "rfsim_support": mapping_summary,
    }
    analysis_manifest = {
        "schema_version": 1,
        "analysis_id": config["name"],
        "status": "catalog_ready",
        "source": source_document,
        "selection": {
            "scenarios": len(traces),
            "applications": sorted({str(trace["app"]) for trace in traces}),
            "trace_ids": [str(trace["trace_id"]) for trace in traces],
            "complete_rf_observations": len(observations),
            "joint_rf_states": int(observations["real_rf_state_id"].nunique()),
        },
        "probabilities": {
            "primary_catalog_weighting": "equal_trace",
            "equal_trace_probability_column": "equal_trace_probability",
            "pooled_time_probability_column": "pooled_time_probability",
            "metrics_are_sampled_jointly": True,
        },
        "temporal": {
            "transition_max_gap_seconds": max_gap,
            "transitions": int(transitions["transitions"].sum()) if not transitions.empty else 0,
            "sequence_runs": len(sequences),
            "observed_segments": int(scenario_index["observed_segments"].sum()),
            "missing_seconds_are_filled": False,
            "transitions_crossing_gaps_are_counted": False,
        },
        "rfsim_support": mapping_summary,
        "outputs": sorted([*output_files, "scenario_catalog.json"]),
        "limitations": [
            "the source catalog contains static one-cell 5G measurement windows",
            "the catalog reproduces empirical RF conditions rather than predicting them",
            "missing UCC seconds are not imputed",
            "application labels reflect the source dataset composition",
            "RFsim support annotations do not alter the empirical distributions",
            "UCC SNR and OAI SS-SINR are not assumed to be the same measurement",
        ],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        base_outputs = {
            "real_rf_observations.csv": observations,
            "joint_rf_distribution.csv": joint,
            "application_joint_rf_distribution.csv": application_joint,
            "scenario_index.csv": scenario_index,
            "scenario_sequences.csv": sequences,
            "transition_distribution.csv": transitions,
            "dwell_distribution.csv": dwell_distribution,
        }
        for name, frame in {**base_outputs, **mapping_outputs}.items():
            _write_csv(frame, staging / name)
        _write_json(staging / "scenario_catalog.json", catalog)
        _write_json(staging / "distribution_manifest.json", analysis_manifest)
        checksums = {
            path.name: _sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        _write_json(staging / "SHA256SUMS.json", checksums)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(output_dir),
        "status": analysis_manifest["status"],
        "scenarios": len(traces),
        "observations": len(observations),
        "joint_states": int(observations["real_rf_state_id"].nunique()),
        "rfsim_mapping_applied": resolved_mapping_dir is not None,
        "rfsim_representable_fraction": representable_fraction,
        "files": len(checksums) + 1,
    }
