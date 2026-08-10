from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

KEY_COLUMNS = ("scenario_id", "observation_index")
REFERENCE_COLUMNS = (
    "trace_id",
    "app",
    "content",
    "real_rf_state_id",
    "primary_rf_state_id",
    "target_rsrp_dbm",
    "target_rsrq_db",
    "target_snr_db",
)
MAPPING_COLUMNS = (
    "mapped_control_state_id",
    "mapped_ploss",
    "mapped_noise_power_dB",
    "within_declared_tolerance",
)


def _sha256(path: Path) -> str:
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


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def _verify_bundle(bundle_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], int]:
    checksums = _read_json(bundle_dir / "SHA256SUMS.json")
    for relative, expected in sorted(checksums.items()):
        path = (bundle_dir / relative).resolve()
        if bundle_dir not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe support-bundle file: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"support-bundle checksum mismatch: {relative}")
    manifest = _read_json(bundle_dir / "distribution_manifest.json")
    support = manifest.get("rfsim_support") or {}
    if support.get("policy") != "nearest_observed_safe_state":
        raise ValueError("family comparison requires observed safe-state support bundles")
    frame = pd.read_csv(bundle_dir / "mapped_observations.csv")
    required = {*KEY_COLUMNS, *REFERENCE_COLUMNS, *MAPPING_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("mapped observations are missing fields: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("mapped observations contain no rows")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("mapped observations contain duplicate observation identifiers")
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True), manifest, len(checksums)


def _boolean_support(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("within_declared_tolerance must contain only true or false")
    return normalized.eq("true")


def _validate_same_reference(
    primary: pd.DataFrame,
    candidate: pd.DataFrame,
    primary_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> None:
    if len(primary) != len(candidate):
        raise ValueError("family support bundles contain different observation counts")
    if primary_manifest.get("analysis_id") != candidate_manifest.get("analysis_id"):
        raise ValueError("family support bundles use different RF analyses")
    if primary_manifest.get("source") != candidate_manifest.get("source"):
        raise ValueError("family support bundles use different RF sources")
    if primary_manifest.get("selection") != candidate_manifest.get("selection"):
        raise ValueError("family support bundles use different RF selections")
    for column in (*KEY_COLUMNS, *REFERENCE_COLUMNS):
        left = primary[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = left.eq(right) | (left.isna() & right.isna())
        else:
            equal = left.astype("string").eq(right.astype("string"))
        if not equal.all():
            raise ValueError(f"family support bundles disagree on {column}")


def _state_contributions(
    aligned: pd.DataFrame,
    *,
    family: str,
    role: str,
    other_role: str,
) -> pd.DataFrame:
    support = aligned[f"{role}_supported"]
    other_support = aligned[f"{other_role}_supported"]
    controls = [
        f"{role}_control_state_id",
        f"{role}_ploss",
        f"{role}_noise_power_dB",
    ]
    working = aligned.loc[:, controls].copy()
    working["supported"] = support
    working["unique_over_other"] = support & ~other_support
    rows = []
    for values, group in working.groupby(controls, dropna=False, sort=True):
        rows.append({
            "family": family,
            "role": role,
            "mapped_control_state_id": values[0],
            "mapped_ploss": values[1],
            "mapped_noise_power_dB": values[2],
            "assigned_observations": len(group),
            "supported_observations": int(group["supported"].sum()),
            "unique_over_other_observations": int(group["unique_over_other"].sum()),
        })
    return pd.DataFrame(rows)


def compare_family_support(
    primary: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    primary_label: str,
    candidate_label: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if not primary_label.strip() or not candidate_label.strip():
        raise ValueError("family labels must be non-empty")
    if primary_label == candidate_label:
        raise ValueError("family labels must be distinct")
    primary_support = _boolean_support(primary["within_declared_tolerance"])
    candidate_support = _boolean_support(candidate["within_declared_tolerance"])
    aligned = primary.loc[:, [*KEY_COLUMNS, *REFERENCE_COLUMNS]].copy()
    aligned["primary_control_state_id"] = primary["mapped_control_state_id"]
    aligned["primary_ploss"] = primary["mapped_ploss"]
    aligned["primary_noise_power_dB"] = primary["mapped_noise_power_dB"]
    aligned["primary_supported"] = primary_support
    aligned["candidate_control_state_id"] = candidate["mapped_control_state_id"]
    aligned["candidate_ploss"] = candidate["mapped_ploss"]
    aligned["candidate_noise_power_dB"] = candidate["mapped_noise_power_dB"]
    aligned["candidate_supported"] = candidate_support
    aligned["both_supported"] = primary_support & candidate_support
    aligned["primary_only"] = primary_support & ~candidate_support
    aligned["candidate_only"] = candidate_support & ~primary_support
    aligned["union_supported"] = primary_support | candidate_support

    total = len(aligned)
    primary_count = int(primary_support.sum())
    candidate_count = int(candidate_support.sum())
    both_count = int(aligned["both_supported"].sum())
    primary_only_count = int(aligned["primary_only"].sum())
    candidate_only_count = int(aligned["candidate_only"].sum())
    union_count = int(aligned["union_supported"].sum())
    recommended = primary_label if primary_count >= candidate_count else candidate_label

    family_summary = pd.DataFrame([
        {
            "family": primary_label,
            "role": "primary",
            "supported_observations": primary_count,
            "supported_fraction": primary_count / total,
            "unique_over_other_observations": primary_only_count,
            "unique_over_other_fraction": primary_only_count / total,
        },
        {
            "family": candidate_label,
            "role": "candidate",
            "supported_observations": candidate_count,
            "supported_fraction": candidate_count / total,
            "unique_over_other_observations": candidate_only_count,
            "unique_over_other_fraction": candidate_only_count / total,
        },
    ])
    overlap = pd.DataFrame([
        {"support_partition": "both", "observations": both_count},
        {"support_partition": "primary_only", "observations": primary_only_count},
        {"support_partition": "candidate_only", "observations": candidate_only_count},
        {"support_partition": "neither", "observations": total - union_count},
        {"support_partition": "union", "observations": union_count},
    ])
    overlap["fraction"] = overlap["observations"] / total
    states = pd.concat([
        _state_contributions(
            aligned,
            family=primary_label,
            role="primary",
            other_role="candidate",
        ),
        _state_contributions(
            aligned,
            family=candidate_label,
            role="candidate",
            other_role="primary",
        ),
    ], ignore_index=True)
    application_rows = []
    for app, group in aligned.groupby("app", sort=True):
        application_rows.append({
            "app": app,
            "observations": len(group),
            "primary_supported": int(group["primary_supported"].sum()),
            "candidate_supported": int(group["candidate_supported"].sum()),
            "both_supported": int(group["both_supported"].sum()),
            "primary_only": int(group["primary_only"].sum()),
            "candidate_only": int(group["candidate_only"].sum()),
            "union_supported": int(group["union_supported"].sum()),
        })
    applications = pd.DataFrame(application_rows)
    for column in (
        "primary_supported",
        "candidate_supported",
        "both_supported",
        "primary_only",
        "candidate_only",
        "union_supported",
    ):
        applications[f"{column}_fraction"] = applications[column] / applications["observations"]

    summary = {
        "observation_count": total,
        "primary_supported_observations": primary_count,
        "primary_supported_fraction": primary_count / total,
        "candidate_supported_observations": candidate_count,
        "candidate_supported_fraction": candidate_count / total,
        "both_supported_observations": both_count,
        "primary_only_observations": primary_only_count,
        "candidate_only_observations": candidate_only_count,
        "candidate_incremental_fraction_over_primary": candidate_only_count / total,
        "union_supported_observations": union_count,
        "union_supported_fraction": union_count / total,
        "recommended_single_family": recommended,
        "recommendation_basis": "maximum representable observations under identical tolerances",
    }
    return summary, {
        "family_summary": family_summary,
        "support_overlap": overlap,
        "state_contributions": states,
        "application_support": applications,
        "support_assignments": aligned,
    }


def run_family_comparison(
    *,
    primary_dir: str | Path,
    candidate_dir: str | Path,
    primary_label: str,
    candidate_label: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    primary_dir = Path(primary_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"family-comparison output already exists: {output_dir}")
    primary, primary_manifest, primary_files = _verify_bundle(primary_dir)
    candidate, candidate_manifest, candidate_files = _verify_bundle(candidate_dir)
    _validate_same_reference(primary, candidate, primary_manifest, candidate_manifest)
    summary, frames = compare_family_support(
        primary,
        candidate,
        primary_label=primary_label,
        candidate_label=candidate_label,
    )
    manifest = {
        "schema_version": 1,
        "analysis_id": "rfsim_family_support_comparison_v1",
        "comparison_scope": "same real RF observations and identical declared tolerances",
        "primary": {
            "label": primary_label,
            "mapping_id": primary_manifest["rfsim_support"]["mapping_id"],
            "verified_bundle_files": primary_files,
            "bundle_sha256": _sha256(primary_dir / "SHA256SUMS.json"),
        },
        "candidate": {
            "label": candidate_label,
            "mapping_id": candidate_manifest["rfsim_support"]["mapping_id"],
            "verified_bundle_files": candidate_files,
            "bundle_sha256": _sha256(candidate_dir / "SHA256SUMS.json"),
        },
        "result": summary,
        "limitations": [
            "support means simultaneous RSRP and RSRQ agreement within declared tolerances",
            "the comparison is restricted to observed, repeatable RFsim control states",
            "SNR and SS-SINR remain diagnostic and do not determine support",
            "family selection does not change the empirical real RF scenario distribution",
        ],
        "outputs": [f"{name}.csv" for name in frames],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, frame in frames.items():
            _write_csv(frame, staging / f"{name}.csv")
        _write_json(staging / "family_comparison_manifest.json", manifest)
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
        "observations": summary["observation_count"],
        "recommended_single_family": summary["recommended_single_family"],
        "primary_supported_fraction": summary["primary_supported_fraction"],
        "candidate_supported_fraction": summary["candidate_supported_fraction"],
        "candidate_incremental_fraction": summary[
            "candidate_incremental_fraction_over_primary"
        ],
        "files": len(checksums) + 1,
    }
