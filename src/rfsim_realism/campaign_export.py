from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .sweep import plan_sha256, verify_checksums, write_json

RESULT_FIELDS = [
    "point_id",
    "parameter",
    "value",
    "repetition",
    "execution_id",
    "measurement_start_utc",
    "measurement_end_utc",
    "measurement_duration_s",
    "ss_rsrp_dbm_segment_mean",
    "ss_rsrq_db_segment_mean",
    "ss_sinr_db_segment_mean",
    "latency_ms_p95",
    "loss_rate",
    "valid_clock_fraction",
    "radio_clock_lag_s_p95",
    "ue_radio_emit_lag_s_p95",
    "verified_files",
]


VERIFY_SCRIPT = '''from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
manifest = json.loads((root / "BUNDLE_SHA256SUMS.json").read_text())
for relative, expected in manifest.items():
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or unsafe file: {relative}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"checksum mismatch: {relative}")
print(f"verified {len(manifest)} files")
'''


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_state(state: dict[str, Any]) -> dict[str, Any]:
    portable = copy.deepcopy(state)
    portable.pop("dashboard_repo", None)
    portable.pop("run_dir", None)
    for repository in portable.get("repositories", {}).values():
        repository.pop("path", None)
    for result in portable["completed"].values():
        result["archive"] = f'executions/{result["execution_id"]}'
    return portable


def _write_results(
    path: Path,
    completed: dict[str, dict[str, Any]],
    plan: dict[str, Any],
) -> None:
    points = {point["point_id"]: point for point in plan["points"]}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for point_id in sorted(completed):
            point = points[point_id]
            result = completed[point_id]
            row = {
                "point_id": point_id,
                "parameter": point["parameter"],
                "value": point["value"],
                "repetition": point["repetition"],
                **{field: result.get(field) for field in RESULT_FIELDS[4:]},
            }
            writer.writerow(row)


def verify_bundle(root: str | Path) -> int:
    root = Path(root)
    manifest = json.loads((root / "BUNDLE_SHA256SUMS.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("bundle checksum manifest is missing or empty")
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"bundle file is missing or unsafe: {relative}")
        if _digest(path) != expected:
            raise ValueError(f"bundle checksum mismatch: {relative}")
    return len(manifest)


def export_campaign(
    state_path: str | Path,
    config_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    state_path = Path(state_path).resolve()
    config_path = Path(config_path).resolve()
    plan_path = Path(plan_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"export destination already exists: {output}")

    state = json.loads(state_path.read_text())
    plan = json.loads(plan_path.read_text())
    point_ids = {point["point_id"] for point in plan["points"]}
    completed_ids = set(state.get("completed", {}))
    if state.get("failures"):
        raise ValueError("campaign contains failures")
    if completed_ids != point_ids:
        missing = sorted(point_ids - completed_ids)
        extra = sorted(completed_ids - point_ids)
        raise ValueError(f"campaign coverage mismatch: missing={missing}, extra={extra}")
    if state.get("plan_sha256") != plan_sha256(plan):
        raise ValueError("campaign state does not match the supplied sweep plan")

    output.mkdir(parents=True)
    (output / "configs").mkdir()
    (output / "manifests").mkdir()
    executions = output / "executions"
    executions.mkdir()
    shutil.copy2(config_path, output / "configs" / config_path.name)
    shutil.copy2(plan_path, output / "manifests" / plan_path.name)

    archive_files = 0
    for point_id in sorted(completed_ids):
        result = state["completed"][point_id]
        source = Path(result["archive"]).resolve()
        archive_files += verify_checksums(source)
        destination = executions / result["execution_id"]
        shutil.copytree(source, destination, copy_function=shutil.copy2)

    portable = _portable_state(state)
    write_json(output / "campaign_state.json", portable)
    _write_results(output / "campaign_results.csv", portable["completed"], plan)
    (output / ".gitattributes").write_text(
        "campaign_results.csv -diff\n"
        "executions/** -diff\n"
        "*.gz binary\n"
        "*.parquet binary\n"
    )
    (output / "verify_bundle.py").write_text(VERIFY_SCRIPT)
    (output / "README.md").write_text(
        "# RFsim AWGN Calibration Dataset\n\n"
        "Private companion dataset for the RFsim Realism calibration project.\n"
        "It contains only the executions accepted by the campaign state. Each\n"
        "execution retains its original immutable checksum manifest.\n\n"
        "Verify the complete repository after cloning with:\n\n"
        "```bash\npython3 verify_bundle.py\n```\n\n"
        "`campaign_state.json` uses repository-relative archive paths. The original\n"
        "raw execution files are preserved without modification.\n"
    )

    bundle_manifest = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name != "BUNDLE_SHA256SUMS.json":
            bundle_manifest[str(path.relative_to(output))] = _digest(path)
    write_json(output / "BUNDLE_SHA256SUMS.json", bundle_manifest)
    verified_bundle_files = verify_bundle(output)
    return {
        "output": str(output),
        "points": len(completed_ids),
        "archive_files": archive_files,
        "bundle_files": verified_bundle_files,
    }
