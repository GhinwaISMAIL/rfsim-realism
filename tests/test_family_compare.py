import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rfsim_realism.family_compare import run_family_comparison


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(path: Path, *, mapping_id: str, support: list[bool]) -> None:
    path.mkdir()
    rows = []
    for index, supported in enumerate(support):
        rows.append({
            "scenario_id": "scenario-1",
            "observation_index": index,
            "trace_id": "trace-1",
            "app": "Netflix" if index < 3 else "Download",
            "content": "sample",
            "real_rf_state_id": f"real-{index}",
            "primary_rf_state_id": f"primary-{index}",
            "target_rsrp_dbm": -90.0 - index,
            "target_rsrq_db": -10.0 - index,
            "target_snr_db": 10.0 + index,
            "mapped_control_state_id": "ploss=0|noise_power_dB=-5",
            "mapped_ploss": 0.0,
            "mapped_noise_power_dB": -5.0,
            "within_declared_tolerance": supported,
        })
    pd.DataFrame(rows).to_csv(path / "mapped_observations.csv", index=False)
    manifest = {
        "schema_version": 1,
        "analysis_id": "same-analysis",
        "source": {"archive_sha256": "same-source"},
        "selection": {"trace_ids": ["trace-1"], "complete_rf_observations": len(rows)},
        "rfsim_support": {
            "policy": "nearest_observed_safe_state",
            "mapping_id": mapping_id,
        },
    }
    (path / "distribution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    checksums = {
        name: _sha256(path / name)
        for name in ("distribution_manifest.json", "mapped_observations.csv")
    }
    (path / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )


def test_family_comparison_reports_overlap_and_recommendation(tmp_path):
    primary = tmp_path / "primary"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_bundle(primary, mapping_id="mapping-primary", support=[True, True, False, False, False])
    _write_bundle(
        candidate,
        mapping_id="mapping-candidate",
        support=[False, True, True, True, False],
    )

    result = run_family_comparison(
        primary_dir=primary,
        candidate_dir=candidate,
        primary_label="TDL_B",
        candidate_label="TDL_C",
        output_dir=output,
    )

    assert result["recommended_single_family"] == "TDL_C"
    assert result["primary_supported_fraction"] == pytest.approx(0.4)
    assert result["candidate_supported_fraction"] == pytest.approx(0.6)
    assert result["candidate_incremental_fraction"] == pytest.approx(0.4)
    overlap = pd.read_csv(output / "support_overlap.csv").set_index("support_partition")
    assert overlap.loc["both", "observations"] == 1
    assert overlap.loc["primary_only", "observations"] == 1
    assert overlap.loc["candidate_only", "observations"] == 2
    assert overlap.loc["neither", "observations"] == 1
    assert overlap.loc["union", "observations"] == 4
    manifest = json.loads((output / "family_comparison_manifest.json").read_text())
    assert manifest["result"]["recommended_single_family"] == "TDL_C"
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    assert len(checksums) == 6
    for name, expected in checksums.items():
        assert _sha256(output / name) == expected


def test_family_comparison_rejects_different_reference_observations(tmp_path):
    primary = tmp_path / "primary"
    candidate = tmp_path / "candidate"
    _write_bundle(primary, mapping_id="mapping-primary", support=[True, False])
    _write_bundle(candidate, mapping_id="mapping-candidate", support=[True, False])
    frame = pd.read_csv(candidate / "mapped_observations.csv")
    frame.loc[0, "target_rsrp_dbm"] = -120.0
    frame.to_csv(candidate / "mapped_observations.csv", index=False)
    checksums = json.loads((candidate / "SHA256SUMS.json").read_text())
    checksums["mapped_observations.csv"] = _sha256(candidate / "mapped_observations.csv")
    (candidate / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="target_rsrp_dbm"):
        run_family_comparison(
            primary_dir=primary,
            candidate_dir=candidate,
            primary_label="TDL_B",
            candidate_label="TDL_C",
            output_dir=tmp_path / "comparison",
        )


def test_family_comparison_rejects_corrupted_bundle(tmp_path):
    primary = tmp_path / "primary"
    candidate = tmp_path / "candidate"
    _write_bundle(primary, mapping_id="mapping-primary", support=[True])
    _write_bundle(candidate, mapping_id="mapping-candidate", support=[False])
    with (candidate / "mapped_observations.csv").open("a") as stream:
        stream.write("corrupt\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        run_family_comparison(
            primary_dir=primary,
            candidate_dir=candidate,
            primary_label="TDL_B",
            candidate_label="TDL_C",
            output_dir=tmp_path / "comparison",
        )
