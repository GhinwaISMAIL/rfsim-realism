import json
from pathlib import Path

from rfsim_realism.report import build_report, trace_frame

ROOT = Path(__file__).parents[1]


def test_committed_manifest_has_expected_inventory():
    manifest = json.loads((ROOT / "manifests/ucc_static_v1.json").read_text())
    assert manifest["source"]["official_archive_verified"] is True
    assert manifest["inventory"]["static_trace_files"] == 23
    assert manifest["inventory"]["calibration_eligible"] == 22
    assert manifest["inventory"]["dynamic_replay_eligible"] == 18
    assert len(trace_frame(manifest)) == 23


def test_static_report_is_generated(tmp_path):
    output = build_report(
        ROOT / "manifests/ucc_static_v1.json", tmp_path / "catalog.html")
    text = output.read_text()
    assert "UCC static 5G trace catalog" in text
    assert "abc729d696b5e0ba" in text
    assert (tmp_path / "figures/static_radio_medians.png").is_file()
