from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from rfsim_realism.upv_phase3d import _read_yaml
from rfsim_realism.upv_version1_release import (
    FIGURE_NAMES,
    finalize_upv_version1_release,
    validate_version1_release_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "upv_version1_final_release_v1.yaml"


def test_version1_release_config_preserves_scope() -> None:
    config = _read_yaml(CONFIG)
    validate_version1_release_config(config)
    assert config["reservation"]["request_now"] is False
    assert config["phase3m_disposition"]["version1_changed"] is False
    assert "confirmatory_test6_validation" in config["claim_scope"]["prohibited"]


def test_version1_release_rejects_test6_reclassification() -> None:
    config = copy.deepcopy(_read_yaml(CONFIG))
    config["evidence_roles"]["test6"] = "confirmatory_validation"
    with pytest.raises(ValueError, match="Test 6"):
        validate_version1_release_config(config)


def test_finalize_version1_release(tmp_path: Path) -> None:
    output = tmp_path / "release"
    figures = tmp_path / "figures"
    report = tmp_path / "UPV_VERSION1_FINAL_RESULT.md"
    result = finalize_upv_version1_release(
        config_path=CONFIG,
        output_dir=output,
        figures_dir=figures,
        report_path=report,
    )
    release = json.loads((output / "release.json").read_text())
    metrics = pd.read_csv(output / "metrics_summary.csv")
    decomposition = pd.read_csv(output / "test6_error_decomposition.csv")
    assert result["reservation_required"] is False
    assert release["release_status"] == "final_supported_version1_kpi_level_emulator"
    assert len(metrics) == 4
    assert decomposition.set_index("row_group").loc["clipped_rows", "rows"] == 21
    assert report.is_file()
    assert "held-out exploratory evidence" in report.read_text()
    for name in FIGURE_NAMES:
        assert (figures / f"{name}.png").stat().st_size > 10_000
        assert (figures / f"{name}.pdf").stat().st_size > 1_000
