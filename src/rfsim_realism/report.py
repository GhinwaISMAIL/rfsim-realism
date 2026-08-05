from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")


def trace_frame(manifest: dict) -> pd.DataFrame:
    rows = []
    for trace in manifest["traces"]:
        window = trace.get("selected_window") or {}
        metrics = window.get("metrics") or {}
        rows.append({
            "trace_id": trace["trace_id"],
            "application": trace["app"],
            "source_path": trace["source_path"],
            "classification": trace["classification"],
            "calibration_eligible": trace["calibration_eligible"],
            "dynamic_replay_eligible": trace["dynamic_replay_eligible"],
            "window_start": window.get("start"),
            "observed_seconds": window.get("observed_seconds"),
            "timestamp_coverage": window.get("timestamp_coverage"),
            "speed_p95_kph": window.get("speed_p95_kph"),
            "rsrp_p50_dbm": (metrics.get("RSRP") or {}).get("p50"),
            "rsrq_p50_db": (metrics.get("RSRQ") or {}).get("p50"),
            "snr_p50_db": (metrics.get("SNR") or {}).get("p50"),
            "cqi_p50": (metrics.get("CQI") or {}).get("p50"),
            "quality_flags": ", ".join(trace.get("quality_flags") or []),
        })
    return pd.DataFrame(rows)


def build_report(manifest_path: str | Path, destination: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    traces = trace_frame(manifest)

    figure_dir = destination.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "static_radio_medians.png"
    colors = {
        "dynamic_static": "#2878B5",
        "steady_anchor": "#55A868",
        "quarantine_mobility": "#C44E52",
        "review": "#8172B2",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for classification, group in traces.groupby("classification"):
        axis.scatter(
            group["rsrp_p50_dbm"], group["snr_p50_db"],
            label=classification, color=colors.get(classification), s=55, alpha=0.85,
        )
    axis.set_xlabel("Selected-window median RSRP (dBm)")
    axis.set_ylabel("Selected-window median SNR (dB)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    inventory = pd.DataFrame([
        {"measure": key, "value": value}
        for key, value in manifest["inventory"].items()
        if not isinstance(value, dict)
    ])
    source = manifest["source"]
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>UCC static 5G trace catalog</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 2rem;
      color: #202124;
    }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem; text-align: left; }}
    th {{ background: #f3f6f9; position: sticky; top: 0; }}
    img {{ max-width: 900px; width: 100%; }}
    code {{ background: #f3f6f9; padding: 0.15rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>UCC static 5G trace catalog</h1>
  <p>Source tag <code>{html.escape(source['ref'])}</code>, commit
  <code>{html.escape(source['commit'])}</code>. Archive SHA-256 verified:
  <code>{html.escape(source['archive_sha256'])}</code>.</p>
  <h2>Inventory</h2>
  {inventory.to_html(index=False, escape=True)}
  <h2>Selected-window radio medians</h2>
  <img src="figures/{figure_path.name}" alt="RSRP and SNR medians by trace classification">
  <h2>Trace catalog</h2>
  {traces.to_html(index=False, escape=True, float_format=lambda value: f'{value:.3f}')}
</body>
</html>
"""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(body)
    temporary.replace(destination)
    return destination
