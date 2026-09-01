from __future__ import annotations

# ruff: noqa: E501, RUF001
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .upv_phase3d import _git_revision, _read_json, _read_yaml, _sha256, _write_csv, _write_json

FIGURE_NAMES = (
    "test1_complete_replay",
    "test6_exploratory_replay",
    "test6_error_decomposition",
    "phase3m_ablation",
)


def validate_version1_release_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Version 1 release schema_version must be 1")
    if config.get("stage") != "version1_final_supported_kpi_emulator_release":
        raise ValueError("unexpected Version 1 release stage")
    if config.get("release_status") != "final_supported_version1_kpi_level_emulator":
        raise ValueError("unexpected Version 1 release status")
    runtime = config["runtime_model"]
    if runtime.get("online_adaptation") != "prohibited":
        raise ValueError("Version 1 must remain open loop")
    if runtime.get("dynamic_compensation") != "prohibited":
        raise ValueError("rejected Phase 3M compensation may not enter Version 1")
    if runtime.get("extrapolation") != "prohibited":
        raise ValueError("translator extrapolation must remain prohibited")
    if config["phase3m_disposition"].get("version1_changed") is not False:
        raise ValueError("Phase 3M may not modify Version 1")
    if config["reservation"].get("request_now") is not False:
        raise ValueError("the final offline release does not require a reservation")
    roles = config["evidence_roles"]
    if roles.get("test6") != "posthoc_held_out_exploratory_replay_not_confirmatory_validation":
        raise ValueError("Test 6 must remain exploratory")
    prohibited = set(config["claim_scope"]["prohibited"])
    required = {
        "absolute_rsrp_calibration",
        "physical_channel_reconstruction",
        "confirmatory_test6_validation",
        "cross_device_or_population_generalization",
        "throughput_calibration",
    }
    if not required.issubset(prohibited):
        raise ValueError("required claim limits are missing")


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise ValueError(f"frozen Version 1 input mismatch: {path}")


def _verify_bundle(directory: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS.json"
    _verify_file(manifest, expected_manifest_sha256)
    checksums = _read_json(manifest)
    for name, digest in checksums.items():
        _verify_file(directory / name, str(digest))
    return {
        "path": str(directory),
        "manifest_sha256": expected_manifest_sha256,
        "verified_files": len(checksums),
    }


def _style_axes(axes: np.ndarray | list[Any]) -> None:
    for axis in np.asarray(axes).reshape(-1):
        axis.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: Any, output_base: Path) -> list[Path]:
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _artifact_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def _plot_test1(frame: pd.DataFrame, output_base: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True, constrained_layout=True)
    target = frame[frame["execution_number"] == frame["execution_number"].min()]
    x = target["trace_t_s"]
    axes[0].plot(x, target["target_relative_rsrp_db"], color="black", linewidth=1.8, label="Target")
    axes[1].plot(x, target["target_sinr_db"], color="black", linewidth=1.8, label="Target")
    colors = ("#0072B2", "#D55E00", "#009E73")
    for color, (execution, group) in zip(colors, frame.groupby("execution_number"), strict=False):
        axes[0].plot(
            group["trace_t_s"],
            group["observed_relative_rsrp_db"],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=f"Execution {execution}",
        )
        axes[1].plot(
            group["trace_t_s"],
            group["ss_sinr_db"],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=f"Execution {execution}",
        )
    axes[0].set_ylabel("Relative RSRP (dB)")
    axes[1].set_ylabel("SINR (dB)")
    axes[1].set_xlabel("Replay time (s)")
    axes[0].set_title("Test 1 complete-trace development replay")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    _style_axes(axes)
    return _save_figure(figure, output_base)


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    ranges: list[list[int]] = []
    for index in indices:
        if not ranges or index > ranges[-1][-1] + 1:
            ranges.append([index])
        else:
            ranges[-1].append(index)
    return [(values[0], values[-1]) for values in ranges]


def _plot_test6(frame: pd.DataFrame, output_base: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True, constrained_layout=True)
    x = frame["trace_t_s"]
    series = (
        ("target_relative_rsrp_db", "projected_relative_rsrp_db", "observed_relative_rsrp_db"),
        ("target_sinr_db", "projected_sinr_db", "ss_sinr_db"),
    )
    for axis, columns in zip(axes, series, strict=True):
        axis.plot(x, frame[columns[0]], color="black", linewidth=1.7, label="Original target")
        axis.plot(x, frame[columns[1]], color="#E69F00", linewidth=1.2, label="Projected target")
        axis.plot(x, frame[columns[2]], color="#0072B2", linewidth=1.0, label="OAI output")
        clipped = frame[frame["clipped"]]
        axis.scatter(
            clipped["trace_t_s"],
            clipped[columns[0]],
            color="#CC79A7",
            marker="x",
            s=28,
            linewidths=1.2,
            label="Clipped target",
            zorder=4,
        )
        for start, end in _contiguous_ranges(clipped.index.to_list()):
            axis.axvspan(
                float(frame.loc[start, "trace_t_s"]) - 0.5,
                float(frame.loc[end, "trace_t_s"]) + 0.5,
                color="#CC79A7",
                alpha=0.08,
            )
    axes[0].set_ylabel("Relative RSRP (dB)")
    axes[1].set_ylabel("SINR (dB)")
    axes[1].set_xlabel("Replay time (s)")
    axes[0].set_title("Test 6 held-out exploratory replay")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    _style_axes(axes)
    return _save_figure(figure, output_base)


def _error_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "complete_trace": frame,
        "supported_rows": frame.loc[~frame["clipped"]],
        "clipped_rows": frame.loc[frame["clipped"]],
    }
    records: list[dict[str, Any]] = []
    for group_name, group in groups.items():
        record: dict[str, Any] = {"row_group": group_name, "rows": len(group)}
        for component in ("translator", "dynamic", "total"):
            for metric in ("relative_rsrp", "sinr"):
                values = group[f"{component}_{metric}_error_db"].astype(float)
                record[f"{component}_{metric}_mae_db"] = float(values.abs().mean())
                record[f"{component}_{metric}_maximum_absolute_error_db"] = float(
                    values.abs().max()
                )
                record[f"{component}_{metric}_signed_bias_db"] = float(values.mean())
        records.append(record)
    return pd.DataFrame(records)


def _plot_error_decomposition(decomposition: pd.DataFrame, output_base: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    components = ("translator", "dynamic", "total")
    labels = ("Clipping", "Dynamic", "Total")
    colors = ("#E69F00", "#0072B2", "#009E73")
    x = np.arange(3)
    width = 0.23
    for axis, metric, title in zip(
        axes,
        ("relative_rsrp", "sinr"),
        ("Relative RSRP", "SINR"),
        strict=True,
    ):
        for offset, (component, label, color) in enumerate(
            zip(components, labels, colors, strict=True)
        ):
            values = [
                float(
                    decomposition.loc[
                        decomposition["row_group"] == row_group,
                        f"{component}_{metric}_mae_db",
                    ].iloc[0]
                )
                for row_group in ("complete_trace", "supported_rows", "clipped_rows")
            ]
            axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axis.set_xticks(x, ("All\n(n=297)", "Supported\n(n=276)", "Clipped\n(n=21)"))
        axis.set_ylabel("Mean absolute error (dB)")
        axis.set_title(title)
    axes[0].legend(fontsize=8)
    figure.suptitle("Test 6 error decomposition")
    _style_axes(axes)
    return _save_figure(figure, output_base)


def _plot_phase3m(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), constrained_layout=True)
    labels = {"static": "Static V1", "memory_only": "Memory-only", "combined": "Combined"}
    order = ["static", "memory_only", "combined"]
    ordered = summary.set_index("model").loc[order]
    colors = ("#0072B2", "#E69F00", "#009E73")
    specifications = (
        ("mean_mae_db", "Mean MAE (dB)"),
        ("mean_p95_absolute_error_db", "Mean p95 error (dB)"),
        ("mean_absolute_residual_lag1_correlation", "Mean |residual lag-1 r|"),
    )
    for axis, (column, title) in zip(axes, specifications, strict=True):
        values = ordered[column].astype(float).to_numpy()
        axis.bar(np.arange(3), values, color=colors)
        axis.set_xticks(np.arange(3), [labels[item] for item in order], rotation=20, ha="right")
        axis.set_title(title)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Phase 3M first-order SINR dynamics ablation (rejected)")
    _style_axes(axes)
    return _save_figure(figure, output_base)


def _metrics_summary(test1: pd.DataFrame, test6: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in test1.to_dict("records"):
        records.append(
            {
                "evidence": "test1_development",
                "execution_number": int(row["execution_number"]),
                "rows": int(row["paired_rows"]),
                "clipped_rows": int(row["clipped_rows"]),
                "relative_rsrp_mae_db": row["total_relative_rsrp_mae_db"],
                "sinr_mae_db": row["total_sinr_mae_db"],
                "relative_rsrp_correlation": row["total_relative_rsrp_pearson_correlation"],
                "sinr_correlation": row["total_sinr_pearson_correlation"],
                "scaled_joint_energy_distance": row["total_scaled_joint_energy_distance"],
                "maximum_command_lateness_seconds": row[
                    "maximum_command_completion_lateness_seconds"
                ],
                "runtime_gate_passed": bool(row["runtime_gate_passed"]),
                "fidelity_gate_passed": bool(row["fidelity_gate_passed"]),
            }
        )
    row = test6.iloc[0]
    records.append(
        {
            "evidence": "test6_held_out_exploratory",
            "execution_number": int(row["execution_number"]),
            "rows": int(row["paired_rows"]),
            "clipped_rows": int(row["clipped_rows"]),
            "relative_rsrp_mae_db": row["total_relative_rsrp_mae_db"],
            "sinr_mae_db": row["total_sinr_mae_db"],
            "relative_rsrp_correlation": row["total_relative_rsrp_pearson_correlation"],
            "sinr_correlation": row["total_sinr_pearson_correlation"],
            "scaled_joint_energy_distance": row["total_scaled_joint_energy_distance"],
            "maximum_command_lateness_seconds": row["maximum_command_completion_lateness_seconds"],
            "runtime_gate_passed": bool(row["runtime_gate_passed"]),
            "fidelity_gate_passed": bool(row["fidelity_gate_passed"]),
        }
    )
    return pd.DataFrame(records)


def _report_text(
    *,
    metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    phase3m: pd.DataFrame,
    release: dict[str, Any],
) -> str:
    test1 = metrics[metrics["evidence"] == "test1_development"]
    test6 = metrics[metrics["evidence"] == "test6_held_out_exploratory"].iloc[0]
    clipped = decomposition[decomposition["row_group"] == "clipped_rows"].iloc[0]
    static = phase3m.set_index("model").loc["static"]
    memory = phase3m.set_index("model").loc["memory_only"]
    combined = phase3m.set_index("model").loc["combined"]
    return f"""# UPV-informed RFsim Version 1 final result

## Research question

Can a deterministic radio-condition model driven by a measured one-cell trace, when translated into bounded RFsim scalar-gain and effective-noise controls, reproduce predefined marginal, joint, and temporal properties of relative RSRP and SINR in OAI?

## Released model

Version 1 is a measurement-driven, KPI-level radio-condition emulator. It consumes synchronized relative RSRP and device-conditioned SINR targets, maps each pair to scalar gain and effective noise through bounded piecewise-affine interpolation, and applies the resulting commands to the AWGN RFsim channel once per second. Targets outside the validated translator hull are projected to the nearest hull boundary and explicitly flagged. The emulator never extrapolates or adapts commands during execution.

This is not a physical channel reconstruction. It does not infer absolute path loss or noise, and it does not reproduce multipath or Doppler.

## Evidence

### Test 1 complete-trace development evaluation

Three fresh UE recreations replayed all 305 target seconds. Every execution passed the frozen runtime and fidelity gates. Mean relative-RSRP MAE was {test1["relative_rsrp_mae_db"].mean():.3f} dB (range {test1["relative_rsrp_mae_db"].min():.3f}–{test1["relative_rsrp_mae_db"].max():.3f}); mean SINR MAE was {test1["sinr_mae_db"].mean():.3f} dB (range {test1["sinr_mae_db"].min():.3f}–{test1["sinr_mae_db"].max():.3f}). Mean correlations were {test1["relative_rsrp_correlation"].mean():.4f} for relative RSRP and {test1["sinr_correlation"].mean():.4f} for SINR. Maximum command lateness across the three executions was {test1["maximum_command_lateness_seconds"].max():.3f} s.

Test 1 is development fidelity and repeatability evidence, not independent final validation.

### Test 6 held-out exploratory evaluation

The unchanged translator replayed all {int(test6["rows"])} Test 6 target seconds in one execution. The predeclared support gate had already classified the trajectory as unsupported because {int(test6["clipped_rows"])}/{int(test6["rows"])} rows ({100 * test6["clipped_rows"] / test6["rows"]:.2f}%) required projection. That verdict remains unchanged.

Despite the modest support violation, complete-trace relative-RSRP MAE was {test6["relative_rsrp_mae_db"]:.3f} dB and SINR MAE was {test6["sinr_mae_db"]:.3f} dB. Correlations were {test6["relative_rsrp_correlation"]:.4f} and {test6["sinr_correlation"]:.4f}, respectively; scaled joint energy distance was {test6["scaled_joint_energy_distance"]:.4f}. All targets produced telemetry, IP reachability remained available, maximum command lateness was {test6["maximum_command_lateness_seconds"]:.3f} s, and rollback completed.

For the {int(clipped["rows"])} clipped rows, mean absolute target-to-projection error was {clipped["translator_relative_rsrp_mae_db"]:.3f} dB in relative RSRP and {clipped["translator_sinr_mae_db"]:.3f} dB in SINR. Complete original-target-to-OAI MAE on those rows was {clipped["total_relative_rsrp_mae_db"]:.3f} dB and {clipped["total_sinr_mae_db"]:.3f} dB, respectively.

Test 6 is genuine held-out exploratory evidence because it did not train or modify the translator. It is not confirmatory validation under the original protocol because its frozen support gate failed before replay, and one execution does not establish Test 6 execution-level repeatability.

## Phase 3M ablation

A post hoc Version 2 diagnostic tested first-order SINR memory using leave-one-complete-execution-out cross-validation. The static model had mean MAE {static["mean_mae_db"]:.3f} dB and mean p95 error {static["mean_p95_absolute_error_db"]:.3f} dB. The memory-only model reduced MAE to {memory["mean_mae_db"]:.3f} dB, while the combined model reduced p95 error to {combined["mean_p95_absolute_error_db"]:.3f} dB. Neither candidate passed every frozen gate: the combined p95 improvement was {100 * (static["mean_p95_absolute_error_db"] - combined["mean_p95_absolute_error_db"]) / static["mean_p95_absolute_error_db"]:.2f}%, below the 8% requirement, and mean absolute residual lag-1 correlation increased from {static["mean_absolute_residual_lag1_correlation"]:.3f} to approximately {combined["mean_absolute_residual_lag1_correlation"]:.3f}.

Dynamic inverse compensation was therefore rejected. Version 1 commands and claims were not changed.

## Reproducibility

- Release status: `{release["release_status"]}`
- Analysis revision: `{release["analysis_repository"]["revision"]}`
- OAI revision: `{release["canonical_revisions"]["oai"]}`
- Primary profile revision: `{release["canonical_revisions"]["phase3j_profile"]}`
- Test 6 exploratory wrapper revision: `{release["canonical_revisions"]["phase3l_exploratory_wrapper"]}`
- Channel family: AWGN
- Command interval: 1 s
- Input bundles, generated artifacts, and checksums are recorded in the release inventory.

## Supported claim

Version 1 achieved high-fidelity KPI-level replay of relative RSRP and SINR across the complete Test 1 development trajectory and a held-out exploratory Test 6 trajectory, while maintaining UE attachment, IP reachability, command timing, and rollback integrity.

## Limitations

- RSRP is relative; absolute NEMO-to-OAI RSRP equivalence is unresolved.
- SINR is an empirical, device-conditioned KPI rather than a calibrated physical noise measurement.
- AWGN scalar gain and effective noise do not reconstruct multipath, Doppler, beam dynamics, or a channel impulse response.
- Test 6 exceeded the frozen support gate and was replayed once; it supports an exploratory generalization claim only.
- The evidence does not establish cross-device, cross-site, or population generalization.
- No real attachment-event distribution or throughput distribution was available for validation.
- Version 1 replays observed trajectories; it does not predict or universally generate radio conditions.
"""


def finalize_upv_version1_release(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    figures_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    figures = Path(figures_dir).resolve()
    report = Path(report_path).resolve()
    if output.exists():
        raise FileExistsError(f"Version 1 release output already exists: {output}")
    if report.exists():
        raise FileExistsError(f"Version 1 report already exists: {report}")
    if any((figures / f"{name}.png").exists() for name in FIGURE_NAMES):
        raise FileExistsError(f"Version 1 figure output already exists: {figures}")

    config = _read_yaml(config_file)
    validate_version1_release_config(config)
    repository = config_file.parent.parent
    bundle_records = []
    bundle_paths: dict[str, Path] = {}
    for name, specification in config["frozen_inputs"].items():
        path = (repository / specification["path"]).resolve()
        bundle_paths[name] = path
        record = _verify_bundle(path, str(specification["manifest_sha256"]))
        record["bundle"] = name
        bundle_records.append(record)

    test1_frame = pd.read_csv(bundle_paths["phase3j_result"] / "paired_full_trace_fidelity.csv")
    test1_metrics = pd.read_csv(bundle_paths["phase3j_result"] / "per_execution_metrics.csv")
    test6_frame = pd.read_csv(bundle_paths["phase3l_result"] / "paired_test6_fidelity.csv")
    test6_metrics = pd.read_csv(
        bundle_paths["phase3l_result"] / "exploratory_execution_metrics.csv"
    )
    phase3m_summary = pd.read_csv(bundle_paths["phase3m_result"] / "model_summary.csv")
    phase3m_decision = _read_json(bundle_paths["phase3m_result"] / "phase3m_decision.json")
    if phase3m_decision.get("candidate_supported") is not False:
        raise ValueError("Phase 3M disposition no longer supports the Version 1 release")
    if not np.allclose(
        test6_frame["total_sinr_error_db"],
        test6_frame["translator_sinr_error_db"] + test6_frame["dynamic_sinr_error_db"],
        atol=1e-9,
    ):
        raise ValueError("Test 6 SINR error decomposition is inconsistent")

    output.mkdir(parents=True)
    figures.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    metrics = _metrics_summary(test1_metrics, test6_metrics)
    decomposition = _error_decomposition(test6_frame)
    _write_csv(output / "metrics_summary.csv", metrics)
    _write_csv(output / "test6_error_decomposition.csv", decomposition)
    _write_json(output / "claim_scope.json", config["claim_scope"])

    generated_paths: list[Path] = []
    generated_paths.extend(_plot_test1(test1_frame, figures / FIGURE_NAMES[0]))
    generated_paths.extend(_plot_test6(test6_frame, figures / FIGURE_NAMES[1]))
    generated_paths.extend(_plot_error_decomposition(decomposition, figures / FIGURE_NAMES[2]))
    generated_paths.extend(_plot_phase3m(phase3m_summary, figures / FIGURE_NAMES[3]))

    revision = _git_revision()
    release = {
        "schema_version": 1,
        "stage": config["stage"],
        "release_status": config["release_status"],
        "analysis_repository": revision,
        "canonical_revisions": config["canonical_revisions"],
        "runtime_model": config["runtime_model"],
        "evidence_roles": config["evidence_roles"],
        "phase3m_disposition": config["phase3m_disposition"],
        "reservation_required": False,
        "source_bundles_verified": len(bundle_records),
        "figures": [_artifact_path(path, repository) for path in generated_paths],
        "report": _artifact_path(report, repository),
    }
    _write_json(output / "release.json", release)
    report.write_text(
        _report_text(
            metrics=metrics,
            decomposition=decomposition,
            phase3m=phase3m_summary,
            release=release,
        ),
        encoding="utf-8",
    )

    artifact_records = bundle_records.copy()
    for path in [*generated_paths, report]:
        artifact_records.append(
            {
                "bundle": "generated_release_artifact",
                "path": _artifact_path(path, repository),
                "manifest_sha256": _sha256(path),
                "verified_files": 1,
            }
        )
    _write_csv(output / "artifact_inventory.csv", pd.DataFrame(artifact_records))
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "release_status": config["release_status"],
        "source_bundles_verified": len(bundle_records),
        "figures_generated": len(generated_paths),
        "report": str(report),
        "reservation_required": False,
    }
