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

SPEED_OF_LIGHT_MPS = 299_792_458.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
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
    return _git_state(Path(__file__).resolve().parents[2])


def _model_arrays(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = config["standard_model"]
    delays = np.asarray(model["normalized_delays"], dtype=float)
    powers_db = np.asarray(model["powers_db"], dtype=float)
    if delays.ndim != 1 or powers_db.ndim != 1 or len(delays) != len(powers_db):
        raise ValueError("TDL delays and powers must be equal-length vectors")
    if len(delays) != 23:
        raise ValueError("the frozen TDL-B profile must contain 23 physical taps")
    powers = np.power(10.0, powers_db / 10.0)
    powers /= powers.sum()
    return delays, powers


def validate_phase3c2_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3C2 schema_version must be 1")
    if config.get("stage") != "phase_3c2_offline_time_varying_trace_validation":
        raise ValueError("unexpected Phase 3C2 stage")
    if any(bool(config.get(key)) for key in (
        "execution_authorized",
        "powder_action_authorized",
        "abc_authorized",
    )):
        raise ValueError("Phase 3C2 is offline and cannot authorize execution or ABC")

    model = config.get("standard_model") or {}
    if (
        model.get("specification") != "3GPP_TR_38.901"
        or model.get("version") != "17.0.0"
        or model.get("profile") != "TDL-B"
        or model.get("profile_table") != "7.7.2-2"
        or model.get("doppler_spectrum") != "classical_Jakes"
    ):
        raise ValueError("the standardized TDL-B/Jakes source must be explicit")
    delays, _ = _model_arrays(config)
    if np.any(delays < 0) or not np.isclose(delays[0], 0.0):
        raise ValueError("TDL-B normalized delays are invalid")

    trace = config.get("trace") or {}
    positive = (
        "carrier_frequency_hz",
        "speed_mps",
        "iq_sample_rate_hz",
        "snapshot_interval_s",
        "duration_s",
        "sinusoids_per_physical_tap",
        "generation_chunk_snapshots",
    )
    if any(float(trace.get(key, 0)) <= 0 for key in positive):
        raise ValueError("all frozen trace dimensions must be positive")
    snapshots = float(trace["duration_s"]) / float(trace["snapshot_interval_s"])
    if not math.isclose(snapshots, round(snapshots), abs_tol=1e-9):
        raise ValueError("trace duration must contain an integer number of snapshots")
    if trace.get("speed_interpretation") != (
        "engineering_validation_speed_not_inferred_from_UPV"
    ):
        raise ValueError("the selected speed must not be attributed to UPV")
    if not bool(trace.get("small_scale_global_mean_power_normalization")):
        raise ValueError("global small-scale mean-power normalization is required")
    if bool(trace.get("per_snapshot_normalization")):
        raise ValueError("per-snapshot normalization would suppress physical fading")

    max_doppler = (
        float(trace["carrier_frequency_hz"])
        * float(trace["speed_mps"])
        / SPEED_OF_LIGHT_MPS
    )
    if 1.0 / (2.0 * float(trace["snapshot_interval_s"])) < 5.0 * max_doppler:
        raise ValueError("the snapshot rate lacks the frozen Doppler oversampling margin")

    desired_ds = float(model["desired_rms_delay_spread_ns"])
    physical_delays_ns = delays * desired_ds
    bins = np.rint(physical_delays_ns * 1e-9 * float(trace["iq_sample_rate_hz"]))
    channel_length = int(bins.max()) + 1
    expected = int(config["discretization"]["expected_channel_length_samples"])
    if channel_length != expected:
        raise ValueError(
            f"discretized channel length {channel_length} does not match {expected}"
        )
    maximum = int(config["transport"]["pinned_constraints"][
        "cirdb_max_published_taps"
    ])
    if channel_length > maximum:
        raise ValueError("the frozen trace would be truncated by CIRDB")
    if config["transport"].get("selected_candidate") != "VRTSIM_CIRDB":
        raise ValueError("Phase 3C2 freezes VRTSIM CIRDB as the replay candidate")
    if bool(config["transport"].get(
        "actual_replay_timing_and_drop_validation_complete"
    )):
        raise ValueError("offline trace validation cannot claim transport replay timing")
    represented_correlation_lengths = (
        float(trace["duration_s"])
        * float(trace["speed_mps"])
        / float(config["large_scale"]["shadowing_correlation_distance_m"])
    )
    minimum_correlation_lengths = float(
        config["validation"]["minimum_shadowing_correlation_lengths"]
    )
    if represented_correlation_lengths < minimum_correlation_lengths:
        raise ValueError("the trace spans too few shadowing correlation lengths")

    claims = config.get("claim_limits") or {}
    prohibited_false = (
        "trace_is_measured_UPV_channel",
        "speed_is_inferred_from_UPV",
        "environmental_realism_established",
        "transport_replay_validated",
        "attachment_robustness_established",
    )
    if any(bool(claims.get(key)) for key in prohibited_false):
        raise ValueError("Phase 3C2 claim limits exceed the offline evidence")
    if claims.get("absolute_rsrp_calibration") != "prohibited":
        raise ValueError("absolute RSRP calibration remains prohibited")
    if claims.get("abc") != "prohibited":
        raise ValueError("ABC remains prohibited")
    reservation = config.get("reservation") or {}
    if reservation.get("gate_state") != "closed" or bool(
        reservation.get("reservation_should_be_requested_now")
    ):
        raise ValueError("the reservation gate must remain closed")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must allow at least 30 minutes")


def _validate_frozen_inputs(
    config: dict[str, Any], phase3c1_result: Path, sample_rate_evidence: Path
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    if _sha256(phase3c1_result) != frozen["phase3c1_result_sha256"]:
        raise ValueError("Phase 3C1 result checksum does not match the frozen input")
    result = _read_json(phase3c1_result)
    gate_name = str(frozen["phase3c1_required_gate"])
    if result.get("decision", {}).get(gate_name) != frozen[
        "phase3c1_required_gate_value"
    ]:
        raise ValueError("the Phase 3C1 deterministic scalar gate did not pass")
    if bool(result.get("decision", {}).get("additional_powder_action_authorized")):
        raise ValueError("Phase 3C1 must not authorize another POWDER action")
    if _sha256(sample_rate_evidence) != frozen["phase3c1_sample_rate_log_sha256"]:
        raise ValueError("sample-rate evidence checksum does not match Phase 3C1")
    expected = float(config["trace"]["iq_sample_rate_hz"])
    marker = f"sample_rate {expected:.6f}"
    if marker not in sample_rate_evidence.read_text(errors="replace"):
        raise ValueError("the frozen IQ sample rate is absent from the runtime evidence")
    return result


def _audit_oai_source(
    config: dict[str, Any], oai_source: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    state = _git_state(oai_source)
    expected = str(config["frozen_inputs"]["oai_expected_revision"])
    if state["revision"] != expected:
        raise ValueError(f"OAI revision {state['revision']} does not match {expected}")
    if bool(state["tracked_worktree_dirty"]):
        raise ValueError("tracked files are dirty in the OAI source checkout")
    rows: list[dict[str, object]] = []
    for assertion_id, assertion in enumerate(config["source_assertions"], start=1):
        relative = str(assertion["path"])
        path = oai_source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe OAI source evidence: {path}")
        text = path.read_text(errors="replace")
        for snippet_id, raw in enumerate(assertion["snippets"], start=1):
            snippet = str(raw)
            if snippet not in text:
                raise ValueError(f"source assertion not found in {relative}: {snippet}")
            line = text[: text.index(snippet)].count("\n") + 1
            rows.append({
                "assertion_id": assertion_id,
                "snippet_id": snippet_id,
                "oai_revision": state["revision"],
                "relative_path": relative,
                "file_sha256": _sha256(path),
                "fact": assertion["fact"],
                "snippet": snippet,
                "first_line": line,
                "verified": True,
            })
    return pd.DataFrame(rows), state


def _normalized_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    series = np.asarray(values)
    if series.ndim != 1 or len(series) < 3:
        raise ValueError("autocorrelation requires a one-dimensional series")
    max_lag = min(int(max_lag), len(series) - 2)
    centered = series - np.mean(series)
    n = len(centered)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.fft(centered, n=size)
    correlation = np.fft.ifft(np.conj(spectrum) * spectrum)[: max_lag + 1]
    correlation /= np.arange(n, n - max_lag - 1, -1)
    zero = float(correlation[0].real)
    if not np.isfinite(zero) or zero <= 0:
        raise ValueError("autocorrelation has non-positive zero-lag power")
    return correlation / zero


def _jakes_autocorrelation(max_doppler_hz: float, lags_s: np.ndarray) -> np.ndarray:
    angles = (np.arange(4096, dtype=float) + 0.5) * (2.0 * np.pi / 4096.0)
    arguments = 2.0 * np.pi * max_doppler_hz * lags_s[:, None]
    return np.cos(arguments * np.cos(angles)[None, :]).mean(axis=1)


def _first_crossing(axis: np.ndarray, values: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(values <= threshold)
    if len(indices) == 0:
        return math.nan
    index = int(indices[0])
    if index == 0:
        return float(axis[0])
    x0, x1 = float(axis[index - 1]), float(axis[index])
    y0, y1 = float(values[index - 1]), float(values[index])
    if y1 == y0:
        return x1
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    selected = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(selected[min(int(np.searchsorted(cumulative, q)), len(selected) - 1)])


def generate_phase3c2_trace(config: dict[str, Any]) -> dict[str, np.ndarray]:
    validate_phase3c2_config(config)
    normalized_delays, physical_powers = _model_arrays(config)
    model = config["standard_model"]
    trace = config["trace"]
    large = config["large_scale"]

    dt = float(trace["snapshot_interval_s"])
    duration = float(trace["duration_s"])
    snapshots = round(duration / dt)
    time_s = np.arange(snapshots, dtype=np.float64) * dt
    distance_m = time_s * float(trace["speed_mps"])
    physical_delays_ns = normalized_delays * float(
        model["desired_rms_delay_spread_ns"]
    )
    delay_bins = np.rint(
        physical_delays_ns * 1e-9 * float(trace["iq_sample_rate_hz"])
    ).astype(np.int32)
    channel_length = int(delay_bins.max()) + 1
    expected_bin_powers = np.bincount(
        delay_bins, weights=physical_powers, minlength=channel_length
    )

    rng = np.random.default_rng(int(trace["random_seed"]))
    max_doppler_hz = (
        float(trace["carrier_frequency_hz"])
        * float(trace["speed_mps"])
        / SPEED_OF_LIGHT_MPS
    )
    sinusoids = int(trace["sinusoids_per_physical_tap"])
    chunk = int(trace["generation_chunk_snapshots"])
    small_scale = np.zeros((snapshots, channel_length), dtype=np.complex64)
    for power, output_bin in zip(physical_powers, delay_bins, strict=True):
        angles = rng.uniform(0.0, 2.0 * np.pi, size=sinusoids)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=sinusoids)
        frequencies = max_doppler_hz * np.cos(angles)
        scale = math.sqrt(float(power) / sinusoids)
        for start in range(0, snapshots, chunk):
            end = min(start + chunk, snapshots)
            argument = (
                2.0
                * np.pi
                * time_s[start:end, None]
                * frequencies[None, :]
                + phases[None, :]
            )
            values = scale * np.exp(1j * argument).sum(axis=1)
            small_scale[start:end, int(output_bin)] += values.astype(np.complex64)

    mean_small_scale_power = float(np.mean(np.sum(np.abs(small_scale) ** 2, axis=1)))
    small_scale /= math.sqrt(mean_small_scale_power)

    sigma_db = float(large["shadowing_sigma_db"])
    correlation_distance_m = float(large["shadowing_correlation_distance_m"])
    distance_step = float(trace["speed_mps"]) * dt
    rho = math.exp(-distance_step / correlation_distance_m)
    shadowing_db = np.empty(snapshots, dtype=np.float64)
    shadowing_db[0] = rng.normal(0.0, sigma_db)
    innovation_scale = sigma_db * math.sqrt(1.0 - rho**2)
    innovations = rng.normal(0.0, innovation_scale, size=snapshots - 1)
    for index in range(1, snapshots):
        shadowing_db[index] = rho * shadowing_db[index - 1] + innovations[index - 1]
    shadowing_db -= shadowing_db.mean()
    realized_sigma = float(shadowing_db.std(ddof=0))
    if realized_sigma <= 0:
        raise ValueError("generated shadowing has zero variance")
    shadowing_db *= sigma_db / realized_sigma
    large_scale_amplitude = np.power(
        10.0,
        (shadowing_db + float(large["constant_path_gain_db"])) / 20.0,
    )
    channel_taps = (small_scale * large_scale_amplitude[:, None]).astype(np.complex64)

    return {
        "time_s": time_s,
        "distance_m": distance_m,
        "small_scale_taps": small_scale,
        "shadowing_db": shadowing_db.astype(np.float32),
        "channel_taps": channel_taps,
        "physical_delays_ns": physical_delays_ns.astype(np.float64),
        "physical_powers": physical_powers.astype(np.float64),
        "physical_delay_bins": delay_bins,
        "bin_delays_samples": np.arange(channel_length, dtype=np.int32),
        "bin_delays_ns": (
            np.arange(channel_length, dtype=np.float64)
            / float(trace["iq_sample_rate_hz"])
            * 1e9
        ),
        "expected_bin_powers": expected_bin_powers.astype(np.float64),
        "max_doppler_hz": np.asarray(max_doppler_hz, dtype=np.float64),
    }


def validate_phase3c2_trace(
    config: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    validate_phase3c2_config(config)
    trace = config["trace"]
    validation = config["validation"]
    small_scale = np.asarray(arrays["small_scale_taps"])
    channel_taps = np.asarray(arrays["channel_taps"])
    time_s = np.asarray(arrays["time_s"], dtype=float)
    shadowing_db = np.asarray(arrays["shadowing_db"], dtype=float)
    expected_pdp = np.asarray(arrays["expected_bin_powers"], dtype=float)
    delays_ns = np.asarray(arrays["bin_delays_ns"], dtype=float)
    dt = float(trace["snapshot_interval_s"])
    expected_snapshots = round(float(trace["duration_s"]) / dt)
    finite = bool(
        np.isfinite(time_s).all()
        and np.isfinite(shadowing_db).all()
        and np.isfinite(small_scale.real).all()
        and np.isfinite(small_scale.imag).all()
        and np.isfinite(channel_taps.real).all()
        and np.isfinite(channel_taps.imag).all()
    )
    timestamp_error = float(np.max(np.abs(time_s - np.arange(len(time_s)) * dt)))
    mean_power = float(np.mean(np.sum(np.abs(small_scale) ** 2, axis=1)))
    mean_power_error = abs(mean_power - 1.0)
    observed_pdp = np.mean(np.abs(small_scale) ** 2, axis=0)
    observed_pdp /= observed_pdp.sum()
    expected_pdp /= expected_pdp.sum()
    pdp_error_db = float(np.max(np.abs(
        10.0 * np.log10(observed_pdp) - 10.0 * np.log10(expected_pdp)
    )))

    mean_delay = float(np.sum(delays_ns * observed_pdp))
    observed_rms_ds = float(np.sqrt(np.sum(observed_pdp * (delays_ns - mean_delay) ** 2)))
    desired_rms_ds = float(config["standard_model"]["desired_rms_delay_spread_ns"])
    rms_ds_relative_error = abs(observed_rms_ds - desired_rms_ds) / desired_rms_ds

    max_doppler_hz = float(np.asarray(arrays["max_doppler_hz"]))
    max_acf_lag = math.ceil(
        float(validation["acf_max_doppler_periods"]) / max_doppler_hz / dt
    )
    strongest_bin = int(np.argmax(expected_pdp))
    acf = _normalized_autocorrelation(small_scale[:, strongest_bin], max_acf_lag)
    lags_s = np.arange(len(acf), dtype=float) * dt
    expected_acf = _jakes_autocorrelation(max_doppler_hz, lags_s)
    acf_mae = float(np.mean(np.abs(acf - expected_acf)))
    acf_imaginary_max = float(np.max(np.abs(acf.imag)))
    lag_one = float(abs(acf[1]))
    coherence_threshold = float(validation["coherence_threshold"])
    observed_coherence_s = _first_crossing(lags_s, np.abs(acf), coherence_threshold)
    expected_coherence_s = _first_crossing(
        lags_s, np.abs(expected_acf), coherence_threshold
    )
    coherence_ratio = observed_coherence_s / expected_coherence_s

    series = small_scale[:, strongest_bin] - np.mean(small_scale[:, strongest_bin])
    window = np.hanning(len(series))
    spectrum = np.fft.fftshift(np.fft.fft(series * window))
    frequencies_hz = np.fft.fftshift(np.fft.fftfreq(len(series), d=dt))
    psd = np.abs(spectrum) ** 2
    total_psd = float(psd.sum())
    guard = float(validation["doppler_guard_fraction"])
    inside = np.abs(frequencies_hz) <= (1.0 + guard) * max_doppler_hz
    in_band_fraction = float(psd[inside].sum() / total_psd)
    occupied_99_hz = _weighted_quantile(np.abs(frequencies_hz), psd, 0.99)
    occupied_edge_ratio = occupied_99_hz / max_doppler_hz

    distance_step = float(trace["speed_mps"]) * dt
    target_correlation_distance = float(
        config["large_scale"]["shadowing_correlation_distance_m"]
    )
    max_shadow_lag = math.ceil(4.0 * target_correlation_distance / distance_step)
    shadow_acf = _normalized_autocorrelation(shadowing_db, max_shadow_lag).real
    shadow_distances = np.arange(len(shadow_acf), dtype=float) * distance_step
    observed_correlation_distance = _first_crossing(
        shadow_distances,
        shadow_acf,
        float(validation["shadowing_correlation_threshold"]),
    )
    shadowing_ratio = observed_correlation_distance / target_correlation_distance

    changed = np.any(np.abs(np.diff(channel_taps, axis=0)) > 1.0e-8, axis=1)
    changed_fraction = float(np.mean(changed))
    coherence_bounds = validation["coherence_time_ratio_range"]
    edge_bounds = validation["doppler_99pct_edge_ratio_range"]
    shadow_bounds = validation["shadowing_correlation_distance_ratio_range"]
    gates = {
        "finite": finite,
        "snapshot_count": len(time_s) == expected_snapshots,
        "timestamp_grid": timestamp_error
        <= float(validation["timestamp_max_abs_error_s"]),
        "small_scale_mean_power": mean_power_error
        <= float(validation["small_scale_mean_power_max_abs_error"]),
        "binned_power_delay_profile": pdp_error_db
        <= float(validation["binned_pdp_max_abs_error_db"]),
        "rms_delay_spread": rms_ds_relative_error
        <= float(validation["discretized_rms_delay_spread_relative_error_maximum"]),
        "autocorrelation_shape": acf_mae
        <= float(validation["acf_mean_abs_error_maximum"]),
        "autocorrelation_imaginary": acf_imaginary_max
        <= float(validation["acf_imaginary_max_abs_maximum"]),
        "lag_one_temporal_correlation": lag_one
        >= float(validation["lag_one_autocorrelation_magnitude_minimum"]),
        "coherence_time": float(coherence_bounds[0])
        <= coherence_ratio
        <= float(coherence_bounds[1]),
        "doppler_in_band_power": in_band_fraction
        >= float(validation["doppler_in_band_power_fraction_minimum"]),
        "doppler_occupied_edge": float(edge_bounds[0])
        <= occupied_edge_ratio
        <= float(edge_bounds[1]),
        "shadowing_correlation_distance": float(shadow_bounds[0])
        <= shadowing_ratio
        <= float(shadow_bounds[1]),
        "snapshots_change": changed_fraction
        >= float(validation["changed_snapshot_fraction_minimum"]),
        "cirdb_length_compatible": small_scale.shape[1]
        <= int(config["transport"]["pinned_constraints"][
            "cirdb_max_published_taps"
        ]),
    }
    gates = {key: bool(value) for key, value in gates.items()}

    pdp_table = pd.DataFrame({
        "delay_bin": arrays["bin_delays_samples"],
        "delay_ns": delays_ns,
        "expected_power_linear": expected_pdp,
        "observed_power_linear": observed_pdp,
        "expected_power_db_relative": 10.0 * np.log10(expected_pdp / expected_pdp.max()),
        "observed_power_db_relative": 10.0 * np.log10(observed_pdp / observed_pdp.max()),
    })
    acf_table = pd.DataFrame({
        "lag": np.arange(len(acf)),
        "lag_s": lags_s,
        "empirical_real": acf.real,
        "empirical_imag": acf.imag,
        "empirical_magnitude": np.abs(acf),
        "expected_jakes": expected_acf,
    })
    doppler_table = pd.DataFrame({
        "frequency_hz": frequencies_hz,
        "power_fraction": psd / total_psd,
        "inside_guarded_support": inside,
    })
    shadow_table = pd.DataFrame({
        "lag": np.arange(len(shadow_acf)),
        "distance_m": shadow_distances,
        "empirical_correlation": shadow_acf,
        "expected_exponential_correlation": np.exp(
            -shadow_distances / target_correlation_distance
        ),
    })
    result = {
        "schema_version": 1,
        "snapshots": len(time_s),
        "duration_s": float(trace["duration_s"]),
        "snapshot_interval_s": dt,
        "channel_length_samples": int(small_scale.shape[1]),
        "physical_taps_before_discretization": len(arrays["physical_delays_ns"]),
        "physical_tap_collisions_merged": int(
            len(arrays["physical_delays_ns"]) - small_scale.shape[1]
        ),
        "max_doppler_hz": max_doppler_hz,
        "doppler_oversampling_ratio": 1.0 / dt / (2.0 * max_doppler_hz),
        "small_scale_mean_power": mean_power,
        "small_scale_mean_power_abs_error": mean_power_error,
        "binned_pdp_max_abs_error_db": pdp_error_db,
        "observed_discretized_rms_delay_spread_ns": observed_rms_ds,
        "desired_continuous_rms_delay_spread_ns": desired_rms_ds,
        "discretized_rms_delay_spread_relative_error": rms_ds_relative_error,
        "autocorrelation_mean_abs_error": acf_mae,
        "autocorrelation_imaginary_max_abs": acf_imaginary_max,
        "lag_one_autocorrelation_magnitude": lag_one,
        "observed_coherence_time_s": observed_coherence_s,
        "expected_jakes_coherence_time_s": expected_coherence_s,
        "coherence_time_ratio": coherence_ratio,
        "doppler_in_guarded_support_power_fraction": in_band_fraction,
        "doppler_99pct_absolute_edge_hz": occupied_99_hz,
        "doppler_99pct_edge_ratio": occupied_edge_ratio,
        "shadowing_sigma_db": float(shadowing_db.std(ddof=0)),
        "observed_shadowing_correlation_distance_m": observed_correlation_distance,
        "target_shadowing_correlation_distance_m": target_correlation_distance,
        "shadowing_correlation_distance_ratio": shadowing_ratio,
        "changed_snapshot_fraction": changed_fraction,
        "gate_results": gates,
        "offline_temporal_trace_gate_pass": all(gates.values()),
        "transport_timing_and_drop_gate_evaluated": False,
        "reservation_should_be_requested_now": False,
    }
    return result, {
        "power_delay_profile": pdp_table,
        "autocorrelation": acf_table,
        "doppler_psd": doppler_table,
        "shadowing_autocorrelation": shadow_table,
    }


def _write_trace_files(
    output: Path,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    validation: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> bool:
    np.savez_compressed(output / "phase3c2_trace.npz", **arrays)
    binary = np.asarray(arrays["channel_taps"], dtype="<c8")
    binary.tofile(output / "cir_db.bin")
    binary_roundtrip = np.fromfile(output / "cir_db.bin", dtype="<c8").reshape(
        binary.shape
    )
    binary_exact = bool(np.array_equal(binary, binary_roundtrip))
    entry = {
        "model_id": int(config["transport"]["cirdb_model_id"]),
        "n_tx": int(config["trace"]["tx_antennas"]),
        "n_rx": int(config["trace"]["rx_antennas"]),
        "L": int(binary.shape[1]),
        "S": int(binary.shape[0]),
        "fs_hz": float(config["trace"]["iq_sample_rate_hz"]),
        "snapshot_dt_s": float(config["trace"]["snapshot_interval_s"]),
        "ds_ns": float(config["standard_model"]["desired_rms_delay_spread_ns"]),
        "speed_mps": float(config["trace"]["speed_mps"]),
        "pair_order": int(config["transport"]["pair_order"]),
        "offset_bytes": 0,
        "nbytes": int((output / "cir_db.bin").stat().st_size),
    }
    (output / "vrtsim.yaml").write_text(
        yaml.safe_dump({"entries": [entry]}, sort_keys=False)
    )
    for name, frame in tables.items():
        _write_csv(output / f"{name}.csv", frame)
    completed = dict(validation)
    completed["binary_roundtrip_exact"] = binary_exact
    completed["gate_results"] = {
        **validation["gate_results"],
        "binary_roundtrip_exact": binary_exact,
    }
    completed["offline_temporal_trace_gate_pass"] = all(
        completed["gate_results"].values()
    )
    _write_json(output / "temporal_validation.json", completed)
    return binary_exact


def run_phase3c2_trace_validation(
    *,
    config_path: str | Path,
    phase3c1_result: str | Path,
    sample_rate_evidence: str | Path,
    oai_source: str | Path,
    output_dir: str | Path,
    manifest_dir: str | Path,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    phase3c1_file = Path(phase3c1_result).resolve()
    sample_rate_file = Path(sample_rate_evidence).resolve()
    source = Path(oai_source).resolve()
    output = Path(output_dir).resolve()
    manifest = Path(manifest_dir).resolve()
    for path in (config_file, phase3c1_file, sample_rate_file):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe Phase 3C2 input: {path}")
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"missing or unsafe OAI checkout: {source}")
    if output.exists() or manifest.exists():
        raise FileExistsError("Phase 3C2 output or manifest already exists")

    config = _read_yaml(config_file)
    validate_phase3c2_config(config)
    _validate_frozen_inputs(config, phase3c1_file, sample_rate_file)
    source_trace, oai_state = _audit_oai_source(config, source)
    implementation = _implementation_state()
    if bool(implementation["tracked_worktree_dirty"]):
        raise ValueError("commit the Phase 3C2 implementation before freezing a trace")

    output_staging = output.parent / f".{output.name}.staging"
    manifest_staging = manifest.parent / f".{manifest.name}.staging"
    for staging in (output_staging, manifest_staging):
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
    try:
        arrays = generate_phase3c2_trace(config)
        validation, tables = validate_phase3c2_trace(config, arrays)
        _write_trace_files(output_staging, config, arrays, validation, tables)
        validation = _read_json(output_staging / "temporal_validation.json")
        raw_checksums = {
            path.name: _sha256(path)
            for path in sorted(output_staging.iterdir())
            if path.is_file()
        }
        trace_metadata = {
            "schema_version": 1,
            "name": config["name"],
            "stage": config["stage"],
            "analysis_implementation_revision": implementation["revision"],
            "analysis_worktree_clean": True,
            "oai_source_state": oai_state,
            "standard_model": config["standard_model"],
            "trace": config["trace"],
            "large_scale": config["large_scale"],
            "discretization": config["discretization"],
            "transport": config["transport"],
            "claim_limits": config["claim_limits"],
            "raw_output_tracking_policy": "gitignored_generated_artifacts_checksums_committed",
            "raw_output_sha256": raw_checksums,
        }
        _write_json(output_staging / "trace_metadata.json", trace_metadata)
        _write_csv(manifest_staging / "source_code_trace.csv", source_trace)
        _write_csv(
            manifest_staging / "power_delay_profile.csv",
            tables["power_delay_profile"],
        )
        _write_json(manifest_staging / "temporal_validation.json", validation)
        _write_json(manifest_staging / "trace_metadata.json", trace_metadata)
        decision = {
            "schema_version": 1,
            "decision_code": (
                "standardized_time_varying_trace_passed_offline_temporal_validation"
                if validation["offline_temporal_trace_gate_pass"]
                else "time_varying_trace_failed_offline_temporal_validation"
            ),
            "offline_temporal_trace_gate_pass": validation[
                "offline_temporal_trace_gate_pass"
            ],
            "selected_trace": config["name"],
            "selected_transport_candidate": config["transport"][
                "selected_candidate"
            ],
            "transport_schema_compatible": validation["gate_results"][
                "cirdb_length_compatible"
            ],
            "transport_replay_timing_validated": False,
            "attachment_parser_validated": False,
            "small_replay_experiment_frozen": False,
            "new_powder_execution_authorized": False,
            "reservation_should_be_requested_now": False,
            "abc_authorized": False,
            "claim_limits": config["claim_limits"],
            "next_action": (
                "validate attachment and PBCH/PUSCH parsers on pinned fixture logs, "
                "then freeze the small CIRDB replay experiment and stopping rules"
                if validation["offline_temporal_trace_gate_pass"]
                else "revise the trace generator without using POWDER"
            ),
        }
        _write_json(manifest_staging / "phase3c2_decision.json", decision)
        reservation = {
            "schema_version": 1,
            "gate_state": "closed",
            "reservation_should_be_requested_now": False,
            "preparation_lead_time_minutes": config["reservation"][
                "preparation_lead_time_minutes"
            ],
            "satisfied_conditions": [
                "deterministic scalar replay passed",
                *(
                    ["pinned time-varying trace passed offline temporal validation"]
                    if validation["offline_temporal_trace_gate_pass"]
                    else []
                ),
            ],
            "blocking_conditions": [
                "attachment and PBCH/PUSCH parsers have not passed on pinned fixture logs",
                "the exact small replay experiment and stopping rules are not committed",
                "actual CIRDB replay timing and drop behavior has not been tested",
            ],
            "notification_rule": (
                "notify the user at least 30 minutes before the first authorized "
                "POWDER-dependent action"
            ),
            "next_action": decision["next_action"],
        }
        _write_json(manifest_staging / "reservation_gate_v5.json", reservation)
        analysis_manifest = {
            "schema_version": 1,
            "name": config["name"],
            "stage": config["stage"],
            "config_sha256": _sha256(config_file),
            "phase3c1_result_sha256": _sha256(phase3c1_file),
            "sample_rate_evidence_sha256": _sha256(sample_rate_file),
            "analysis_implementation_revision": implementation["revision"],
            "oai_source_state": oai_state,
            "source_assertions_verified": len(source_trace),
            "raw_artifact_directory": str(output),
            "raw_output_sha256": raw_checksums,
            "offline_temporal_trace_gate_pass": validation[
                "offline_temporal_trace_gate_pass"
            ],
            "reservation_should_be_requested_now": False,
            "powder_action_performed": False,
            "abc_performed": False,
        }
        _write_json(manifest_staging / "analysis_manifest.json", analysis_manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(manifest_staging.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.json"
        }
        _write_json(manifest_staging / "SHA256SUMS.json", checksums)
        output_staging.replace(output)
        manifest_staging.replace(manifest)
    except Exception:
        shutil.rmtree(output_staging, ignore_errors=True)
        shutil.rmtree(manifest_staging, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "manifest": str(manifest),
        "decision_code": decision["decision_code"],
        "offline_temporal_trace_gate_pass": validation[
            "offline_temporal_trace_gate_pass"
        ],
        "channel_length_samples": validation["channel_length_samples"],
        "reservation_should_be_requested_now": False,
        "powder_action_performed": False,
        "abc_authorized": False,
    }
