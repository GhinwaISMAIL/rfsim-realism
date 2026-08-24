from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rfsim_realism.upv_phase3c2 import (
    generate_phase3c2_trace,
    validate_phase3c2_config,
    validate_phase3c2_trace,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (REPOSITORY / "configs/upv_phase3c2_trace_v1.yaml").read_text()
    )


def _small_config() -> dict[str, object]:
    config = deepcopy(_config())
    config["trace"]["snapshot_interval_s"] = 0.005
    config["trace"]["duration_s"] = 6.0
    config["trace"]["sinusoids_per_physical_tap"] = 32
    config["trace"]["generation_chunk_snapshots"] = 512
    config["large_scale"]["shadowing_correlation_distance_m"] = 0.25
    config["validation"]["acf_mean_abs_error_maximum"] = 0.35
    config["validation"]["acf_imaginary_max_abs_maximum"] = 0.35
    config["validation"]["lag_one_autocorrelation_magnitude_minimum"] = 0.75
    config["validation"]["coherence_time_ratio_range"] = [0.4, 2.0]
    config["validation"]["shadowing_correlation_distance_ratio_range"] = [0.25, 2.5]
    return config


def test_phase3c2_config_is_offline_scoped_and_transport_safe() -> None:
    config = _config()

    validate_phase3c2_config(config)

    assert config["execution_authorized"] is False
    assert config["powder_action_authorized"] is False
    assert config["abc_authorized"] is False
    assert config["transport"]["selected_candidate"] == "VRTSIM_CIRDB"
    assert config["discretization"]["expected_channel_length_samples"] == 8
    assert config["reservation"]["reservation_should_be_requested_now"] is False


def test_phase3c2_rejects_per_snapshot_normalization() -> None:
    config = _config()
    config["trace"]["per_snapshot_normalization"] = True

    with pytest.raises(ValueError, match="per-snapshot"):
        validate_phase3c2_config(config)


def test_phase3c2_rejects_cirdb_truncation() -> None:
    config = _config()
    config["trace"]["iq_sample_rate_hz"] = 61_440_000.0

    with pytest.raises(ValueError, match="does not match"):
        validate_phase3c2_config(config)


def test_phase3c2_trace_is_deterministic_and_time_correlated() -> None:
    config = _small_config()

    first = generate_phase3c2_trace(config)
    second = generate_phase3c2_trace(config)
    result, _ = validate_phase3c2_trace(config, first)

    assert np.array_equal(first["channel_taps"], second["channel_taps"])
    assert first["small_scale_taps"].shape == (1200, 8)
    assert result["gate_results"]["finite"] is True
    assert result["gate_results"]["small_scale_mean_power"] is True
    assert result["gate_results"]["lag_one_temporal_correlation"] is True
    assert result["gate_results"]["snapshots_change"] is True


def test_phase3c2_validation_rejects_independent_snapshots() -> None:
    config = _small_config()
    arrays = generate_phase3c2_trace(config)
    rng = np.random.default_rng(7)
    independent = (
        rng.normal(size=arrays["small_scale_taps"].shape)
        + 1j * rng.normal(size=arrays["small_scale_taps"].shape)
    ).astype(np.complex64)
    independent /= np.sqrt(np.mean(np.sum(np.abs(independent) ** 2, axis=1)))
    arrays["small_scale_taps"] = independent
    arrays["channel_taps"] = independent

    result, _ = validate_phase3c2_trace(config, arrays)

    assert result["gate_results"]["lag_one_temporal_correlation"] is False
    assert result["offline_temporal_trace_gate_pass"] is False


def test_phase3c2_validation_rejects_timestamp_gap() -> None:
    config = _small_config()
    arrays = generate_phase3c2_trace(config)
    arrays["time_s"] = arrays["time_s"].copy()
    arrays["time_s"][400:] += 0.01

    result, _ = validate_phase3c2_trace(config, arrays)

    assert result["gate_results"]["timestamp_grid"] is False
    assert result["offline_temporal_trace_gate_pass"] is False
