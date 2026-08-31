"""RFsim control-semantics checks used by the validity audit."""

from __future__ import annotations

import math

NOISE_REFERENCE_RMS = 256.0


def corrected_noise_rms(
    noise_power_db: float,
    reference_rms: float = NOISE_REFERENCE_RMS,
) -> float:
    """Convert relative noise power in dB to per-component RMS amplitude."""

    return reference_rms * math.pow(10.0, noise_power_db / 20.0)


def legacy_noise_rms(
    noise_power_db: float,
    reference_rms: float = NOISE_REFERENCE_RMS,
) -> float:
    """Reproduce the pinned legacy /10 amplitude conversion."""

    return reference_rms * math.pow(10.0, noise_power_db / 10.0)


def legacy_equivalent_corrected_db(legacy_noise_db: float) -> float:
    """Return the corrected command that reproduces a legacy command's RMS."""

    return 2.0 * legacy_noise_db


def relative_power_db_from_rms(
    rms: float,
    reference_rms: float = NOISE_REFERENCE_RMS,
) -> float:
    """Recover relative power in dB from a positive RMS amplitude."""

    if rms <= 0.0 or reference_rms <= 0.0:
        raise ValueError("RMS amplitudes must be positive")
    return 20.0 * math.log10(rms / reference_rms)
