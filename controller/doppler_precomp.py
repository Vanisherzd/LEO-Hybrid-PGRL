"""Doppler pre-compensation: convert PGRL Doppler prediction to TX frequency command."""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


def compensated_tx_frequency(
    base_freq_hz: float,
    predicted_doppler_hz: float,
    precomp_ratio: float = 1.0,
) -> float:
    """Apply Doppler pre-compensation: f_tx = f_c − α · f̂_D

    Args:
        base_freq_hz:          Nominal carrier frequency [Hz]
        predicted_doppler_hz:  PGRL-predicted Doppler shift [Hz] (positive = approaching)
        precomp_ratio:         Pre-compensation fraction (1.0 = full, 0.0 = none)

    Returns:
        Compensated TX frequency [Hz]
    """
    return base_freq_hz - precomp_ratio * predicted_doppler_hz


def residual_doppler_after_precomp(
    true_doppler_hz: float,
    predicted_doppler_hz: float,
    precomp_ratio: float = 1.0,
) -> float:
    """Residual Doppler after pre-compensation [Hz]."""
    return true_doppler_hz - precomp_ratio * predicted_doppler_hz


def doppler_rate_from_series(doppler_hz_list: list[float], dt_s: float) -> np.ndarray:
    """Estimate Doppler rate d|f_D|/dt from a Doppler time series [Hz/s]."""
    doppler = np.asarray(doppler_hz_list, dtype=float)
    return np.gradient(doppler, dt_s)


def cfo_to_radial_velocity(cfo_hz: float, f_carrier_hz: float) -> float:
    """Inverse Doppler: Δv = Δf · c / f_c [m/s]."""
    return cfo_hz * C_LIGHT / f_carrier_hz