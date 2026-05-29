"""EVM as RF-quality proxy under controlled impairment conditions.

This is NOT LR-FHSS PER. It demonstrates that Doppler/CFO pre-compensation
improves physical-layer signal quality, which translates to better demodulation
margin for standards-compliant LR-FHSS receivers.
"""
from __future__ import annotations

import numpy as np


def evm_percent(received_symbols: np.ndarray, reference_symbols: np.ndarray) -> float:
    """Compute EVM % = 100 × RMS(error) / RMS(reference).

    Args:
        received_symbols:  Received constellation points (complex), shape (N,)
        reference_symbols:  Ideal constellation points (complex), shape (N,)

    Returns:
        EVM [%]
    """
    error  = received_symbols - reference_symbols
    ref    = reference_symbols
    rms_err = np.sqrt(np.mean(np.abs(error) ** 2))
    rms_ref = np.sqrt(np.mean(np.abs(ref) ** 2))
    if rms_ref == 0:
        return float("inf")
    return float(100.0 * rms_err / rms_ref)


def evm_db(e: float) -> float:
    """Convert EVM % to dB."""
    if e <= 0:
        return float("inf")
    return float(20.0 * np.log10(e / 100.0))


def simulate_qpsk_with_cfo(
    snr_db: float,
    cfo_hz: float,
    fs_hz: float,
    n_symbols: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate QPSK constellation under CFO impairment.

    Returns:
        (received, reference) as complex arrays
    """
    np.random.seed(seed)
    # QPSK symbols
    s = np.random.choice([1+1j, -1+1j, -1-1j, 1-1j], size=n_symbols) / np.sqrt(2)

    # CFO rotation
    n = np.arange(n_symbols)
    cfo_phase = 2 * np.pi * cfo_hz / fs_hz * n
    received = s * np.exp(1j * cfo_phase)

    # AWGN
    signal_power = np.mean(np.abs(received)**2)
    noise_power  = signal_power * 10**(-snr_db / 10.0)
    noise = np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)
    received += np.sqrt(noise_power) * noise / np.sqrt(2)

    return received, s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-db", type=float, default=40.0)
    ap.add_argument("--cfo-hz", type=float, default=1000.0)
    ap.add_argument("--fs-hz", type=float, default=1e6)
    args = ap.parse_args()

    rx, ref = simulate_qpsk_with_cfo(args.snr_db, args.cfo_hz, args.fs_hz)
    e = evm_percent(rx, ref)
    print(f"EVM: {e:.4f} %  ({evm_db(e):.2f} dB)")