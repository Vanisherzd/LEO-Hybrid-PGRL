"""Estimate residual CFO from captured IQ samples using phase-difference method."""
from __future__ import annotations

import numpy as np


def estimate_cfo_from_phase(iq: np.ndarray, sample_rate_hz: float) -> float:
    """Estimate CFO from IQ samples using mean phase derivative.

    Δφ̂ = mean(diff(unwrap(angle(iq))))
    CFO = Δφ̂ × fs / (2π)   [Hz]

    Args:
        iq:              Complex IQ samples [N]
        sample_rate_hz:  Sample rate [Hz]

    Returns:
        Estimated CFO [Hz]
    """
    phase     = np.unwrap(np.angle(iq))
    dphi      = np.diff(phase)
    mean_dphi = np.mean(dphi)
    return float(mean_dphi * sample_rate_hz / (2.0 * np.pi))


def estimate_cfo_fft(iq: np.ndarray, sample_rate_hz: float) -> float:
    """Rough CFO estimate via FFT peak (for sanity check)."""
    n = len(iq)
    fft_mag = np.abs(np.fft.fft(iq[: n - n % 2]))
    freqs   = np.fft.fftfreq(len(fft_mag), 1.0 / sample_rate_hz)
    peak_idx = np.argmax(fft_mag)
    return float(freqs[peak_idx])


def load_cfile(path: str) -> tuple[np.ndarray, float]:
    """Load a binary IQ file (complex64) and infer sample rate from filename.

    Filename format: <label>_fs<rate>mhz_<timestamp>.cfile
    e.g. baseline_fs20mhz_1690000000.cfile → 20 MHz
    """
    import re
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.complex64)
    m = re.search(r"fs(\d+)mhz", path, re.IGNORECASE)
    fs_mhz = int(m.group(1)) if m else 20
    return raw, fs_mhz * 1e6


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile", help="Binary IQ capture file")
    ap.add_argument("--method", choices=["phase", "fft"], default="phase")
    args = ap.parse_args()

    iq, fs = load_cfile(args.cfile)
    cfo = estimate_cfo_from_phase(iq, fs) if args.method == "phase" else estimate_cfo_fft(iq, fs)
    print(f"CFO estimate: {cfo:.2f} Hz  (fs={fs/1e6:.0f} MHz, N={len(iq)})")