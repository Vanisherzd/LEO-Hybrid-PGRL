"""Energy-based packet detection proxy."""
from __future__ import annotations

import numpy as np


def energy_detection(
    iq: np.ndarray,
    threshold: float,
    window_samples: int = 100,
) -> tuple[bool, float]:
    """Detect packet presence via sliding-window energy threshold.

    Args:
        iq:             IQ samples [N]
        threshold:      Energy threshold [W]
        window_samples: Sliding window length

    Returns:
        (detected: bool, max_energy: float)
    """
    energy = np.convolve(np.abs(iq)**2, np.ones(window_samples) / window_samples, mode="same")
    max_energy = float(np.max(energy))
    return bool(max_energy > threshold), max_energy


def detect_and_localize(
    iq: np.ndarray,
    threshold: float,
    window_samples: int = 100,
) -> list[int]:
    """Return sample indices where energy exceeds threshold."""
    energy = np.convolve(np.abs(iq)**2, np.ones(window_samples) / window_samples, mode="same")
    return [int(i) for i in np.where(energy > threshold)[0]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    import struct
    with open(args.cfile, "rb") as f:
        iq = np.frombuffer(f.read(), dtype=np.complex64)
    det, en = energy_detection(iq, args.threshold, args.window)
    print(f"Detected: {det}  max_energy: {en:.6f}")