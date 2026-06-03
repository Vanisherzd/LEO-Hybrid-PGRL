#!/usr/bin/env python3
"""
quick_maxhold.py
Fast max-hold spectrum check for a .fc32 USRP B210 capture.

Computes a spectrogram over time, takes the max across time per frequency bin
(max-hold), excludes the B210 DC/LO guard band, and reports a single
candidate-signal flag. Intended as a quick triage tool before the full
analyze_capture.py pipeline.

No hard scipy dependency (numpy FFT fallback).

Usage:
    uv run python hardware/usrp_scripts/quick_maxhold.py input.fc32 \
        --sample-rate 1000000 \
        --out out.png \
        --json out.json \
        --threshold-db 8 \
        --nfft 4096
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def maxhold_spectrum(samples: np.ndarray, fs: float, nfft: int):
    """
    Spectrogram over time, max across time per bin. Returns (freqs, maxhold_db),
    fftshifted so DC is centered. numpy-only.
    """
    nfft = int(min(nfft, len(samples)))
    if nfft < 8:
        nfft = len(samples)
    if nfft <= 0:
        return np.array([0.0]), np.array([-99.0])

    window = np.hanning(nfft)
    win_norm = np.sum(window ** 2)
    if win_norm <= 0:
        win_norm = 1.0
    step = max(1, nfft // 2)
    maxhold = None
    for s in range(0, len(samples) - nfft + 1, step):
        seg = samples[s : s + nfft] * window
        spec = np.fft.fft(seg, n=nfft)
        p = (np.abs(spec) ** 2) / (fs * win_norm)
        maxhold = p if maxhold is None else np.maximum(maxhold, p)

    if maxhold is None:
        # Capture shorter than one full segment.
        seg = samples[:nfft]
        w = np.hanning(len(seg)) if len(seg) > 1 else np.ones(len(seg))
        wn = np.sum(w ** 2) if np.sum(w ** 2) > 0 else 1.0
        spec = np.fft.fft(seg * w, n=nfft)
        maxhold = (np.abs(spec) ** 2) / (fs * wn)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    maxhold = np.fft.fftshift(maxhold)
    with np.errstate(divide="ignore"):
        maxhold_db = 10.0 * np.log10(np.maximum(maxhold, 1e-30))
    return freqs, maxhold_db


def dc_guard_mask(freqs: np.ndarray, fs: float, guard_frac: float = 0.02, min_bins: int = 3):
    """Boolean keep-mask excluding a DC guard band (B210 LO spike)."""
    if len(freqs) <= 1:
        return np.ones(len(freqs), dtype=bool)
    df = float(np.abs(freqs[1] - freqs[0]))
    guard_hz = max(guard_frac * fs, min_bins * df)
    mask = np.abs(freqs) > guard_hz
    if not np.any(mask):
        mask = np.ones(len(freqs), dtype=bool)
        center = int(np.argmin(np.abs(freqs)))
        lo = max(0, center - min_bins)
        hi = min(len(freqs), center + min_bins + 1)
        mask[lo:hi] = False
        if not np.any(mask):
            mask = np.ones(len(freqs), dtype=bool)
    return mask


def write_plot(freqs, maxhold_db, peak_freq, candidate, threshold_db, out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[maxhold] matplotlib not available — skipping PNG")
        return
    status = "candidate_signal" if candidate else "noise_floor_only"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, maxhold_db, linewidth=0.8, color="navy")
    if candidate:
        ax.axvline(peak_freq, color="green", linestyle="--", linewidth=1.0, label="peak")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Frequency offset (Hz)")
    ax.set_ylabel("Max-hold power (dB)")
    ax.set_title(f"Max-hold — {out_path.name}  [{status}, thr={threshold_db:g} dB]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[maxhold] PNG → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick max-hold spectrum check for .fc32 capture")
    parser.add_argument("input", type=str, help="Input .fc32 file")
    parser.add_argument("--sample-rate", type=float, default=1e6, help="Sample rate in Hz")
    parser.add_argument("--out", type=str, default=None, help="Output max-hold PNG path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--threshold-db", type=float, default=8.0, help="Candidate threshold (dB)")
    parser.add_argument("--nfft", type=int, default=4096, help="FFT size (default 4096)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    samples = np.fromfile(in_path, dtype=np.complex64)
    if len(samples) == 0:
        print("ERROR: capture is empty (0 samples)", file=sys.stderr)
        sys.exit(1)
    fs = float(args.sample_rate)

    freqs, maxhold_db = maxhold_spectrum(samples, fs, args.nfft)

    # Median in dB as a robust noise reference, peak excluding DC guard.
    median_db = float(np.median(maxhold_db))
    mask = dc_guard_mask(freqs, fs)
    masked = np.where(mask, maxhold_db, -np.inf)
    peak_idx = int(np.argmax(masked))
    peak_db = float(maxhold_db[peak_idx])
    peak_freq = float(freqs[peak_idx])
    peak_to_median_db = peak_db - median_db

    candidate = bool(peak_to_median_db >= args.threshold_db)

    result = {
        "peak_frequency_offset_hz": round(peak_freq, 2),
        "peak_to_median_db": round(peak_to_median_db, 2),
        "candidate_signal": candidate,
        "threshold_db": args.threshold_db,
    }

    print(f"[maxhold] peak_to_median_db = {peak_to_median_db:.2f} dB "
          f"@ {peak_freq:.0f} Hz  → candidate_signal={candidate}")

    if args.out:
        write_plot(freqs, maxhold_db, peak_freq, candidate, args.threshold_db, Path(args.out))

    if args.json:
        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[maxhold] JSON → {out_json}")


if __name__ == "__main__":
    main()
