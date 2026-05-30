#!/usr/bin/env python3
"""
analyze_capture.py
Offline IQ analysis of captured .fc32 USRP B210 data.

Computes and reports:
- Estimated residual CFO (phase-derivative method)
- QPSK EVM proxy (constellation rotation)
- SNR estimate
- Waterfall / spectrogram

Usage:
    uv run python hardware/usrp_scripts/analyze_capture.py \
        hardware/captures/baseline.fc32 \
        --output hardware/captures/baseline_results.json \
        --plot hardware/captures/baseline_waterfall.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def estimate_cfo(samples: np.ndarray, fs: float) -> dict:
    """
    Estimate residual carrier frequency offset using phase-derivative (Viterbi) method.
    Works on arbitrary baseband signal — no preamble required.

    Returns dict with cfo_hz, cfo_std_hz, method, n_samples.
    """
    n = len(samples)
    # Phase derivative: unwrapped angle difference
    phase = np.unwrap(np.angle(samples))
    # Fit linear trend to phase → slope = 2π * cfo / fs
    t = np.arange(n)
    slope, intercept = np.polyfit(t, phase, 1)
    cfo_hz = slope * fs / (2 * math.pi)

    # Confidence from residual
    phase_fit = slope * t + intercept
    residual = phase - phase_fit
    residuals_std = np.std(residual)
    # Convert residual phase std to Hz equivalent
    cfo_std_hz = residuals_std * fs / (2 * math.pi)

    return {
        "cfo_hz": round(cfo_hz, 2),
        "cfo_std_hz": round(cfo_std_hz, 2),
        "method": "phase_derivative_viterbi",
        "n_samples": n,
    }


def estimate_evm_qpsk(samples: np.ndarray) -> dict:
    """
    QPSK EVM proxy: measures constellation rotation under residual CFO.
    Works by:
    1. Removing estimated CFO rotation
    2. Mapping to nearest QPSK constellation points
    3. Computing normalized MSE

    Returns EVM as percentage (lower is better; <10% = acceptable for production).
    """
    n = len(samples)

    # Step 1: Remove CFO
    cfo_info = estimate_cfo(samples, 1.0)  # normalized fs
    dt = np.arange(n) / 1.0
    rot = np.exp(-1j * 2 * math.pi * cfo_info["cfo_hz"] * dt)
    samples_rotated = samples * rot

    # Step 2: QPSK constellation points
    qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / math.sqrt(2)

    # Step 3: Nearest QPSK mapping
    closest = np.array([qpsk[np.argmin(np.abs(s - qpsk))] for s in samples_rotated])

    # Step 4: EVM = sqrt(MSE / signal_power) * 100
    error = samples_rotated - closest
    mse = np.mean(np.abs(error) ** 2)
    power = np.mean(np.abs(samples_rotated) ** 2)
    evm_percent = math.sqrt(mse / power) * 100.0 if power > 0 else float("inf")

    return {
        "evm_percent": round(evm_percent, 2),
        "mse": round(mse, 6),
        "signal_power_linear": round(power, 4),
        "method": "qpsk_constellation_mse",
        "note": "RF-quality proxy only; not a standard LR-FHSS PER measurement",
    }


def estimate_snr(samples: np.ndarray) -> dict:
    """
    Simple SNR estimate via signal variance vs noise floor.
    Assumes signal has higher variance than noise alone.
    """
    power_linear = np.mean(np.abs(samples) ** 2)
    # Noise floor estimate: remove top 10% (likely signal peaks)
    sorted_pow = np.sort(np.abs(samples) ** 2)
    noise_floor = np.mean(sorted_pow[: len(sorted_pow) // 10])
    snr_linear = (power_linear - noise_floor) / noise_floor
    snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else -99.0

    return {
        "snr_db": round(snr_db, 2),
        "signal_power_dBfs": round(10 * math.log10(power_linear), 2),
        "noise_floor_dBfs": round(10 * math.log10(noise_floor), 2) if noise_floor > 0 else -99.0,
    }


def write_waterfall(samples: np.ndarray, fs: float, output_path: Path, duration_s: float = 1.0):
    """Generate waterfall/spectrogram PNG for visual inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nfft = 512
        overlap = int(nfft * 0.75)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.specgram(samples, NFFT=nfft, Fs=fs, noverlap=overlap, cmap="viridis")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Waterfall — {output_path.name}")
        plt.colorbar(ax.images[0], ax=ax, label="Power (dB)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[analyze] Waterfall → {output_path}")
    except ImportError:
        print("[analyze] matplotlib not available — skipping waterfall plot")


def main():
    parser = argparse.ArgumentParser(description="Analyze USRP B210 .fc32 IQ capture")
    parser.add_argument("input", type=str, help="Input .fc32 file")
    parser.add_argument("--sample-rate", type=float, default=1e6, help="Sample rate in Hz")
    parser.add_argument("--output-json", type=str, help="Output results.json path")
    parser.add_argument("--plot", type=str, help="Output waterfall PNG path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"[analyze] Loading {in_path} ({in_path.stat().st_size // 1024} KB)...")
    samples = np.fromfile(in_path, dtype=np.complex64)
    n = len(samples)
    duration_s = n / args.sample_rate
    print(f"[analyze] {n:,} samples, {duration_s:.2f} s at {args.sample_rate/1e6:.1f} Msps")

    # Run analyses
    print("[analyze] Estimating residual CFO...")
    cfo = estimate_cfo(samples, args.sample_rate)
    print(f"[analyze]   Residual CFO: {cfo['cfo_hz']:.2f} ± {cfo['cfo_std_hz']:.2f} Hz")

    print("[analyze] Computing QPSK EVM proxy...")
    evm = estimate_evm_qpsk(samples)
    print(f"[analyze]   EVM: {evm['evm_percent']:.2f}%  (lower is better; <10% production-ready)")

    print("[analyze] Estimating SNR...")
    snr = estimate_snr(samples)
    print(f"[analyze]   SNR: {snr['snr_db']:.2f} dB")

    # Consolidate
    results = {
        "rx_cfo_hz": cfo["cfo_hz"],
        "rx_cfo_std_hz": cfo["cfo_std_hz"],
        "rx_evm_percent": evm["evm_percent"],
        "rx_snr_db": snr["snr_db"],
        "capture_sample_count": n,
        "capture_duration_s": round(duration_s, 4),
        "validation_type": "hardware",
        "cfo_method": cfo["method"],
        "evm_method": evm["method"],
        "evm_note": evm["note"],
        "input_file": str(in_path),
    }

    # Save JSON
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[analyze] Results → {out_json}")

    # Plot waterfall
    if args.plot:
        out_png = Path(args.plot)
        write_waterfall(samples, args.sample_rate, out_png, duration_s)

    # Print summary
    print("\n=== Hardware Capture Results ===")
    print(f"  Residual CFO: {cfo['cfo_hz']:.2f} Hz  (±{cfo['cfo_std_hz']:.2f} Hz)")
    print(f"  QPSK EVM:     {evm['evm_percent']:.2f}%  ({evm['note']})")
    print(f"  SNR:          {snr['snr_db']:.2f} dB")
    print("================================")
    print("\nNOTE: EVM is an RF-quality proxy only.")
    print("  LR-FHSS PER requires a standards-compliant decoder.")


if __name__ == "__main__":
    main()