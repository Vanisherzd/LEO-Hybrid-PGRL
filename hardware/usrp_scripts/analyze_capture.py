#!/usr/bin/env python3
"""
analyze_capture.py
Offline IQ analysis of captured .fc32 USRP B210 data.

Conservative, defensible signal-detection pipeline for an LR-FHSS hardware
validation paper. The core philosophy: do NOT overclaim. CFO/EVM proxies are
ONLY computed once an actual modulated signal has been detected above the noise
floor. If the capture is just the noise floor (or a bare DC/LO spike from the
B210 front end), we say so explicitly and refuse to emit meaningless
CFO/EVM numbers.

Pipeline:
1. Load .fc32 as complex64.
2. Estimate PSD (Welch via scipy if available, else numpy-only averaged
   periodogram fallback). fftshift so DC is band-centered.
3. Exclude a DC guard band (B210 has a strong LO/DC spike) when searching for
   the signal peak.
4. peak_to_median_db, peak_frequency_offset_hz, burst_energy_excess_db, and a
   supporting max-hold excess decide detection.
5. CFO/EVM proxies gated behind detection. SNR (coarse power ratio) always
   reported.

Usage:
    uv run python hardware/usrp_scripts/analyze_capture.py \
        hardware/captures/baseline.fc32 \
        --sample-rate 1000000 \
        --output-json hardware/captures/baseline_analysis.json \
        --plot hardware/captures/baseline_waterfall.png \
        --signal-threshold-db 8 \
        --maxhold-plot hardware/captures/baseline_maxhold.png \
        --spectrogram-nfft 4096
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy: try to import for Welch PSD / spectrogram, but never hard-fail.
# ---------------------------------------------------------------------------
try:
    from scipy import signal as _scipy_signal  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - environment dependent
    _scipy_signal = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Existing helpers (kept, but now gated behind signal detection in main()).
# ---------------------------------------------------------------------------
def estimate_cfo(samples: np.ndarray, fs: float) -> dict:
    """
    Estimate residual carrier frequency offset using phase-derivative (Viterbi)
    method. Works on arbitrary baseband signal — no preamble required.

    Returns dict with cfo_hz, cfo_std_hz, method, n_samples.
    """
    n = len(samples)
    phase = np.unwrap(np.angle(samples))
    t = np.arange(n)
    slope, intercept = np.polyfit(t, phase, 1)
    cfo_hz = slope * fs / (2 * math.pi)

    phase_fit = slope * t + intercept
    residual = phase - phase_fit
    residuals_std = np.std(residual)
    cfo_std_hz = residuals_std * fs / (2 * math.pi)

    return {
        "cfo_hz": round(float(cfo_hz), 2),
        "cfo_std_hz": round(float(cfo_std_hz), 2),
        "method": "phase_derivative_viterbi",
        "n_samples": n,
    }


def estimate_evm_qpsk(samples: np.ndarray) -> dict:
    """
    QPSK EVM proxy: measures constellation rotation under residual CFO.

    Returns EVM as percentage (lower is better). This is an RF-quality proxy
    only; it is NOT a standards-compliant LR-FHSS PER measurement.
    """
    n = len(samples)

    cfo_info = estimate_cfo(samples, 1.0)  # normalized fs
    dt = np.arange(n) / 1.0
    rot = np.exp(-1j * 2 * math.pi * cfo_info["cfo_hz"] * dt)
    samples_rotated = samples * rot

    qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / math.sqrt(2)

    # Vectorized nearest-constellation mapping.
    diffs = samples_rotated[:, None] - qpsk[None, :]
    idx = np.argmin(np.abs(diffs), axis=1)
    closest = qpsk[idx]

    error = samples_rotated - closest
    mse = float(np.mean(np.abs(error) ** 2))
    power = float(np.mean(np.abs(samples_rotated) ** 2))
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
    Coarse SNR estimate via total power vs. low-percentile noise floor.
    Safe to report regardless of detection outcome (it is just a power ratio).
    """
    inst_pow = np.abs(samples) ** 2
    power_linear = float(np.mean(inst_pow))
    sorted_pow = np.sort(inst_pow)
    k = max(1, len(sorted_pow) // 10)
    noise_floor = float(np.mean(sorted_pow[:k]))
    snr_linear = (power_linear - noise_floor) / noise_floor if noise_floor > 0 else 0.0
    snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else -99.0

    return {
        "snr_db": round(snr_db, 2),
        "signal_power_dBfs": round(10 * math.log10(power_linear), 2) if power_linear > 0 else -99.0,
        "noise_floor_dBfs": round(10 * math.log10(noise_floor), 2) if noise_floor > 0 else -99.0,
    }


# ---------------------------------------------------------------------------
# Signal detection.
# ---------------------------------------------------------------------------
def _compute_psd(samples: np.ndarray, fs: float, nfft: int):
    """
    Compute a (frequency, PSD) pair, fftshifted so DC is centered.
    Uses scipy.signal.welch when available, else a numpy-only averaged
    periodogram (Hann-windowed, 50% overlap).
    """
    nfft = int(min(nfft, len(samples)))
    if nfft < 8:
        nfft = min(8, len(samples)) if len(samples) >= 8 else len(samples)
    if nfft <= 0:
        # Degenerate: single-bin PSD.
        freqs = np.array([0.0])
        psd = np.array([float(np.mean(np.abs(samples) ** 2)) if len(samples) else 0.0])
        return freqs, psd

    if _HAVE_SCIPY:
        freqs, psd = _scipy_signal.welch(
            samples,
            fs=fs,
            window="hann",
            nperseg=nfft,
            noverlap=nfft // 2,
            nfft=nfft,
            return_onesided=False,
            scaling="density",
            detrend=False,
        )
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        return freqs, np.abs(psd)

    # numpy-only fallback: averaged periodogram.
    window = np.hanning(nfft)
    win_norm = np.sum(window ** 2)
    step = max(1, nfft // 2)
    starts = range(0, len(samples) - nfft + 1, step)
    accum = None
    count = 0
    for s in starts:
        seg = samples[s : s + nfft] * window
        spec = np.fft.fft(seg, n=nfft)
        p = (np.abs(spec) ** 2) / (fs * win_norm)
        accum = p if accum is None else accum + p
        count += 1
    if accum is None or count == 0:
        # Capture shorter than one segment: single FFT.
        seg = samples[:nfft]
        w = np.hanning(len(seg)) if len(seg) > 1 else np.ones(len(seg))
        wn = np.sum(w ** 2) if np.sum(w ** 2) > 0 else 1.0
        spec = np.fft.fft(seg * w, n=nfft)
        accum = (np.abs(spec) ** 2) / (fs * wn)
        count = 1
    psd = accum / count
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    return freqs, np.abs(psd)


def _dc_guard_mask(freqs: np.ndarray, fs: float, guard_frac: float = 0.02, min_bins: int = 3):
    """
    Boolean mask of bins to KEEP (True) when searching for the signal peak.
    Excludes a DC guard band around f=0 to suppress the B210 LO/DC spike.
    Guard half-width = max(guard_frac * fs, min_bins worth of resolution).
    """
    if len(freqs) <= 1:
        return np.ones(len(freqs), dtype=bool)
    df = float(np.abs(freqs[1] - freqs[0]))
    guard_hz = max(guard_frac * fs, min_bins * df)
    mask = np.abs(freqs) > guard_hz
    if not np.any(mask):
        # Guard ate everything (tiny capture); fall back to excluding only the
        # single center bin.
        mask = np.ones(len(freqs), dtype=bool)
        center = int(np.argmin(np.abs(freqs)))
        lo = max(0, center - min_bins)
        hi = min(len(freqs), center + min_bins + 1)
        mask[lo:hi] = False
        if not np.any(mask):
            mask = np.ones(len(freqs), dtype=bool)
    return mask, guard_hz


def _burst_energy_excess_db(samples: np.ndarray, frame_len: int = 1024) -> float:
    """
    Short-time framing: per-frame energy, then 95th-percentile / median in dB.
    Captures bursty (FHSS) energy that a time-averaged PSD might smear out.
    """
    n = len(samples)
    if n < frame_len * 2:
        frame_len = max(8, n // 2)
    n_frames = n // frame_len
    if n_frames < 2:
        return 0.0
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_energy = np.sum(np.abs(trimmed) ** 2, axis=1)
    med = float(np.median(frame_energy))
    p95 = float(np.percentile(frame_energy, 95))
    if med <= 0:
        return 0.0
    return 10.0 * math.log10(p95 / med)


def _maxhold_spectrum(samples: np.ndarray, fs: float, nfft: int):
    """
    Spectrogram over time, then max across time per frequency bin (max-hold).
    Returns (freqs, maxhold_db) fftshifted. Helps catch intermittent hops.
    """
    nfft = int(min(nfft, len(samples)))
    if nfft < 8:
        freqs, psd = _compute_psd(samples, fs, nfft)
        with np.errstate(divide="ignore"):
            return freqs, 10.0 * np.log10(np.maximum(psd, 1e-30))

    window = np.hanning(nfft)
    win_norm = np.sum(window ** 2)
    step = max(1, nfft // 2)
    maxhold = None
    for s in range(0, len(samples) - nfft + 1, step):
        seg = samples[s : s + nfft] * window
        spec = np.fft.fft(seg, n=nfft)
        p = (np.abs(spec) ** 2) / (fs * win_norm)
        maxhold = p if maxhold is None else np.maximum(maxhold, p)
    if maxhold is None:
        freqs, psd = _compute_psd(samples, fs, nfft)
        with np.errstate(divide="ignore"):
            return freqs, 10.0 * np.log10(np.maximum(psd, 1e-30))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    maxhold = np.fft.fftshift(maxhold)
    with np.errstate(divide="ignore"):
        maxhold_db = 10.0 * np.log10(np.maximum(maxhold, 1e-30))
    return freqs, maxhold_db


def detect_signal(samples: np.ndarray, fs: float, nfft: int) -> dict:
    """
    Core detection. Returns a dict of all detection metrics plus the
    PSD/freqs/guard used (for plotting).
    """
    freqs, psd = _compute_psd(samples, fs, nfft)
    median_psd = float(np.median(psd))
    if median_psd <= 0:
        median_psd = float(np.mean(psd)) if np.mean(psd) > 0 else 1e-30

    mask, guard_hz = _dc_guard_mask(freqs, fs)
    psd_masked = np.where(mask, psd, -np.inf)
    peak_idx = int(np.argmax(psd_masked))
    peak_val = float(psd[peak_idx])
    peak_freq = float(freqs[peak_idx])
    peak_to_median_db = 10.0 * math.log10(peak_val / median_psd) if peak_val > 0 else -99.0

    burst_db = _burst_energy_excess_db(samples)

    # Supporting max-hold excess (peak of max-hold over median noise, DC-excluded).
    mh_freqs, mh_db = _maxhold_spectrum(samples, fs, nfft)
    mh_mask, _ = _dc_guard_mask(mh_freqs, fs)
    mh_median_db = float(np.median(mh_db))
    mh_masked = np.where(mh_mask, mh_db, -np.inf)
    maxhold_excess_db = float(np.max(mh_masked) - mh_median_db) if np.any(mh_mask) else 0.0

    return {
        "peak_to_median_db": round(peak_to_median_db, 2),
        "peak_frequency_offset_hz": round(peak_freq, 2),
        "burst_energy_excess_db": round(burst_db, 2),
        "maxhold_excess_db": round(maxhold_excess_db, 2),
        "dc_guard_hz": round(guard_hz, 2),
        "_freqs": freqs,
        "_psd": psd,
        "_median_psd": median_psd,
    }


def classify(det: dict, threshold_db: float) -> tuple:
    """
    Conservative decision logic.
    Returns (validation_status, signal_detected).
    """
    p2m = det["peak_to_median_db"]
    burst = det["burst_energy_excess_db"]

    if p2m >= threshold_db and burst >= 3.0:
        return "signal_detected", True
    if (threshold_db - 3.0) <= p2m < threshold_db:
        return "weak_signal_candidate", False
    return "noise_floor_only", False


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def write_waterfall(
    samples: np.ndarray,
    fs: float,
    output_path: Path,
    validation_status: str,
    peak_freq_hz: float,
    dc_guard_hz: float,
    nfft: int = 1024,
):
    """Spectrogram PNG. Marks peak when signal detected; states status in title."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping waterfall plot")
        return

    nfft = int(min(max(64, nfft), max(64, len(samples))))
    overlap = int(nfft * 0.75)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.specgram(samples, NFFT=nfft, Fs=fs, noverlap=overlap, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency offset (Hz)")
    ax.set_title(f"Waterfall — {output_path.name}  [{validation_status}]")

    # DC guard band (visual).
    if dc_guard_hz > 0:
        ax.axhspan(-dc_guard_hz, dc_guard_hz, color="red", alpha=0.12)

    if validation_status == "signal_detected":
        ax.axhline(peak_freq_hz, color="white", linestyle="--", linewidth=1.0)
        ax.text(
            0.02,
            0.95,
            f"signal_detected @ {peak_freq_hz:.0f} Hz",
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            va="top",
        )
    else:
        ax.text(
            0.02,
            0.95,
            validation_status,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            va="top",
        )

    if ax.images:
        plt.colorbar(ax.images[0], ax=ax, label="Power (dB)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[analyze] Waterfall → {output_path}")


def write_maxhold_plot(
    samples: np.ndarray,
    fs: float,
    output_path: Path,
    nfft: int,
    validation_status: str,
    peak_freq_hz: float,
    dc_guard_hz: float,
):
    """Max-hold spectrum line plot (x=freq offset Hz, y=dB)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping max-hold plot")
        return

    freqs, mh_db = _maxhold_spectrum(samples, fs, nfft)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, mh_db, linewidth=0.8, color="navy")
    ax.set_xlabel("Frequency offset (Hz)")
    ax.set_ylabel("Max-hold power (dB)")
    ax.set_title(f"Max-hold spectrum — {output_path.name}  [{validation_status}]")
    if dc_guard_hz > 0:
        ax.axvspan(-dc_guard_hz, dc_guard_hz, color="red", alpha=0.12, label="DC guard")
    if validation_status == "signal_detected":
        ax.axvline(peak_freq_hz, color="green", linestyle="--", linewidth=1.0, label="peak")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[analyze] Max-hold → {output_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze USRP B210 .fc32 IQ capture")
    parser.add_argument("input", type=str, help="Input .fc32 file")
    parser.add_argument("--sample-rate", type=float, default=1e6, help="Sample rate in Hz")
    parser.add_argument("--output-json", type=str, help="Output analysis.json path")
    parser.add_argument("--plot", type=str, help="Output waterfall PNG path")
    parser.add_argument(
        "--signal-threshold-db",
        type=float,
        default=8.0,
        help="peak-to-median threshold (dB) for signal detection (default 8)",
    )
    parser.add_argument(
        "--maxhold-plot",
        type=str,
        default=None,
        help="Optional output PNG for max-hold spectrum line plot",
    )
    parser.add_argument(
        "--spectrogram-nfft",
        type=int,
        default=4096,
        help="FFT size for detection PSD / max-hold (default 4096)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"[analyze] Loading {in_path} ({in_path.stat().st_size // 1024} KB)...")
    samples = np.fromfile(in_path, dtype=np.complex64)
    n = len(samples)
    if n == 0:
        print("ERROR: capture is empty (0 samples)", file=sys.stderr)
        sys.exit(1)
    fs = float(args.sample_rate)
    duration_s = n / fs
    print(f"[analyze] {n:,} samples, {duration_s:.2f} s at {fs/1e6:.3f} Msps")
    print(f"[analyze] scipy {'available' if _HAVE_SCIPY else 'NOT available (numpy fallback)'}")

    # --- Signal detection (always) ---
    print("[analyze] Running signal detection...")
    det = detect_signal(samples, fs, args.spectrogram_nfft)
    validation_status, signal_detected = classify(det, args.signal_threshold_db)

    print(f"[analyze]   peak_to_median_db    = {det['peak_to_median_db']:.2f} dB")
    print(f"[analyze]   burst_energy_excess  = {det['burst_energy_excess_db']:.2f} dB")
    print(f"[analyze]   maxhold_excess_db    = {det['maxhold_excess_db']:.2f} dB")
    print(f"[analyze]   peak_freq_offset     = {det['peak_frequency_offset_hz']:.1f} Hz")
    print(f"[analyze]   DC guard half-width  = {det['dc_guard_hz']:.1f} Hz")

    # --- SNR always (coarse power ratio, safe) ---
    snr = estimate_snr(samples)

    # --- CFO / EVM gated behind detection ---
    rx_cfo_hz = None
    rx_cfo_std_hz = None
    rx_evm_percent = None
    cfo_method = None
    evm_method = None

    if signal_detected:
        print("[analyze] Signal detected → computing CFO / EVM proxies...")
        cfo = estimate_cfo(samples, fs)
        evm = estimate_evm_qpsk(samples)
        rx_cfo_hz = cfo["cfo_hz"]
        rx_cfo_std_hz = cfo["cfo_std_hz"]
        rx_evm_percent = evm["evm_percent"]
        cfo_method = cfo["method"]
        evm_method = evm["method"]
        analysis_note = (
            f"Signal detected: peak/median={det['peak_to_median_db']:.1f} dB, "
            f"burst_excess={det['burst_energy_excess_db']:.1f} dB at "
            f"{det['peak_frequency_offset_hz']:.0f} Hz offset. CFO/EVM are RF-quality "
            f"proxies only, not LR-FHSS PER."
        )
        print(f"[analyze]   rx_cfo_hz = {rx_cfo_hz:.2f} Hz")
        print(f"[analyze]   rx_evm_percent = {rx_evm_percent:.2f} %  (RF proxy, not PER)")
    else:
        if validation_status == "weak_signal_candidate":
            analysis_note = (
                f"Weak signal candidate: peak/median={det['peak_to_median_db']:.1f} dB is "
                f"below the {args.signal_threshold_db:.1f} dB detection threshold. "
                f"CFO/EVM proxy not meaningful and intentionally withheld (conservative)."
            )
            print("[analyze] Weak candidate below threshold → CFO/EVM SKIPPED.")
        else:
            analysis_note = (
                "No modulated signal detected; CFO/EVM proxy not meaningful. "
                "Capture is consistent with noise floor (DC/LO spike excluded)."
            )
            print("[analyze] Noise floor only → CFO/EVM SKIPPED.")

    results = {
        "validation_status": validation_status,
        "signal_detected": bool(signal_detected),
        "peak_to_median_db": det["peak_to_median_db"],
        "burst_energy_excess_db": det["burst_energy_excess_db"],
        "peak_frequency_offset_hz": det["peak_frequency_offset_hz"],
        "analysis_note": analysis_note,
        "rx_cfo_hz": rx_cfo_hz,
        "rx_cfo_std_hz": rx_cfo_std_hz,
        "rx_evm_percent": rx_evm_percent,
        "rx_snr_db": snr["snr_db"],
        "capture_sample_count": int(n),
        "capture_duration_s": round(duration_s, 4),
        "sample_rate_hz": fs,
        "signal_threshold_db": float(args.signal_threshold_db),
        "validation_type": "hardware",
        "input_file": str(in_path),
        # Supporting / provenance fields (non-canonical but useful, do not break schema).
        "maxhold_excess_db": det["maxhold_excess_db"],
        "dc_guard_hz": det["dc_guard_hz"],
        "detection_nfft": int(args.spectrogram_nfft),
        "psd_method": "scipy_welch" if _HAVE_SCIPY else "numpy_periodogram",
        "cfo_method": cfo_method,
        "evm_method": evm_method,
        "evm_note": "RF-quality proxy only; not a standard LR-FHSS PER measurement",
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[analyze] Results → {out_json}")

    if args.plot:
        write_waterfall(
            samples,
            fs,
            Path(args.plot),
            validation_status,
            det["peak_frequency_offset_hz"],
            det["dc_guard_hz"],
        )

    if args.maxhold_plot:
        write_maxhold_plot(
            samples,
            fs,
            Path(args.maxhold_plot),
            args.spectrogram_nfft,
            validation_status,
            det["peak_frequency_offset_hz"],
            det["dc_guard_hz"],
        )

    print("\n=== Hardware Capture Analysis ===")
    print(f"  validation_status:    {validation_status}")
    print(f"  signal_detected:      {signal_detected}")
    print(f"  peak_to_median_db:    {det['peak_to_median_db']:.2f} dB")
    print(f"  burst_energy_excess:  {det['burst_energy_excess_db']:.2f} dB")
    print(f"  peak_freq_offset_hz:  {det['peak_frequency_offset_hz']:.1f} Hz")
    print(f"  rx_snr_db:            {snr['snr_db']:.2f} dB")
    if signal_detected:
        print(f"  rx_cfo_hz:            {rx_cfo_hz:.2f} Hz")
        print(f"  rx_evm_percent:       {rx_evm_percent:.2f} %  (RF proxy)")
    else:
        print("  rx_cfo_hz / rx_evm:   SKIPPED (no detected signal)")
    print("=================================")
    print("\nNOTE: EVM is an RF-quality proxy only.")
    print("  LR-FHSS PER requires a standards-compliant decoder.")


if __name__ == "__main__":
    main()
