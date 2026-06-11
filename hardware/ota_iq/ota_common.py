#!/usr/bin/env python3
"""
ota_common.py
=============
Shared helpers for the OTA IQ replay experiments (CFO residual + adjacent-bin
leakage). IQ loading, config loading, burst detection in time-frequency space,
and per-burst carrier estimation.

SCOPE DISCIPLINE: every metric produced from these helpers is an IQ-level
physical-layer proxy from a short-range room OTA / near-field capture. Nothing
here decodes packets, measures PER, checks CRC, or validates a receiver.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── provenance ────────────────────────────────────────────────────────────────
def git_commit() -> str:
    """Return short git commit hash of the repo, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── config ────────────────────────────────────────────────────────────────────
def load_config(path: str | Path) -> dict:
    """Load a replay YAML config and normalise numeric fields."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # YAML may parse 868.0e6 as str on some loaders; coerce known numerics.
    for k in ("nominal_center_freq_hz", "sample_rate_hz", "rx_gain_db",
              "replay_window_s", "burst_interval_s", "burst_duration_ms",
              "grid_spacing_hz", "lo_offset_hz"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    return cfg


# ── IQ IO ─────────────────────────────────────────────────────────────────────
def load_iq(path: str | Path) -> np.ndarray:
    """
    Load complex64 IQ from .npy or raw interleaved float32 (.fc32 / .cfile).

    Raises FileNotFoundError if the capture is missing — callers must NOT
    fabricate samples when no real capture exists.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"IQ capture not found: {path}. Run usrp_capture_ota_iq.py on real "
            f"hardware first; this analysis never synthesises IQ."
        )
    if path.suffix == ".npy":
        iq = np.load(path)
        return iq.astype(np.complex64)
    # raw interleaved float32 I,Q (UHD fc32 / GNU Radio .cfile)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ── burst detection in time-frequency space ───────────────────────────────────
@dataclass
class Burst:
    index: int
    t_start_s: float
    t_end_s: float
    peak_offset_hz: float       # carrier offset from F0 (nominal center), signed
    peak_power_db: float
    snr_db: float


def _stft(iq: np.ndarray, fs: float, nfft: int, hop: int):
    """Simple STFT magnitude (linear power), fftshifted. Returns (P[t,f], f_hz)."""
    win = np.hanning(nfft).astype(np.float32)
    n_frames = 1 + (len(iq) - nfft) // hop if len(iq) >= nfft else 0
    P = np.empty((n_frames, nfft), dtype=np.float32)
    for i in range(n_frames):
        seg = iq[i * hop: i * hop + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        P[i] = (spec.real ** 2 + spec.imag ** 2)
    f_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return P, f_hz


def detect_bursts(
    iq: np.ndarray,
    fs: float,
    nfft: int = 1024,
    hop: int = 256,
    inband_frac: float = 0.8,
    snr_gate_db: float = 6.0,
    exclude_dc_hz: float = 5e3,
    min_burst_ms: float = 5.0,
) -> list[Burst]:
    """
    Detect short bursts as contiguous time frames whose in-band peak power
    exceeds the per-capture noise floor by snr_gate_db. For each burst, estimate
    the carrier offset from F0 by parabolic interpolation of the peak bin.

    The DC/LO region (|f| < exclude_dc_hz) is masked so the USRP LO leakage is
    not mistaken for a burst peak.
    """
    P, f_hz = _stft(iq, fs, nfft, hop)
    if P.shape[0] == 0:
        return []

    inband = np.abs(f_hz) <= (inband_frac * fs / 2.0)
    dc_mask = np.abs(f_hz) < exclude_dc_hz
    usable = inband & ~dc_mask

    # per-frame in-band peak power
    Pu = P.copy()
    Pu[:, ~usable] = 0.0
    frame_peak = Pu.max(axis=1)
    frame_peak_db = 10.0 * np.log10(frame_peak + 1e-20)

    # Noise reference = median of the per-frame in-band PEAK. Noise-only frames
    # cluster near this value (the natural peak-of-noise level); burst frames sit
    # well above it. Comparing against the per-bin median instead would flag the
    # spectral peak of every noise frame and merge the whole capture into 1 burst.
    noise_db = float(np.median(frame_peak_db))

    above = frame_peak_db > (noise_db + snr_gate_db)

    frame_dt = hop / fs
    min_frames = max(1, int((min_burst_ms / 1e3) / frame_dt))

    bursts: list[Burst] = []
    i = 0
    n = len(above)
    bidx = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if (j - i) >= min_frames:
            # aggregate spectrum over the burst frames, estimate peak
            seg_P = Pu[i:j].mean(axis=0)
            kpk = int(np.argmax(seg_P))
            peak_hz = _parabolic_peak_hz(seg_P, f_hz, kpk)
            ppow_db = 10.0 * np.log10(seg_P[kpk] + 1e-20)
            bursts.append(Burst(
                index=bidx,
                t_start_s=i * frame_dt,
                t_end_s=j * frame_dt,
                peak_offset_hz=peak_hz,
                peak_power_db=ppow_db,
                snr_db=ppow_db - noise_db,
            ))
            bidx += 1
        i = j
    return bursts


def burst_spectrum(iq: np.ndarray, b: "Burst", fs: float,
                   nfft: int = 1024, hop: int = 256):
    """Linear-power spectrum averaged over a burst's frames. Returns (P, f_hz)."""
    P, f_hz = _stft(iq, fs, nfft, hop)
    frame_dt = hop / fs
    i = int(b.t_start_s / frame_dt)
    j = max(i + 1, int(b.t_end_s / frame_dt))
    seg = P[i:j]
    if seg.shape[0] == 0:
        seg = P[i:i + 1]
    return seg.mean(axis=0), f_hz


def band_power(spec: np.ndarray, f_hz: np.ndarray,
               center_hz: float, half_bw_hz: float) -> float:
    """Sum of linear power within [center-half_bw, center+half_bw]."""
    m = np.abs(f_hz - center_hz) <= half_bw_hz
    return float(spec[m].sum())


def _parabolic_peak_hz(spec: np.ndarray, f_hz: np.ndarray, k: int) -> float:
    """Parabolic (quadratic) interpolation around bin k for sub-bin peak freq."""
    if k <= 0 or k >= len(spec) - 1:
        return float(f_hz[k])
    a = np.log(spec[k - 1] + 1e-20)
    b = np.log(spec[k] + 1e-20)
    c = np.log(spec[k + 1] + 1e-20)
    denom = (a - 2 * b + c)
    delta = 0.0 if abs(denom) < 1e-20 else 0.5 * (a - c) / denom
    df = f_hz[1] - f_hz[0]
    return float(f_hz[k] + delta * df)


# ── small JSON helper ─────────────────────────────────────────────────────────
def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def nearest_grid_hz(offset_hz: float, grid_spacing_hz: Optional[float]) -> float:
    """Nearest LR-FHSS grid point to a carrier offset. If grid spacing is None,
    the grid reference collapses to the nominal center (single-tone assumption)."""
    if not grid_spacing_hz:
        return 0.0
    return round(offset_hz / grid_spacing_hz) * grid_spacing_hz
