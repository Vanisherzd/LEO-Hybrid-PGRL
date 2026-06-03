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

LR-FHSS specifics
-----------------
LR-FHSS (Semtech LR1121) is a GMSK-like, *sparse time-frequency hopping*
waveform: short narrow-band tones (dashes) that jump across the band over time.
It is NOT a continuous carrier and NOT QPSK. The old burst_energy_excess_db /
QPSK-EVM path was tuned for continuous bursts and is the WRONG indicator for
sparse hopping. The relevant indicator here is lr_fhss_candidate_score, built
from a spectrogram "hot tile" analysis:

  * per-frequency-bin noise floor = median over time (robust to sparse hops),
  * hot tiles = spectrogram cells exceeding (per-bin floor + N dB),
  * SHORT horizontal runs of hot tiles in time = "hop-like" dashes,
  * multiple narrow max-hold peaks across the band = frequency hopping,
  * frequency occupancy spread WIDE while time occupancy stays LOW = hopping
    rather than a continuous carrier.

To keep a fixed-dB hot-tile threshold robust against the high variance of a
single Hann periodogram bin (which would otherwise flag ~1% of pure-noise tiles
at 8 dB), the spectrogram is Welch-averaged over a few adjacent time frames
before the hot-tile test. This drives pure-noise false-hot tiles to ~0 while a
short hop spanning several raw frames still survives as one short averaged dash.

Pipeline:
1. Load .fc32 as complex64.
2. Estimate PSD (Welch via scipy if available, else numpy-only averaged
   periodogram fallback). fftshift so DC is band-centered.
3. Exclude a DC guard band (B210 has a strong LO/DC spike) when searching for
   the signal peak AND in every spectrogram/max-hold count.
4. peak_to_median_db, peak_frequency_offset_hz, burst_energy_excess_db, a
   supporting max-hold excess, and the LR-FHSS spectrogram metrics decide
   detection.
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
        --spectrogram-nfft 4096 \
        --lr-fhss-mode \
        --lr-fhss-score-threshold 0.5 \
        --uart-packet-sent-count 10 \
        --tx-off-reference hardware/captures/tx_off.fc32
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
    only; it is NOT a standards-compliant LR-FHSS PER measurement, and is NOT
    the LR-FHSS detection metric (LR-FHSS is GMSK-like sparse hopping, not
    QPSK). The relevant LR-FHSS indicator is lr_fhss_candidate_score.
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
        "note": "RF-quality proxy only; NOT the LR-FHSS metric (LR-FHSS is "
        "GMSK-like sparse hopping, not QPSK)",
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
    Returns (keep_mask, guard_hz). keep_mask is True for bins to KEEP when
    searching for the signal peak / counting hot tiles. Excludes a DC guard
    band around f=0 to suppress the B210 LO/DC spike.
    Guard half-width = max(guard_frac * fs, min_bins worth of resolution).
    """
    if len(freqs) <= 1:
        return np.ones(len(freqs), dtype=bool), 0.0
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
    NOTE: tuned for continuous bursts; for sparse LR-FHSS hops this is a weak /
    misleading indicator (de-emphasized in --lr-fhss-mode).
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


def compute_spectrogram_db(samples: np.ndarray, fs: float, nfft: int, overlap_frac: float = 0.5):
    """
    Compute a power spectrogram in dB, fftshifted so DC is centered.

    Returns (freqs, times, spec_db) where spec_db has shape
    (n_freq_bins, n_time_frames). numpy-only (Hann window). This is the shared
    time-frequency surface used by BOTH the max-hold and the LR-FHSS hot-tile
    analysis, so callers never duplicate the STFT math.
    """
    nfft = int(min(nfft, len(samples)))
    if nfft < 8:
        nfft = len(samples) if len(samples) >= 1 else 1
    if nfft <= 0:
        return np.array([0.0]), np.array([0.0]), np.full((1, 1), -300.0)

    window = np.hanning(nfft)
    win_norm = np.sum(window ** 2)
    if win_norm <= 0:
        win_norm = 1.0
    step = max(1, int(round(nfft * (1.0 - overlap_frac))))

    cols = []
    centers = []
    for s in range(0, len(samples) - nfft + 1, step):
        seg = samples[s : s + nfft] * window
        spec = np.fft.fft(seg, n=nfft)
        p = (np.abs(spec) ** 2) / (fs * win_norm)
        cols.append(p)
        centers.append((s + nfft / 2.0) / fs)

    if not cols:
        # Capture shorter than one full segment: one frame.
        seg = samples[:nfft]
        w = np.hanning(len(seg)) if len(seg) > 1 else np.ones(len(seg))
        wn = np.sum(w ** 2) if np.sum(w ** 2) > 0 else 1.0
        spec = np.fft.fft(seg * w, n=nfft)
        cols.append((np.abs(spec) ** 2) / (fs * wn))
        centers.append((len(seg) / 2.0) / fs)

    power = np.stack(cols, axis=1)  # (nfft, n_frames)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    power = np.fft.fftshift(power, axes=0)
    times = np.asarray(centers)
    with np.errstate(divide="ignore"):
        spec_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    return freqs, times, spec_db


def _avg_spectrogram_frames(spec_db: np.ndarray, times: np.ndarray, k: int):
    """
    Welch-average k adjacent time frames (in linear power) to cut periodogram
    variance, then back to dB. Returns (avg_spec_db, avg_times). With k>=4 the
    per-bin spread of pure noise collapses so a fixed-dB hot-tile threshold no
    longer flags noise, while a short hop spanning several raw frames survives.
    """
    if k <= 1 or spec_db.shape[1] < 2:
        return spec_db, times
    n_freq, n_time = spec_db.shape
    n_blocks = n_time // k
    if n_blocks < 1:
        return spec_db, times
    lin = np.power(10.0, spec_db / 10.0)
    lin = lin[:, : n_blocks * k].reshape(n_freq, n_blocks, k).mean(axis=2)
    with np.errstate(divide="ignore"):
        avg_db = 10.0 * np.log10(np.maximum(lin, 1e-30))
    t = np.asarray(times)[: n_blocks * k].reshape(n_blocks, k).mean(axis=1)
    return avg_db, t


def _maxhold_from_spec(freqs: np.ndarray, spec_db: np.ndarray):
    """Max across time per frequency bin → max-hold spectrum (dB)."""
    if spec_db.size == 0:
        return freqs, np.full(len(freqs), -300.0)
    return freqs, np.max(spec_db, axis=1)


def _maxhold_spectrum(samples: np.ndarray, fs: float, nfft: int):
    """
    Backward-compatible wrapper: (freqs, maxhold_db) fftshifted, DC centered.
    Now built on the shared compute_spectrogram_db().
    """
    freqs, _times, spec_db = compute_spectrogram_db(samples, fs, nfft)
    return _maxhold_from_spec(freqs, spec_db)


def maxhold_excess_db(freqs: np.ndarray, maxhold_db: np.ndarray, fs: float) -> float:
    """Max-hold peak excess over the (DC-excluded) median, in dB."""
    if maxhold_db.size == 0:
        return 0.0
    mask, _ = _dc_guard_mask(freqs, fs)
    median_db = float(np.median(maxhold_db))
    masked = np.where(mask, maxhold_db, -np.inf)
    if not np.any(np.isfinite(masked)):
        return 0.0
    return float(np.max(masked) - median_db)


def count_maxhold_peaks(freqs: np.ndarray, maxhold_db: np.ndarray, fs: float,
                        threshold_db: float, min_separation_bins: int = 2) -> int:
    """
    Count distinct narrow peaks in the max-hold spectrum that exceed
    median + threshold_db, with DC guard excluded. Simple local-maxima +
    prominence; adjacent above-threshold bins within min_separation_bins are
    merged into a single peak.
    """
    n = len(maxhold_db)
    if n < 3:
        return 0
    mask, _ = _dc_guard_mask(freqs, fs)
    median_db = float(np.median(maxhold_db))
    level = median_db + threshold_db

    above = (maxhold_db >= level) & mask
    if not np.any(above):
        return 0

    # Local maxima among above-threshold, DC-allowed bins.
    peak_idxs = []
    for i in range(n):
        if not above[i]:
            continue
        left = maxhold_db[i - 1] if i - 1 >= 0 else -np.inf
        right = maxhold_db[i + 1] if i + 1 < n else -np.inf
        if maxhold_db[i] >= left and maxhold_db[i] >= right:
            peak_idxs.append(i)

    if not peak_idxs:
        # Plateau case: count contiguous above-threshold runs instead.
        peak_idxs = [i for i in range(n) if above[i]]

    # Merge peaks closer than min_separation_bins, keep the strongest.
    peak_idxs.sort()
    merged = []
    cluster = [peak_idxs[0]]
    for idx in peak_idxs[1:]:
        if idx - cluster[-1] <= min_separation_bins:
            cluster.append(idx)
        else:
            best = max(cluster, key=lambda j: maxhold_db[j])
            merged.append(best)
            cluster = [idx]
    best = max(cluster, key=lambda j: maxhold_db[j])
    merged.append(best)
    return len(merged)


def compute_lr_fhss_metrics(samples: np.ndarray, fs: float, nfft: int,
                            threshold_db: float, hop_max_frames: int = 8,
                            time_avg_frames: int = 4) -> dict:
    """
    Spectrogram hot-tile analysis for the LR-FHSS sparse-hopping signature.

    The raw spectrogram is first Welch-averaged over `time_avg_frames` adjacent
    time frames (linear power) to cut single-periodogram variance, so a fixed-dB
    hot-tile threshold is robust against pure-noise false positives. All counts
    below are on this averaged spectrogram.

    Per-frequency-bin noise floor = MEDIAN over time for that bin (robust to
    sparse hops: a bin that is hot in only a few frames still has a low median).
    A tile is "hot" if its power exceeds (per-bin floor + threshold_db). DC
    guard bins are excluded from ALL counts.

    hop_like_segment_count: for each freq bin, find contiguous-in-time hot runs;
    a run counts as "hop-like" if its length is in [1, hop_max_frames] AND it
    does NOT span (nearly) the whole capture. A continuous carrier produces ONE
    long run per occupied bin → it does NOT inflate this count.

    Returns a dict of LR-FHSS metrics plus the spectrogram surface (for plots).
    """
    freqs, times, spec_db_raw = compute_spectrogram_db(samples, fs, nfft)
    spec_db, times_avg = _avg_spectrogram_frames(spec_db_raw, times, time_avg_frames)
    n_freq, n_time = spec_db.shape

    keep_mask, guard_hz = _dc_guard_mask(freqs, fs)

    # Per-bin noise floor = median over time.
    per_bin_floor = np.median(spec_db, axis=1, keepdims=True)  # (n_freq, 1)
    hot = spec_db > (per_bin_floor + threshold_db)  # (n_freq, n_time)

    # Exclude DC guard bins from every count.
    hot = hot & keep_mask[:, None]

    hot_bin_count = int(np.sum(hot))
    occupied_frequency_bins = int(np.sum(np.any(hot, axis=1)))
    occupied_time_bins = int(np.sum(np.any(hot, axis=0)))

    # hop-like short horizontal runs (per freq bin, contiguous in time).
    # A run spanning >= 80% of the capture is treated as "continuous",
    # not a hop, regardless of hop_max_frames.
    span_limit = max(hop_max_frames + 1, int(round(0.8 * n_time)))
    hop_like_segment_count = 0
    if n_time >= 1:
        for f in range(n_freq):
            if not keep_mask[f]:
                continue
            row = hot[f]
            if not row.any():
                continue
            run = 0
            for t in range(n_time):
                if row[t]:
                    run += 1
                else:
                    if run > 0:
                        if 1 <= run <= hop_max_frames and run < span_limit:
                            hop_like_segment_count += 1
                        run = 0
            if run > 0:
                if 1 <= run <= hop_max_frames and run < span_limit:
                    hop_like_segment_count += 1

    # Max-hold surfaces are computed on the RAW (un-averaged) spectrogram so the
    # max-hold peak/excess reflects true instantaneous peaks.
    mh_freqs, mh_db = _maxhold_from_spec(freqs, spec_db_raw)

    return {
        "time_frequency_hot_bin_count": hot_bin_count,
        "occupied_frequency_bins": occupied_frequency_bins,
        "occupied_time_bins": occupied_time_bins,
        "hop_like_segment_count": int(hop_like_segment_count),
        "_freqs": freqs,
        "_times": times_avg,
        "_spec_db": spec_db,         # averaged (for hot-tile/plot consistency)
        "_spec_db_raw": spec_db_raw,  # raw (for waterfall)
        "_mh_freqs": mh_freqs,
        "_mh_db": mh_db,
        "_n_freq_total": int(np.sum(keep_mask)),
        "_n_time_total": int(n_time),
    }


def compute_lr_fhss_score(fhss: dict, maxhold_peak_count: int,
                          peak_to_median_db: float, maxhold_excess_db_val: float) -> float:
    """
    Composite LR-FHSS candidate score in [0, 1]. Monotonic and bounded.

    Evidence channels (each normalized to [0, 1] via a saturating ramp, then
    averaged with documented weights):

      (i)   peaks  : multiple distinct narrow max-hold peaks across the band.
                     maxhold_peak_count / PEAK_SAT, capped at 1.
                     LR-FHSS hops across many channels → several peaks.
      (ii)  hops   : sparse SHORT horizontal dashes in the spectrogram.
                     hop_like_segment_count / HOP_SAT, capped at 1.
                     This is the strongest hopping-specific cue.
      (iii) spread : frequency occupancy spread WIDE while time occupancy stays
                     LOW. spread = freq_occ_frac * (1 - time_occ_frac), rescaled.
                     A continuous carrier has high time_occ_frac → spread ~ 0.
                     Pure noise has freq_occ_frac ~ 0 → spread ~ 0. Only sparse
                     hopping (wide in freq, sparse in time) scores high.
      (iv)  level  : raw magnitude of excess above the floor.
                     max(peak_to_median_db, maxhold_excess_db) / LEVEL_SAT.

    score = 0.30*peaks + 0.35*hops + 0.20*spread + 0.15*level   (weights sum 1)

    Weights emphasize the two hopping-specific cues (peaks, hops) while keeping
    raw level as a corroborating but non-dominant term so that a strong
    *continuous* tone (one fat peak, no short dashes, time_occ_frac high) cannot
    by itself push the score high.
    """
    PEAK_SAT = 4.0    # >= 4 distinct narrow peaks saturates the peaks channel
    HOP_SAT = 8.0     # >= 8 hop-like dashes saturates the hops channel
    LEVEL_SAT = 20.0  # dB excess that saturates the level channel

    n_freq_total = max(1, int(fhss.get("_n_freq_total", 1)))
    n_time_total = max(1, int(fhss.get("_n_time_total", 1)))
    freq_occ_frac = fhss["occupied_frequency_bins"] / n_freq_total
    time_occ_frac = fhss["occupied_time_bins"] / n_time_total

    peaks = min(1.0, max(0, maxhold_peak_count) / PEAK_SAT)
    hops = min(1.0, max(0, fhss["hop_like_segment_count"]) / HOP_SAT)
    spread = max(0.0, min(1.0, freq_occ_frac)) * max(0.0, 1.0 - time_occ_frac)
    spread = min(1.0, spread * 4.0)  # rescale: sparse hopping rarely fills band
    level_db = max(peak_to_median_db, maxhold_excess_db_val, 0.0)
    level = min(1.0, level_db / LEVEL_SAT)

    score = 0.30 * peaks + 0.35 * hops + 0.20 * spread + 0.15 * level
    return float(max(0.0, min(1.0, score)))


def detect_signal(samples: np.ndarray, fs: float, nfft: int,
                  threshold_db: float, hop_max_frames: int = 8) -> dict:
    """
    Core detection. Returns a dict of all detection metrics (legacy + LR-FHSS)
    plus the PSD/freqs/guard/spectrogram used (for plotting).
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

    # LR-FHSS spectrogram metrics (shares its STFT with the max-hold below).
    fhss = compute_lr_fhss_metrics(samples, fs, nfft, threshold_db, hop_max_frames)
    mh_freqs, mh_db = fhss["_mh_freqs"], fhss["_mh_db"]
    mh_excess = maxhold_excess_db(mh_freqs, mh_db, fs)
    mh_peak_count = count_maxhold_peaks(mh_freqs, mh_db, fs, threshold_db)

    score = compute_lr_fhss_score(fhss, mh_peak_count, peak_to_median_db, mh_excess)

    return {
        "peak_to_median_db": round(peak_to_median_db, 2),
        "peak_frequency_offset_hz": round(peak_freq, 2),
        "burst_energy_excess_db": round(burst_db, 2),
        "maxhold_excess_db": round(mh_excess, 2),
        "dc_guard_hz": round(guard_hz, 2),
        # LR-FHSS metrics
        "time_frequency_hot_bin_count": fhss["time_frequency_hot_bin_count"],
        "occupied_frequency_bins": fhss["occupied_frequency_bins"],
        "occupied_time_bins": fhss["occupied_time_bins"],
        "hop_like_segment_count": fhss["hop_like_segment_count"],
        "maxhold_peak_count_excluding_dc": int(mh_peak_count),
        "lr_fhss_candidate_score": round(score, 4),
        # surfaces for plotting / downstream
        "_freqs": freqs,
        "_psd": psd,
        "_median_psd": median_psd,
        "_mh_freqs": mh_freqs,
        "_mh_db": mh_db,
        "_spec_freqs": fhss["_freqs"],
        "_spec_times": fhss["_times"],
        "_spec_db": fhss["_spec_db"],
        "_n_freq_total": fhss["_n_freq_total"],
        "_n_time_total": fhss["_n_time_total"],
    }


def classify(det: dict, threshold_db: float, score_threshold: float,
             lr_fhss_mode: bool, uart_packet_count, off_metrics) -> tuple:
    """
    Conservative decision logic.

    Returns (validation_status, signal_detected, gates_applied: list[str]).

    validation_status is one of:
      - noise_floor_only      : no evidence.
      - weak_signal_candidate : some LR-FHSS-like evidence but not corroborated.
      - signal_detected       : STRONG evidence; ALL provided gates must hold.

    Gates for signal_detected:
      a) lr_fhss_candidate_score >= score_threshold (REQUIRED, always applied).
      b) if uart_packet_count is not None: must be > 0 (else downgrade).
      c) if off_metrics provided: TX-ON must be SIGNIFICANTLY stronger than OFF
         (occupancy delta AND maxhold delta above documented margins).
    """
    p2m = det["peak_to_median_db"]
    burst = det["burst_energy_excess_db"]
    score = det["lr_fhss_candidate_score"]
    candidate = score >= score_threshold

    gates = []

    # ---- noise_floor_only screen ----
    # Evidence requires the score OR a moderate peak OR genuine hop-like dashes
    # (after Welch-averaging, noise produces ~0 hops). maxhold_excess alone is
    # NOT counted as evidence to avoid a lone DC-adjacent glitch tripping it.
    has_any_evidence = (
        candidate
        or det["hop_like_segment_count"] > 0
        or p2m >= (threshold_db - 3.0)
    )
    if not has_any_evidence:
        return "noise_floor_only", False, gates

    if lr_fhss_mode:
        # Score-driven path (burst de-emphasized — wrong for sparse hops).
        gates.append("score>=threshold")
        gate_a = candidate
        gate_b = True
        gate_c = True

        if uart_packet_count is not None:
            gates.append("uart_packet_count>0")
            gate_b = uart_packet_count > 0
        if off_metrics is not None:
            gates.append("tx_on_stronger_than_off")
            gate_c = bool(_on_stronger_than_off(det, off_metrics))

        if gate_a and gate_b and gate_c:
            return "signal_detected", True, gates
        # Strong-ish but a corroboration gate failed → weak.
        return "weak_signal_candidate", False, gates

    # ---- legacy / non-lr-fhss-mode path ----
    # Score still drives detection (Task B: signal_detected requires score>=thr),
    # but the legacy burst term remains a contributor for backward sanity.
    gates.append("score>=threshold")
    gate_a = candidate
    gate_b = True
    gate_c = True
    if uart_packet_count is not None:
        gates.append("uart_packet_count>0")
        gate_b = uart_packet_count > 0
    if off_metrics is not None:
        gates.append("tx_on_stronger_than_off")
        gate_c = bool(_on_stronger_than_off(det, off_metrics))

    legacy_strong = (p2m >= threshold_db and burst >= 3.0)
    if gate_a and gate_b and gate_c and legacy_strong:
        return "signal_detected", True, gates
    if candidate or (threshold_db - 3.0) <= p2m < threshold_db or legacy_strong:
        return "weak_signal_candidate", False, gates
    return "noise_floor_only", False, gates


# Margins documented for gate (c): ON must beat OFF by these to corroborate.
_ONOFF_FREQ_OCC_MARGIN = 3       # occupied_frequency_bins delta
_ONOFF_MAXHOLD_MARGIN_DB = 3.0   # maxhold_excess_db delta


def _on_stronger_than_off(det: dict, off_metrics: dict) -> bool:
    """ON significantly stronger than OFF: BOTH occupancy AND maxhold deltas."""
    on_freq = det["occupied_frequency_bins"]
    off_freq = int(off_metrics.get("occupied_frequency_bins", 0))
    on_mh = det["maxhold_excess_db"]
    off_mh = float(off_metrics.get("maxhold_excess_db", 0.0))
    freq_ok = (on_freq - off_freq) >= _ONOFF_FREQ_OCC_MARGIN
    mh_ok = (on_mh - off_mh) >= _ONOFF_MAXHOLD_MARGIN_DB
    return freq_ok and mh_ok


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
    lr_fhss_candidate: bool = False,
    hop_like_segment_count: int = 0,
    occupied_frequency_bins: int = 0,
):
    """Spectrogram PNG. Marks peak when signal detected; states status in title.
    Annotates LR-FHSS hop evidence when lr_fhss_candidate is True."""
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

    if lr_fhss_candidate:
        title_tag = (
            f"LR-FHSS candidate: {occupied_frequency_bins} freq bins, "
            f"{hop_like_segment_count} hop-like dashes"
        )
    else:
        title_tag = validation_status
    ax.set_title(f"Waterfall — {output_path.name}  [{title_tag}]")

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
        label = title_tag if lr_fhss_candidate else validation_status
        ax.text(
            0.02,
            0.95,
            label,
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
# Reference (TX-OFF) loading helpers — importable by compare_tx_on_off.py.
# ---------------------------------------------------------------------------
def load_samples(path) -> np.ndarray:
    """Load a .fc32 capture as complex64."""
    return np.fromfile(Path(path), dtype=np.complex64)


def compute_reference_metrics(samples: np.ndarray, fs: float, nfft: int,
                              threshold_db: float) -> dict:
    """
    Compute the metrics needed for ON/OFF comparison and gate (c) from raw IQ.
    Returns the canonical comparison keys for one capture.
    """
    det = detect_signal(samples, fs, nfft, threshold_db)
    return {
        "maxhold_excess_db": det["maxhold_excess_db"],
        "hot_bin_count": det["time_frequency_hot_bin_count"],
        "occupied_frequency_bins": det["occupied_frequency_bins"],
        "occupied_time_bins": det["occupied_time_bins"],
        "lr_fhss_candidate_score": det["lr_fhss_candidate_score"],
    }


def load_off_metrics(off_reference, off_json, fs: float, nfft: int,
                     threshold_db: float):
    """
    Load OFF-reference metrics for gate (c). Prefer recomputing from a raw
    capture (--tx-off-reference); otherwise read a prior analysis JSON
    (--tx-off-json). Returns a dict with at least occupied_frequency_bins and
    maxhold_excess_db, or None if neither provided / loadable.
    """
    if off_reference:
        p = Path(off_reference)
        if not p.exists():
            print(f"[analyze] WARN: tx-off-reference {p} not found — gate (c) skipped",
                  file=sys.stderr)
            return None
        samples = load_samples(p)
        if len(samples) == 0:
            print("[analyze] WARN: tx-off-reference empty — gate (c) skipped", file=sys.stderr)
            return None
        m = compute_reference_metrics(samples, fs, nfft, threshold_db)
        m["_source"] = str(p)
        return m
    if off_json:
        p = Path(off_json)
        if not p.exists():
            print(f"[analyze] WARN: tx-off-json {p} not found — gate (c) skipped",
                  file=sys.stderr)
            return None
        with open(p) as f:
            data = json.load(f)
        m = {
            "occupied_frequency_bins": int(data.get("occupied_frequency_bins", 0)),
            "maxhold_excess_db": float(data.get("maxhold_excess_db", 0.0)),
            "_source": str(p),
        }
        return m
    return None


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
        help="peak-to-median / hot-tile threshold (dB) for detection (default 8)",
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
        help="FFT size for detection PSD / spectrogram / max-hold (default 4096)",
    )
    # --- New LR-FHSS flags ---
    parser.add_argument(
        "--lr-fhss-mode",
        action="store_true",
        help="Enable LR-FHSS-driven decision (score drives detection; "
        "burst_energy_excess_db de-emphasized as it is wrong for sparse hops)",
    )
    parser.add_argument(
        "--lr-fhss-score-threshold",
        type=float,
        default=0.5,
        help="lr_fhss_candidate_score threshold for candidate/detection (default 0.5)",
    )
    parser.add_argument(
        "--uart-packet-sent-count",
        type=int,
        default=None,
        help="Optional UART-confirmed packet count; gate (b): must be > 0 for "
        "signal_detected when provided",
    )
    parser.add_argument(
        "--tx-off-reference",
        type=str,
        default=None,
        help="Optional TX-OFF .fc32 capture; gate (c): ON must be significantly "
        "stronger than OFF",
    )
    parser.add_argument(
        "--tx-off-json",
        type=str,
        default=None,
        help="Optional prior TX-OFF analysis JSON (alternative to "
        "--tx-off-reference for gate (c))",
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
    print(f"[analyze] LR-FHSS mode: {'ON' if args.lr_fhss_mode else 'off'}")

    # --- Signal detection (always; computes LR-FHSS metrics regardless of mode) ---
    print("[analyze] Running signal detection...")
    det = detect_signal(samples, fs, args.spectrogram_nfft, args.signal_threshold_db)

    # --- TX-OFF reference for gate (c) ---
    off_metrics = load_off_metrics(
        args.tx_off_reference, args.tx_off_json, fs, args.spectrogram_nfft,
        args.signal_threshold_db,
    )

    validation_status, signal_detected, gates = classify(
        det,
        args.signal_threshold_db,
        args.lr_fhss_score_threshold,
        args.lr_fhss_mode,
        args.uart_packet_sent_count,
        off_metrics,
    )
    lr_fhss_candidate = bool(det["lr_fhss_candidate_score"] >= args.lr_fhss_score_threshold)

    print(f"[analyze]   peak_to_median_db        = {det['peak_to_median_db']:.2f} dB")
    print(f"[analyze]   burst_energy_excess      = {det['burst_energy_excess_db']:.2f} dB")
    print(f"[analyze]   maxhold_excess_db        = {det['maxhold_excess_db']:.2f} dB")
    print(f"[analyze]   peak_freq_offset         = {det['peak_frequency_offset_hz']:.1f} Hz")
    print(f"[analyze]   DC guard half-width      = {det['dc_guard_hz']:.1f} Hz")
    print("[analyze]   --- LR-FHSS metrics ---")
    print(f"[analyze]   time_frequency_hot_bin_count      = {det['time_frequency_hot_bin_count']}")
    print(f"[analyze]   occupied_frequency_bins           = {det['occupied_frequency_bins']}")
    print(f"[analyze]   occupied_time_bins                = {det['occupied_time_bins']}")
    print(f"[analyze]   hop_like_segment_count            = {det['hop_like_segment_count']}")
    print(f"[analyze]   maxhold_peak_count_excluding_dc   = {det['maxhold_peak_count_excluding_dc']}")
    print(f"[analyze]   lr_fhss_candidate_score           = {det['lr_fhss_candidate_score']:.4f}")
    print(f"[analyze]   lr_fhss_candidate                 = {lr_fhss_candidate}")
    print(f"[analyze]   gates applied: {gates if gates else 'none'}")
    if off_metrics is not None:
        print(f"[analyze]   tx-off ref: occ_freq={off_metrics.get('occupied_frequency_bins')}, "
              f"maxhold_excess={off_metrics.get('maxhold_excess_db')} dB "
              f"(src={off_metrics.get('_source','?')})")

    # --- SNR always (coarse power ratio, safe) ---
    snr = estimate_snr(samples)

    # --- CFO / EVM gated behind detection ---
    rx_cfo_hz = None
    rx_cfo_std_hz = None
    rx_evm_percent = None
    cfo_method = None
    evm_method = None

    gates_str = ", ".join(gates) if gates else "none"

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
            f"Signal detected (gates applied: {gates_str}): "
            f"lr_fhss_candidate_score={det['lr_fhss_candidate_score']:.3f}>=thr, "
            f"hop_like={det['hop_like_segment_count']}, "
            f"occ_freq={det['occupied_frequency_bins']}, "
            f"maxhold_peaks={det['maxhold_peak_count_excluding_dc']}. "
            f"CFO/EVM are RF-quality proxies only and are NOT the LR-FHSS metric "
            f"(LR-FHSS is GMSK-like sparse hopping, not QPSK); "
            f"lr_fhss_candidate_score is the relevant indicator."
        )
        print(f"[analyze]   rx_cfo_hz = {rx_cfo_hz:.2f} Hz")
        print(f"[analyze]   rx_evm_percent = {rx_evm_percent:.2f} %  (RF proxy, not the LR-FHSS metric)")
    else:
        if validation_status == "weak_signal_candidate":
            analysis_note = (
                f"Weak signal candidate (gates applied: {gates_str}): "
                f"lr_fhss_candidate_score={det['lr_fhss_candidate_score']:.3f} "
                f"(threshold {args.lr_fhss_score_threshold:.2f}), "
                f"hop_like={det['hop_like_segment_count']}, "
                f"peak/median={det['peak_to_median_db']:.1f} dB. Evidence present "
                f"but not corroborated/strong enough for signal_detected. CFO/EVM "
                f"proxies intentionally withheld (conservative). EVM is NOT the "
                f"LR-FHSS metric; lr_fhss_candidate_score is the relevant indicator."
            )
            print("[analyze] Weak candidate → CFO/EVM SKIPPED.")
        else:
            analysis_note = (
                "No LR-FHSS evidence detected; CFO/EVM proxy not meaningful. "
                "Capture is consistent with noise floor (DC/LO spike excluded). "
                "EVM is NOT the LR-FHSS metric; lr_fhss_candidate_score is the "
                "relevant indicator."
            )
            print("[analyze] Noise floor only → CFO/EVM SKIPPED.")

    results = {
        "validation_status": validation_status,
        "signal_detected": bool(signal_detected),
        "peak_to_median_db": det["peak_to_median_db"],
        "burst_energy_excess_db": det["burst_energy_excess_db"],
        "maxhold_excess_db": det["maxhold_excess_db"],
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
        # --- LR-FHSS metrics (new, additive) ---
        "time_frequency_hot_bin_count": det["time_frequency_hot_bin_count"],
        "occupied_frequency_bins": det["occupied_frequency_bins"],
        "occupied_time_bins": det["occupied_time_bins"],
        "hop_like_segment_count": det["hop_like_segment_count"],
        "maxhold_peak_count_excluding_dc": det["maxhold_peak_count_excluding_dc"],
        "lr_fhss_candidate_score": det["lr_fhss_candidate_score"],
        "lr_fhss_candidate": bool(lr_fhss_candidate),
        "lr_fhss_mode": bool(args.lr_fhss_mode),
        "lr_fhss_score_threshold": float(args.lr_fhss_score_threshold),
        "gates_applied": gates,
        "uart_packet_sent_count": args.uart_packet_sent_count,
        "tx_off_reference_used": off_metrics.get("_source") if off_metrics else None,
        # Supporting / provenance fields (non-canonical but useful, do not break schema).
        "dc_guard_hz": det["dc_guard_hz"],
        "detection_nfft": int(args.spectrogram_nfft),
        "psd_method": "scipy_welch" if _HAVE_SCIPY else "numpy_periodogram",
        "cfo_method": cfo_method,
        "evm_method": evm_method,
        "evm_note": "RF-quality proxy only; NOT the LR-FHSS metric (LR-FHSS is "
        "GMSK-like sparse hopping, not QPSK). lr_fhss_candidate_score is the "
        "relevant indicator.",
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
            lr_fhss_candidate=lr_fhss_candidate,
            hop_like_segment_count=det["hop_like_segment_count"],
            occupied_frequency_bins=det["occupied_frequency_bins"],
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
    print(f"  validation_status:        {validation_status}")
    print(f"  signal_detected:          {signal_detected}")
    print(f"  peak_to_median_db:        {det['peak_to_median_db']:.2f} dB")
    print(f"  burst_energy_excess:      {det['burst_energy_excess_db']:.2f} dB")
    print(f"  maxhold_excess_db:        {det['maxhold_excess_db']:.2f} dB")
    print(f"  peak_freq_offset_hz:      {det['peak_frequency_offset_hz']:.1f} Hz")
    print(f"  lr_fhss_candidate_score:  {det['lr_fhss_candidate_score']:.4f}")
    print(f"  lr_fhss_candidate:        {lr_fhss_candidate}")
    print(f"  hop_like_segment_count:   {det['hop_like_segment_count']}")
    print(f"  occupied_frequency_bins:  {det['occupied_frequency_bins']}")
    print(f"  occupied_time_bins:       {det['occupied_time_bins']}")
    print(f"  maxhold_peak_count:       {det['maxhold_peak_count_excluding_dc']}")
    print(f"  gates_applied:            {gates_str}")
    print(f"  rx_snr_db:                {snr['snr_db']:.2f} dB")
    if signal_detected:
        print(f"  rx_cfo_hz:                {rx_cfo_hz:.2f} Hz")
        print(f"  rx_evm_percent:           {rx_evm_percent:.2f} %  (RF proxy, NOT the LR-FHSS metric)")
    else:
        print("  rx_cfo_hz / rx_evm:       SKIPPED (no detected signal)")
    print("=================================")
    print("\nNOTE: EVM is an RF-quality proxy only and is NOT the LR-FHSS metric.")
    print("  LR-FHSS is GMSK-like sparse hopping (not QPSK);")
    print("  lr_fhss_candidate_score is the relevant detection indicator.")


if __name__ == "__main__":
    main()
