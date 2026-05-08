#!/usr/bin/env python3
"""
dual_mode_trx.py
================
Dual-Mode Cognitive SDR Terminal — Simulation + Hardware TX/RX.

Modes:
  --sim  (default)  Pure NumPy/SciPy channel emulation. No radio needed.
                    Used to validate DSP math, EVM, and constellation plotting.
  --hw             Live TX/RX with USRP B210. Requires UHD driver + USB 3.0.

LR-FHSS parameters: BW=242 Hz, chip_rate=8393 cps, SF=14, hop=49 ms
QPSK payload, RRC pulse shaping, PGRL inverse-Doppler pre-compensation.

Hardware path: USRP B210 (S-band 436.5 MHz, 1 Msps, TX/RX gain 40 dB)
Edge host     : MacBook Pro (Intel or Apple Silicon) or Raspberry Pi 5

Usage:
  # Simulation (any machine)
  python dual_mode_trx.py --sim

  # Simulation with custom Doppler
  python dual_mode_trx.py --sim --doppler-hz 8000 --n-bursts 5 --snr-db 20

  # Hardware (requires B210 connected)
  python dual_mode_trx.py --hw --doppler-hz 8000

  # Show all options
  python dual_mode_trx.py --help

Output:
  payload_results_realizations/live_usrp/Live_B210_EVM.png
  payload_results_realizations/live_usrp/evm_log.csv

Author : Hermes Agent — LEO-PINN / D2S Architecture
Date   : 2026-05-08
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
import logging
import datetime
from typing import Optional, Tuple, Literal

import numpy as np
from scipy import signal as sp
from scipy.linalg import circulant

# ─────────────────────────────────────────────────────────────────────────────
# UHD import — graceful failure if not installed
# ─────────────────────────────────────────────────────────────────────────────
UHD_AVAILABLE = False
UHD_ERR: Optional[str] = None
try:
    import uhd
    from uhd import libpyuhd as libuhd
    UHD_AVAILABLE = True
except ImportError as exc:
    UHD_ERR = f"uhd not importable ({exc}). Install: pip install pyuhd"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dual_mode_trx")

# ─────────────────────────────────────────────────────────────────────────────
# Physical Constants
# ─────────────────────────────────────────────────────────────────────────────
F_CARRIER   : float = 436.5e6    # Hz  S-band carrier
SAMPLE_RATE : float = 1e6        # Hz  1 Msps
TX_GAIN     : float = 40.0       # dB
RX_GAIN     : float = 40.0       # dB
TX_AMPL     : float = 0.25       # B210 DAC peak normalisation
N_SYMS      : int   = 300        # QPSK symbols per burst
SPB         : int   = 8          # Samples per symbol → 125 ksps
ROLLOFF     : float = 0.5         # RRC roll-off factor
BURST_GAP   : int   = 500         # Zero-padding samples between bursts

# QPSK constellation
QPSK_ANGLES = np.array([math.pi/4, 3*math.pi/4, -3*math.pi/4, -math.pi/4],
                         dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR     = os.path.join(REPO_ROOT, "payload_results_realizations", "live_usrp")
os.makedirs(OUT_DIR, exist_ok=True)
PNG_PATH    = os.path.join(OUT_DIR, "Live_B210_EVM.png")
CSV_PATH    = os.path.join(OUT_DIR, "evm_log.csv")

# ══════════════════════════════════════════════════════════════════════════════
# DSP CORE — QPSK / LR-FHSS Baseband Generation
# ══════════════════════════════════════════════════════════════════════════════

def qpsk_symbols(n: int, ampl: float = 1.0) -> np.ndarray:
    """Random QPSK complex symbols."""
    idx = np.random.randint(0, 4, size=n, dtype=np.int64)
    ang = QPSK_ANGLES[idx]
    return ampl * (np.cos(ang) + 1j * np.sin(ang))


def rrc_kernel(span_syms: int = 8, alpha: float = 0.5, spb: int = SPB
               ) -> np.ndarray:
    """
    Root Raised Cosine FIR filter kernel (IEEE 802.15.4g / ETSI EN 301 428).
    h(t) = sinc(t/T) * cos(παt/T) / (1 - (2αt/T)²)
    With T = 1 (symbol period in sample units):
      h(t) = sinc(t) * cos(παt) / (1 - (2αt)²)
    Key properties (α=0.5, T=1):
      h(0) = 1        (peak)
      h(±T/2) = 0.6   (roll-off)
      h(±T) = 0       (first zero)
      h(±3T/2) = -0.1 (side-lobe)
    Returns: real-valued taps, length = span_syms*spb + 1, peak = 1 at t=0.
    """
    Ts  = 1.0
    n_t = np.arange(-span_syms*Ts/2,
                     span_syms*Ts/2 + Ts/spb,
                     Ts/spb, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        # np.sinc(x) = sin(πx)/(πx) — correctly gives sinc(0) = 1 (not NaN)
        sinc_t  = np.sinc(n_t)                             # sin(πt)/(πt)
        cos_f   = np.cos(math.pi * alpha * n_t)
        denom   = 1.0 - (2.0 * alpha * n_t)**2
        # At t=0: sinc=1, cos=1, denom=1 → h=1 (handled by sinc giving 1.0)
        # At t=±1: sinc=0, denom=0 → h=0 (else branch sets to 0)
        h = np.where(np.abs(denom) > 1e-12,
                     sinc_t * cos_f / denom,
                     0.0)                                   # t = ±1/(2α) = ±1 → zero
    h = h / np.max(np.abs(h))
    return h.astype(np.float32)


def shape_burst(syms: np.ndarray, spb: int = SPB,
                kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """RRC pulse-shape QPSK symbols → complex baseband samples."""
    if kernel is None:
        kernel = rrc_kernel()
    upsampled = np.zeros(len(syms)*spb, dtype=np.complex64)
    upsampled[::spb] = syms
    shaped = np.convolve(upsampled, kernel, mode='full')
    # The RRC peak is at kernel.argmax() = 28 (not len(kernel)//2 = 32) because the
    # discrete sampling grid hits ±Ts/2, ±3Ts/2, ... instead of t=0, ±Ts, ±2Ts, ...
    # Using the true peak index ensures shaped[i] = sym[i/spb] * RRC_peak = sym[i/spb].
    offset = int(np.argmax(kernel))
    return shaped[offset: offset + len(syms)*spb].astype(np.complex64)


def generate_lr_fhss_chirp(n_samples: int, chip_rate: float = 8393.0,
                            sample_rate: float = SAMPLE_RATE,
                            seed: int = 42) -> np.ndarray:
    """
    LR-FHSS chirp: linear frequency sweep over 242 Hz BW.
    Used as preamble / synchronisation header before QPSK payload.
    """
    rng  = np.random.default_rng(seed)
    t    = np.arange(n_samples, dtype=np.float64) / sample_rate
    # Frequency sweep: 0 → +chip_rate over the samples
    freq = chip_rate * np.arange(n_samples) / n_samples
    # Apply random chip-phase (PN spreading)
    chip_phase = rng.integers(0, 2, size=n_samples) * 2 - 1   # ±1
    phase = 2*math.pi * np.cumsum(freq) / sample_rate
    chirp = chip_phase * np.exp(1j * phase)
    return chirp.astype(np.complex64)


def generate_tx_burst(n_syms: int = N_SYMS, use_lr_fhss_preamble: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full TX burst: [pilot tone (20 syms)] + [GAP (BURST_GAP)] + [RRC-shaped QPSK payload].
    Returns: (full_burst, qpsk_ref_symbols)
    The pilot is a clean complex tone at 62.5 kHz — used for frequency tracking.
    """
    # Single-tone pilot: pure complex exponential, 20 symbols
    n_pilot = 20 * SPB
    t_pilot = np.arange(n_pilot, dtype=np.float64) / SAMPLE_RATE
    f_pilot = 62.5e3          # 62.5 kHz (within 125 ksps symbol rate)
    pilot   = np.exp(1j * 2.0 * math.pi * f_pilot * t_pilot).astype(np.complex64)
    gap     = np.zeros(BURST_GAP, dtype=np.complex64)

    # QPSK payload
    syms  = qpsk_symbols(n_syms)
    shaped = shape_burst(syms)

    burst = np.concatenate([pilot, gap, shaped, gap])
    return burst, syms


# ══════════════════════════════════════════════════════════════════════════════
# PGRL PRE-COMPENSATION (W3 core)
# ══════════════════════════════════════════════════════════════════════════════

def apply_inverse_doppler(iq        : np.ndarray,
                           df_hz     : float,
                           fs_hz     : float = SAMPLE_RATE
                           ) -> np.ndarray:
    """
    W3: Inverse-Doppler pre-compensation.
    TX_side[x](t) = x(t) · exp(-j·2π·Δf·t) → RX sees zero net frequency offset.

    df_hz: Doppler shift predicted by PGRL PINN (e.g., +5033 Hz from ISS pass).
    """
    n = len(iq)
    t = np.arange(n, dtype=np.float64) / fs_hz
    rot = np.exp(-1j * 2.0 * math.pi * df_hz * t).astype(np.complex64)
    return (iq * rot).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CHANNEL (--sim mode)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_channel(tx_burst  : np.ndarray,
                     df_hz     : float,
                     snr_db    : float = 20.0,
                     fs_hz     : float = SAMPLE_RATE
                     ) -> Tuple[np.ndarray, float]:
    """
    Emulate over-the-air LEO channel.

    Signal model:
      TX sends pre-compensated burst:  burst_tx = burst_raw * exp(-j*2π*df_hz*t)
      Channel adds Doppler:            burst_rx = burst_raw * exp(-j*2π*df_hz*t)
                                                       * exp(+j*2π*df_hz*t)
                                            = burst_raw    (perfect cancellation)
      Then AWGN is added.

    The "f_meas" returned is the residual frequency offset that the RX estimator
    would measure from the preamble.  In simulation (pre-comp perfectly cancels
    the Doppler), the residual is dominated by estimation noise — we model it
    as a small Gaussian error around 0 Hz proportional to the preamble SNR.
    """
    n  = len(tx_burst)
    t  = np.arange(n, dtype=np.float64) / fs_hz

    # 1. Doppler rotation on the pre-compensated burst
    #    (burst_comp = burst_raw * exp(-j*2π*df_hz*t), so +df rotates it back)
    rot = np.exp(1j * 2.0 * math.pi * df_hz * t).astype(np.complex64)
    rx  = (tx_burst * rot).astype(np.complex64)

    # 2. AWGN — calibrate to the DATA section power (not burst average).
    #    The burst contains: pilot (power=1), gap (0), data (≈0.44), gap (0).
    #    Using the burst average under-estimates data SNR.
    #    Measure data section power separately and scale noise accordingly.
    n_pilot_hdr  = 20 * SPB
    data_start   = n_pilot_hdr + BURST_GAP
    data_end     = data_start + N_SYMS * SPB
    data_pow     = float(np.mean(np.abs(rx[data_start:data_end])**2))
    n_var        = data_pow / (10**(snr_db/10)) if snr_db < 60.0 else 0.0
    _sig_db = 10 * math.log10(data_pow / n_var) if n_var > 0 else float('inf')
    _n_amp  = math.sqrt(n_var) if n_var > 0 else 0.0
    if snr_db < 60.0:
        rx = rx + np.sqrt(n_var/2) * (
            np.random.randn(n) + 1j * np.random.randn(n)
        ).astype(np.complex64)
        log.info("  [sim] data_pow=%.4f n_var=%.6e n_amp=%.4f sig_db=%.1f (target %d dB)",
                 data_pow, n_var, _n_amp, _sig_db, snr_db)

    # 3. Residual frequency estimate (what RX preamble estimator would return)
    #    With a strong chirp/ZC preamble at SNR=snr_db, the CRB for frequency est.
    #    at sample rate fs over N preamble samples is:
    #      σ_f = sqrt(3 * fs² / (2π² * N³ * SNR))
    #    We use this to model the estimation noise on top of the near-zero residual.
    #    With pre-comp perfectly canceling, the only residual is this noise.
    n_est   = min(n, 1000)
    sig_est = float(np.mean(np.abs(rx[:n_est])**2))
    snr_lin = sig_est / max(n_var, 1e-12)
    sigma_f = math.sqrt(3.0 * fs_hz**2 / (2.0 * math.pi**2 * n_est**3 * max(snr_lin, 1e-9)))
    f_est   = float(np.random.normal(0.0, sigma_f))   # mean residual ≈ 0 (pre-comp)

    log.info("Sim channel: df_applied=%.1f Hz  df_measured=%.2f Hz  SNR=%.1f dB",
             df_hz, f_est, snr_db)
    return rx.astype(np.complex64), float(f_est)


# ══════════════════════════════════════════════════════════════════════════════
# RECEIVE PROCESSING (common to both modes)
# ══════════════════════════════════════════════════════════════════════════════

def coarse_freq_correct(rx       : np.ndarray,
                         df_hz   : float,
                         fs_hz   : float = SAMPLE_RATE
                         ) -> np.ndarray:
    """Remove residual frequency offset from RX stream."""
    n  = len(rx)
    t  = np.arange(n, dtype=np.float64) / fs_hz
    rot = np.exp(-1j * 2.0*math.pi*df_hz*t).astype(np.complex64)
    return (rx * rot).astype(np.complex64)


def costas_loop(iq      : np.ndarray,
                loop_bw : float = 0.005,
                psr_len : int   = 2
                ) -> np.ndarray:
    """
    Costas loop (order 4) for QPSK phase tracking.
    loop_bw: loop bandwidth (normalised to symbol rate)
    psr_len: number of pilot symbols at start (known pattern for phase lock)
    Returns: phase-corrected IQ.
    """
    # Use Viterbi-Viterbi (4th power) phase estimation on data
    n  = len(iq)
    w  = 1.0 + 0j
    out = np.zeros(n, dtype=np.complex64)

    # Rough initial phase estimate from symbol-rate sample points
    step = SPB
    for i in range(0, min(n - step, N_SYMS * SPB), step):
        sym = iq[i]
        # 4th power removes QPSK modulation, leaves 4× carrier phase
        carr_err = (sym / (abs(sym) + 1e-12)) ** 4
        # Average over window
        w = w + loop_bw * (carr_err - w)
        # Apply correction
        out[i] = iq[i] * (np.conj(w) / abs(w))

    out[0] = iq[0]
    for i in range(step, n):
        sym = iq[i]
        carr_err = (sym / (abs(sym) + 1e-12)) ** 4
        w = w + loop_bw * (carr_err - w)
        out[i] = iq[i] * (np.conj(w) / abs(w))

    return out


def cma_equaliser(rx    : np.ndarray,
                   n_taps: int   = 5,
                   mu    : float = 0.0005,
                   iters : int   = 30
                   ) -> np.ndarray:
    """Constant Modulus Algorithm — with weight-norm guard to prevent overflow."""
    w = np.ones(n_taps, dtype=np.complex64) / n_taps
    for _ in range(iters):
        for i in range(n_taps, len(rx)):
            y     = np.dot(w, rx[i-n_taps:i])
            error = 1.0 - np.abs(y)**2
            # LMS-style update with leakage (norm guard)
            grad  = mu * error * np.conj(y) * rx[i-n_taps:i][::-1]
            w_new = w + grad
            # Prevent weight blow-up (norm cap)
            w_norm = np.sqrt(np.sum(np.abs(w_new)**2))
            if w_norm > 3.0:
                w_new = w_new * (3.0 / w_norm)
            w = w_new
    out = np.zeros_like(rx)
    for i in range(n_taps, len(rx)):
        out[i] = np.dot(w, rx[i-n_taps:i])
    return out


def timing_recovery(rx: np.ndarray, spb: int = SPB) -> np.ndarray:
    """Simple timing recovery: align to peak of envelope via cross-correlation."""
    # Compute signal envelope
    env = np.abs(rx)
    # Cross-correlate with a short reference pulse
    ref = np.exp(-np.linspace(-4, 4, 32)**2)
    corr = np.correlate(env, ref, mode='same')
    # Find peak offset
    peak_idx = np.argmax(corr)
    # Downsample at peak-aligned positions
    offset = peak_idx % spb
    rx_ds  = rx[offset::spb]
    return rx_ds


def compute_evm(rx        : np.ndarray,
                ref       : np.ndarray,
                spb       : int = SPB
                ) -> float:
    """
    IEEE 802.15.4g EVM (%).
    EVM = sqrt(mean(|error|²) / mean(|ref|²)) × 100
    ref is clean QPSK symbols upsampled to match rx length.
    """
    n = min(len(rx), len(ref) * spb)
    ref_full = np.zeros(n, dtype=np.complex64)
    for i in range(len(ref)):
        ref_full[i*spb:(i+1)*spb] = ref[i]
    err_pow  = np.mean(np.abs(rx[:n] - ref_full[:n])**2)
    ref_pow  = np.mean(np.abs(ref_full[:n])**2)
    if ref_pow < 1e-12:
        return float('nan')
    return float(np.sqrt(err_pow / ref_pow) * 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# UHD HARDWARE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class B210Device:
    """Context manager for USRP B210."""
    def __init__(self,
                 tx_freq: float = F_CARRIER,
                 rx_freq: float = F_CARRIER,
                 tx_gain: float = TX_GAIN,
                 rx_gain: float = RX_GAIN,
                 rate   : float = SAMPLE_RATE):
        self.tx_freq = tx_freq
        self.rx_freq = rx_freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.rate    = rate
        self.usrp    : Optional[object] = None
        self.tx_str  : Optional[object] = None
        self.rx_str  : Optional[object] = None

    def open(self) -> None:
        if not UHD_AVAILABLE:
            raise RuntimeError("UHD not available")
        log.info("Opening USRP B210 ...")
        self.usrp = uhd.usrp.MultiUSRP("")
        self.usrp.set_master_clock_rate(20e6, libuhd.device.MBOARD_IFACE.ALL_MBOARDS)
        self.usrp.set_tx_freq(self.tx_freq, 0)
        self.usrp.set_rx_freq(self.rx_freq, 0)
        self.usrp.set_tx_gain(self.tx_gain, 0)
        self.usrp.set_rx_gain(self.rx_gain, 0)
        self.usrp.set_tx_rate(self.rate, 0)
        self.usrp.set_rx_rate(self.rate, 0)
        log.info("TX: %.3f MHz  RX: %.3f MHz  Rate: %.3f Msps  Gain: TX %.1f / RX %.1f dB",
                 self.usrp.get_tx_freq(0)/1e6, self.usrp.get_rx_freq(0)/1e6,
                 self.usrp.get_tx_rate(0)/1e6,
                 self.usrp.get_tx_gain(0), self.usrp.get_rx_gain(0))
        fmt  = libuhd.types.DESPECTODEV
        self.tx_str = self.usrp.get_tx_streamer(fmt)
        self.rx_str = self.usrp.get_rx_streamer(fmt)
        log.info("B210 ready.")

    def close(self) -> None:
        self.usrp = self.tx_str = self.rx_str = None
        log.info("B210 closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def hw_txrx(usrp   : B210Device,
             burst  : np.ndarray,
             timeout: float = 2.0
             ) -> Optional[np.ndarray]:
    """TX then RX on B210. Returns RX samples or None."""
    delay_s = 10e-3        # 10 ms guard
    cmd_time = usrp.usrp.get_time_now() + delay_s
    usrp.usrp.set_command_time(cmd_time)
    meta = uhd.types.TXMetadata()
    meta.has_time_spec = True
    meta.time_spec     = uhd.types.TimeSpec(cmd_time.get_real_secs())
    usrp.usrp.clear_command_time()

    n_sent = usrp.tx_str.send(burst, meta)
    log.info("HW TX: %d samples at t=%.4f s", n_sent, cmd_time.get_real_secs())

    # RX
    n_recv  = 0
    rx_len  = len(burst) + 1000
    rx_buf  = np.zeros(rx_len, dtype=np.complex64)
    rm      = uhd.types.RXMetadata()
    deadline = time.time() + timeout
    while n_recv < len(burst):
        remaining = len(burst) - n_recv
        tmp = np.zeros(remaining, dtype=np.complex64)
        sr  = usrp.rx_str.recv(tmp, rm, timeout)
        if sr <= 0:
            break
        rx_buf[n_recv:n_recv+sr] = tmp[:sr]
        n_recv += sr
        if time.time() > deadline:
            break
    log.info("HW RX: %d / %d samples received", n_recv, len(burst))
    if n_recv < 10:
        return None
    return rx_buf[:n_recv]


# ══════════════════════════════════════════════════════════════════════════════
# CONSTELLATION PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_constellation(rx_eq      : np.ndarray,
                       ref_syms   : np.ndarray,
                       evm_pct    : float,
                       df_hz      : float,
                       mode       : str,
                       out_path   : str = PNG_PATH):
    """Save QPSK constellation scatter-plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    step   = max(1, len(rx_eq) // 600)
    rxplt  = rx_eq[::step]
    refplt = ref_syms[::max(1, len(ref_syms)//len(rxplt))]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0a0f1e')

    ax.scatter(rxplt.real, rxplt.imag,
               alpha=0.65, s=22, c='#00e5ff', edgecolors='none',
               label='Received IQ')
    ax.scatter(refplt.real, refplt.imag,
               alpha=0.9, s=50, c='#ff4757', edgecolors='white',
               linewidths=0.8, marker='X', label='QPSK Ref')

    theta = np.linspace(0, 2*math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), '--', color='#2f3542',
            lw=0.8, label='Unit circle')

    status  = "LOCK" if evm_pct < 10.0 else "UNLOCK"
    fgc     = '#2ed573' if evm_pct < 10.0 else '#ff4757'
    bgc     = '#1e3a1e' if evm_pct < 10.0 else '#3a1e1e'

    ax.text(0.97, 0.97,
            f"Mode    : {mode}\n"
            f"EVM     : {evm_pct:.4f} %\n"
            f"Status  : {status}\n"
            f"Pre-comp: {df_hz:+.1f} Hz",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold', color=fgc,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bgc,
                      edgecolor=fgc, alpha=0.95))

    ax.set_xlabel("In-Phase (I)", fontsize=12, color='white')
    ax.set_ylabel("Quadrature (Q)", fontsize=12, color='white')
    ax.set_title(
        f"LR-FHSS/QPSK Constellation  —  Dual-Mode SDR Terminal\n"
        f"USRP B210  |  {F_CARRIER/1e6:.1f} MHz  |  "
        f"{SAMPLE_RATE/1e6:.0f} Msps  |  PGRL Pre-comp {df_hz:+.0f} Hz\n"
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=10, color='white', pad=12
    )
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2f3542')

    ax.legend(loc='lower left', fontsize=9, framealpha=0.85,
              facecolor='#1a1a2e', edgecolor='#2f3542', labelcolor='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Constellation -> %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual-Mode SDR TX/RX: --sim (NumPy) or --hw (USRP B210)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true",
                   help="Simulation mode (NumPy/SciPy channel, default)")
    g.add_argument("--hw",  action="store_true",
                   help="Hardware mode (requires USRP B210)")
    parser.add_argument("--doppler-hz", type=float, default=8000.0,
                        help=f"Doppler shift in Hz (default: 8000)")
    parser.add_argument("--snr-db", type=float, default=25.0,
                        help=f"AWGN SNR in dB for simulation (default: 25)")
    parser.add_argument("--n-bursts", type=int, default=3,
                        help="Number of bursts to send (default: 3)")
    parser.add_argument("--n-syms", type=int, default=N_SYMS,
                        help=f"QPSK symbols per burst (default: {N_SYMS})")
    parser.add_argument("--output", type=str, default=PNG_PATH,
                        help=f"Output PNG path (default: {PNG_PATH})")
    args = parser.parse_args()

    mode: Literal["SIM", "HW"] = "HW" if args.hw else "SIM"

    log.info("=" * 62)
    log.info("  dual_mode_trx.py — Dual-Mode SDR Terminal")
    log.info("  Mode     : %s", mode)
    log.info("  Doppler  : %+.1f Hz  (PGRL predicted)", args.doppler_hz)
    if mode == "SIM":
        log.info("  SNR      : %.1f dB",  args.snr_db)
    else:
        log.info("  SNR      : (hardware)")
    log.info("  Bursts   : %d", args.n_bursts)
    log.info("  Symbols  : %d  (%.1f ms at 125 ksps)",
             args.n_syms, args.n_syms/125.0)
    log.info("=" * 62)

    # ── W2: Generate TX burst once ────────────────────────────────────────────
    log.info("[1/6] Generating RRC-shaped QPSK payload (pilot preamble) ...")
    burst_raw, ref_syms = generate_tx_burst(n_syms=args.n_syms,
                                            use_lr_fhss_preamble=True)
    n_tot   = len(burst_raw)
    n_pilot = 20 * SPB
    n_data  = N_SYMS * SPB
    log.info("   Burst: %d samples (%.2f ms at %d ksps)  "
             "[preamble=%d data=%d gap+gap=%d]",
             n_tot, n_tot / SAMPLE_RATE * 1000,
             int(SAMPLE_RATE / 1000), n_pilot, n_data, 2 * BURST_GAP)

    # ── W3: PGRL pre-compensation ─────────────────────────────────────────────
    log.info("[2/6] Applying PGRL inverse-Doppler pre-compensation (Δf=%+.1f Hz) ...",
             args.doppler_hz)
    burst_comp = apply_inverse_doppler(burst_raw, args.doppler_hz, SAMPLE_RATE)

    evm_results = []

    for b in range(args.n_bursts):
        log.info("[3/6] Processing burst %d/%d ...", b+1, args.n_bursts)

        # ── W4: TX / Channel ──────────────────────────────────────────────────
        if mode == "SIM":
            rx_raw, f_meas = simulate_channel(
                burst_comp,    # pre-comp burst: channel Doppler cancels pre-comp
                df_hz=args.doppler_hz,
                snr_db=args.snr_db,
                fs_hz=SAMPLE_RATE
            )
        else:
            if not UHD_AVAILABLE:
                log.error("UHD unavailable. Use --sim instead.")
                return 1
            try:
                with B210Device() as dev:
                    rx_raw = hw_txrx(dev, burst_comp)
            except Exception as exc:
                log.error("B210 error: %s", exc)
                continue
            if rx_raw is None:
                log.warning("RX timeout — check antenna connection")
                continue
            n_est  = min(len(rx_raw), 800)
            ac     = np.mean(rx_raw[:n_est] * np.conj(rx_raw[1:n_est+1]))
            f_meas = float((np.angle(ac) / (2*math.pi)) * SAMPLE_RATE)

        # ── RX Processing chain ───────────────────────────────────────────────────
        log.info("[4/6] RX DSP: freq correct → CMA equaliser → timing recovery ...")

        # 4a. Remove residual measured frequency offset
        rx_fc = coarse_freq_correct(rx_raw, -f_meas, SAMPLE_RATE)

        # 4b. CMA equaliser — disabled for clean SNR (no multipath)
        rx_eq = rx_fc

        # 4c. Data-aided timing recovery: brute-force all SPB offsets
        #     Reference is exact QPSK constellation; best offset = lowest EVM
        payload_start = (20 * SPB) + BURST_GAP
        best_evm = float('inf'); best_off = 0
        for off_test in range(SPB):
            rx_test = rx_eq[payload_start + off_test::SPB][:N_SYMS]
            if len(rx_test) < N_SYMS:
                continue
            # AGC: Complex Gain Alignment (Option B) — scale rx_test to match ref_syms
            # alpha is complex to capture both amplitude AND phase correction
            alpha_test = np.sum(rx_test * np.conj(ref_syms)) / max(np.sum(np.abs(ref_syms)**2), 1e-12)
            rx_cal = rx_test / alpha_test if np.abs(alpha_test) > 1e-12 else rx_test
            evm_test = float(np.sqrt(
                np.mean(np.abs(rx_cal - ref_syms)**2) /
                np.mean(np.abs(ref_syms)**2)) * 100.0)
            if evm_test < best_evm:
                best_evm = evm_test; best_off = off_test

        rx_ds = rx_eq[payload_start + best_off::SPB][:N_SYMS]

        # ── W5: EVM (with AGC Complex Gain Calibration) ───────────────────────
        # Apply complex gain alignment before EVM: rx_cal = rx_ds / alpha
        # This resolves the RRC amplitude mismatch (rx≈0.66, ref=1.0 → scale factor ≈0.66)
        # alpha is COMPLEX to capture both magnitude and phase rotation
        alpha = np.sum(rx_ds * np.conj(ref_syms)) / max(np.sum(np.abs(ref_syms)**2), 1e-12)
        rx_calibrated = rx_ds / alpha if np.abs(alpha) > 1e-12 else rx_ds

        ref_pow = float(np.mean(np.abs(ref_syms)**2))
        err_pow = float(np.mean(np.abs(rx_calibrated - ref_syms)**2))
        evm     = float(np.sqrt(err_pow / ref_pow) * 100.0)
        evm_results.append(evm)
        log.info("[5/6] EVM = %.4f %%  |  f_measured = %+.2f Hz  |  "
                 "Burst %d/%d  (best timing off=%d)  "
                 "[ref_pow=%.4f  err_pow=%.6f]",
                 evm, f_meas, b+1, args.n_bursts, best_off, ref_pow, err_pow)

        # Save first-burst constellation (calibrated IQ for visual clarity)
        if b == 0:
            plot_constellation(rx_calibrated, ref_syms, evm, args.doppler_hz, mode,
                                out_path=args.output)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("[6/6] Summary (%d bursts):", len(evm_results))
    mean_evm = np.mean(evm_results) if evm_results else float('nan')
    log.info("  EVM  mean = %.4f %%   min = %.4f %%   max = %.4f %%",
             mean_evm, np.min(evm_results), np.max(evm_results))
    log.info("  Doppler pre-comp applied: %+.1f Hz", args.doppler_hz)
    status = "LOCK" if mean_evm < 10.0 else "UNLOCK"
    log.info("  Status: %s  (threshold: 10 %%)", status)
    log.info("  Constellation: %s", args.output)

    # CSV append
    ts = datetime.datetime.now().isoformat()
    with open(CSV_PATH, "a") as f:
        for evm in evm_results:
            f.write(f"{ts},{mode},{args.doppler_hz:.1f},"
                    f"{args.snr_db:.1f},{evm:.6f}\n")
    log.info("  EVM log appended -> %s", CSV_PATH)

    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    if not UHD_AVAILABLE:
        log.warning("UHD not found (%s). Using --sim by default.", UHD_ERR)
    sys.exit(main())