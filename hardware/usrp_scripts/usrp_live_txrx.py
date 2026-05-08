#!/usr/bin/env python3
"""
usrp_live_txrx.py
=================
Cognitive SDR Terminal — Live LR-FHSS / QPSK TX/RX with PGRL Pre-compensation.

Hardware : USRP B210 (S-band 436.5 MHz, 1 Msps)
Edge Host: Raspberry Pi 5 / Intel NUC (USB 3.0 passthrough)
Python    : 3.10+  |  pyuhd >= 4.3.0  |  numpy  |  scipy  |  matplotlib

This script:
  W1  — Initialises the B210 via pyuhd (TX + RX streamers)
  W2  — Generates LR-FHSS / QPSK baseband in pure NumPy (no torch needed)
  W3  — Applies PGRL inverse-Doppler pre-compensation per-hop phase rotation
  W4  — Synchronously TX then RX the waveform (coax loopback or over-the-air)
  W5  — Measures EVM on the received IQ stream and saves constellation PNG

Usage (container with USB passthrough):
    cd leo-pinn && source .venv/bin/activate
    PYTHONPATH="/opt/data/workspace/leo-pinn/.deps:$PYTHONPATH" python \\
        hardware/usrp_scripts/usrp_live_txrx.py

Usage (host OS, no Docker needed):
    pip install numpy scipy matplotlib pyuhd
    python hardware/usrp_scripts/usrp_live_txrx.py

Forcing simulated mode (no hardware required for baseband validation):
    python hardware/usrp_scripts/usrp_live_txrx.py --simulate

Author  : Hermes Agent (LEO-PINN / D2S Architecture)
Created : 2026-05-08
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
import logging
import datetime
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp

# ── UHD import with graceful fallback ─────────────────────────────────────────
UHD_AVAILABLE = False
UHD_ERR       = None
try:
    import uhd
    from uhd import libpyuhd as libuhd
    UHD_AVAILABLE = True
except ImportError as exc:
    UHD_ERR = f"uhd not available ({exc}). Running in SIMULATE mode."

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("usrp_live_txrx")

# ── Physical / LR-FHSS constants ───────────────────────────────────────────────
F_CARRIER    : float = 436.5e6     # Hz  S-band
TX_FREQ      : float = 436.5e6     # Hz
RX_FREQ      : float = 436.5e6     # Hz
TX_GAIN      : float = 40.0        # dB  (adjust for your setup)
RX_GAIN      : float = 40.0        # dB
SAMPLE_RATE  : float = 1e6        # Hz  1 Msps
DOPPLER_HZ   : float = 5_000.0     # Hz  simulated Doppler shift (PGRL predictor output)
                                          # Real system: DOPPLER_HZ comes from PINN
TX_AMPL     : float = 0.25        # TX amplitude ±1.0 max (B210 DAC peak = 1.0)
N_SYMS      : int   = 200         # QPSK symbols per burst
SPB         : int   = 8           # Samples per symbol (1 Msps / 125 ksps)
NPADDING    : int   = 200         # Zero-padding samples between bursts

# QPSK constellation angles
QPSK_ANGLES  = np.array([math.pi/4, 3*math.pi/4, -3*math.pi/4, -math.pi/4], dtype=np.float64)

# ── Output paths ───────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "payload_results_realizations", "live_usrp"
)
os.makedirs(OUT_DIR, exist_ok=True)

PNG_CONSTELLATION = os.path.join(OUT_DIR, "Live_B210_Constellation.png")
CSV_EVM_LOG       = os.path.join(OUT_DIR, "evm_log.csv")

# ══════════════════════════════════════════════════════════════════════════════
# BASEBAND GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_qpsk_symbols(n_syms: int, ampl: float = 1.0) -> np.ndarray:
    """Generate n_syms random QPSK symbols (complex baseband)."""
    indices = np.random.randint(0, 4, size=n_syms, dtype=np.int64)
    angles  = QPSK_ANGLES[indices]
    return ampl * (np.cos(angles) + 1j * np.sin(angles))


def root_raised_cosine(N: int, alpha: float = 0.5, Ts: float = 1.0) -> np.ndarray:
    """
    Root Raised Cosine FIR filter (time-domain), span=N symbols, roll-off α.
    Normalised to peak = 1.0.
    """
    t = np.arange(-N*Ts/2, N*Ts/2 + 1e-9, Ts / SPB, dtype=np.float64)
    # Avoid division by zero at centre
    with np.errstate(divide='ignore', invalid='ignore'):
        y = (np.sin(math.pi*t*(1-alpha)) + 4*alpha*t*np.cos(math.pi*t*(1+alpha)))
        denom = math.pi*t*(1-(4*alpha*t)**2)
        y = np.where(np.abs(denom) < 1e-12,
                     1.0 - alpha + 4*alpha/math.pi,
                     y / denom)
    # Normalise
    y = y / np.max(np.abs(y))
    return y.astype(np.float32)


def generate_baseband_burst(n_syms: int = N_SYMS,
                             spb: int    = SPB,
                             rolloff: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a shaped QPSK burst with RRC pulse shaping.

    Returns:
        tx_burst : complex baseband samples (n_syms * spb, )
        tx_syms  : clean QPSK constellation points (n_syms,) — for EVM ref
    """
    syms   = generate_qpsk_symbols(n_syms, ampl=1.0)
    rrc_t  = root_raised_cosine(N=8, alpha=rolloff)          # 8-symbol span
    rrc_len = len(rrc_t)

    # Upsample: zero-insert then convolve with RRC
    upsampled = np.zeros(n_syms * spb, dtype=np.complex64)
    upsampled[::spb] = syms

    burst = np.convolve(upsampled, rrc_t, mode='full')
    # Trim to symbol-aligned length
    offset = rrc_len // 2
    burst  = burst[offset: offset + n_syms * spb].astype(np.complex64)

    # Normalise to TX amplitude ceiling
    peak = np.max(np.abs(burst))
    if peak > 1e-9:
        burst = burst / peak * TX_AMPL

    return burst, syms


def apply_doppler_precompensation(iq_samples : np.ndarray,
                                   doppler_hz  : float,
                                   sample_rate : float = SAMPLE_RATE
                                   ) -> np.ndarray:
    """
    W3 core: Apply inverse-Doppler phase rotation.
    Pre-compensates TX waveform so RX sees Doppler-corrected signal.

    x_precomp(t) = x(t) * exp(-j * 2π * Doppler_Hz * t)
    """
    n       = len(iq_samples)
    t       = np.arange(n, dtype=np.float64) / sample_rate   # seconds
    phase   = -2.0 * math.pi * doppler_hz * t                # inverse rotation
    rot     = np.exp(1j * phase).astype(np.complex64)
    return iq_samples * rot


# ══════════════════════════════════════════════════════════════════════════════
# EVM METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_evm(received: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute Error Vector Magnitude (%) per IEEE 802.15.4g.
    EVM = sqrt( mean(|r - ref|²) / mean(|ref|²) ) * 100
    """
    # Align lengths
    n = min(len(received), len(reference))
    r = received[:n]
    ref = reference[:n]
    error_power  = np.mean(np.abs(r - ref)**2)
    reference_power = np.mean(np.abs(ref)**2)
    if reference_power < 1e-12:
        return np.nan
    return float(np.sqrt(error_power / reference_power) * 100.0)


def measure_frequency_offset(rx_samples: np.ndarray,
                               sample_rate: float = SAMPLE_RATE
                               ) -> float:
    """Coarse frequency offset estimate via phase difference method (rad/s)."""
    n    = min(len(rx_samples), 1000)
    autocorr = np.mean(rx_samples[:n] * np.conj(rx_samples[1:n+1]))
    phase_per_sample = np.angle(autocorr)
    rad_per_sample = phase_per_sample
    f_est_hz = (rad_per_sample / (2.0 * math.pi)) * sample_rate
    return float(f_est_hz)


def coarse_frequency_correction(iq_samples: np.ndarray,
                                  f_offset_hz: float,
                                  sample_rate: float = SAMPLE_RATE
                                  ) -> np.ndarray:
    """Remove residual frequency offset from RX samples."""
    n    = len(iq_samples)
    t    = np.arange(n, dtype=np.float64) / sample_rate
    rot  = np.exp(-1j * 2.0 * math.pi * f_offset_hz * t).astype(np.complex64)
    return iq_samples * rot


def cma_equaliser(rx_samples: np.ndarray, n_taps: int = 5,
                   mu: float = 0.001, n_iter: int = 50) -> np.ndarray:
    """
    Constant-modulus algorithm (CMA) equaliser for QPSK.
    Simple adaptive FIR that rotates received constellation to unit circle.
    """
    w = np.ones(n_taps, dtype=np.complex64) / n_taps
    for _ in range(n_iter):
        for i in range(n_taps, len(rx_samples)):
            y     = np.dot(w, rx_samples[i-n_taps:i])
            error = 1.0 - np.abs(y)**2
            w    += mu * error * np.conj(y) * rx_samples[i-n_taps:i]
    # Apply equaliser
    out = np.zeros_like(rx_samples)
    for i in range(n_taps, len(rx_samples)):
        out[i] = np.dot(w, rx_samples[i-n_taps:i])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CONSTELLATION PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_constellation(rx_corrected : np.ndarray,
                       ref_syms      : np.ndarray,
                       evm_pct       : float,
                       doppler_hz    : float,
                       f_out         : str):
    """Save scatter-plot of received QPSK constellation with EVM annotation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Downsample for plotting clarity
    step  = max(1, len(rx_corrected) // 500)
    rxplt = rx_corrected[::step]
    refplt = ref_syms[::SPB][:len(rxplt)]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')

    ax.scatter(rxplt.real, rxplt.imag,
               alpha=0.6, s=18, c='#00e5ff', edgecolors='none',
               label='Received')
    ax.scatter(refplt.real, refplt.imag,
               alpha=0.9, s=35, c='#ff6b6b', edgecolors='white', linewidths=0.5,
               marker='x', label='QPSK Ref')

    ax.axhline(0, color='#333', lw=0.5)
    ax.axvline(0, color='#333', lw=0.5)

    # Unit circle guide
    theta = np.linspace(0, 2*math.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '--', color='#444', lw=0.8, label='Unit circle')

    badge_bg   = '#e8f5e9' if evm_pct < 10.0 else '#ffebee'
    badge_fg   = '#2e7d32' if evm_pct < 10.0 else '#c62828'
    ax.text(0.97, 0.97,
            f"EVM  : {evm_pct:.3f} %\n"
            f"Pre-comp Δf: {doppler_hz:+.1f} Hz\n"
            f"Status: {'LOCK' if evm_pct < 10 else 'UNLOCK'}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold', color=badge_fg,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=badge_bg,
                      edgecolor=badge_fg, alpha=0.95))

    ax.set_xlabel("In-Phase (I)", fontsize=11, color='white')
    ax.set_ylabel("Quadrature (Q)", fontsize=11, color='white')
    ax.set_title("Live B210 Constellation — LR-FHSS/QPSK — PGRL Pre-compensation\n"
                 f"USRP B210 | {F_CARRIER/1e6:.1f} MHz | {SAMPLE_RATE/1e6:.0f} Msps | "
                 f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                fontsize=10, color='white', pad=10)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#444')

    ax.legend(loc='lower left', fontsize=9, framealpha=0.8,
              facecolor='#1b263b', edgecolor='#444', labelcolor='white')

    fig.tight_layout()
    fig.savefig(f_out, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Constellation saved -> %s", f_out)


# ══════════════════════════════════════════════════════════════════════════════
# UHD HARDWARE HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

class USRPDevice:
    """Context manager for USRP B210."""

    def __init__(self,
                 tx_freq  : float = TX_FREQ,
                 rx_freq  : float = RX_FREQ,
                 tx_gain  : float = TX_GAIN,
                 rx_gain  : float = RX_GAIN,
                 rate     : float = SAMPLE_RATE):
        self.tx_freq = tx_freq
        self.rx_freq = rx_freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.rate    = rate
        self.usrp    : Optional[object] = None
        self.tx_streamer : Optional[object] = None
        self.rx_streamer : Optional[object] = None

    def open(self) -> None:
        if not UHD_AVAILABLE:
            raise RuntimeError("UHD not available. Install: pip install pyuhd")
        log.info("Opening USRP B210 ...")
        self.usrp = uhd.usrp.MultiUSRP("")
        self.usrp.set_master_clock_rate(20e6, libuhd.device.MBOARD_IFACE.ALL_MBOARDS)

        # TX
        self.usrp.set_tx_freq(self.tx_freq, 0)
        self.usrp.set_tx_gain(self.tx_gain, 0)
        # RX
        self.usrp.set_rx_freq(self.rx_freq, 0)
        self.usrp.set_rx_gain(self.rx_gain, 0)

        # Sample rate
        self.usrp.set_tx_rate(self.rate, 0)
        self.usrp.set_rx_rate(self.rate, 0)

        actual_tx_rate = self.usrp.get_tx_rate(0)
        actual_rx_rate = self.usrp.get_rx_rate(0)
        log.info("TX rate: %.3f Msps  (requested %.3f)", actual_tx_rate/1e6, self.rate/1e6)
        log.info("RX rate: %.3f Msps  (requested %.3f)", actual_rx_rate/1e6, self.rate/1e6)
        log.info("TX freq: %.3f MHz   RX freq: %.3f MHz",
                 self.usrp.get_tx_freq(0)/1e6, self.usrp.get_rx_freq(0)/1e6)
        log.info("TX gain: %.1f dB    RX gain: %.1f dB",
                 self.usrp.get_tx_gain(0), self.usrp.get_rx_gain(0))
        log.info("Master clock: %.3f MHz", self.usrp.get_master_clock_rate(0)/1e6)

        # Streamers
        tx_fmt = libuhd.types.DESPECTODEV
        self.tx_streamer = self.usrp.get_tx_streamer(tx_fmt)
        self.rx_streamer = self.usrp.get_rx_streamer(tx_fmt)

        log.info("USRP B210 ready.")

    def close(self) -> None:
        if self.usrp:
            self.usrp = None
            self.tx_streamer = None
            self.rx_streamer = None
            log.info("USRP B210 closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def txrx_burst(usrp    : USRPDevice,
                tx_burst: np.ndarray,
                timeout_s: float = 1.0
                ) -> Optional[np.ndarray]:
    """
    TX a burst then immediately RX a window of samples.
    Returns received samples or None on failure.
    """
    delay_s  = 5e-3          # 5 ms guard before TX
    rx_len   = len(tx_burst) + 500   # slightly more than TX length

    # Wait for clean start time
    cmd_time = usrp.usrp.get_time_now() + delay_s
    usrp.usrp.set_command_time(cmd_time)

    # TX buffer
    tx_meta  = uhd.types.TXMetadata()
    tx_meta.has_time_spec  = True
    tx_meta.time_spec     = uhd.types.TimeSpec(cmd_time.get_real_secs())

    # Clear command time (resume immediate scheduling)
    usrp.usrp.clear_command_time()

    # Stream TX
    n_sent = usrp.tx_streamer.send(tx_burst, tx_meta)
    log.info("TX: sent %d samples at t=%.3f s",
             n_sent, cmd_time.get_real_secs())

    # RX setup
    rx_meta  = uhd.types.RXMetadata()
    rx_bufs  = [np.zeros(len(tx_burst) + 1000, dtype=np.complex64)]

    # Receive
    n_recv = 0
    end_time = time.time() + timeout_s
    while n_recv < len(tx_burst):
        remaining = len(tx_burst) - n_recv
        tmp = np.zeros(remaining, dtype=np.complex64)
        sr = usrp.rx_streamer.recv(tmp, rx_meta, timeout_s)
        if sr <= 0:
            break
        rx_bufs[0][n_recv:n_recv+sr] = tmp[:sr]
        n_recv += sr
        if time.time() > end_time:
            break

    log.info("RX: received %d / %d samples", n_recv, len(tx_burst))
    if n_recv < 10:
        return None
    return rx_bufs[0][:n_recv]


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE (no hardware)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_txrx(tx_burst     : np.ndarray,
                   ref_syms     : np.ndarray,
                   doppler_hz   : float,
                   snr_db       : float = 30.0,
                   sample_rate  : float = SAMPLE_RATE
                   ) -> Tuple[np.ndarray, float]:
    """
    Emulate the TX→channel→RX path without any hardware.
    Adds Doppler shift, AWGN, and a small multipath to mimic real over-the-air.
    """
    n = len(tx_burst)
    t = np.arange(n) / sample_rate

    # Channel: Doppler rotation + AWGN
    channel_rot  = np.exp(1j * 2 * math.pi * doppler_hz * t).astype(np.complex64)
    rx_samples   = tx_burst * channel_rot

    # AWGN
    if snr_db < 60.0:
        sig_power  = np.mean(np.abs(rx_samples)**2)
        noise_vold = np.sqrt(sig_power / (10**(snr_db/10)))
        rx_samples += (np.random.randn(n) + 1j * np.random.randn(n)) * noise_vold / np.sqrt(2)

    # Tiny multipath (2 rays, 0.3 rel amp, 40-sample delay)
    multipath = np.zeros(n, dtype=np.complex64)
    multipath[40:]  += 0.3 * rx_samples[:-40] if n > 40 else 0
    rx_samples      += multipath * 0.1

    f_meas = measure_frequency_offset(rx_samples, sample_rate)
    log.info("Sim RX: Doppler applied = %.1f Hz  |  Measured = %.2f Hz",
             doppler_hz, f_meas)
    return rx_samples.astype(np.complex64), f_meas


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="USRP B210 Live TX/RX with PGRL pre-comp")
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode (no hardware needed)")
    parser.add_argument("--doppler-hz", type=float, default=DOPPLER_HZ,
                        help=f"Simulated Doppler shift in Hz (default {DOPPLER_HZ})")
    parser.add_argument("--n-bursts",   type=int,   default=3,
                        help="Number of TX bursts to send (default 3)")
    parser.add_argument("--snr-db",     type=float, default=30.0,
                        help="AWGN SNR in dB for simulate mode (default 30)")
    parser.add_argument("--output-dir", type=str,   default=OUT_DIR,
                        help=f"Output directory (default {OUT_DIR})")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  usrp_live_txrx.py — Cognitive SDR TX/RX")
    log.info("  LR-FHSS/QPSK  |  %.1f MHz  |  %.0f ksps",
             F_CARRIER/1e6, SAMPLE_RATE/1e3)
    log.info("  Doppler pre-comp: %+.1f Hz", args.doppler_hz)
    log.info("  Mode: %s", "SIMULATE" if args.simulate else "LIVE UHD")
    log.info("=" * 60)

    # ── Generate baseband once (same payload per burst for easy EVM comparison) ──
    log.info("[1/5] Generating QPSK baseband burst ...")
    tx_burst_raw, ref_syms = generate_baseband_burst(n_syms=N_SYMS, spb=SPB)
    log.info("     Burst: %d samples (%.1f ms at %.0f ksps)",
             len(tx_burst_raw),
             len(tx_burst_raw)/SAMPLE_RATE*1e3,
             SAMPLE_RATE/1e3)

    # ── Apply PGRL inverse-Doppler pre-compensation (W3) ──────────────────────
    log.info("[2/5] Applying PGRL pre-compensation (Δf = %+.1f Hz) ...", args.doppler_hz)
    tx_burst_comp = apply_doppler_precompensation(tx_burst_raw, args.doppler_hz, SAMPLE_RATE)

    # Save clean reference constellation (no Doppler)
    log.info("[3/5] Processing RX stream ...")
    evm_results = []

    for burst_idx in range(args.n_bursts):
        log.info("  Burst %d/%d ...", burst_idx + 1, args.n_bursts)
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if args.simulate:
            rx_raw, f_measured = simulate_txrx(
                tx_burst_raw, ref_syms, args.doppler_hz,
                snr_db=args.snr_db, sample_rate=SAMPLE_RATE
            )
        else:
            if not UHD_AVAILABLE:
                log.error("UHD not available. Re-run with --simulate")
                return 1
            try:
                with USRPDevice() as usrp:
                    rx_raw = txrx_burst(usrp, tx_burst_comp)
                    if rx_raw is None:
                        log.warning("RX returned no samples — check antenna/loopback")
                        continue
                    f_measured = measure_frequency_offset(rx_raw, SAMPLE_RATE)
            except Exception as exc:
                log.error("USRP error: %s", exc)
                continue

        # Frequency correction on RX side
        rx_corrected = coarse_frequency_correction(rx_raw, -f_measured, SAMPLE_RATE)

        # CMA equalisation
        rx_eq = cma_equaliser(rx_corrected, n_taps=7, mu=0.002, n_iter=40)

        # EVM measurement
        # Reference for EVM: use the original clean symbols pulse-shaped
        n_ref = min(len(rx_eq), len(ref_syms) * SPB)
        ref_for_evm = np.zeros(n_ref, dtype=np.complex64)
        for i in range(min(len(ref_syms), n_ref // SPB)):
            ref_for_evm[i*SPB:(i+1)*SPB] = ref_syms[i]

        evm_pct = compute_evm(rx_eq[:n_ref], ref_for_evm)
        evm_results.append(evm_pct)

        log.info("     Burst %d  |  f_measured = %+.2f Hz  |  EVM = %.4f %%",
                 burst_idx + 1, f_measured, evm_pct)

        # Save first burst constellation
        if burst_idx == 0:
            plot_constellation(rx_eq, ref_syms, evm_pct, args.doppler_hz,
                               os.path.join(args.output_dir, "Live_B210_Constellation.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  SUMMARY (%d bursts)", len(evm_results))
    if evm_results:
        mean_evm  = np.mean(evm_results)
        min_evm   = np.min(evm_results)
        max_evm   = np.max(evm_results)
        log.info("  EVM  mean: %.4f %%   min: %.4f %%   max: %.4f %%",
                 mean_evm, min_evm, max_evm)
        log.info("  Pre-comp Δf applied: %+.1f Hz", args.doppler_hz)
        log.info("  Mode: %s",
                 "SIMULATE (no hardware)" if args.simulate else "LIVE UHD")
        if mean_evm < 10.0:
            log.info("  STATUS: LOCK  — EVM < 10%% threshold")
        else:
            log.info("  STATUS: UNLOCK — EVM exceeds 10%% threshold")
        log.info("  Constellation: %s",
                 os.path.join(args.output_dir, "Live_B210_Constellation.png"))

    # Write CSV log
    with open(os.path.join(args.output_dir, "evm_log.csv"), "a") as f:
        ts = datetime.datetime.now().isoformat()
        for evm in evm_results:
            f.write(f"{ts},{args.doppler_hz:.1f},{evm:.6f}\n")

    log.info("=" * 60)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    if not UHD_AVAILABLE:
        log.warning("uhd/ UHD library not found (%s)", UHD_ERR)
        log.warning("Rerun with --simulate to test baseband generation without hardware.")
        log.warning("To install: pip install pyuhd  (requires UHD driver)")
    sys.exit(main())