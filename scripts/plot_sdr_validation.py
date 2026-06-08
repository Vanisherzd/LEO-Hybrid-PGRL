#!/usr/bin/env python3
"""Plot conducted LR1121 -> USRP B210 IQ capture: TX ON/OFF max-hold spectrum.

Reads RAW IQ (.fc32, interleaved complex float32) from a conducted capture sweep
and computes a max-hold power spectrum for the TX-ON and TX-OFF captures.

Evidence (conducted capture; IQ-level RF signal detection ONLY, NOT decode/PER):
  hardware/captures/<sweep>/cap_868000000_txrx_on.fc32 / _off.fc32  (raw IQ)
  hardware/captures/<sweep>/cap_868000000_txrx_comparison.json      (ON/OFF delta)
  hardware/captures/<sweep>/cap_868000000_txrx_on_uart.log          (LR1121 TX log)
Default sweep has on_off_delta_db = 9.82, validation_status = signal_detected.

NOTE: the raw .fc32 captures are large and live locally (not committed). This
script regenerates the figure from them on the capture machine.
Output: paper/figures/fig5_sdr_capture_validation.pdf
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
SWEEP = os.environ.get(
    "SDR_SWEEP",
    os.path.join(ROOT, "hardware", "captures", "auto_sweep_20260604_011519"))
FS_HZ = 1_000_000.0
NFFT = 4096
OUT = os.path.join(ROOT, "paper", "figures", "fig5_sdr_capture_validation.pdf")


def maxhold_psd(path, fs=FS_HZ, nfft=NFFT, max_frames=2000):
    iq = np.fromfile(path, dtype=np.complex64)
    n = (len(iq) // nfft) * nfft
    iq = iq[:n].reshape(-1, nfft)
    if iq.shape[0] > max_frames:                       # subsample frames evenly
        idx = np.linspace(0, iq.shape[0] - 1, max_frames).astype(int)
        iq = iq[idx]
    win = np.hanning(nfft)
    spec = np.fft.fftshift(np.fft.fft(iq * win, axis=1), axes=1)
    psd_db = 10.0 * np.log10(np.maximum((np.abs(spec) ** 2).max(axis=0), 1e-12))
    freqs_khz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs)) / 1e3
    return freqs_khz, psd_db


def main():
    on = os.path.join(SWEEP, "cap_868000000_txrx_on.fc32")
    off = os.path.join(SWEEP, "cap_868000000_txrx_off.fc32")
    cmp_path = os.path.join(SWEEP, "cap_868000000_txrx_comparison.json")
    delta = None
    if os.path.exists(cmp_path):
        with open(cmp_path) as fh:
            delta = json.load(fh).get("on_off_delta_db")

    f_on, p_on = maxhold_psd(on)
    f_off, p_off = maxhold_psd(off)
    # DC/LO guard band (exclude +/-5 kHz around 0 from the visual peak)
    guard = np.abs(f_on) < 5.0

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(f_off, p_off, color="#7f7f7f", lw=0.8, label="TX off")
    ax.plot(f_on, p_on, color="#1f77b4", lw=0.8, label="TX on")
    ax.axvspan(-5, 5, color="orange", alpha=0.15)
    ax.text(0, ax.get_ylim()[1], "DC/LO\nguard", fontsize=6, ha="center", va="top",
            color="darkorange")
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Max-hold power (dB)")
    lbl = "ON/OFF max-hold"
    if delta is not None:
        lbl += f"  ($\\Delta={delta:.2f}$ dB)"
    ax.set_title(lbl, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
