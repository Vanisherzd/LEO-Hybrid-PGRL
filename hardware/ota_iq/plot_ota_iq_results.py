#!/usr/bin/env python3
"""
plot_ota_iq_results.py
======================
Render figures for the OTA IQ proxy experiments from the per-run CSV/JSON
products of analyze_cfo_residual.py and analyze_adjacent_bin_leakage.py.

Figures (PDF + PNG):
  fig_cfo_residual_vs_time     residual |CFO| per burst over the replay window
  fig_cfo_residual_cdf         CDF of |residual CFO| across modes
  fig_ablr_histogram           histogram of per-burst ABLR_dB across modes
  fig_ablr_cdf                 CDF of per-burst ABLR_dB across modes

Optional Experiment 3 (qualitative only):
  fig_two_node_waterfall_fixed     spectrogram of the fixed-guard schedule
  fig_two_node_waterfall_riskaware spectrogram of the risk-aware-guard schedule

All figures are labelled "short-range OTA IQ proxy". No PER / decoding claims.
Two-node waterfalls are QUALITATIVE OTA demonstrations only (no shared trigger,
so no measured collision probability). See docs/ota_iq_validation_scope.md.

Usage
-----
  uv run python hardware/ota_iq/plot_ota_iq_results.py \
      --run no_comp=hardware/ota_iq/runs/nocomp_001 \
      --run sgp4=hardware/ota_iq/runs/sgp4_001 \
      --run pgrl=hardware/ota_iq/runs/pgrl_001 \
      --out-dir hardware/ota_iq/figures

  # optional qualitative two-node waterfalls:
  uv run python hardware/ota_iq/plot_ota_iq_results.py \
      --waterfall fixed=hardware/ota_iq/runs/twonode_fixed/capture_iq.npy \
      --waterfall riskaware=hardware/ota_iq/runs/twonode_riskaware/capture_iq.npy \
      --sample-rate 1e6 --out-dir hardware/ota_iq/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ota_common import load_iq  # noqa: E402

_COLORS = {"no_comp": "#c0392b", "sgp4": "#e67e22", "pgrl": "#27ae60"}
_FOOTER = "Short-range OTA IQ proxy — IQ-level physical-layer evidence only; not PER / not decoding"


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {stem}.pdf / .png → {out_dir}")


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _color(label: str) -> str:
    for k, c in _COLORS.items():
        if k in label.lower():
            return c
    return "#34495e"


def plot_cfo(runs: dict[str, Path], out_dir: Path):
    series = {}
    for label, d in runs.items():
        ts = d / "cfo_residual_timeseries.csv"
        if not ts.exists():
            print(f"[plot] skip CFO for '{label}': {ts.name} missing")
            continue
        rows = _read_csv(ts)
        t = np.array([float(r["t_start_s"]) for r in rows])
        a = np.array([float(r["abs_residual_cfo_hz"]) for r in rows])
        series[label] = (t, a)
    if not series:
        print("[plot] no CFO data; skipping CFO figures")
        return

    # vs time
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for label, (t, a) in series.items():
        ax.plot(t, a, "o-", ms=3, lw=1, color=_color(label), label=label, alpha=0.85)
    ax.set_xlabel("Replay time (s)")
    ax.set_ylabel("|Residual CFO| (Hz)")
    ax.set_title("Per-burst residual CFO over replay window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, _FOOTER, ha="center", fontsize=6.5, style="italic", color="#666")
    _save(fig, out_dir, "fig_cfo_residual_vs_time")

    # CDF
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, (_, a) in series.items():
        s = np.sort(a)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, lw=1.6, color=_color(label), label=label)
    ax.set_xlabel("|Residual CFO| (Hz)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of per-burst residual CFO")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, _FOOTER, ha="center", fontsize=6.5, style="italic", color="#666")
    _save(fig, out_dir, "fig_cfo_residual_cdf")


def plot_ablr(runs: dict[str, Path], out_dir: Path):
    series = {}
    for label, d in runs.items():
        pc = d / "ablr_per_burst.csv"
        if not pc.exists():
            print(f"[plot] skip ABLR for '{label}': {pc.name} missing")
            continue
        rows = _read_csv(pc)
        series[label] = np.array([float(r["ablr_db"]) for r in rows])
    if not series:
        print("[plot] no ABLR data; skipping ABLR figures")
        return

    # histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lo = min(v.min() for v in series.values())
    hi = max(v.max() for v in series.values())
    bins = np.linspace(lo, hi, 30)
    for label, v in series.items():
        ax.hist(v, bins=bins, alpha=0.5, color=_color(label), label=label)
    ax.set_xlabel("Adjacent-bin leakage ratio (dB)")
    ax.set_ylabel("Burst count")
    ax.set_title("Per-burst adjacent-bin leakage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, _FOOTER, ha="center", fontsize=6.5, style="italic", color="#666")
    _save(fig, out_dir, "fig_ablr_histogram")

    # CDF
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, v in series.items():
        s = np.sort(v)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, lw=1.6, color=_color(label), label=label)
    ax.set_xlabel("Adjacent-bin leakage ratio (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of per-burst adjacent-bin leakage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, _FOOTER, ha="center", fontsize=6.5, style="italic", color="#666")
    _save(fig, out_dir, "fig_ablr_cdf")


def plot_waterfall(label: str, iq_path: Path, fs: float, out_dir: Path):
    iq = load_iq(iq_path)
    f, t, Sxx = signal.spectrogram(
        iq, fs=fs, nperseg=1024, noverlap=768,
        return_onesided=False, scaling="density",
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    im = ax.pcolormesh(t, f / 1e3, Sxx_db, shading="auto", cmap="magma", rasterized=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency offset from F0 (kHz)")
    guard = "fixed small guard" if "fixed" in label.lower() else "risk-aware wider guard"
    ax.set_title(f"Two-node OTA waterfall — {guard} (QUALITATIVE)")
    fig.colorbar(im, ax=ax, label="Power (dB, uncal.)")
    fig.text(0.5, -0.02,
             "Qualitative OTA demonstration only — no shared trigger, no measured "
             "collision probability. " + _FOOTER,
             ha="center", fontsize=6, style="italic", color="#666")
    stem = "fig_two_node_waterfall_fixed" if "fixed" in label.lower() \
        else "fig_two_node_waterfall_riskaware"
    _save(fig, out_dir, stem)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot OTA IQ proxy results")
    ap.add_argument("--run", action="append", default=[],
                    metavar="label=run_dir", help="repeatable; e.g. pgrl=runs/pgrl_001")
    ap.add_argument("--waterfall", action="append", default=[],
                    metavar="label=iq_file", help="repeatable; fixed=... riskaware=...")
    ap.add_argument("--sample-rate", type=float, default=1e6)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    runs = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects label=dir, got '{spec}'")
        label, d = spec.split("=", 1)
        runs[label] = Path(d)

    if runs:
        plot_cfo(runs, out_dir)
        plot_ablr(runs, out_dir)

    for spec in args.waterfall:
        if "=" not in spec:
            raise SystemExit(f"--waterfall expects label=iq_file, got '{spec}'")
        label, p = spec.split("=", 1)
        plot_waterfall(label, Path(p), args.sample_rate, out_dir)

    if not runs and not args.waterfall:
        raise SystemExit("Nothing to plot: pass --run and/or --waterfall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
