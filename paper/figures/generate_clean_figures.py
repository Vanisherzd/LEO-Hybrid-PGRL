#!/usr/bin/env python3
"""Regenerate paper figures from existing experiment and hardware artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
FIG = PAPER / "figures"
HW_FIG = FIG / "hardware"
EXP = REPO / "experiments"
TABLES = PAPER / "tables"
HW_ART = REPO / "hardware" / "artifacts" / "lr1121_signal_detected_repeatability_20260604"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.9,
    }
)

BLACK = "0.10"
GRAY = "0.45"
LIGHT = "0.86"
PGRL = "#1f77b4"
SGP4 = "#d62728"
ORACLE = "#2ca02c"
BASE = "#7f7f7f"
ALT = "#ff7f0e"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_pdf_png(fig, stem: Path, png: bool = False):
    fig.savefig(stem.with_suffix(".pdf"))
    if png:
        fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def float_from_text(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        raise ValueError(f"Cannot parse numeric value from {value!r}")
    return float(match.group(0))


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge=BLACK, lw=0.8):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.08)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.9, shrinkA=2, shrinkB=2),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom", fontsize=7.2, color=GRAY)

    ax.text(0.25, 4.35, "Orbit baseline", fontsize=8.3, weight="bold", color=BLACK)
    ax.text(0.25, 2.32, "Uncertainty-aware LR-FHSS uplink control", fontsize=8.3, weight="bold", color=BLACK)
    ax.text(0.25, 0.60, "Hardware evidence path", fontsize=8.3, weight="bold", color=BLACK)

    box(0.25, 3.35, 1.55, 0.58, "TLE", "0.96")
    box(2.25, 3.35, 1.65, 0.58, "SGP4/SDP4", "0.96")
    box(4.45, 3.20, 1.95, 0.88, "Bayesian\nresidual learner", "#e8f2fb")
    box(7.00, 3.20, 1.75, 0.88, "Residuals\nand variance", "#e8f2fb")

    arrow(1.80, 3.64, 2.25, 3.64)
    arrow(3.90, 3.64, 4.45, 3.64, "state")
    arrow(6.40, 3.64, 7.00, 3.64, "mean, sigma")

    box(0.25, 1.45, 1.72, 0.70, "Adaptive\nguard time", "#f1f1f1")
    box(2.55, 1.45, 1.72, 0.70, "TX timing\nselection", "#f1f1f1")
    box(4.85, 1.45, 1.85, 0.70, "Doppler\npre-comp.", "#f1f1f1")
    box(7.30, 1.45, 1.80, 0.70, "LR-FHSS TX\nconfiguration", "#f1f1f1")

    arrow(7.88, 3.20, 1.10, 2.18)
    arrow(1.97, 1.80, 2.55, 1.80)
    arrow(4.27, 1.80, 4.85, 1.80)
    arrow(6.70, 1.80, 7.30, 1.80)

    box(3.10, 0.25, 2.10, 0.62, "Semtech LR1121 TX", "#fbf4df")
    box(6.15, 0.25, 2.05, 0.62, "USRP B210 IQ capture", "#eef7e9")
    arrow(8.08, 1.45, 4.15, 0.88, "868 MHz")
    arrow(5.20, 0.56, 6.15, 0.56, "RF")
    ax.text(9.35, 0.56, "signal detection only", fontsize=7.4, color=GRAY, va="center")
    arrow(8.20, 0.56, 9.28, 0.56)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.06)
    save_pdf_png(fig, FIG / "fig1_architecture")


def fig2_uncertainty():
    nominal = np.array([0.68, 0.90, 0.95, 0.99])
    timing = np.array([0.661, 0.883, 0.932, 0.978])
    doppler = np.array([0.674, 0.891, 0.945, 0.981])

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.plot(nominal, nominal, color=BLACK, linestyle="--", lw=1.0, label="ideal")
    ax.plot(nominal, timing, "o-", color=PGRL, label="timing, ECE=0.28%")
    ax.plot(nominal, doppler, "s-", color=ORACLE, label="Doppler, ECE=0.12%")
    ax.set_xlabel("Nominal confidence")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.66, 1.005)
    ax.set_ylim(0.66, 1.005)
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.18)
    save_pdf_png(fig, FIG / "fig2_pgrl_uncertainty")


def fig3_guard_energy():
    data = load_json(TABLES / "main_results.json")["table_3_guard_band"]["rows"]
    labels = ["Fixed\n30 ms", "SGP4\n3 sigma", "PGRL\nmean", "PGRL\nunc."]
    overhead = [float_from_text(r[1]) for r in data]
    missed = [float_from_text(r[2]) for r in data]
    energy = [float_from_text(r[3]) for r in data]
    colors = [BASE, SGP4, ALT, PGRL]

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35))
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, overhead, color=colors, edgecolor=BLACK, linewidth=0.35)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Guard overhead (%)")
    ax.set_ylim(0.08, max(overhead) * 1.8)
    for bar, val in zip(bars, overhead):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.12, f"{val:g}%", ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    bars = ax.bar(x, energy, color=colors, edgecolor=BLACK, linewidth=0.35)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Energy per opportunity (J)")
    ax.set_ylim(0, max(energy) * 1.24)
    for bar, val, miss in zip(bars, energy, missed):
        note = "miss=1" if miss >= 0.99 else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(energy) * 0.025, note, ha="center", va="bottom", fontsize=7)

    handles = [patches.Patch(facecolor=c, edgecolor=BLACK, linewidth=0.35, label=l.replace("\n", " ")) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 1.05))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.23, wspace=0.34)
    save_pdf_png(fig, FIG / "fig3_guard_energy")


def fig4_lrfhss_grid():
    data = load_json(EXP / "exp3_lrfhss_grid_proxy" / "results.json")
    x = np.array(data["residual_doppler_hz"], dtype=float)
    series = data["orthogonality_score"]
    styles = {
        "oracle_comp": ("Oracle", BLACK, "--", "D"),
        "pgrl_comp": ("PGRL", PGRL, "-", "o"),
        "no_comp": ("No comp.", ALT, "-.", "^"),
        "sgp4_comp": ("SGP4", SGP4, ":", "s"),
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for key in ["oracle_comp", "pgrl_comp", "no_comp", "sgp4_comp"]:
        label, color, ls, marker = styles[key]
        y = series[key]["orthogonality"]
        ax.plot(x, y, marker=marker, linestyle=ls, color=color, label=label)
    ax.set_xlabel("Residual Doppler (Hz)")
    ax.set_ylabel("Orthogonality score")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlim(-20, 1020)
    ax.set_xticks([0, 300, 500, 1000])
    ax.axvline(300, color=PGRL, lw=0.8, alpha=0.35)
    ax.text(310, 0.13, "300 Hz", fontsize=7.2, color=PGRL, rotation=90, va="bottom")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, ncol=2, columnspacing=0.8)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.18)
    save_pdf_png(fig, FIG / "fig4_lrfhss_grid_proxy")


def fig5_sdr_proxy():
    data = load_json(EXP / "exp5_sdr_doppler_precomp" / "results.json")["results"]
    order = ["oracle_comp_SNR40", "pgrl_comp_SNR40", "no_comp_SNR40"]
    labels = ["Oracle", "PGRL", "No comp."]
    cfo = [data[k]["cfo_hz"] for k in order]
    evm = [data[k]["evm_percent"] for k in order]
    colors = [BLACK, PGRL, SGP4]

    fig, ax1 = plt.subplots(figsize=(3.45, 2.15))
    x = np.arange(len(labels))
    bars = ax1.bar(x, evm, color=colors, edgecolor=BLACK, linewidth=0.35)
    ax1.set_ylabel("QPSK EVM proxy (%)")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, max(evm) * 1.22)
    for bar, val in zip(bars, evm):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(evm) * 0.035, f"{val:.1f}", ha="center", fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(x, cfo, color=GRAY, marker="o", linestyle="--", lw=1.0, label="residual CFO")
    ax2.set_ylabel("Residual CFO (Hz)", color=GRAY)
    ax2.tick_params(axis="y", colors=GRAY)
    ax2.set_ylim(0, max(cfo) * 1.30)
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")
    fig.subplots_adjust(left=0.16, right=0.85, top=0.96, bottom=0.20)
    save_pdf_png(fig, FIG / "fig5_sdr_synthetic_pipeline")


def maxhold_from_fc32(path: Path, fs: float, nfft: int = 4096):
    samples = np.fromfile(path, dtype=np.complex64)
    nfft = min(nfft, len(samples))
    window = np.hanning(nfft)
    win_norm = max(float(np.sum(window**2)), 1.0)
    step = nfft // 2
    maxhold = None
    for start in range(0, len(samples) - nfft + 1, step):
        seg = samples[start : start + nfft] * window
        power = (np.abs(np.fft.fft(seg, n=nfft)) ** 2) / (fs * win_norm)
        if maxhold is None:
            maxhold = power
        else:
            np.maximum(maxhold, power, out=maxhold)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    db = 10.0 * np.log10(np.maximum(np.fft.fftshift(maxhold), 1e-30))
    return freqs, db


def hardware_figures():
    summary = load_json(HW_ART / "repeatability_summary.json")
    run1_on = load_json(HW_ART / "run1_000358" / "run1_on_analysis.json")
    run1_off = load_json(HW_ART / "run1_000358" / "run1_off_analysis.json")
    fs = float(run1_on["sample_rate_hz"])
    freqs_on, db_on = maxhold_from_fc32(Path(run1_on["input_file"]), fs)
    freqs_off, db_off = maxhold_from_fc32(Path(run1_off["input_file"]), fs)
    mh_delta = load_json(HW_ART / "run1_000358" / "run1_comparison.json")["on_off_delta_db"]

    runs = summary["runs"]
    labels = [r["run_id"].replace("run", "trial ") for r in runs]
    deltas = [float(r["on_off_delta_db"]) for r in runs]
    scores = [float(r["lr_fhss_candidate_score"]) for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
    ax = axes[0]
    ax.plot(freqs_off / 1e3, db_off, color=GRAY, lw=0.75, label="TX off")
    ax.plot(freqs_on / 1e3, db_on, color=PGRL, lw=0.75, label="TX on")
    ax.axvspan(-20, 20, color="0.9", alpha=0.9, zorder=-1, label="DC guard")
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Max-hold power (dB)")
    ax.set_xlim(-500, 500)
    ax.set_ylim(min(db_off.min(), db_on.min()) - 0.5, max(db_on.max(), db_off.max()) + 0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.text(0.02, 0.93, f"trial 1: delta={mh_delta:.2f} dB", transform=ax.transAxes, fontsize=7.5, va="top")

    ax = axes[1]
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=[PGRL, "#4c78a8", "#6baed6"], edgecolor=BLACK, linewidth=0.35)
    ax.axhline(3.0, color=GRAY, linestyle="--", lw=0.85, label="3 dB gate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ON/OFF occupancy delta (dB)")
    ax.set_ylim(0, max(deltas) * 1.30)
    for bar, val, score in zip(bars, deltas, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", fontsize=7.5)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.42, f"score\n{score:.2f}", ha="center", va="bottom", fontsize=6.8, color="white")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.93, "detected in all trials", transform=ax.transAxes, fontsize=7.2, ha="right", va="top")

    fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.20, wspace=0.32)
    save_pdf_png(fig, HW_FIG / "fig_hw_lrfhss_evidence", png=True)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.20))
    ax = axes[0]
    ax.plot(freqs_off / 1e3, db_off, color=GRAY, lw=0.75, label="TX off")
    ax.plot(freqs_on / 1e3, db_on, color=PGRL, lw=0.75, label="TX on")
    ax.axvspan(-20, 20, color="0.9", alpha=0.9, zorder=-1)
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Max-hold power (dB)")
    ax.set_xlim(-500, 500)
    ax.set_ylim(min(db_off.min(), db_on.min()) - 0.5, max(db_on.max(), db_off.max()) + 0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.text(0.02, 0.93, f"(a) trial 1, delta={mh_delta:.2f} dB", transform=ax.transAxes, fontsize=7.4, va="top")

    ax = axes[1]
    bars = ax.bar(x, deltas, color=[PGRL, "#4c78a8", "#6baed6"], edgecolor=BLACK, linewidth=0.35)
    ax.axhline(3.0, color=GRAY, linestyle="--", lw=0.85, label="3 dB gate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ON/OFF delta (dB)")
    ax.set_ylim(0, max(deltas) * 1.30)
    for bar, val, score in zip(bars, deltas, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", fontsize=7.5)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.35, f"score {score:.2f}", ha="center", va="bottom", fontsize=6.6, color="white", rotation=90)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.92, "(b) detected in all trials", transform=ax.transAxes, fontsize=7.1, ha="right", va="top")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.10, hspace=0.38)
    save_pdf_png(fig, HW_FIG / "fig_hw_lrfhss_evidence_column", png=True)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(freqs_off / 1e3, db_off, color=GRAY, lw=0.75, label="TX off")
    ax.plot(freqs_on / 1e3, db_on, color=PGRL, lw=0.75, label="TX on")
    ax.axvspan(-20, 20, color="0.9", alpha=0.9, zorder=-1, label="DC guard")
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Max-hold power (dB)")
    ax.set_xlim(-500, 500)
    ax.set_ylim(min(db_off.min(), db_on.min()) - 0.5, max(db_on.max(), db_off.max()) + 0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.text(0.02, 0.93, f"trial 1, delta={mh_delta:.2f} dB", transform=ax.transAxes, fontsize=7.4, va="top")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.18)
    save_pdf_png(fig, HW_FIG / "fig_hw_lrfhss_onoff_comparison", png=True)

    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    bars = ax.bar(x, deltas, color=[PGRL, "#4c78a8", "#6baed6"], edgecolor=BLACK, linewidth=0.35)
    ax.axhline(3.0, color=GRAY, linestyle="--", lw=0.85, label="3 dB gate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ON/OFF delta (dB)")
    ax.set_ylim(0, max(deltas) * 1.30)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.20)
    save_pdf_png(fig, HW_FIG / "fig_hw_lrfhss_repeatability", png=True)


def main():
    fig1_architecture()
    fig2_uncertainty()
    fig3_guard_energy()
    fig4_lrfhss_grid()
    fig5_sdr_proxy()
    hardware_figures()


if __name__ == "__main__":
    main()
