#!/usr/bin/env python3
"""Generate manuscript figures from checked-in experiment artifacts."""

import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "paper" / "figures"
HW_DIR = OUT_DIR / "hardware"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HW_DIR.mkdir(parents=True, exist_ok=True)

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, text=True
    ).strip()
except Exception:
    COMMIT = "unknown"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
    }
)

BLACK = "#202020"
GRAY = "#6d6d6d"
LIGHT = "#f2f2f2"
PGRL = "#1f77b4"
SGP4 = "#d62728"
ORACLE = "#2ca02c"
ACCENT = "#9467bd"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.relative_to(REPO)}")


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(3.55, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.035",
            facecolor=fc,
            edgecolor=BLACK,
            linewidth=0.65,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.1)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.75, shrinkA=1.5, shrinkB=1.5),
        )

    box(0.25, 4.45, 1.55, 0.75, "TLE", LIGHT)
    box(2.35, 4.45, 1.75, 0.75, "SGP4/\nSDP4", "#e8f1fb")
    box(4.75, 4.45, 2.05, 0.75, "Bayesian\nPGRL", "#eaf5ea")
    box(7.55, 4.45, 1.95, 0.75, "timing,\nDoppler, sigma", "#fff3dd")

    box(0.65, 2.45, 2.1, 0.8, "adaptive\nguard time", "#f5f5f5")
    box(3.25, 2.45, 2.1, 0.8, "TX timing\nselection", "#f5f5f5")
    box(5.85, 2.45, 2.1, 0.8, "Doppler\npre-comp.", "#f5f5f5")
    box(3.0, 0.65, 2.0, 0.8, "LR1121\nTX path", "#e8f1fb")
    box(5.95, 0.65, 2.0, 0.8, "USRP B210\nIQ capture", "#eaf5ea")

    for x1, x2 in [(1.8, 2.35), (4.1, 4.75), (6.8, 7.55)]:
        arrow(x1, 4.825, x2, 4.825)
    arrow(8.55, 4.45, 7.0, 3.25)
    arrow(7.55, 4.45, 4.3, 3.25)
    arrow(7.55, 4.45, 1.7, 3.25)
    arrow(6.9, 2.45, 4.0, 1.45)
    arrow(5.0, 1.05, 5.95, 1.05)

    ax.text(0.25, 5.55, "orbit prediction", fontsize=7.2, weight="bold", color=GRAY)
    ax.text(0.25, 3.62, "uncertainty-aware uplink control", fontsize=7.2, weight="bold", color=GRAY)
    ax.text(0.25, 1.58, "hardware signal-detection path", fontsize=7.2, weight="bold", color=GRAY)
    save(fig, OUT_DIR / "fig1_architecture.pdf")


def fig2_uncertainty():
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    nominal = np.array([0.68, 0.80, 0.90, 0.95, 0.99])
    timing = np.array([0.661, 0.800, 0.883, 0.932, 0.978])
    doppler = np.array([0.674, 0.799, 0.891, 0.945, 0.981])

    ax.plot(nominal, nominal, "--", color=GRAY, lw=1.0, label="ideal")
    ax.plot(nominal, timing, "o-", color=PGRL, ms=3.5, label="timing")
    ax.plot(nominal, doppler, "s-", color=ORACLE, ms=3.3, label="Doppler")
    ax.set_xlabel("Nominal confidence")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.66, 1.0)
    ax.set_ylim(0.64, 1.0)
    ax.set_xticks(nominal)
    ax.set_xticklabels([f"{int(x*100)}%" for x in nominal])
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.text(0.675, 0.975, "ECE: timing 0.28%, Doppler 0.12%", fontsize=7.2)
    save(fig, OUT_DIR / "fig2_pgrl_uncertainty.pdf")


def fig3_guard_energy():
    data = load_json(REPO / "paper" / "tables" / "main_results.json")["table_3_guard_band"]["rows"]
    labels = ["Fixed\n30 ms", "SGP4\n3 sigma", "PGRL\nmean", "PGRL\nuncert."]
    overhead = [0.013, 2.41, 0.51, 0.23]
    energy = [0.07955, 0.09506, 0.174, 0.09665]
    colors = ["#bdbdbd", SGP4, ACCENT, PGRL]

    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.25))
    axes[0].bar(labels, overhead, color=colors, edgecolor=BLACK, linewidth=0.35)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Guard overhead (%)")
    axes[0].set_ylim(0.008, 4.0)
    for i, v in enumerate(overhead):
        axes[0].text(i, v * 1.18, f"{v:g}%", ha="center", va="bottom", fontsize=6.7)

    axes[1].bar(labels, energy, color=colors, edgecolor=BLACK, linewidth=0.35)
    axes[1].set_ylabel("Energy (J/opportunity)")
    axes[1].set_ylim(0, 0.2)
    for i, v in enumerate(energy):
        axes[1].text(i, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.7)

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    save(fig, OUT_DIR / "fig3_guard_energy.pdf")


def fig4_lrfhss_grid():
    data = load_json(REPO / "experiments" / "exp3_lrfhss_grid_proxy" / "results.json")
    residual = np.array(data["residual_doppler_hz"])
    # Use the proxy curve as a function of residual frequency, not the inconsistent
    # per-baseline labels in the artifact.
    score = np.array(data["orthogonality_score"]["no_comp"]["orthogonality"])

    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    ax.plot(residual, score, "o-", color=BLACK, ms=3.5, label="grid proxy")
    ax.axvspan(200, 350, color=PGRL, alpha=0.12, lw=0)
    ax.axvline(300, color=PGRL, ls="--", lw=1.0)
    ax.text(315, 0.91, "PGRL\nresidual", color=PGRL, fontsize=7.0, va="top")
    ax.annotate(
        "SGP4-only residuals\nextend to kHz scale",
        xy=(1000, score[-1]),
        xytext=(520, 0.52),
        arrowprops=dict(arrowstyle="->", lw=0.7, color=SGP4),
        color=SGP4,
        fontsize=7.0,
    )
    ax.set_xlabel("Residual Doppler (Hz)")
    ax.set_ylabel("Orthogonality score")
    ax.set_xlim(0, 1050)
    ax.set_ylim(0.25, 1.02)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    save(fig, OUT_DIR / "fig4_lrfhss_grid_proxy.pdf")


def fig_hw_lrfhss():
    summary = load_json(
        REPO
        / "hardware"
        / "artifacts"
        / "lr1121_signal_detected_repeatability_20260604"
        / "repeatability_summary.json"
    )
    run1 = summary["runs"][0]
    comp = load_json(
        REPO
        / "hardware"
        / "artifacts"
        / "lr1121_signal_detected_repeatability_20260604"
        / "run1_000358"
        / "run1_comparison.json"
    )
    deltas = [r["on_off_delta_db"] for r in summary["runs"]]
    runs = [r["run_id"].replace("run", "trial ") for r in summary["runs"]]

    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.2))
    axes[0].bar(
        ["TX OFF", "TX ON"],
        [comp["tx_off"]["maxhold_excess_db"], comp["tx_on"]["maxhold_excess_db"]],
        color=["#d9d9d9", PGRL],
        edgecolor=BLACK,
        linewidth=0.4,
    )
    axes[0].axhline(comp["threshold_db"], color=GRAY, ls="--", lw=0.9)
    axes[0].set_ylabel("Max-hold excess (dB)")
    axes[0].set_ylim(0, 14.5)
    axes[0].text(0.03, 0.93, "trial 1", transform=axes[0].transAxes, fontsize=7.0)
    axes[0].text(0.74, 0.58, "detector\nthreshold", transform=axes[0].transAxes, fontsize=6.7, color=GRAY)

    bars = axes[1].bar(runs, deltas, color=[PGRL, ORACLE, ACCENT], edgecolor=BLACK, linewidth=0.4)
    axes[1].axhline(3.0, color=GRAY, ls="--", lw=0.9)
    axes[1].set_ylabel("ON/OFF delta (dB)")
    axes[1].set_ylim(0, 13.5)
    axes[1].tick_params(axis="x", rotation=18)
    for bar, v in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.2f}", ha="center", fontsize=6.8)
    axes[1].text(0.02, 0.93, "all signal-detected", transform=axes[1].transAxes, fontsize=7.0)

    save(fig, HW_DIR / "fig_hw_lrfhss_evidence.pdf")


if __name__ == "__main__":
    print(f"Generating figures in {OUT_DIR} (commit {COMMIT})")
    fig1_architecture()
    fig2_uncertainty()
    fig3_guard_energy()
    fig4_lrfhss_grid()
    fig_hw_lrfhss()
