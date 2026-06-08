#!/usr/bin/env python3
"""Generate detector-focused hardware evidence from curated repeatability artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PAPER_HW = ROOT / "paper" / "figures" / "hardware"
ART = ROOT / "hardware" / "artifacts" / "lr1121_signal_detected_repeatability_20260604"
RUN1 = ART / "run1_000358"

COMPARISON = RUN1 / "run1_comparison.png"
COMPARISON_JSON = RUN1 / "run1_comparison.json"
REPEATABILITY_JSON = ART / "repeatability_summary.json"
RUN1_ON = RUN1 / "run1_on_analysis.json"
RUN1_OFF = RUN1 / "run1_off_analysis.json"
OUT_STEM = PAPER_HW / "fig_hw_lrfhss_detection_evidence"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.6,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def crop_comparison_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[52 : h - 10, 0 : int(w * 0.515)]


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.5,
            alpha=0.96,
            boxstyle="round,pad=0.18",
        ),
    )


def add_panel_a(ax, comparison_metrics):
    img = crop_comparison_left(mpimg.imread(COMPARISON))
    ax.imshow(img)
    ax.axis("off")
    add_panel_label(ax, "(a) ON/OFF max-hold")
    ax.text(
        0.98,
        0.08,
        f"run1 delta = {comparison_metrics['on_off_delta_db']:.2f} dB",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        color="0.2",
        bbox=dict(facecolor="white", edgecolor="0.85", linewidth=0.4, alpha=0.9, boxstyle="round,pad=0.16"),
    )


def add_panel_b(ax, comparison_metrics, on_metrics, off_metrics):
    categories = ["hot bins", "occ. freq", "occ. time", "peaks"]
    tx_on = [
        comparison_metrics["tx_on"]["hot_bin_count"],
        comparison_metrics["tx_on"]["occupied_frequency_bins"],
        comparison_metrics["tx_on"]["occupied_time_bins"],
        on_metrics["maxhold_peak_count_excluding_dc"],
    ]
    tx_off = [
        comparison_metrics["tx_off"]["hot_bin_count"],
        comparison_metrics["tx_off"]["occupied_frequency_bins"],
        comparison_metrics["tx_off"]["occupied_time_bins"],
        off_metrics["maxhold_peak_count_excluding_dc"],
    ]
    y = np.arange(len(categories))
    h = 0.32
    ax.barh(y - h / 2, tx_on, height=h, color="#2c7fb8", edgecolor="0.25", linewidth=0.45, label="TX-ON")
    ax.barh(y + h / 2, tx_off, height=h, color="#9aa3b2", edgecolor="0.25", linewidth=0.45, label="TX-OFF")
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_xlabel("Count (log scale)")
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.96)
    add_panel_label(ax, "(b) Occupancy summary")
    ax.text(
        0.02,
        0.05,
        f"hot-bin delta = {comparison_metrics['hot_bin_delta']}, freq-bin delta = {comparison_metrics['freq_occupancy_delta']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="0.32",
    )


def add_panel_c(ax, runs):
    labels = ["trial 1", "trial 2", "trial 3"]
    deltas = [run["on_off_delta_db"] for run in runs]
    colors = ["#2c7fb8", "#567fae", "#6baed6"]
    bars = ax.bar(labels, deltas, color=colors, edgecolor="0.25", linewidth=0.55)
    ax.axhline(3.0, color="0.45", linestyle="--", linewidth=0.9)
    ax.text(0.08, 3.23, "3 dB gate", color="0.38", fontsize=6.8)
    ax.set_ylabel("ON/OFF delta (dB)")
    ax.set_ylim(0, 14.8)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            delta + 0.18,
            f"{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.35,
            "detected",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#1f6b3a",
        )
    add_panel_label(ax, "(c) Repeatability")


def add_panel_d(ax, runs):
    ax.axis("off")
    add_panel_label(ax, "(d) Detector summary")
    ax.text(0.03, 0.80, "run   delta   score   UART   detected", transform=ax.transAxes, fontsize=7.1, weight="bold")
    y = 0.66
    for run in runs:
        line = (
            f"{run['run_id'].replace('run', 'run '):<5} "
            f"{run['on_off_delta_db']:>5.2f}   "
            f"{run['lr_fhss_candidate_score']:>5.3f}    "
            f"{run['uart_packet_sent_count']:>2}      "
            f"{'true' if run['signal_detected'] else 'false'}"
        )
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=6.9, family="monospace")
        y -= 0.12
    ax.text(
        0.03,
        0.30,
        "Gate: UART>0, score>=0.50, ON>OFF, DC/LO guard excluded",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.8,
        color="0.28",
    )
    ax.text(
        0.03,
        0.17,
        "run1 TX-OFF: score=0.113, hot bins=2, occ. freq=2, occ. time=2",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.6,
        color="0.35",
    )


def main() -> None:
    comparison_metrics = load_json(COMPARISON_JSON)
    repeatability = load_json(REPEATABILITY_JSON)
    on_metrics = load_json(RUN1_ON)
    off_metrics = load_json(RUN1_OFF)

    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.85))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.08, hspace=0.26, wspace=0.20)

    add_panel_a(axes[0, 0], comparison_metrics)
    add_panel_b(axes[0, 1], comparison_metrics, on_metrics, off_metrics)
    add_panel_c(axes[1, 0], repeatability["runs"])
    add_panel_d(axes[1, 1], repeatability["runs"])

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
