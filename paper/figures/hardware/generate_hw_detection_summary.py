#!/usr/bin/env python3
"""Generate Fig. 6 detection-summary hardware evidence from curated artifacts."""

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
REPEATABILITY_JSON = ART / "repeatability_summary.json"
OUT_STEM = PAPER_HW / "fig_hw_lrfhss_detection_summary"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.3,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def crop_comparison_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[52 : h - 10, 0 : int(w * 0.515)]


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.018,
        0.982,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.5,
            alpha=0.96,
            boxstyle="round,pad=0.18",
        ),
    )


def load_repeatability():
    with REPEATABILITY_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [run["on_off_delta_db"] for run in data["runs"]]


def main() -> None:
    comparison_img = crop_comparison_left(mpimg.imread(COMPARISON))
    deltas = load_repeatability()

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.35))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.06, hspace=0.12)

    axes[0].imshow(comparison_img)
    axes[0].axis("off")
    add_panel_label(axes[0], "(a) Max-hold ON/OFF")

    labels = ["trial 1", "trial 2", "trial 3"]
    colors = ["#2c7fb8", "#567fae", "#6baed6"]
    bars = axes[1].bar(labels, deltas, color=colors, edgecolor="0.25", linewidth=0.55)
    axes[1].axhline(3.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].text(0.08, 3.28, "3 dB gate", color="0.38", fontsize=7.2)
    axes[1].set_ylabel("ON/OFF delta (dB)")
    axes[1].set_ylim(0, 14.8)
    axes[1].grid(axis="y", color="0.85", linewidth=0.6)
    for spine in axes[1].spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.25")
    for bar, delta in zip(bars, deltas):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            delta + 0.18,
            f"{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
    add_panel_label(axes[1], "(b) Repeatability")

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
