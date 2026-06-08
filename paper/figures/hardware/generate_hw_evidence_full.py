#!/usr/bin/env python3
"""Build the Fig. 5 hardware evidence panel from curated successful artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PAPER_HW = ROOT / "paper" / "figures" / "hardware"
ART = ROOT / "hardware" / "artifacts" / "lr1121_signal_detected_repeatability_20260604"
RUN1 = ART / "run1_000358"

ON_WATERFALL = RUN1 / "run1_on_waterfall.png"
OFF_WATERFALL = RUN1 / "run1_off_waterfall.png"
COMPARISON = RUN1 / "run1_comparison.png"
REPEATABILITY_JSON = ART / "repeatability_summary.json"
OUT_STEM = PAPER_HW / "fig_hw_lrfhss_evidence_full"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.4,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_image(path: Path) -> np.ndarray:
    return mpimg.imread(path)


def crop_waterfall(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[94 : h - 14, 10 : w - 8]


def crop_comparison_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[52 : h - 10, 0 : int(w * 0.515)]


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.5,
            alpha=0.95,
            boxstyle="round,pad=0.18",
        ),
    )


def add_image_panel(ax, image: np.ndarray, label: str) -> None:
    ax.imshow(image)
    ax.axis("off")
    add_panel_label(ax, label)


def load_repeatability():
    with REPEATABILITY_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [run["on_off_delta_db"] for run in data["runs"]]


def add_repeatability_panel(ax) -> None:
    deltas = load_repeatability()
    labels = ["trial 1", "trial 2", "trial 3"]
    colors = ["#2c7fb8", "#567fae", "#6baed6"]

    bars = ax.bar(labels, deltas, color=colors, edgecolor="0.25", linewidth=0.55)
    ax.axhline(3.0, color="0.45", linestyle="--", linewidth=0.9)
    ax.text(0.06, 3.28, "3 dB gate", color="0.38", fontsize=7.0)
    ax.set_ylabel("ON/OFF delta (dB)")
    ax.set_ylim(0, 14.8)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.25")
    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            delta + 0.18,
            f"{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    add_panel_label(ax, "(d) Repeatability")


def main() -> None:
    on_img = crop_waterfall(load_image(ON_WATERFALL))
    off_img = crop_waterfall(load_image(OFF_WATERFALL))
    comparison_img = crop_comparison_left(load_image(COMPARISON))

    fig = plt.figure(figsize=(7.15, 4.85))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.035,
        right=0.992,
        top=0.985,
        bottom=0.055,
        hspace=0.20,
        wspace=0.15,
    )

    add_image_panel(fig.add_subplot(gs[0, 0]), on_img, "(a) TX-ON waterfall")
    add_image_panel(fig.add_subplot(gs[0, 1]), off_img, "(b) TX-OFF reference")
    add_image_panel(fig.add_subplot(gs[1, 0]), comparison_img, "(c) Max-hold ON/OFF")
    add_repeatability_panel(fig.add_subplot(gs[1, 1]))

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
