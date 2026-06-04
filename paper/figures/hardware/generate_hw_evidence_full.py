#!/usr/bin/env python3
"""Build the full 2x2 hardware evidence figure from curated artifact images."""

from __future__ import annotations

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

ON_WATERFALL = RUN1 / "run1_on_waterfall.png"
OFF_WATERFALL = RUN1 / "run1_off_waterfall.png"
MAXHOLD_PANEL = PAPER_HW / "fig_hw_lrfhss_onoff_comparison.png"
REPEATABILITY_PANEL = PAPER_HW / "fig_hw_lrfhss_repeatability.png"
OUT_STEM = PAPER_HW / "fig_hw_lrfhss_evidence_full"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10.0,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_image(path: Path) -> np.ndarray:
    return mpimg.imread(path)


def crop_waterfall(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    top = 56
    bottom = h - 14
    left = 10
    right = w - 8
    return img[top:bottom, left:right]


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="0.75", linewidth=0.5, alpha=0.92, boxstyle="round,pad=0.18"),
    )


def main() -> None:
    on_img = crop_waterfall(load_image(ON_WATERFALL))
    off_img = crop_waterfall(load_image(OFF_WATERFALL))
    maxhold_img = load_image(MAXHOLD_PANEL)
    repeatability_img = load_image(REPEATABILITY_PANEL)

    fig = plt.figure(figsize=(7.1, 3.85))
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.99, top=0.972, bottom=0.055, hspace=0.06, wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(on_img)
    ax.axis("off")
    add_panel_label(ax, "(a) TX-ON waterfall")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(off_img)
    ax.axis("off")
    add_panel_label(ax, "(b) TX-OFF reference")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(maxhold_img)
    ax.axis("off")
    add_panel_label(ax, "(c) Max-hold ON/OFF")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(repeatability_img)
    ax.axis("off")
    add_panel_label(ax, "(d) Repeatability")

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
