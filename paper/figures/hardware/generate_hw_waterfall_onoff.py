#!/usr/bin/env python3
"""Generate Fig. 5 waterfall-only hardware evidence from curated run-1 artifacts."""

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
OUT_STEM = PAPER_HW / "fig_hw_lrfhss_waterfall_onoff"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
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


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.018,
        0.982,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.5,
            alpha=0.96,
            boxstyle="round,pad=0.18",
        ),
    )


def main() -> None:
    on_img = crop_waterfall(load_image(ON_WATERFALL))
    off_img = crop_waterfall(load_image(OFF_WATERFALL))

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.06, wspace=0.13)

    for ax, img, label in (
        (axes[0], on_img, "(a) TX-ON waterfall"),
        (axes[1], off_img, "(b) TX-OFF reference"),
    ):
        ax.imshow(img)
        ax.axis("off")
        add_panel_label(ax, label)

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
