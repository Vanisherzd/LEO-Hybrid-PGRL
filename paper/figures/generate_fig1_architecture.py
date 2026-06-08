#!/usr/bin/env python3
"""Generate the cleaner systems-style Fig. 1 architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "fig1_architecture.pdf"


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


def region(ax, x, y, w, h, title, face):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=0.8,
        edgecolor="#cfd3d8",
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.015,
        y + h - 0.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.6,
        fontweight="bold",
        color="#1c1f23",
    )


def box(ax, x, y, w, h, text, face="#f5f6f7", edge="#2f3338", fs=10.0, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color="#111315",
    )
    return (x, y, w, h)


def arrow(ax, start, end, text=None, dy=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=11.5,
        linewidth=1.0,
        color="#6a6f75",
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2.0
        my = (start[1] + end[1]) / 2.0 + dy
        ax.text(mx, my, text, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="#5c6168")


def line(ax, start, end, lw=1.0):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        transform=ax.transAxes,
        color="#6a6f75",
        linewidth=lw,
        solid_capstyle="round",
    )


def center_right(rect):
    x, y, w, h = rect
    return (x + w, y + h / 2.0)


def center_left(rect):
    x, y, w, h = rect
    return (x, y + h / 2.0)


def center_bottom(rect):
    x, y, w, h = rect
    return (x + w / 2.0, y)


def center_top(rect):
    x, y, w, h = rect
    return (x + w / 2.0, y + h)


def main():
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    region(ax, 0.02, 0.70, 0.96, 0.24, "Prediction Layer", "#f4f7fb")
    region(ax, 0.02, 0.36, 0.96, 0.26, "LR-FHSS Control Layer", "#f8f8f8")
    region(ax, 0.02, 0.05, 0.96, 0.22, "Validation / Evidence Layer", "#fbfaf6")

    tle = box(ax, 0.07, 0.77, 0.13, 0.09, "TLE /\nquery epoch")
    sgp4 = box(ax, 0.28, 0.77, 0.15, 0.09, "SGP4 baseline")
    pgrl = box(ax, 0.51, 0.745, 0.19, 0.14, "PGRL\nresidual + uncertainty", face="#dfeaf6", edge="#466d93", fs=10.8, bold=True)
    arrow(ax, center_right(tle), center_left(sgp4))
    arrow(ax, center_right(sgp4), center_left(pgrl), text="physical anchor", dy=0.05)

    guard = box(ax, 0.11, 0.425, 0.19, 0.10, "timing $\\sigma$\n$\\rightarrow$ adaptive guard time", face="#eef3f8", edge="#6684a5", fs=9.4)
    doppler = box(ax, 0.40, 0.425, 0.21, 0.10, "Doppler mean / rate\n$\\rightarrow$ carrier pre-comp.", face="#eef3f8", edge="#6684a5", fs=9.4)
    timing = box(ax, 0.71, 0.425, 0.17, 0.10, "link score\n$\\rightarrow$ TX timing", face="#eef3f8", edge="#6684a5", fs=9.4)
    txcfg = box(ax, 0.37, 0.255, 0.28, 0.085, "LR-FHSS TX configuration", face="#f4efe2", edge="#8a6f39", fs=10.2, bold=True)

    bus_y = 0.60
    pgrl_mid_x = center_bottom(pgrl)[0]
    line(ax, center_bottom(pgrl), (pgrl_mid_x, bus_y))
    line(ax, (0.16, bus_y), (0.80, bus_y))
    for target in (guard, doppler, timing):
        arrow(ax, (center_top(target)[0], bus_y), center_top(target))
    arrow(ax, (pgrl_mid_x, bus_y + 0.002), (pgrl_mid_x, bus_y - 0.002))
    arrow(ax, center_bottom(guard), (0.46, 0.34))
    arrow(ax, center_bottom(doppler), (0.51, 0.34))
    arrow(ax, center_bottom(timing), (0.56, 0.34))
    arrow(ax, (0.51, 0.34), center_top(txcfg))

    trace = box(ax, 0.10, 0.095, 0.20, 0.085, "trace-driven\nevaluation", face="#eef4ea", edge="#6c8362", fs=9.6)
    proxy = box(ax, 0.40, 0.095, 0.22, 0.085, "LR-FHSS grid /\nQPSK EVM proxy", face="#eef4ea", edge="#6c8362", fs=9.6)
    hw = box(ax, 0.72, 0.095, 0.20, 0.085, "LR1121 + USRP B210\nIQ signal detection", face="#eef4ea", edge="#6c8362", fs=9.4)
    hw_bus_y = 0.225
    tx_mid_x = center_bottom(txcfg)[0]
    hw_top_x = center_top(hw)[0]
    line(ax, center_bottom(txcfg), (tx_mid_x, hw_bus_y))
    line(ax, (tx_mid_x, hw_bus_y), (hw_top_x, hw_bus_y))
    arrow(ax, (hw_top_x, hw_bus_y), center_top(hw))

    ax.text(
        0.84,
        0.072,
        "IQ-level evidence only",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#55604c",
        style="italic",
    )

    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
