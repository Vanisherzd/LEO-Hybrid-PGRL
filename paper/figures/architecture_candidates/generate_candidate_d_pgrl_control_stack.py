#!/usr/bin/env python3
"""Generate candidate D: a PGRL-centered control-stack architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "figures" / "architecture_candidates" / "candidate_d_pgrl_control_stack.pdf"


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


def box(ax, x, y, w, h, text, face="#f5f6f7", edge="#2f3338", fs=10.0, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.0,
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


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11.0,
            linewidth=1.0,
            color="#6a6f75",
            shrinkA=1,
            shrinkB=1,
        )
    )


def cr(rect):
    x, y, w, h = rect
    return (x + w, y + h / 2.0)


def cl(rect):
    x, y, w, h = rect
    return (x, y + h / 2.0)


def cb(rect):
    x, y, w, h = rect
    return (x + w / 2.0, y)


def ct(rect):
    x, y, w, h = rect
    return (x + w / 2.0, y + h)


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "PGRL-centered control stack", transform=ax.transAxes, fontsize=11.0, fontweight="bold", ha="left", va="top")

    inp = box(ax, 0.05, 0.72, 0.16, 0.11, "Orbit / ephemeris\ninputs")
    sgp4 = box(ax, 0.28, 0.72, 0.17, 0.11, "SGP4 / SDP4\nbaseline")
    pgrl = box(ax, 0.52, 0.66, 0.20, 0.19, "PGRL\nresidual + uncertainty", face="#dfeaf6", edge="#466d93", fs=11.2, bold=True)
    txcfg = box(ax, 0.40, 0.36, 0.26, 0.10, "LR-FHSS TX configuration", face="#f4efe2", edge="#8a6f39", bold=True)
    guard = box(ax, 0.08, 0.47, 0.20, 0.09, "timing $\\sigma$\n$\\rightarrow$ guard time", face="#eef3f8", edge="#6684a5", fs=9.5)
    doppler = box(ax, 0.40, 0.49, 0.22, 0.09, "Doppler mean / rate\n$\\rightarrow$ pre-comp.", face="#eef3f8", edge="#6684a5", fs=9.5)
    timing = box(ax, 0.74, 0.47, 0.18, 0.09, "link score\n$\\rightarrow$ TX timing", face="#eef3f8", edge="#6684a5", fs=9.5)
    trace = box(ax, 0.08, 0.12, 0.18, 0.08, "trace-driven", face="#eef4ea", edge="#6c8362", fs=9.5)
    proxy = box(ax, 0.39, 0.12, 0.20, 0.08, "proxy", face="#eef4ea", edge="#6c8362", fs=9.5)
    hw = box(ax, 0.71, 0.12, 0.21, 0.08, "LR1121 + USRP IQ", face="#eef4ea", edge="#6c8362", fs=9.5)

    arrow(ax, cr(inp), cl(sgp4))
    arrow(ax, cr(sgp4), cl(pgrl))
    arrow(ax, cb(pgrl), ct(guard))
    arrow(ax, cb(pgrl), ct(doppler))
    arrow(ax, cb(pgrl), ct(timing))
    arrow(ax, cb(guard), ct(txcfg))
    arrow(ax, cb(doppler), ct(txcfg))
    arrow(ax, cb(timing), ct(txcfg))
    arrow(ax, cb(txcfg), ct(trace))
    arrow(ax, cb(txcfg), ct(proxy))
    arrow(ax, cb(txcfg), ct(hw))

    ax.text(0.50, 0.26, "validation evidence", transform=ax.transAxes, ha="center", va="center", fontsize=9.0, color="0.35")

    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
