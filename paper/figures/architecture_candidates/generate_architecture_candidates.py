#!/usr/bin/env python3
"""Generate alternative architecture candidates for future review."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "paper" / "figures" / "architecture_candidates"


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


def arrow(ax, start, end, color="#6a6f75", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11.0,
            linewidth=1.0,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
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


def save_three_layer():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for y, title, face in (
        (0.70, "Prediction layer", "#f4f7fb"),
        (0.38, "LR-FHSS control layer", "#f8f8f8"),
        (0.07, "Validation / evidence layer", "#fbfaf6"),
    ):
        ax.add_patch(
            FancyBboxPatch((0.03, y), 0.94, 0.21, boxstyle="round,pad=0.012,rounding_size=0.014",
                           linewidth=0.8, edgecolor="#cfd3d8", facecolor=face, transform=ax.transAxes)
        )
        ax.text(0.05, y + 0.18, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")
    tle = box(ax, 0.07, 0.77, 0.13, 0.08, "TLE /\nquery epoch")
    sgp4 = box(ax, 0.29, 0.77, 0.14, 0.08, "SGP4 baseline")
    pgrl = box(ax, 0.52, 0.75, 0.18, 0.12, "PGRL\nresidual + uncertainty", face="#dfeaf6", edge="#466d93", bold=True, fs=10.8)
    arrow(ax, cr(tle), cl(sgp4))
    arrow(ax, cr(sgp4), cl(pgrl))
    g = box(ax, 0.11, 0.45, 0.20, 0.09, "timing $\\sigma$\n$\\rightarrow$ guard time", face="#eef3f8", edge="#6684a5", fs=9.4)
    d = box(ax, 0.40, 0.45, 0.21, 0.09, "Doppler mean / rate\n$\\rightarrow$ pre-comp.", face="#eef3f8", edge="#6684a5", fs=9.4)
    t = box(ax, 0.72, 0.45, 0.16, 0.09, "link score\n$\\rightarrow$ TX timing", face="#eef3f8", edge="#6684a5", fs=9.4)
    tx = box(ax, 0.38, 0.27, 0.25, 0.08, "LR-FHSS TX configuration", face="#f4efe2", edge="#8a6f39", bold=True, fs=10.0)
    for r in (g, d, t):
        arrow(ax, cb(pgrl), ct(r))
        arrow(ax, cb(r), ct(tx))
    trace = box(ax, 0.10, 0.12, 0.21, 0.08, "trace-driven\nevaluation", face="#eef4ea", edge="#6c8362", fs=9.6)
    proxy = box(ax, 0.40, 0.12, 0.22, 0.08, "grid / QPSK EVM\nproxy", face="#eef4ea", edge="#6c8362", fs=9.6)
    hw = box(ax, 0.72, 0.12, 0.18, 0.08, "LR1121 + USRP\nIQ evidence", face="#eef4ea", edge="#6c8362", fs=9.4)
    arrow(ax, cb(tx), ct(proxy))
    arrow(ax, cb(tx), ct(hw))
    fig.savefig(OUTDIR / "candidate_a_three_layer.pdf")
    plt.close(fig)


def save_closed_loop():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = {
        "pgrl": (0.5, 0.82, "PGRL\nresidual + uncertainty", "#dfeaf6", "#466d93"),
        "ctrl": (0.82, 0.52, "LR-FHSS\ncontrol", "#eef3f8", "#6684a5"),
        "tx": (0.62, 0.17, "LR1121 TX /\nIQ capture", "#f4efe2", "#8a6f39"),
        "ev": (0.20, 0.20, "IQ evidence /\nartifacts", "#eef4ea", "#6c8362"),
        "base": (0.17, 0.58, "TLE + SGP4\nbaseline", "#f5f6f7", "#2f3338"),
    }
    rects = {}
    for k, (cx, cy, text, face, edge) in nodes.items():
        rects[k] = box(ax, cx - 0.12, cy - 0.07, 0.24, 0.14, text, face=face, edge=edge, fs=10.0, bold=(k == "pgrl"))
    arrow(ax, cr(rects["base"]), cl(rects["pgrl"]), rad=0.0)
    arrow(ax, cr(rects["pgrl"]), cl(rects["ctrl"]), rad=0.15)
    arrow(ax, cb(rects["ctrl"]), ct(rects["tx"]), rad=0.12)
    arrow(ax, cl(rects["tx"]), cr(rects["ev"]), rad=0.1)
    arrow(ax, ct(rects["ev"]), cb(rects["base"]), rad=0.12)
    ax.text(0.50, 0.50, "review artifact loop", transform=ax.transAxes, ha="center", va="center", fontsize=10.0, color="#5c6168")
    fig.savefig(OUTDIR / "candidate_b_closed_loop.pdf")
    plt.close(fig)


def save_swimlane():
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lanes = [
        ("Offline", 0.70, "#f4f7fb"),
        ("Online", 0.39, "#f8f8f8"),
        ("Hardware", 0.08, "#fbfaf6"),
    ]
    for title, y, face in lanes:
        ax.add_patch(
            FancyBboxPatch((0.03, y), 0.94, 0.22, boxstyle="round,pad=0.012,rounding_size=0.014",
                           linewidth=0.8, edgecolor="#cfd3d8", facecolor=face, transform=ax.transAxes)
        )
        ax.text(0.05, y + 0.18, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")
    b1 = box(ax, 0.12, 0.77, 0.17, 0.08, "TLE / ephemeris")
    b2 = box(ax, 0.39, 0.77, 0.18, 0.08, "SGP4 + residual labels")
    b3 = box(ax, 0.68, 0.75, 0.18, 0.12, "Bayesian PGRL\n+ calibration", face="#dfeaf6", edge="#466d93", bold=True)
    arrow(ax, cr(b1), cl(b2))
    arrow(ax, cr(b2), cl(b3))
    c1 = box(ax, 0.10, 0.46, 0.19, 0.08, "new TLE / pass query")
    c2 = box(ax, 0.38, 0.46, 0.20, 0.08, "PGRL residual + $\\sigma$", face="#eef3f8", edge="#6684a5")
    c3 = box(ax, 0.69, 0.44, 0.19, 0.12, "guard / TX timing /\npre-comp.", face="#eef3f8", edge="#6684a5")
    arrow(ax, cr(c1), cl(c2))
    arrow(ax, cr(c2), cl(c3))
    h1 = box(ax, 0.13, 0.16, 0.20, 0.08, "LR1121 + NUCLEO", face="#eef4ea", edge="#6c8362")
    h2 = box(ax, 0.43, 0.16, 0.17, 0.08, "USRP B210 IQ", face="#eef4ea", edge="#6c8362")
    h3 = box(ax, 0.72, 0.14, 0.16, 0.12, "TX ON/OFF\nsignal-detected", face="#eef4ea", edge="#6c8362")
    arrow(ax, cr(h1), cl(h2))
    arrow(ax, cr(h2), cl(h3))
    arrow(ax, cb(b3), ct(c2))
    arrow(ax, cb(c3), ct(h1))
    fig.savefig(OUTDIR / "candidate_c_swimlane.pdf")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    save_three_layer()
    save_closed_loop()
    save_swimlane()


if __name__ == "__main__":
    main()
