#!/usr/bin/env python3
"""Sweep k for the uncertainty-aware guard policy and generate reviewer-facing outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller.energy_model import total_opportunity_energy
from controller.guard_band_policy import adaptive_guard_time, guard_overhead_fraction, missed_opportunity_probability
from paper_hardening_common import (
    PAPER_FIGURES_DIR,
    PAPER_TABLES_DIR,
    RESULTS_DIR,
    git_commit,
    make_tex_table,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def main() -> None:
    ks = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
    rng = np.random.default_rng(42)
    sigma_nominal = 0.016
    base_guard = 0.5
    pass_duration = 240.0

    rows = []
    for k in ks:
        guards = []
        misses = []
        energies = []
        for _ in range(5000):
            sigma = max(1e-6, sigma_nominal * (1.0 + 0.2 * rng.standard_normal()))
            guard = adaptive_guard_time(base_guard, sigma, k=k)
            guards.append(guard)
            misses.append(missed_opportunity_probability(sigma, guard))
            energies.append(total_opportunity_energy(guard, rx_on_s=2.0, tx_s=0.5)["total_j"])

        rows.append(
            {
                "k_sigma": k,
                "guard_overhead_percent": round(float(np.mean([guard_overhead_fraction(g, pass_duration) for g in guards])) * 100, 4),
                "missed_opportunity_rate": float(np.mean(misses)),
                "energy_per_opportunity_j": round(float(np.mean(energies)), 5),
                "status": "available",
                "notes": "Base guard follows the current evaluation definition (0.5 s lower bound + k·sigma).",
            }
        )

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "validation_type": "simulation",
        "source": [
            "controller/guard_band_policy.py",
            "controller/energy_model.py",
        ],
        "assumptions": {
            "base_guard_s": base_guard,
            "timing_sigma_s_nominal": sigma_nominal,
            "timing_sigma_jitter": "20% Gaussian multiplicative jitter",
            "pass_duration_s": pass_duration,
            "seed": 42,
        },
        "operating_point": "k=3.5",
    }

    out_json = RESULTS_DIR / "guard_k_sweep.json"
    out_csv = RESULTS_DIR / "guard_k_sweep.csv"
    out_tex = PAPER_TABLES_DIR / "guard_k_sweep_table.tex"
    out_fig = PAPER_FIGURES_DIR / "guard_k_sweep.pdf"

    write_json(out_json, {"metadata": metadata, "rows": rows})
    write_csv(out_csv, rows, rows[0].keys())

    tex_rows = [
        [
            f"{row['k_sigma']:.1f}",
            f"{row['guard_overhead_percent']:.4f}\\%",
            f"{row['missed_opportunity_rate']:.2e}",
            f"{row['energy_per_opportunity_j']:.5f}",
        ]
        for row in rows
    ]
    tex = make_tex_table(
        caption="Guard-$k$ Sweep for the Uncertainty-Aware Policy",
        label="tab:guard_k_sweep",
        headers=["$k$", "Guard OH", "Miss rate", "Energy (J)"],
        rows=tex_rows,
        note=(
            "Under the current evaluation definition, all $k$ values inherit a 0.5 s base guard. "
            "The sweep therefore mainly shows the small overhead/energy increase around the conservative operating point."
        ),
        colfmt="c c c c",
    )
    write_text(out_tex, tex)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, ax1 = plt.subplots(figsize=(3.35, 2.2))
    x = [row["k_sigma"] for row in rows]
    ax1.plot(x, [row["guard_overhead_percent"] for row in rows], marker="o", color="#1f77b4", label="Guard overhead")
    ax1.set_xlabel("$k$")
    ax1.set_ylabel("Guard overhead (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax1.axvline(3.5, color="0.4", linestyle="--", linewidth=0.9)
    ax1.text(3.52, max(row["guard_overhead_percent"] for row in rows) * 0.98, "operating point", fontsize=6.8, va="top", color="0.35")

    ax2 = ax1.twinx()
    miss_rates = [max(row["missed_opportunity_rate"], 1e-18) for row in rows]
    ax2.plot(x, miss_rates, marker="s", color="#d62728", label="Miss rate")
    ax2.set_yscale("log")
    ax2.set_ylabel("Missed opportunity rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)

    print(f"[guard-k] wrote {out_json}")
    print(f"[guard-k] wrote {out_csv}")
    print(f"[guard-k] wrote {out_tex}")
    print(f"[guard-k] wrote {out_fig}")


if __name__ == "__main__":
    main()
