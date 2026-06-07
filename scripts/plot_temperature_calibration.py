#!/usr/bin/env python3
"""Plot Stage 3F post-hoc uncertainty temperature calibration sweep.

Source of values (do NOT change without re-running the experiment):
  docs/uncertainty_temperature_stage3f_results.txt
Deterministic mean head is frozen; RMSE is invariant at 5.35 m across all T.
Output: paper/figures/fig2_temperature_calibration.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- exact values from docs/uncertainty_temperature_stage3f_results.txt ---
T      = [0.6,   0.7,   0.8,   0.9,   1.0,   1.1,   1.2,   1.3,   1.4]
COV68  = [0.275, 0.389, 0.503, 0.606, 0.713, 0.783, 0.850, 0.893, 0.928]
COV95  = [0.601, 0.748, 0.845, 0.907, 0.947, 0.974, 0.988, 0.994, 0.996]
CALERR = [0.7569,0.4958,0.2852,0.1197,0.0331,0.1242,0.2053,0.2548,0.2909]
BEST_T = 1.0

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures",
                   "fig2_temperature_calibration.pdf")


def main():
    fig, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.plot(T, COV68, "o-", color="#1f77b4", label="Cov68")
    ax1.plot(T, COV95, "s-", color="#2ca02c", label="Cov95")
    ax1.axhline(0.68, ls=":", color="#1f77b4", lw=0.8)
    ax1.axhline(0.95, ls=":", color="#2ca02c", lw=0.8)
    ax1.set_xlabel("Temperature $T$")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(T, CALERR, "^--", color="#d62728", label="CalErr")
    ax2.set_ylabel("Calibration error", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.axvline(BEST_T, color="gray", lw=1.0, ls="-")
    ax1.annotate("best $T=1.0$", xy=(BEST_T, 0.10), xytext=(1.02, 0.18),
                 fontsize=7, color="gray")

    lines = ax1.get_lines()[:2] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
