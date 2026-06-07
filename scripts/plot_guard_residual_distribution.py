#!/usr/bin/env python3
"""Plot outage proxy vs guard margin (operating curve) for Stage 4.

NOTE ON SCOPE: per-sample residual errors are NOT exported in the repository
(they require the trained checkpoint, which lives on the training machine), so
this figure is NOT a per-sample residual histogram. It plots the *measured
operating points* of the alpha sweep --- outage proxy as a function of the
guard margin in metres --- which is fully backed by:
  docs/risk_aware_control_stage4_results.txt
Vertical markers: deterministic guard (alpha=0) = 9.56 m, and the
uncertainty-aware best-reward guard (alpha=0.25) = 10.88 m.
Output: paper/figures/fig4_guard_residual_distribution.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- exact values from docs/risk_aware_control_stage4_results.txt ---
GUARD_M = [9.56, 10.88, 12.20, 13.51, 14.83]   # metres, per alpha
OUTAGE  = [0.050, 0.017, 0.004, 0.001, 0.000]
DET_GUARD = 9.56     # alpha=0.00
UNC_GUARD = 10.88    # alpha=0.25 (best reward)

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures",
                   "fig4_guard_residual_distribution.pdf")


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(GUARD_M, [100 * o for o in OUTAGE], "o-", color="#d62728")
    ax.set_xlabel("Guard margin (m)")
    ax.set_ylabel("Outage proxy (%)")
    ax.set_ylim(-0.3, 6)

    ax.axvline(DET_GUARD, color="#7f7f7f", ls="--", lw=1.0)
    ax.annotate("deterministic\nguard 9.56 m\n(outage 5.0%)",
                xy=(DET_GUARD, 5.0), xytext=(9.7, 4.0), fontsize=6.5, color="#7f7f7f")
    ax.axvline(UNC_GUARD, color="#1f77b4", ls="-", lw=1.0)
    ax.annotate(r"uncertainty-aware ($\alpha=0.25$)" + "\nguard 10.88 m\n(outage 1.7%)",
                xy=(UNC_GUARD, 1.7), xytext=(11.2, 2.4), fontsize=6.5, color="#1f77b4")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
