#!/usr/bin/env python3
"""Plot Stage 4 risk-aware guard-adaptation ablation (alpha sweep).

Source of values (do NOT change without re-running the experiment):
  docs/risk_aware_control_stage4_results.txt
Policy: guard_uncertain = guard_base + alpha * radial_sigma_m.
RMSE is invariant at 5.35 m and collision proxy is 0.000 for all alpha.
outage/success/reward are control PROXIES (not measured PER).
Output: paper/figures/fig3_risk_aware_ablation.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- exact values from docs/risk_aware_control_stage4_results.txt ---
ALPHA   = [0.00,  0.25,  0.50,  0.75,  1.00]
OUTAGE  = [0.050, 0.017, 0.004, 0.001, 0.000]
SUCCESS = [0.9500,0.9828,0.9961,0.9989,1.0000]
REWARD  = [0.9500,0.9690,0.9685,0.9575,0.9448]
NGUARD  = [1.000, 1.138, 1.276, 1.414, 1.552]
BEST_ALPHA = 0.25

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures",
                   "fig3_risk_aware_ablation.pdf")


def main():
    fig, ax1 = plt.subplots(figsize=(3.4, 2.5))
    # success proxy is reported exactly in the alpha-sweep table; omitted here
    # to keep the figure uncluttered.
    ax1.plot(ALPHA, OUTAGE, "o-", color="#d62728", label="Outage proxy")
    ax1.plot(ALPHA, REWARD, "s-", color="#1f77b4", label="Risk-adj. reward")
    ax1.set_xlabel(r"$\alpha$ (uncertainty guard scale)")
    ax1.set_ylabel("Proxy value")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(ALPHA, NGUARD, "x--", color="#7f7f7f", label="Norm. guard")
    ax2.set_ylabel("Normalized guard", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")

    ax1.axvline(BEST_ALPHA, color="gray", lw=1.0, ls=":")

    lines = ax1.get_lines()[:2] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
