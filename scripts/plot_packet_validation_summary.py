#!/usr/bin/env python3
"""Plot packet-validation summaries (mock/decoded runs only).

Consumes ONLY packet_validation_summary.{json,csv} produced by the framework.
Does NOT fabricate hardware results: if a run has no decoding (PER null), it is
skipped with a printed notice rather than plotted as a delivered result.

Usage:
  python scripts/plot_packet_validation_summary.py RUN_DIR [RUN_DIR ...] \
         --out paper/figures/fig6_packet_validation.pdf
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(run_dir):
    p = os.path.join(run_dir, "packet_validation_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="paper/figures/fig6_packet_validation.pdf")
    args = ap.parse_args(argv)

    runs = []
    for d in args.run_dirs:
        s = load_summary(d)
        if s is None:
            print(f"[skip] no summary in {d}")
            continue
        if not s.get("decoding_available") or s.get("packet_error_rate") is None:
            print(f"[skip] {s.get('run_id')}: PER unavailable (no decoding)")
            continue
        runs.append(s)

    if not runs:
        print("No decoded runs to plot (PER unavailable for all). "
              "Nothing written.")
        return

    ids = [r["run_id"] for r in runs]
    pdr = [r["packet_delivery_ratio"] for r in runs]
    per = [r["packet_error_rate"] for r in runs]
    miss = [r.get("missing_seq_count", 0) for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    x = range(len(runs))
    ax1.plot(x, pdr, "o-", color="#2ca02c", label="PDR")
    ax1.plot(x, per, "s-", color="#d62728", label="PER")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(ids, rotation=30, ha="right",
                                                 fontsize=6)
    ax1.set_ylabel("Ratio"); ax1.set_ylim(0, 1.05); ax1.legend(fontsize=7)
    ax1.set_title("PDR / PER over runs", fontsize=8)

    ax2.bar(list(x), miss, color="#1f77b4")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(ids, rotation=30, ha="right",
                                                 fontsize=6)
    ax2.set_ylabel("Missing packets"); ax2.set_title("Missing by run", fontsize=8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
