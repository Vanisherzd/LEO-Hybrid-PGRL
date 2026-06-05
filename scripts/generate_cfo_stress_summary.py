#!/usr/bin/env python3
"""Generate a conservative residual-CFO stress proxy from existing analytical models."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdr_hwil.evm_proxy import evm_percent, simulate_qpsk_with_cfo
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


def orthogonality_proxy(abs_cfo_hz: float) -> float:
    return max(0.0, 0.979 * math.exp(-((abs_cfo_hz / 1000.0) ** 2)))


def collision_proxy(abs_cfo_hz: float) -> float:
    bin_width_hz = 200e3 / 64.0
    return min(1.0, abs_cfo_hz / bin_width_hz * 0.5)


def main() -> None:
    offsets = [0, -300, 300, -500, 500, -1000, 1000, -5000, 5000]
    rows = []
    for offset in offsets:
        abs_offset = abs(offset)
        rx, ref = simulate_qpsk_with_cfo(snr_db=40.0, cfo_hz=float(offset), fs_hz=1e6, n_symbols=1000, seed=42)
        rows.append(
            {
                "cfo_offset_hz": offset,
                "abs_cfo_hz": abs_offset,
                "grid_orthogonality_proxy": round(orthogonality_proxy(abs_offset), 4),
                "collision_probability_proxy": round(collision_proxy(abs_offset), 4),
                "qpsk_evm_proxy_percent": round(evm_percent(rx, ref), 4),
                "validation_type": "proxy-simulation",
                "notes": "Analytical LR-FHSS-inspired grid proxy plus synthetic QPSK EVM proxy; not LR-FHSS PER or decoding.",
            }
        )

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "validation_type": "proxy-simulation",
        "source_files": [
            "experiments/exp3_lrfhss_grid_proxy/run.sh",
            "sdr_hwil/evm_proxy.py",
        ],
        "interpretation": (
            "This stress test links residual CFO to the sparse-hop/grid-alignment proxy and a synthetic RF-quality proxy. "
            "It is not a decoding or PER result."
        ),
    }

    out_json = RESULTS_DIR / "cfo_stress_summary.json"
    out_csv = RESULTS_DIR / "cfo_stress_summary.csv"
    out_tex = PAPER_TABLES_DIR / "cfo_stress_table.tex"
    out_fig = PAPER_FIGURES_DIR / "cfo_stress_proxy.pdf"

    write_json(out_json, {"metadata": metadata, "rows": rows})
    write_csv(out_csv, rows, rows[0].keys())

    tex_rows = [
        [
            f"{row['cfo_offset_hz']:+d}",
            f"{row['grid_orthogonality_proxy']:.3f}",
            f"{row['collision_probability_proxy']:.3f}",
            f"{row['qpsk_evm_proxy_percent']:.2f}\\%",
        ]
        for row in rows
    ]
    tex = make_tex_table(
        caption="Residual-CFO Stress Proxy",
        label="tab:cfo_stress",
        headers=["CFO (Hz)", "Grid orth.", "Coll. proxy", "QPSK EVM"],
        rows=tex_rows,
        note="All rows are proxy-simulation outputs. They should not be interpreted as LR-FHSS decoding or PER.",
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
    pos_rows = [row for row in rows if row["cfo_offset_hz"] >= 0]
    xs = [row["cfo_offset_hz"] for row in pos_rows]
    orth = [row["grid_orthogonality_proxy"] for row in pos_rows]
    coll = [row["collision_probability_proxy"] for row in pos_rows]
    fig, ax1 = plt.subplots(figsize=(3.35, 2.2))
    ax1.plot(xs, orth, marker="o", color="#1f77b4", label="Grid orthogonality")
    ax1.plot(xs, coll, marker="s", color="#d62728", label="Collision proxy")
    ax1.set_xlabel("Residual CFO (Hz)")
    ax1.set_ylabel("Proxy score")
    ax1.set_xscale("symlog", linthresh=300)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)

    print(f"[cfo-stress] wrote {out_json}")
    print(f"[cfo-stress] wrote {out_csv}")
    print(f"[cfo-stress] wrote {out_tex}")
    print(f"[cfo-stress] wrote {out_fig}")


if __name__ == "__main__":
    main()
