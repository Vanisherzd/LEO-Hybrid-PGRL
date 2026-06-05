#!/usr/bin/env python3
"""Generate a reviewer-facing ablation summary from existing repo artifacts."""

from __future__ import annotations

from pathlib import Path

from paper_hardening_common import (
    PAPER_TABLES_DIR,
    RESULTS_DIR,
    git_commit,
    load_json,
    make_tex_table,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def main() -> None:
    main_results = load_json(PAPER_TABLES_DIR / "main_results.json")
    ablation = load_json(PAPER_TABLES_DIR / "ablation_baselines.json")

    table2_rows = {row[0]: row for row in main_results["table_2_ablation"]["rows"]}
    guard_rows = {row[0]: row for row in main_results["table_3_guard_band"]["rows"]}
    ablation_rows = {row["method"]: row for row in ablation["rows"]}

    rows = [
        {
            "variant": "SGP4-only",
            "timing_rmse": ablation_rows["SGP4-only"]["timing_rmse"],
            "doppler_metric_hz": ablation_rows["SGP4-only"]["doppler_rmse_hz"],
            "nll": ablation_rows["SGP4-only"]["nll"],
            "ece": "TODO",
            "guard_overhead": guard_rows["SGP4 3$\\sigma$ ($\\sigma$=1.5 s)"][1],
            "missed_opportunity_rate": guard_rows["SGP4 3$\\sigma$ ($\\sigma$=1.5 s)"][2],
            "energy_per_opportunity_j": guard_rows["SGP4 3$\\sigma$ ($\\sigma$=1.5 s)"][3],
            "validation_type": "simulation",
            "status": "available",
            "notes": "Baseline propagated from paper/tables/main_results.json and ablation_baselines.json.",
        },
        {
            "variant": "SGP4 + deterministic residual MLP",
            "timing_rmse": ablation_rows["PGRL w/o uncertainty (SGP4 + deterministic residual)"]["timing_rmse"],
            "doppler_metric_hz": ablation_rows["PGRL w/o uncertainty (SGP4 + deterministic residual)"]["doppler_rmse_hz"],
            "nll": "TODO",
            "ece": "TODO",
            "guard_overhead": "TODO",
            "missed_opportunity_rate": "TODO",
            "energy_per_opportunity_j": "TODO",
            "validation_type": "simulation",
            "status": "partial",
            "notes": "Predictor-only deterministic residual variant is available; controller metrics are not separately reported.",
        },
        {
            "variant": "PGRL without physics regularization",
            "timing_rmse": table2_rows["PGRL (no physics loss)"][1],
            "doppler_metric_hz": table2_rows["PGRL (no physics loss)"][2],
            "nll": table2_rows["PGRL (no physics loss)"][3],
            "ece": "TODO",
            "guard_overhead": "TODO",
            "missed_opportunity_rate": "TODO",
            "energy_per_opportunity_j": "TODO",
            "validation_type": "simulation",
            "status": "partial",
            "notes": "No standalone controller sweep for the no-physics-loss predictor is present in the repo.",
        },
        {
            "variant": "PGRL without uncertainty-aware guard",
            "timing_rmse": ablation_rows["PGRL with uncertainty (PGRL full)"]["timing_rmse"],
            "doppler_metric_hz": ablation_rows["PGRL with uncertainty (PGRL full)"]["doppler_rmse_hz"],
            "nll": ablation_rows["PGRL with uncertainty (PGRL full)"]["nll"],
            "ece": ablation_rows["PGRL with uncertainty (PGRL full)"]["uncertainty_ece"],
            "guard_overhead": guard_rows["PGRL mean 3$\\sigma$ ($\\sigma$=0.2 s)"][1],
            "missed_opportunity_rate": guard_rows["PGRL mean 3$\\sigma$ ($\\sigma$=0.2 s)"][2],
            "energy_per_opportunity_j": guard_rows["PGRL mean 3$\\sigma$ ($\\sigma$=0.2 s)"][3],
            "validation_type": "trace-driven / simulation",
            "status": "available",
            "notes": "Uses the full predictor with the mean-only guard policy row already reported in Table IV.",
        },
        {
            "variant": "Full PGRL + uncertainty-aware control",
            "timing_rmse": ablation_rows["PGRL with uncertainty (PGRL full)"]["timing_rmse"],
            "doppler_metric_hz": ablation_rows["PGRL with uncertainty (PGRL full)"]["doppler_rmse_hz"],
            "nll": ablation_rows["PGRL with uncertainty (PGRL full)"]["nll"],
            "ece": ablation_rows["PGRL with uncertainty (PGRL full)"]["uncertainty_ece"],
            "guard_overhead": guard_rows["PGRL uncertainty-aware ($\\sigma$=16 ms)"][1],
            "missed_opportunity_rate": guard_rows["PGRL uncertainty-aware ($\\sigma$=16 ms)"][2],
            "energy_per_opportunity_j": guard_rows["PGRL uncertainty-aware ($\\sigma$=16 ms)"][3],
            "validation_type": "trace-driven / simulation",
            "status": "available",
            "notes": "Current paper operating point.",
        },
    ]

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "validation_labels": [
            "trace-driven",
            "simulation",
            "proxy-simulation",
            "hardware-signal-detected",
        ],
        "source_files": [
            "paper/tables/main_results.json",
            "paper/tables/ablation_baselines.json",
        ],
        "limitations": (
            "This summary does not fabricate missing controller metrics for variants "
            "that are only described at predictor level. Missing cells are marked TODO."
        ),
    }

    out_json = RESULTS_DIR / "ablation_summary.json"
    out_csv = RESULTS_DIR / "ablation_summary.csv"
    out_tex = PAPER_TABLES_DIR / "ablation_table.tex"

    write_json(out_json, {"metadata": metadata, "rows": rows})
    write_csv(
        out_csv,
        rows,
        [
            "variant",
            "timing_rmse",
            "doppler_metric_hz",
            "nll",
            "ece",
            "guard_overhead",
            "missed_opportunity_rate",
            "energy_per_opportunity_j",
            "validation_type",
            "status",
            "notes",
        ],
    )

    tex_rows = [
        [
            row["variant"],
            row["timing_rmse"],
            row["doppler_metric_hz"],
            row["guard_overhead"],
            row["missed_opportunity_rate"],
        ]
        for row in rows
    ]
    tex = make_tex_table(
        caption="Ablation Summary for Predictor and Controller Variants",
        label="tab:ablation_hardening",
        headers=["Variant", "Timing RMSE", "Doppler", "Guard OH", "Miss rate"],
        rows=tex_rows,
        note=(
            "Rows reuse existing repository summary artifacts only. Missing per-variant "
            "controller numbers are marked TODO rather than inferred."
        ),
        colfmt="p{2.45cm}p{1.05cm}p{1.2cm}p{0.9cm}p{0.9cm}",
    )
    write_text(out_tex, tex)

    print(f"[ablation] wrote {out_json}")
    print(f"[ablation] wrote {out_csv}")
    print(f"[ablation] wrote {out_tex}")


if __name__ == "__main__":
    main()
