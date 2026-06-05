#!/usr/bin/env python3
"""Generate hard-split / leakage-control summary from current configs and planned scaffolds."""

from __future__ import annotations

from pathlib import Path
import re

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
    exp1_text = (Path(__file__).resolve().parent.parent / "experiments" / "exp1_pgrl_prediction" / "config.yaml").read_text()
    exp6 = load_json(Path(__file__).resolve().parent.parent / "experiments" / "exp6_robustness" / "results.json")

    def _extract_int(key: str) -> int:
        match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9]+)\s*$", exp1_text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not extract {key} from exp1 config")
        return int(match.group(1))

    sat_count = _extract_int("num_satellites")
    pass_count = _extract_int("num_test_passes")
    horizon_hours = _extract_int("horizon_hours")

    rows = [
        {
            "split_name": "chronological_future_pass",
            "status": "configured",
            "prediction_horizon_min": horizon_hours * 60,
            "num_passes": pass_count,
            "num_satellites": sat_count,
            "sgp4_timing_rmse": "TODO",
            "pgrl_timing_rmse": "TODO",
            "sgp4_doppler_error": "TODO",
            "pgrl_doppler_error": "TODO",
            "notes": "Chronological train-before-test separation is described in the paper but no standalone result file exists yet.",
        },
        {
            "split_name": "held_out_satellite",
            "status": exp6["results"]["satellite_held_out_split"]["status"],
            "prediction_horizon_min": horizon_hours * 60,
            "num_passes": "TODO",
            "num_satellites": sat_count,
            "sgp4_timing_rmse": "TODO",
            "pgrl_timing_rmse": "TODO",
            "sgp4_doppler_error": "TODO",
            "pgrl_doppler_error": "TODO",
            "notes": "Held-out satellite sweep scaffold exists in exp6_robustness; no measured values are present.",
        },
        {
            "split_name": "tle_age_breakdown",
            "status": exp6["results"]["tle_age_perturbation"]["status"],
            "prediction_horizon_min": "0/1/3/7/14 day age bins",
            "num_passes": "TODO",
            "num_satellites": sat_count,
            "sgp4_timing_rmse": "TODO",
            "pgrl_timing_rmse": "TODO",
            "sgp4_doppler_error": "TODO",
            "pgrl_doppler_error": "TODO",
            "notes": "TLE-age control is configured but not yet evaluated in a saved results artifact.",
        },
    ]

    for horizon in [30, 60, 120, 360]:
        rows.append(
            {
                "split_name": f"prediction_horizon_{horizon}min",
                "status": "template",
                "prediction_horizon_min": horizon,
                "num_passes": "TODO",
                "num_satellites": sat_count,
                "sgp4_timing_rmse": "TODO",
                "pgrl_timing_rmse": "TODO",
                "sgp4_doppler_error": "TODO",
                "pgrl_doppler_error": "TODO",
                "notes": "Horizon-specific breakdown requested for reviewer defense; exact counts remain configurable in current scripts.",
            }
        )

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "validation_type": "planned / leakage-control scaffold",
        "source_files": [
            "experiments/exp1_pgrl_prediction/config.yaml",
            "experiments/exp6_robustness/results.json",
            "experiments/exp6_robustness/config.yaml",
        ],
        "leakage_controls": [
            "chronological train-before-test separation",
            "held-out-satellite testing where available",
            "identical TLE input for SGP4-only and PGRL-assisted runs",
            "explicit TLE-age and prediction-horizon breakdown scaffolds",
        ],
    }

    out_json = RESULTS_DIR / "hard_split_summary.json"
    out_csv = RESULTS_DIR / "hard_split_summary.csv"
    out_tex = PAPER_TABLES_DIR / "hard_split_table.tex"

    write_json(out_json, {"metadata": metadata, "rows": rows})
    write_csv(
        out_csv,
        rows,
        [
            "split_name",
            "status",
            "prediction_horizon_min",
            "num_passes",
            "num_satellites",
            "sgp4_timing_rmse",
            "pgrl_timing_rmse",
            "sgp4_doppler_error",
            "pgrl_doppler_error",
            "notes",
        ],
    )

    tex_rows = [
        [
            row["split_name"].replace("_", "\\_"),
            row["status"],
            row["prediction_horizon_min"],
            row["num_passes"],
            row["num_satellites"],
            row["pgrl_timing_rmse"],
        ]
        for row in rows
    ]
    tex = make_tex_table(
        caption="Leakage-Control and Hard-Split Evaluation Summary",
        label="tab:hard_split",
        headers=["Split", "Status", "Horizon", "Passes", "Sats", "PGRL timing"],
        rows=tex_rows,
        note=(
            "The current repository records the split policy and robustness scaffold, "
            "but not measured hard-split metrics. Missing values remain TODO by design."
        ),
        colfmt="p{2.1cm}p{0.9cm}p{1.2cm}p{0.8cm}p{0.6cm}p{1.0cm}",
    )
    write_text(out_tex, tex)

    print(f"[hard-split] wrote {out_json}")
    print(f"[hard-split] wrote {out_csv}")
    print(f"[hard-split] wrote {out_tex}")


if __name__ == "__main__":
    main()
