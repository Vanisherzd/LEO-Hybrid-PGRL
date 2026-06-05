#!/usr/bin/env python3
"""Generate hard-split / leakage-control summary from current repo-backed artifacts."""

from __future__ import annotations

from pathlib import Path
import re

from paper_hardening_common import (
    REPO_ROOT,
    RESULTS_DIR,
    git_commit,
    load_json,
    make_tex_table,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


DOCS_SCAFFOLD_DIR = REPO_ROOT / "docs" / "reviewer_scaffolds"
NOMINAL_SPLIT_CANDIDATE = DOCS_SCAFFOLD_DIR / "nominal_split_summary_candidate.tex"


def main() -> None:
    exp1_text = (Path(__file__).resolve().parent.parent / "experiments" / "exp1_pgrl_prediction" / "config.yaml").read_text()
    summary_script_text = (Path(__file__).resolve().parent.parent / "experiments" / "summary_table.py").read_text()
    exp6 = load_json(Path(__file__).resolve().parent.parent / "experiments" / "exp6_robustness" / "results.json")
    main_results = load_json(Path(__file__).resolve().parent.parent / "paper" / "tables" / "main_results.json")

    def _extract_int(key: str) -> int:
        match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9]+)\s*$", exp1_text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not extract {key} from exp1 config")
        return int(match.group(1))

    def _extract_float(name: str) -> float:
        match = re.search(rf"^{re.escape(name)}\s*=\s*([0-9.]+)", summary_script_text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not extract {name} from experiments/summary_table.py")
        return float(match.group(1))

    sat_count = _extract_int("num_satellites")
    pass_count = _extract_int("num_test_passes")
    horizon_hours = _extract_int("horizon_hours")
    horizon_min = horizon_hours * 60

    sgp4_timing_rmse_s = _extract_float("SGP4_TIMING_RMSE_S")
    pgrl_timing_rmse_s = _extract_float("PGRL_TIMING_RMSE_MS") / 1000.0
    sgp4_doppler_rmse_hz = _extract_float("SGP4_DOPPLER_RMSE_HZ")
    pgrl_doppler_rmse_hz = _extract_float("PGRL_DOPPLER_RMSE_HZ")

    trace_rows = [
        row
        for row in main_results["table_1_main_results"]["rows"]
        if row[0] == "PGRL predictor" and row[4] == "Trace-driven"
    ]
    if len(trace_rows) < 2:
        raise RuntimeError("main_results.json does not contain the expected trace-driven PGRL rows")

    rows = [
        {
            "split_name": "chronological_future_pass",
            "status": "measured",
            "prediction_horizon_min": horizon_min,
            "num_passes": pass_count,
            "num_satellites": sat_count,
            "sgp4_timing_rmse": round(sgp4_timing_rmse_s, 4),
            "pgrl_timing_rmse": round(pgrl_timing_rmse_s, 4),
            "sgp4_doppler_error": round(sgp4_doppler_rmse_hz, 1),
            "pgrl_doppler_error": round(pgrl_doppler_rmse_hz, 1),
            "notes": (
                "Repo-backed nominal trace-driven evaluation reused from experiments/summary_table.py "
                "and paper/tables/main_results.json for the configured 100-pass, 20-satellite, 6 h test horizon. "
                "Chronological train-before-test separation is documented, but per-pass split membership is not stored separately."
            ),
        },
        {
            "split_name": "held_out_satellite",
            "status": "unsupported",
            "prediction_horizon_min": horizon_min,
            "num_passes": "",
            "num_satellites": "",
            "sgp4_timing_rmse": "",
            "pgrl_timing_rmse": "",
            "sgp4_doppler_error": "",
            "pgrl_doppler_error": "",
            "notes": (
                "Unsupported: experiments/exp6_robustness marks satellite_held_out_split as planned, "
                "the held_out_satellites list is empty, and no saved per-satellite prediction residual artifact exists."
            ),
        },
        {
            "split_name": "tle_age_breakdown",
            "status": "unsupported",
            "prediction_horizon_min": "0/1/3/7/14 day age bins",
            "num_passes": "",
            "num_satellites": "",
            "sgp4_timing_rmse": "",
            "pgrl_timing_rmse": "",
            "sgp4_doppler_error": "",
            "pgrl_doppler_error": "",
            "notes": (
                "Unsupported: experiments/exp6_robustness/config.yaml defines the TLE-age bins, "
                "but experiments/exp6_robustness/results.json contains no evaluated metrics and the repo has no saved per-age prediction outputs."
            ),
        },
    ]

    for horizon in [30, 60, 120, 360]:
        if horizon == horizon_min:
            rows.append(
                {
                    "split_name": f"prediction_horizon_{horizon}min",
                    "status": "measured",
                    "prediction_horizon_min": horizon,
                    "num_passes": pass_count,
                    "num_satellites": sat_count,
                    "sgp4_timing_rmse": round(sgp4_timing_rmse_s, 4),
                    "pgrl_timing_rmse": round(pgrl_timing_rmse_s, 4),
                    "sgp4_doppler_error": round(sgp4_doppler_rmse_hz, 1),
                    "pgrl_doppler_error": round(pgrl_doppler_rmse_hz, 1),
                    "notes": (
                        "Repo-backed nominal 6 h evaluation reused as the only materialized horizon-specific artifact. "
                        "This row reflects the configured 360 min horizon, not a separate multi-horizon sweep."
                    ),
                }
            )
            continue

        rows.append(
            {
                "split_name": f"prediction_horizon_{horizon}min",
                "status": "unsupported",
                "prediction_horizon_min": horizon,
                "num_passes": "",
                "num_satellites": "",
                "sgp4_timing_rmse": "",
                "pgrl_timing_rmse": "",
                "sgp4_doppler_error": "",
                "pgrl_doppler_error": "",
                "notes": (
                    f"Unsupported: the repo stores only the nominal {horizon_min} min aggregate evaluation; "
                    f"no saved {horizon} min breakdown artifact or per-pass prediction export is present."
                ),
            }
        )

    measured_rows = [row for row in rows if row["status"] == "measured"]
    unsupported_rows = [row["split_name"] for row in rows if row["status"] != "measured"]

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "validation_type": "partial measured / partial unsupported",
        "source_files": [
            "experiments/summary_table.py",
            "paper/tables/main_results.json",
            "experiments/exp1_pgrl_prediction/config.yaml",
            "experiments/exp6_robustness/results.json",
            "experiments/exp6_robustness/config.yaml",
        ],
        "leakage_controls": [
            "chronological train-before-test separation",
            "held-out-satellite testing where available",
            "identical TLE input for SGP4-only and PGRL-assisted runs",
            "explicit TLE-age and prediction-horizon breakdown when saved artifacts exist",
        ],
        "measured_row_count": len(measured_rows),
        "unsupported_rows": unsupported_rows,
    }

    out_json = RESULTS_DIR / "hard_split_summary.json"
    out_csv = RESULTS_DIR / "hard_split_summary.csv"

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

    if len(measured_rows) >= 2:
        candidate_rows = [
            [
                row["split_name"].replace("_", "\\_"),
                row["prediction_horizon_min"],
                row["num_passes"],
                row["num_satellites"],
                row["sgp4_timing_rmse"],
                row["pgrl_timing_rmse"],
                row["sgp4_doppler_error"],
                row["pgrl_doppler_error"],
            ]
            for row in measured_rows
        ]
        candidate_tex = make_tex_table(
            caption="Candidate Nominal Split Summary for Reviewer Response",
            label="tab:hard_split_candidate",
            headers=["Split", "Horizon", "Passes", "Sats", "SGP4 t", "PGRL t", "SGP4 f", "PGRL f"],
            rows=candidate_rows,
            note=(
                "Only repo-backed measured rows are shown here. Held-out-satellite, TLE-age, "
                "and shorter-horizon breakdowns remain unsupported because the current repository does not "
                "materialize per-split prediction artifacts for those cases."
            ),
            colfmt="p{2.2cm}p{0.8cm}p{0.7cm}p{0.6cm}p{0.7cm}p{0.7cm}p{0.8cm}p{0.8cm}",
        )
        candidate_tex = (
            "% NOTE:\n"
            "% This is not a complete hard-split robustness sweep.\n"
            "% It records the repository-backed nominal chronological/360-min aggregate\n"
            "% and unsupported split reasons. Do not include this table in the paper\n"
            "% until held-out-satellite, TLE-age, and shorter-horizon metrics are measured.\n\n"
            f"{candidate_tex}"
        )
        write_text(NOMINAL_SPLIT_CANDIDATE, candidate_tex)

    print(f"[hard-split] wrote {out_json}")
    print(f"[hard-split] wrote {out_csv}")
    if len(measured_rows) >= 2:
        print(f"[hard-split] wrote {NOMINAL_SPLIT_CANDIDATE}")


if __name__ == "__main__":
    main()
