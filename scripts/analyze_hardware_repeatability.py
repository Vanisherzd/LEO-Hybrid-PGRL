#!/usr/bin/env python3
"""Summarize current LR1121/USRP hardware repeatability and negative controls."""

from __future__ import annotations

import math
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


REPO_ROOT = Path(__file__).resolve().parent.parent


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def detection_rate(values: list[bool]) -> str:
    if not values:
        return "TODO"
    return f"{sum(bool(v) for v in values)}/{len(values)}"


def main() -> None:
    repeatability = load_json(REPO_ROOT / "hardware" / "artifacts" / "lr1121_signal_detected_repeatability_20260604" / "repeatability_summary.json")
    wrong_freq = load_json(REPO_ROOT / "hardware" / "captures" / "auto_sweep_20260603_231940" / "sweep_summary.json")

    onoff_deltas = [run["on_off_delta_db"] for run in repeatability["runs"]]
    onoff_detect = [run["signal_detected"] for run in repeatability["runs"]]
    onoff_mean, onoff_std = mean_std(onoff_deltas)

    txoff_scores = []
    txoff_detect = []
    for run_dir in ["run1_000358", "run2_011203", "run3_011519"]:
        off = load_json(
            REPO_ROOT
            / "hardware"
            / "artifacts"
            / "lr1121_signal_detected_repeatability_20260604"
            / run_dir
            / f"{run_dir.split('_')[0]}_off_analysis.json"
        )
        txoff_scores.append(off.get("lr_fhss_candidate_score"))
        txoff_detect.append(bool(off.get("signal_detected", False)))

    wrong_captures = [
        c
        for c in wrong_freq["captures"]
        if c["frequency_hz"] in (915000000, 923000000)
    ]
    wrong_detect = [bool(c.get("signal_detected", False)) for c in wrong_captures]

    idle_controls = [
        c
        for c in wrong_freq["captures"]
        if c["frequency_hz"] == 923000000 and c.get("uart_packet_sent_count", 0) == 0
    ]
    idle_detect = [bool(c.get("signal_detected", False)) for c in idle_controls]

    rows = [
        {
            "condition": "TX ON/OFF",
            "runs": len(onoff_deltas),
            "mean_on_off_delta_db": round(onoff_mean, 2),
            "std_db": round(onoff_std, 2),
            "detection_rate": detection_rate(onoff_detect),
            "validation_type": "hardware-signal-detected",
            "notes": "Curated 868 MHz repeatability artifact; all three runs report signal_detected=true.",
        },
        {
            "condition": "TX-OFF only",
            "runs": len(txoff_detect),
            "mean_on_off_delta_db": "N/A",
            "std_db": "N/A",
            "detection_rate": detection_rate(txoff_detect),
            "validation_type": "hardware-signal-detected",
            "notes": f"Reference captures only. Mean TX-OFF candidate score = {sum(txoff_scores)/len(txoff_scores):.3f}.",
        },
        {
            "condition": "wrong frequency",
            "runs": len(wrong_captures),
            "mean_on_off_delta_db": "N/A",
            "std_db": "N/A",
            "detection_rate": detection_rate(wrong_detect),
            "validation_type": "hardware negative control",
            "notes": "915/923 MHz pre-signal-detected sweep entries; no LR1121 868 MHz detection expected.",
        },
        {
            "condition": "idle firmware / UART-no-TX",
            "runs": len(idle_controls),
            "mean_on_off_delta_db": "N/A",
            "std_db": "N/A",
            "detection_rate": detection_rate(idle_detect),
            "validation_type": "hardware negative control",
            "notes": "Subset of wrong-frequency sweep with uart_packet_sent_count=0; retained as a conservative no-TX control.",
        },
    ]

    metadata = {
        "generated_at": utc_now(),
        "commit": git_commit(),
        "source_files": [
            "hardware/artifacts/lr1121_signal_detected_repeatability_20260604/repeatability_summary.json",
            "hardware/artifacts/lr1121_signal_detected_repeatability_20260604/run*/run*_off_analysis.json",
            "hardware/captures/auto_sweep_20260603_231940/sweep_summary.json",
        ],
        "claim_boundary": (
            "All rows remain IQ-level RF-path evidence only. None implies LR-FHSS decoding, PER, or gateway interoperability."
        ),
    }

    out_json = RESULTS_DIR / "hardware_repeatability_summary.json"
    out_csv = RESULTS_DIR / "hardware_repeatability_summary.csv"
    out_tex = PAPER_TABLES_DIR / "hardware_repeatability_table.tex"

    write_json(out_json, {"metadata": metadata, "rows": rows})
    write_csv(
        out_csv,
        rows,
        [
            "condition",
            "runs",
            "mean_on_off_delta_db",
            "std_db",
            "detection_rate",
            "validation_type",
            "notes",
        ],
    )

    tex_rows = [
        [
            row["condition"],
            row["runs"],
            row["mean_on_off_delta_db"],
            row["std_db"],
            row["detection_rate"],
        ]
        for row in rows
    ]
    tex = make_tex_table(
        caption="Hardware Repeatability and Negative-Control Summary",
        label="tab:hw_repeatability",
        headers=["Condition", "Runs", "Mean delta", "Std", "Detection rate"],
        rows=tex_rows,
        note="Negative-control rows are included only as RF-path sanity checks. They do not extend the claim boundary beyond IQ-level signal detection.",
        colfmt="p{1.8cm}c c c c",
    )
    write_text(out_tex, tex)

    print(f"[hardware] wrote {out_json}")
    print(f"[hardware] wrote {out_csv}")
    print(f"[hardware] wrote {out_tex}")


if __name__ == "__main__":
    main()
