#!/usr/bin/env python3
"""Analyze CW (C-command) tones: per-tone carrier vs commanded, de-biased.

This is an offline IQ analysis script. It does not transmit, does not call USRP,
and does not claim satellite validation.

The nominal center frequency must be explicitly provided by the user after
confirming the local frequency plan. No 868 MHz default is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "hardware/ota_iq")
from ota_common import load_iq, detect_bursts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline CW tone CFO analysis for conducted/lab IQ captures."
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory, e.g., hardware/ota_iq/runs/20260611_154458",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        required=True,
        help="Mode directories to analyze, e.g., no_compensation sgp4_only pgrl_corrected",
    )
    parser.add_argument(
        "--nominal-center-hz",
        type=float,
        required=True,
        help=(
            "Nominal local carrier center frequency in Hz. Must be explicitly "
            "provided after confirming local frequency plan. No default is used."
        ),
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=1_000_000.0,
        help="IQ sample rate in Hz. Default: 1e6.",
    )
    parser.add_argument(
        "--lo-offset-hz",
        type=float,
        default=200_000.0,
        help="RX LO offset from nominal center frequency in Hz. Default: 200 kHz.",
    )
    return parser.parse_args()


def tone_offsets(mode_dir: Path, fs: float, lo: float):
    iq_path = mode_dir / "capture_iq.fc32"
    sched_path = mode_dir / "burst_schedule.csv"

    if not iq_path.exists():
        raise FileNotFoundError(f"Missing IQ file: {iq_path}")
    if not sched_path.exists():
        raise FileNotFoundError(f"Missing schedule CSV: {sched_path}")

    iq = load_iq(iq_path)
    bursts = detect_bursts(iq, fs, nfft=1024, hop=256, snr_gate_db=6.0)
    sched = list(csv.DictReader(open(sched_path)))

    out = []
    for b in bursts:
        i0 = int(b.t_start_s * fs)
        i1 = int(b.t_end_s * fs)
        seg = iq[i0:i1]
        if seg.size < 4096:
            continue

        nfft = 1 << 20
        w = np.hanning(seg.size)
        sp = np.abs(np.fft.fftshift(np.fft.fft(seg * w, nfft)))
        fr = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))

        # Exclude DC/LO baseband region.
        mask = np.abs(fr) > 5e3
        k = int(np.argmax(sp * mask))

        # Measured carrier offset from nominal F0.
        meas_from_f0 = fr[k] + lo
        out.append((b.index, b.t_start_s, meas_from_f0))

    return out, sched


def main() -> int:
    args = parse_args()

    run = args.run
    modes = args.modes
    fs = float(args.sample_rate_hz)
    lo = float(args.lo_offset_hz)
    f0 = float(args.nominal_center_hz)

    if f0 <= 0:
        raise SystemExit(
            "Nominal center frequency must be explicitly provided after confirming "
            "local frequency plan. No default carrier is used."
        )

    if not run.exists():
        raise SystemExit(f"Run directory does not exist: {run}")

    allpairs = []
    per_mode = {}

    for mode in modes:
        mode_dir = run / mode
        if not mode_dir.exists():
            print(f"[WARN] mode directory missing, skip: {mode_dir}", file=sys.stderr)
            continue

        tones, sched = tone_offsets(mode_dir, fs, lo)
        rows = []

        for n, (idx, t_start, meas) in enumerate(tones):
            cmd = float(sched[n]["commanded_offset_hz"]) if n < len(sched) else 0.0
            rows.append([idx, t_start, meas, cmd])
            allpairs.append(meas - cmd)

        per_mode[mode] = rows

    bias = float(np.median(allpairs)) if allpairs else 0.0
    print(f"global_oscillator_bias_hz = {bias:.1f}  (from {len(allpairs)} tones across modes)")

    summary = {
        "global_oscillator_bias_hz": round(bias, 2),
        "grid_reference": "nominal_center_F0",
        "nominal_center_freq_hz": round(f0, 2),
        "evidence_type": "offline_conducted_or_lab_iq_analysis",
        "limitation": (
            "CW tone analysis is an RF diagnostic only. It is not satellite validation, "
            "not measured Doppler truth, and not PER/BER/CRC."
        ),
        "modes": {},
    }

    for mode, rows in per_mode.items():
        mode_dir = run / mode
        if not rows:
            print(f"[WARN] no valid tones for mode: {mode}")
            summary["modes"][mode] = {"n_tones": 0, "valid": False}
            continue

        csv_rows = []
        res_grid = []
        real_err = []

        for idx, t_start, meas, cmd in rows:
            residual_to_grid = meas - bias
            realization_error = meas - cmd - bias

            csv_rows.append({
                "burst_index": idx,
                "t_start_s": round(t_start, 4),
                "measured_offset_from_f0_hz": round(meas, 2),
                "commanded_offset_hz": round(cmd, 2),
                "residual_to_grid_hz": round(residual_to_grid, 2),
                "realization_error_hz": round(realization_error, 2),
            })
            res_grid.append(abs(residual_to_grid))
            real_err.append(abs(realization_error))

        out_csv = mode_dir / "cw_cfo_per_tone.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        rg = np.array(res_grid)
        re = np.array(real_err)

        stats = {
            "n_tones": len(rows),
            "valid": True,
            "median_abs_residual_to_grid_hz": round(float(np.median(rg)), 2),
            "p95_abs_residual_to_grid_hz": round(float(np.percentile(rg, 95)), 2),
            "max_abs_residual_to_grid_hz": round(float(np.max(rg)), 2),
            "median_abs_realization_error_hz": round(float(np.median(re)), 2),
            "p95_abs_realization_error_hz": round(float(np.percentile(re, 95)), 2),
            "max_abs_realization_error_hz": round(float(np.max(re)), 2),
        }
        summary["modes"][mode] = stats

        print(
            f"{mode}: n={stats['n_tones']}  "
            f"|residual_to_grid| med/p95/max = "
            f"{stats['median_abs_residual_to_grid_hz']}/"
            f"{stats['p95_abs_residual_to_grid_hz']}/"
            f"{stats['max_abs_residual_to_grid_hz']} Hz  | "
            f"|realization_err| med/p95 = "
            f"{stats['median_abs_realization_error_hz']}/"
            f"{stats['p95_abs_realization_error_hz']} Hz"
        )

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    summary["commit"] = commit
    summary["analyzed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_summary = run / "cw_cfo_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
