#!/usr/bin/env python3
"""
analyze_cfo_residual.py
=======================
Estimate residual carrier-frequency-offset (CFO) per burst from a real OTA IQ
capture under a programmed-Doppler replay.

For each detected burst the analyzer estimates the carrier offset from the
nominal channel center F0 (parabolic-interpolated FFT peak, DC/LO excluded) and
reports the residual CFO relative to the nearest LR-FHSS grid point:

    residual_cfo_hz = peak_offset_hz - nearest_grid(peak_offset_hz)

If grid_spacing_hz is null (not yet verified from the LR1121 config), the grid
reference collapses to F0 and residual_cfo_hz = peak_offset_hz; this is recorded
in the summary as grid_reference="nominal_center".

Where a burst_schedule.csv exists, the measured offset is paired with the
commanded_offset_hz (= expected residual) as a realization-error cross-check.

SCOPE: short-range room OTA / near-field IQ-level proxy. NOT PER / decoding /
CRC / receiver validation. See docs/ota_iq_validation_scope.md.

Outputs (into --out-dir):
  cfo_residual_timeseries.csv
  cfo_residual_summary.json

Usage
-----
  uv run python hardware/ota_iq/analyze_cfo_residual.py \
      --run-dir hardware/ota_iq/runs/pgrl_001 \
      --config  hardware/ota_iq/configs/replay_pgrl_corrected.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ota_common import (  # noqa: E402
    detect_bursts, git_commit, load_config, load_iq, nearest_grid_hz, write_json,
)


def _find_iq(run_dir: Path) -> Path:
    for name in ("capture_iq.npy", "capture_iq.fc32", "capture_iq.cfile"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No capture_iq.* in {run_dir}. Capture real IQ first; this analyzer "
        f"never synthesises samples."
    )


def _load_schedule(run_dir: Path) -> list[dict]:
    p = run_dir / "burst_schedule.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-burst residual CFO from OTA IQ")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--snr-gate-db", type=float, default=6.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config)
    fs = cfg["sample_rate_hz"]
    grid = cfg.get("grid_spacing_hz")
    lo_off = cfg.get("lo_offset_hz") or 0.0  # baseband peak + lo_off = offset from F0
    grid_ref = "lr_fhss_grid" if grid else "nominal_center"

    iq_path = _find_iq(run_dir)
    iq = load_iq(iq_path)
    print(f"[cfo] loaded {iq.size} samples from {iq_path.name}  fs={fs/1e6:.3f} MS/s")

    bursts = detect_bursts(iq, fs, nfft=args.nfft, hop=args.hop,
                           snr_gate_db=args.snr_gate_db)
    print(f"[cfo] detected {len(bursts)} bursts (snr_gate={args.snr_gate_db} dB)")

    # --- no-burst path: write stub summary, exit 0 (TX-OFF / noise-floor check) ---
    if not bursts:
        stub_fields = [
            "burst_index", "t_start_s", "t_end_s", "peak_offset_hz",
            "nearest_grid_hz", "residual_cfo_hz", "abs_residual_cfo_hz",
            "norm_abs_residual_cfo_grid", "snr_db",
            "commanded_offset_hz", "realization_error_hz",
        ]
        ts_csv = run_dir / "cfo_residual_timeseries.csv"
        with open(ts_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=stub_fields, extrasaction="ignore")
            w.writeheader()   # header-only; no data rows

        summary = dict(
            kind="cfo_residual_summary",
            mode=cfg.get("mode"),
            compensation_mode=cfg.get("compensation_mode"),
            grid_reference=grid_ref,
            grid_spacing_hz=grid,
            lo_offset_hz=lo_off,
            n_bursts_detected=0,
            status="no_bursts_detected",
            median_abs_residual_cfo_hz=None,
            p95_abs_residual_cfo_hz=None,
            max_abs_residual_cfo_hz=None,
            mean_abs_residual_cfo_hz=None,
            median_norm_abs_residual_cfo_grid=None,
            p95_norm_abs_residual_cfo_grid=None,
            max_norm_abs_residual_cfo_grid=None,
            mean_norm_abs_residual_cfo_grid=None,
            sample_rate_hz=fs,
            nfft=args.nfft,
            hop=args.hop,
            snr_gate_db=args.snr_gate_db,
            iq_source=iq_path.name,
            measurement_type="short_range_ota_iq",
            validation_scope="short_range_ota_iq_proxy",
            commit=git_commit(),
            analyzed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        write_json(run_dir / "cfo_residual_summary.json", summary)
        print("[cfo] no bursts detected; stub summary written (TX-OFF / noise-floor check)")
        print(f"[cfo] timeseries → {ts_csv}")
        print(f"[cfo] summary    → {run_dir / 'cfo_residual_summary.json'}")
        return 0

    schedule = _load_schedule(run_dir)

    rows = []
    abs_res = []
    for b in bursts:
        offset_f0 = b.peak_offset_hz + lo_off   # carrier offset from F0
        nearest = nearest_grid_hz(offset_f0, grid)
        residual = offset_f0 - nearest
        commanded = ""
        realiz_err = ""
        if b.index < len(schedule):
            try:
                commanded = float(schedule[b.index]["commanded_offset_hz"])
                realiz_err = offset_f0 - commanded
            except (KeyError, ValueError):
                pass
        rows.append(dict(
            burst_index=b.index,
            t_start_s=round(b.t_start_s, 4),
            t_end_s=round(b.t_end_s, 4),
            peak_offset_hz=round(offset_f0, 3),
            nearest_grid_hz=round(nearest, 3),
            residual_cfo_hz=round(residual, 3),
            abs_residual_cfo_hz=round(abs(residual), 3),
            norm_abs_residual_cfo_grid=(
                round(abs(residual) / grid, 6) if grid else ""
            ),
            snr_db=round(b.snr_db, 2),
            commanded_offset_hz=("" if commanded == "" else round(commanded, 3)),
            realization_error_hz=("" if realiz_err == "" else round(realiz_err, 3)),
        ))
        abs_res.append(abs(residual))

    ts_csv = run_dir / "cfo_residual_timeseries.csv"
    with open(ts_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    abs_res = np.asarray(abs_res)
    summary = dict(
        kind="cfo_residual_summary",
        mode=cfg.get("mode"),
        compensation_mode=cfg.get("compensation_mode"),
        grid_reference=grid_ref,
        grid_spacing_hz=grid,
        lo_offset_hz=lo_off,
        n_bursts_detected=len(bursts),
        median_abs_residual_cfo_hz=round(float(np.median(abs_res)), 3),
        p95_abs_residual_cfo_hz=round(float(np.percentile(abs_res, 95)), 3),
        max_abs_residual_cfo_hz=round(float(np.max(abs_res)), 3),
        mean_abs_residual_cfo_hz=round(float(np.mean(abs_res)), 3),
        median_norm_abs_residual_cfo_grid=(
            round(float(np.median(abs_res) / grid), 6) if grid else None
        ),
        p95_norm_abs_residual_cfo_grid=(
            round(float(np.percentile(abs_res, 95) / grid), 6) if grid else None
        ),
        max_norm_abs_residual_cfo_grid=(
            round(float(np.max(abs_res) / grid), 6) if grid else None
        ),
        mean_norm_abs_residual_cfo_grid=(
            round(float(np.mean(abs_res) / grid), 6) if grid else None
        ),
        sample_rate_hz=fs,
        nfft=args.nfft,
        hop=args.hop,
        snr_gate_db=args.snr_gate_db,
        iq_source=iq_path.name,
        measurement_type="short_range_ota_iq",  # NOT conducted
        validation_scope="short_range_ota_iq_proxy",  # NOT PER / decoding
        commit=git_commit(),
        analyzed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(run_dir / "cfo_residual_summary.json", summary)

    print(f"[cfo] median|res|={summary['median_abs_residual_cfo_hz']} Hz  "
          f"p95={summary['p95_abs_residual_cfo_hz']} Hz  "
          f"max={summary['max_abs_residual_cfo_hz']} Hz  "
          f"(grid_ref={grid_ref})")
    print(f"[cfo] timeseries → {ts_csv}")
    print(f"[cfo] summary    → {run_dir / 'cfo_residual_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
