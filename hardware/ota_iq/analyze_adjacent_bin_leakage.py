#!/usr/bin/env python3
"""
analyze_adjacent_bin_leakage.py
===============================
Per-burst adjacent-bin leakage ratio (ABLR) from a real OTA IQ capture.

For each detected burst, power is integrated in the target LR-FHSS grid bin
(nearest grid point to the measured carrier) and in the two immediately adjacent
grid bins (target ± grid_spacing). The leakage ratio is

    ABLR    = P_adjacent / P_target
    ABLR_dB = 10 * log10(P_adjacent / P_target)

Lower (more negative) ABLR_dB = tighter grid confinement. SGP4-only and
PGRL-corrected runs are compared by running this analyzer on each run dir.

REQUIRES grid_spacing_hz to be set from the VERIFIED LR1121 LR-FHSS configuration
(datasheet/firmware). If it is null the analyzer refuses to run rather than
assume a spacing. Do NOT hard-code 137 Hz; use the configured LR-FHSS grid.

SCOPE: short-range room OTA / near-field IQ-level proxy. NOT PER / decoding /
CRC / receiver validation. See docs/ota_iq_validation_scope.md.

Outputs (into --run-dir):
  ablr_per_burst.csv
  ablr_summary.json

Usage
-----
  uv run python hardware/ota_iq/analyze_adjacent_bin_leakage.py \
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
    band_power, burst_spectrum, detect_bursts, git_commit, load_config,
    load_iq, nearest_grid_hz, write_json,
)


def _find_iq(run_dir: Path) -> Path:
    for name in ("capture_iq.npy", "capture_iq.fc32", "capture_iq.cfile"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No capture_iq.* in {run_dir}. Capture real IQ first.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-burst adjacent-bin leakage from OTA IQ")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--snr-gate-db", type=float, default=6.0)
    ap.add_argument("--half-bw-frac", type=float, default=0.5,
                    help="integration half-bandwidth as a fraction of grid spacing")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config)
    fs = cfg["sample_rate_hz"]
    grid = cfg.get("grid_spacing_hz")
    lo_off = cfg.get("lo_offset_hz") or 0.0   # baseband freq + lo_off = offset from F0

    if not grid:
        raise SystemExit(
            "ERROR: grid_spacing_hz is null. Set it ONLY from the verified LR1121 "
            "LR-FHSS configuration before running ABLR (do NOT assume / hard-code). "
            "Adjacent-bin leakage is undefined without a verified grid spacing."
        )

    half_bw = args.half_bw_frac * grid
    bin_res = fs / args.nfft
    if half_bw < bin_res:
        print(f"[ablr] WARNING half_bw={half_bw:.1f} Hz < FFT bin res {bin_res:.1f} Hz; "
              f"reduce nfft or note coarse resolution.")

    iq_path = _find_iq(run_dir)
    iq = load_iq(iq_path)
    print(f"[ablr] loaded {iq.size} samples  grid={grid:.1f} Hz  half_bw={half_bw:.1f} Hz")

    bursts = detect_bursts(iq, fs, nfft=args.nfft, hop=args.hop,
                           snr_gate_db=args.snr_gate_db)
    print(f"[ablr] detected {len(bursts)} bursts")
    if not bursts:
        raise SystemExit("ERROR: no bursts detected; no ABLR emitted.")

    rows = []
    ablr_db_all = []
    for b in bursts:
        spec, f_hz = burst_spectrum(iq, b, fs, nfft=args.nfft, hop=args.hop)
        offset_f0 = b.peak_offset_hz + lo_off       # carrier offset from F0
        g0 = nearest_grid_hz(offset_f0, grid)        # target grid bin (F0-referenced)
        g0_bb = g0 - lo_off                          # back to baseband for band_power
        p_target = band_power(spec, f_hz, g0_bb, half_bw)
        p_lo = band_power(spec, f_hz, g0_bb - grid, half_bw)
        p_hi = band_power(spec, f_hz, g0_bb + grid, half_bw)
        p_adj = p_lo + p_hi
        ratio = p_adj / (p_target + 1e-20)
        ablr_db = 10.0 * np.log10(ratio + 1e-20)
        rows.append(dict(
            burst_index=b.index,
            t_start_s=round(b.t_start_s, 4),
            target_grid_hz=round(g0, 3),
            p_target=p_target,
            p_adjacent=p_adj,
            ablr=round(float(ratio), 6),
            ablr_db=round(float(ablr_db), 3),
            snr_db=round(b.snr_db, 2),
        ))
        ablr_db_all.append(ablr_db)

    per_csv = run_dir / "ablr_per_burst.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ablr_db_all = np.asarray(ablr_db_all)
    summary = dict(
        kind="ablr_summary",
        mode=cfg.get("mode"),
        compensation_mode=cfg.get("compensation_mode"),
        grid_spacing_hz=grid,
        lo_offset_hz=lo_off,
        integration_half_bw_hz=half_bw,
        n_bursts=len(bursts),
        median_ablr_db=round(float(np.median(ablr_db_all)), 3),
        p95_ablr_db=round(float(np.percentile(ablr_db_all, 95)), 3),
        max_ablr_db=round(float(np.max(ablr_db_all)), 3),
        mean_ablr_db=round(float(np.mean(ablr_db_all)), 3),
        iq_source=iq_path.name,
        measurement_type="short_range_ota_iq",
        validation_scope="short_range_ota_iq_proxy",
        commit=git_commit(),
        analyzed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(run_dir / "ablr_summary.json", summary)

    print(f"[ablr] median={summary['median_ablr_db']} dB  "
          f"p95={summary['p95_ablr_db']} dB  max={summary['max_ablr_db']} dB")
    print(f"[ablr] per-burst → {per_csv}")
    print(f"[ablr] summary   → {run_dir / 'ablr_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
