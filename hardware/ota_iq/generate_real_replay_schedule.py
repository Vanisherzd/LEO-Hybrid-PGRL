#!/usr/bin/env python3
"""
generate_real_replay_schedule.py
================================
Build a model-derived per-burst replay schedule for the OTA/CW/CFO
diagnostic from user-supplied prediction/reference CSV files.

Important scope:
  - This tool does NOT create measured Doppler truth.
  - This tool does NOT validate PGRL against a live satellite.
  - This tool only prepares replay offsets for conducted/lab RF diagnostics.
  - Carrier frequency must be explicitly supplied after local frequency plan review.

Per-mode commanded offset, relative to nominal grid F0, is:

    no_compensation_offset_hz = reference_doppler_hz
    sgp4_only_offset_hz      = reference_doppler_hz - sgp4_model_doppler_hz
    pgrl_corrected_offset_hz = reference_doppler_hz - pgrl_model_doppler_hz

Accepted input CSV columns:
    t_s or t_rel_s or time_s
    reference_doppler_hz or modeled_sgp4_doppler_hz or replayed_doppler_hz
    sgp4_model_doppler_hz or sgp4_pred_doppler_hz or sgp4_doppler_hz
    pgrl_model_doppler_hz or pgrl_pred_doppler_hz or pgrl_doppler_hz

The output intentionally uses `reference_doppler_hz`, not `true_doppler_hz`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REQUIRED_COLS = {
    "t": ["t_s", "t_rel_s", "time_s"],
    "reference": [
        "reference_doppler_hz",
        "modeled_sgp4_doppler_hz",
        "modeled_reference_doppler_hz",
        "replayed_doppler_hz",
    ],
    "sgp4": ["sgp4_model_doppler_hz", "sgp4_pred_doppler_hz", "sgp4_doppler_hz"],
    "pgrl": ["pgrl_model_doppler_hz", "pgrl_pred_doppler_hz", "pgrl_doppler_hz"],
}

PROHIBITED_INPUT_COLS = {
    "true_doppler_hz",
    "truth_doppler_hz",
    "measured_doppler_hz",
    "measured_doppler_truth_hz",
}

SYNTHETIC_MARKERS = (
    "synthetic",
    "demo",
    "example",
    "placeholder",
    "99999",
    "sgp4-feat-0.1.0-example",
)

INPUTS_DIR = Path("hardware/ota_iq/inputs")


def _pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _md5_col(rows, key):
    h = hashlib.md5()
    for row in rows:
        h.update(f"{row[key]:.6f}\n".encode())
    return h.hexdigest()


def load_predictions(path: Path):
    """Load and validate a model/reference predictions CSV.

    Returns:
        samples, colmap, warnings
    """
    if not path.exists():
        raise SystemExit(
            f"MISSING: predictions CSV not found: {path}\n"
            "Provide a CSV with accepted reference/model Doppler columns. "
            "This tool does not fabricate model outputs."
        )

    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        rows = list(rdr)

    prohibited = sorted(PROHIBITED_INPUT_COLS.intersection(cols))
    if prohibited:
        raise SystemExit(
            f"REFUSED: misleading Doppler-truth column(s) present: {prohibited}\n"
            "Rename the input reference column to `reference_doppler_hz` or "
            "`modeled_sgp4_doppler_hz`. This pipeline does not accept "
            "`true_doppler_hz` or measured-truth naming unless a future live-pass "
            "reference dataset is explicitly supported."
        )

    colmap = {k: _pick(cols, v) for k, v in REQUIRED_COLS.items()}
    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        raise SystemExit(
            f"MISSING required columns in {path}: {missing}\n"
            f"  present columns: {cols}\n"
            f"  accepted names: {REQUIRED_COLS}"
        )

    blob = (
        " ".join(cols)
        + " "
        + path.name
        + " "
        + " ".join(str(v) for v in (rows[0] if rows else {}).values())
    ).lower()
    hits = [m for m in SYNTHETIC_MARKERS if m in blob]
    if hits:
        raise SystemExit(
            f"REFUSED: predictions CSV looks synthetic/placeholder (markers {hits}).\n"
            "For paper-use replay, provide provenance-confirmed model/reference "
            "outputs. For software-only tests, use a separate dry-run/toy pipeline "
            "and do not treat outputs as paper evidence."
        )

    samples = []
    for row in rows:
        samples.append({
            "t": float(row[colmap["t"]]),
            "reference": float(row[colmap["reference"]]),
            "sgp4": float(row[colmap["sgp4"]]),
            "pgrl": float(row[colmap["pgrl"]]),
        })

    warnings = [
        "reference_doppler_hz is user-supplied/model-derived reference, not measured Doppler truth",
        "schedule outputs are for conducted/lab RF diagnostics, not live satellite validation",
    ]
    return samples, colmap, warnings


def build_schedule(samples, f0_hz, interval_s, n_bursts, pass_id, source_file):
    t = np.array([s["t"] for s in samples])
    reference = np.array([s["reference"] for s in samples])
    sgp4 = np.array([s["sgp4"] for s in samples])
    pgrl = np.array([s["pgrl"] for s in samples])

    tb = np.arange(n_bursts) * interval_s
    if tb.max() > t.max() + 1e-6:
        raise SystemExit(
            f"MISSING coverage: predictions span {t.min()}..{t.max()} s but schedule "
            f"needs up to {tb.max()} s ({n_bursts} bursts x {interval_s} s)."
        )

    d_ref = np.interp(tb, t, reference)
    d_sgp4 = np.interp(tb, t, sgp4)
    d_pgrl = np.interp(tb, t, pgrl)

    rows = []
    for i in range(n_bursts):
        rows.append({
            "burst_index": i,
            "t_rel_s": round(float(tb[i]), 4),
            "nominal_center_hz": int(f0_hz),
            "reference_doppler_hz": round(float(d_ref[i]), 3),
            "sgp4_model_doppler_hz": round(float(d_sgp4[i]), 3),
            "pgrl_model_doppler_hz": round(float(d_pgrl[i]), 3),
            "reference_doppler_type": "user_supplied_or_model_derived_not_measured_truth",
            "reference_is_measured_truth": "false",
            "no_compensation_offset_hz": round(float(d_ref[i]), 3),
            "sgp4_only_offset_hz": round(float(d_ref[i] - d_sgp4[i]), 3),
            "pgrl_corrected_offset_hz": round(float(d_ref[i] - d_pgrl[i]), 3),
            "pass_id": pass_id,
            "source_file": source_file,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Generate model/reference-derived replay schedule for conducted RF diagnostics"
    )
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        help=(
            "Reference/model CSV with columns like: "
            "t_s,reference_doppler_hz,sgp4_model_doppler_hz,pgrl_model_doppler_hz. "
            "Do not use true/measured Doppler column names."
        ),
    )
    ap.add_argument(
        "--pgrl-checkpoint",
        type=Path,
        help="trained PGRL weights metadata/provenance only; this script does not run the model",
    )
    ap.add_argument(
        "--nominal-center-hz",
        type=float,
        default=None,
        help=(
            "Carrier frequency in Hz. Must be explicitly provided after confirming "
            "local frequency plan. No default is used."
        ),
    )
    ap.add_argument("--burst-interval-s", type=float, default=2.0)
    ap.add_argument("--n-bursts", type=int, default=10)
    ap.add_argument("--pass-id", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print only; do not write files")
    ap.add_argument("--write", action="store_true", help="write schedule + manifest")
    args = ap.parse_args()

    if args.nominal_center_hz is None:
        raise SystemExit(
            "Carrier frequency must be explicitly provided after confirming local frequency plan. "
            "No default carrier is used."
        )

    if args.nominal_center_hz <= 0:
        raise SystemExit("Carrier frequency must be a positive Hz value.")

    if args.predictions_csv is None:
        ck = args.pgrl_checkpoint
        raise SystemExit(
            "STOP: no --predictions-csv given.\n"
            "Replay schedule generation needs per-sample reference/model Doppler columns:\n"
            "  t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz\n"
            + (
                "MISSING PGRL checkpoint: " + str(ck) + " (not found)\n"
                if (ck and not ck.exists())
                else "Provide --predictions-csv produced by the prediction/reference pipeline.\n"
            )
            + "This tool will NOT fabricate or hand-make offsets."
        )

    samples, colmap, warns = load_predictions(args.predictions_csv)
    pass_id = args.pass_id or args.predictions_csv.stem
    rows = build_schedule(
        samples,
        args.nominal_center_hz,
        args.burst_interval_s,
        args.n_bursts,
        pass_id,
        str(args.predictions_csv),
    )

    modes = {
        "no_compensation": "no_compensation_offset_hz",
        "sgp4_only": "sgp4_only_offset_hz",
        "pgrl_corrected": "pgrl_corrected_offset_hz",
    }

    print(f"[gen] predictions: {args.predictions_csv}  pass_id={pass_id}  bursts={len(rows)}")
    print("[gen] first 10 rows:")
    for row in rows[:10]:
        print("   ", {
            k: row[k]
            for k in (
                "burst_index",
                "t_rel_s",
                "reference_doppler_hz",
                "sgp4_model_doppler_hz",
                "pgrl_model_doppler_hz",
                "reference_is_measured_truth",
                "no_compensation_offset_hz",
                "sgp4_only_offset_hz",
                "pgrl_corrected_offset_hz",
            )
        })

    print("[gen] per-mode offset stats + md5:")
    for mode, col in modes.items():
        vals = np.array([row[col] for row in rows])
        print(
            f"    {mode:16s} min={vals.min():10.2f} "
            f"median={np.median(vals):10.2f} "
            f"max={vals.max():10.2f}  md5={_md5_col(rows, col)}"
        )

    if not args.write or args.dry_run:
        print("[gen] DRY-RUN: nothing written. Re-run with --write and without --dry-run to save.")
        return 0

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = INPUTS_DIR / "real_model_replay_schedule.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schedule_source": "model_reference_derived",
        "predictions_csv": str(args.predictions_csv),
        "source_files": [str(args.predictions_csv)],
        "pgrl_checkpoint": str(args.pgrl_checkpoint) if args.pgrl_checkpoint else None,
        "pass_id": pass_id,
        "nominal_center_hz": args.nominal_center_hz,
        "burst_interval_s": args.burst_interval_s,
        "n_bursts": args.n_bursts,
        "column_map": colmap,
        "reference_is_measured_truth": False,
        "evidence_type": "model_reference_schedule_for_conducted_rf_diagnostic",
        "limitation": (
            "Schedule is not measured Doppler truth, not live satellite validation, "
            "and not PER/BER/CRC evidence."
        ),
        "per_mode_md5": {mode: _md5_col(rows, col) for mode, col in modes.items()},
        "generation_command": " ".join(["generate_real_replay_schedule.py"] + sys.argv[1:]),
        "warning": "; ".join(warns) if warns else None,
        "commit": _git_commit(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = INPUTS_DIR / "real_model_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] wrote {out_csv} and manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
