#!/usr/bin/env python3
"""
generate_real_replay_schedule.py
================================
Build a REAL model-derived per-burst replay schedule for the OTA CW/CFO
diagnostic, from actual repo model outputs. Replaces the demo_synthetic
schedules.

Per-mode commanded offset (carrier offset from the nominal grid F0) is:

    no_compensation_offset_hz = true_doppler_hz                 (c = 0)
    sgp4_only_offset_hz       = true_doppler_hz - sgp4_pred_hz  (SGP4 open-loop residual)
    pgrl_corrected_offset_hz  = true_doppler_hz - pgrl_pred_hz  (PGRL open-loop residual)

This tool NEVER fabricates or hand-makes offsets. It requires a real predictions
CSV that already contains, per time sample:
    t_s, true_doppler_hz (or replayed_doppler_hz),
    sgp4_pred_doppler_hz, pgrl_pred_doppler_hz
plus provenance (pass_id, source_file). It refuses inputs that look synthetic /
placeholder (e.g. NORAD 99999, "EXAMPLE", "synthetic", "demo").

If you instead have a trained PGRL checkpoint + a real TLE + truth ephemeris,
generate that predictions CSV with the prediction pipeline first; this tool does
not run an untrained model.

Outputs:
    hardware/ota_iq/inputs/real_model_replay_schedule.csv
    hardware/ota_iq/inputs/real_model_replay_manifest.json

Dry-run (default; never transmits, never writes unless --write):
    uv run python hardware/ota_iq/generate_real_replay_schedule.py \
        --predictions-csv hardware/ota_iq/inputs/<real_predictions>.csv --dry-run
"""

from __future__ import annotations
import argparse, csv, hashlib, json, sys, time, subprocess
from pathlib import Path
import numpy as np

REQUIRED_COLS = {
    "t": ["t_s", "t_rel_s", "time_s"],
    "true": ["true_doppler_hz", "replayed_doppler_hz", "truth_doppler_hz"],
    "sgp4": ["sgp4_pred_doppler_hz", "sgp4_doppler_hz"],
    "pgrl": ["pgrl_pred_doppler_hz", "pgrl_doppler_hz"],
}
SYNTHETIC_MARKERS = ("synthetic", "demo", "example", "placeholder", "99999", "sgp4-feat-0.1.0-example")
INPUTS_DIR = Path("hardware/ota_iq/inputs")


def _pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _md5_col(rows, key):
    h = hashlib.md5()
    for r in rows:
        h.update(f"{r[key]:.6f}\n".encode())
    return h.hexdigest()


def load_predictions(path: Path):
    """Load + validate a real predictions CSV. Returns (samples, colmap, warnings)."""
    if not path.exists():
        raise SystemExit(
            f"MISSING: predictions CSV not found: {path}\n"
            f"  This tool does not fabricate model outputs. Provide a real CSV with "
            f"columns {REQUIRED_COLS}.")
    with open(path) as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        rows = list(rdr)
    colmap = {k: _pick(cols, v) for k, v in REQUIRED_COLS.items()}
    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        raise SystemExit(
            f"MISSING required columns in {path}: {missing}\n"
            f"  present columns: {cols}\n  accepted names: {REQUIRED_COLS}")
    # synthetic / placeholder refusal
    blob = (" ".join(cols) + " " + path.name + " " +
            " ".join(str(v) for v in (rows[0] if rows else {}).values())).lower()
    hits = [m for m in SYNTHETIC_MARKERS if m in blob]
    if hits:
        raise SystemExit(
            f"REFUSED: predictions CSV looks synthetic/placeholder (markers {hits}).\n"
            f"  Real model-derived schedule requires a real pass + trained PGRL "
            f"checkpoint output. Not fabricating.")
    samples = []
    for r in rows:
        samples.append({
            "t": float(r[colmap["t"]]),
            "true": float(r[colmap["true"]]),
            "sgp4": float(r[colmap["sgp4"]]),
            "pgrl": float(r[colmap["pgrl"]]),
        })
    return samples, colmap, []


def build_schedule(samples, f0_hz, interval_s, n_bursts, pass_id, source_file):
    t = np.array([s["t"] for s in samples])
    true = np.array([s["true"] for s in samples])
    sgp4 = np.array([s["sgp4"] for s in samples])
    pgrl = np.array([s["pgrl"] for s in samples])
    tb = np.arange(n_bursts) * interval_s
    if tb.max() > t.max() + 1e-6:
        raise SystemExit(
            f"MISSING coverage: predictions span {t.min()}..{t.max()} s but schedule "
            f"needs up to {tb.max()} s ({n_bursts} bursts x {interval_s} s).")
    d_true = np.interp(tb, t, true); d_sgp4 = np.interp(tb, t, sgp4); d_pgrl = np.interp(tb, t, pgrl)
    rows = []
    for i in range(n_bursts):
        rows.append({
            "burst_index": i, "t_rel_s": round(float(tb[i]), 4),
            "nominal_center_hz": int(f0_hz),
            "true_doppler_hz": round(float(d_true[i]), 3),
            "sgp4_pred_doppler_hz": round(float(d_sgp4[i]), 3),
            "pgrl_pred_doppler_hz": round(float(d_pgrl[i]), 3),
            "no_compensation_offset_hz": round(float(d_true[i]), 3),
            "sgp4_only_offset_hz": round(float(d_true[i] - d_sgp4[i]), 3),
            "pgrl_corrected_offset_hz": round(float(d_true[i] - d_pgrl[i]), 3),
            "pass_id": pass_id, "source_file": source_file,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate REAL model-derived replay schedule")
    ap.add_argument("--predictions-csv", type=Path,
                    help="real CSV: t_s,true_doppler_hz,sgp4_pred_doppler_hz,pgrl_pred_doppler_hz")
    ap.add_argument("--pgrl-checkpoint", type=Path, help="trained PGRL weights (to generate predictions)")
    ap.add_argument("--nominal-center-hz", type=float, default=868_000_000.0)
    ap.add_argument("--burst-interval-s", type=float, default=2.0)
    ap.add_argument("--n-bursts", type=int, default=10)
    ap.add_argument("--pass-id", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print only; do not write files")
    ap.add_argument("--write", action="store_true", help="write schedule + manifest")
    args = ap.parse_args()

    if args.predictions_csv is None:
        # No real predictions provided. If a checkpoint path was given, we still
        # cannot run an untrained/absent model -> report missing precisely.
        ck = args.pgrl_checkpoint
        raise SystemExit(
            "STOP: no --predictions-csv given.\n"
            "Real model-derived schedule needs per-sample: true/replayed Doppler, "
            "SGP4-predicted Doppler, PGRL-predicted Doppler.\n"
            + ("MISSING PGRL checkpoint: " + str(ck) + " (not found)\n" if (ck and not ck.exists())
               else "Provide --predictions-csv produced by the prediction pipeline.\n")
            + "This tool will NOT fabricate or hand-make offsets.")

    samples, colmap, warns = load_predictions(args.predictions_csv)
    pass_id = args.pass_id or args.predictions_csv.stem
    rows = build_schedule(samples, args.nominal_center_hz, args.burst_interval_s,
                          args.n_bursts, pass_id, str(args.predictions_csv))

    modes = {
        "no_compensation": "no_compensation_offset_hz",
        "sgp4_only": "sgp4_only_offset_hz",
        "pgrl_corrected": "pgrl_corrected_offset_hz",
    }
    print(f"[gen] predictions: {args.predictions_csv}  pass_id={pass_id}  bursts={len(rows)}")
    print("[gen] first 10 rows:")
    for r in rows[:10]:
        print("   ", {k: r[k] for k in ("burst_index", "t_rel_s", "true_doppler_hz",
              "sgp4_pred_doppler_hz", "pgrl_pred_doppler_hz",
              "no_compensation_offset_hz", "sgp4_only_offset_hz", "pgrl_corrected_offset_hz")})
    print("[gen] per-mode offset stats + md5:")
    for m, col in modes.items():
        vals = np.array([r[col] for r in rows])
        print(f"    {m:16s} min={vals.min():10.2f} median={np.median(vals):10.2f} "
              f"max={vals.max():10.2f}  md5={_md5_col(rows, col)}")

    if not args.write or args.dry_run:
        print("[gen] DRY-RUN: nothing written. Re-run with --write (and without --dry-run) to save.")
        return 0

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = INPUTS_DIR / "real_model_replay_schedule.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    manifest = dict(
        schedule_source="real_model_derived",
        predictions_csv=str(args.predictions_csv),
        source_files=[str(args.predictions_csv)],
        pgrl_checkpoint=str(args.pgrl_checkpoint) if args.pgrl_checkpoint else None,
        pass_id=pass_id,
        nominal_center_hz=args.nominal_center_hz,
        burst_interval_s=args.burst_interval_s, n_bursts=args.n_bursts,
        column_map=colmap,
        per_mode_md5={m: _md5_col(rows, col) for m, col in modes.items()},
        generation_command=" ".join(["generate_real_replay_schedule.py"] + sys.argv[1:]),
        warning=("; ".join(warns) if warns else None),
        commit=_git_commit(),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (INPUTS_DIR / "real_model_replay_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gen] wrote {out_csv} and manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
