#!/usr/bin/env python3
"""
usrp_capture_ota_iq.py
======================
Plan and capture a programmed-Doppler LR-FHSS replay for the OTA IQ proxy
experiments (CFO residual + adjacent-bin leakage).

The USRP B210 is RX-ONLY here. Transmission is performed by the NUCLEO + LR1121
terminal under operator control. This script:

  plan     -- read a replay YAML config, build the per-burst schedule manifest
              the operator programs into the LR1121 firmware, and the analysis
              uses as ground-truth nominal grid bins. (No hardware needed.)

  capture  -- drive the USRP B210 RX path and record raw IQ to .npy (complex64)
              plus a JSON metadata sidecar over the replay window. If python-uhd
              is unavailable it prints the exact uhd_rx_cfile fallback command
              and exits non-zero. It NEVER writes synthetic IQ.

SAFETY (read before any TX):
  * Do NOT key the LR1121 unless an antenna or a 50-ohm load is attached.
  * Use the lowest practical LR1121 TX power.
  * USRP is RX only; never enable its TX in these experiments.
  * Use only locally permitted ISM/lab frequencies and short controlled bursts.

SCOPE: short-range room OTA / near-field IQ capture only. NOT conducted (no coax
/ no calibrated attenuator). NOT PER / decoding / CRC / receiver validation.
See docs/ota_iq_validation_scope.md.

Examples
--------
  # 1) Build the burst schedule the operator programs into the LR1121:
  uv run python hardware/ota_iq/usrp_capture_ota_iq.py plan \
      --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \
      --out-dir hardware/ota_iq/runs/pgrl_001

  # 2) Capture (USRP RX only) while the operator runs the programmed replay:
  uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
      --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \
      --out-dir hardware/ota_iq/runs/pgrl_001
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ota_common import require_valid_nominal_center_freq, git_commit, load_config, load_iq, write_json  # noqa: E402

# Prebuilt USRP B210 capture backend shipped in the repo (arm64/x86 Mach-O/ELF).
_RX_CPP = (Path(__file__).resolve().parents[1] / "usrp_scripts" / "rx_capture_to_file_cpp")


# ── schedule generation (INPUT, not a result) ─────────────────────────────────
def _load_profile_csv(path):
    """Load a t_s,<value>_hz CSV → (t[s], v[hz]) arrays."""
    ts, vs = [], []
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        # find numeric columns by name if possible
        cols = [c.strip().lower() for c in header]
        ti = cols.index("t_s") if "t_s" in cols else 0
        vi = 1
        for cand in ("doppler_hz", "comp_hz", "value_hz", "hz"):
            if cand in cols:
                vi = cols.index(cand)
                break
        for row in rdr:
            if not row:
                continue
            ts.append(float(row[ti]))
            vs.append(float(row[vi]))
    return np.asarray(ts), np.asarray(vs)


def build_schedule(cfg: dict, demo_synthetic: bool) -> list[dict]:
    """
    Build per-burst schedule. programmed_offset_hz = doppler(t) - comp(t), the
    net carrier offset from F0 the LR1121 is commanded to emit per burst.

    Requires real doppler_profile_csv (+ compensation_csv unless mode=none),
    UNLESS --demo-synthetic is passed (wiring/dry-run only; clearly tagged).
    """
    window = cfg["replay_window_s"]
    interval = cfg["burst_interval_s"]
    n_bursts = int(window // interval)
    t_onsets = np.arange(n_bursts) * interval

    comp_mode = cfg.get("compensation_mode", "none")

    if demo_synthetic:
        # SYNTHETIC schedule for hardware wiring tests ONLY. Tagged so it can
        # never be mistaken for a real replay. Do not use for paper results.
        peak = 18_000.0  # Hz, illustrative LEO-pass scale
        d = peak * np.sin(np.pi * t_onsets / window)        # half-sine pass shape
        if comp_mode == "none":
            c = np.zeros_like(d)
        elif comp_mode == "sgp4":
            c = d - 900.0 * np.sin(7 * np.pi * t_onsets / window)   # coarse residual
        else:  # pgrl
            c = d - 120.0 * np.sin(11 * np.pi * t_onsets / window)  # tight residual
        prog = d - c
    else:
        dop_csv = cfg.get("doppler_profile_csv")
        if not dop_csv:
            raise SystemExit(
                "ERROR: doppler_profile_csv is null. Supply a real SGP4-propagated "
                "LEO-pass reference Doppler CSV (t_s,doppler_hz) or pass --demo-synthetic "
                "for a wiring dry-run. This tool does not invent Doppler profiles."
            )
        td, d_full = _load_profile_csv(dop_csv)
        d = np.interp(t_onsets, td, d_full)
        if comp_mode == "none":
            c = np.zeros_like(d)
        else:
            comp_csv = cfg.get("compensation_csv")
            if not comp_csv:
                raise SystemExit(
                    f"ERROR: compensation_mode={comp_mode} but compensation_csv is null. "
                    f"Supply the real {comp_mode.upper()} feedforward CSV (t_s,comp_hz)."
                )
            tc, c_full = _load_profile_csv(comp_csv)
            c = np.interp(t_onsets, tc, c_full)
        prog = d - c

    schedule = []
    for i in range(n_bursts):
        schedule.append(dict(
            burst_index=i,
            t_rel_s=round(float(t_onsets[i]), 4),
            nominal_center_freq_hz=cfg["nominal_center_freq_hz"],
            compensation_mode=comp_mode,
            emulated_doppler_hz=round(float(d[i]), 3),
            commanded_offset_hz=round(float(prog[i]), 3),  # = expected residual CFO
            burst_duration_ms=cfg["burst_duration_ms"],
        ))
    return schedule


def cmd_plan(args) -> int:
    cfg = load_config(args.config)
    require_valid_nominal_center_freq(cfg, context="usrp_capture_ota_iq plan config")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(cfg, args.demo_synthetic)

    sched_csv = out_dir / "burst_schedule.csv"
    with open(sched_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(schedule[0].keys()))
        w.writeheader()
        w.writerows(schedule)

    meta = dict(
        kind="ota_iq_replay_schedule",
        config=str(args.config),
        mode=cfg.get("mode"),
        compensation_mode=cfg.get("compensation_mode"),
        nominal_center_freq_hz=cfg["nominal_center_freq_hz"],
        sample_rate_hz=cfg["sample_rate_hz"],
        rx_gain_db=cfg["rx_gain_db"],
        replay_window_s=cfg["replay_window_s"],
        burst_interval_s=cfg["burst_interval_s"],
        burst_duration_ms=cfg["burst_duration_ms"],
        n_bursts=len(schedule),
        demo_synthetic=bool(args.demo_synthetic),
        validation_scope="short_range_ota_iq_proxy",
        commit=git_commit(),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(out_dir / "schedule_meta.json", meta)

    print(f"[plan] mode={cfg.get('mode')}  bursts={len(schedule)}  "
          f"window={cfg['replay_window_s']}s")
    if args.demo_synthetic:
        print("[plan] *** demo_synthetic=TRUE — wiring dry-run only, NOT for results ***")
    print(f"[plan] schedule → {sched_csv}")
    print(f"[plan] meta     → {out_dir / 'schedule_meta.json'}")
    print("[plan] Program the LR1121 feedforward so each burst emits at "
          "F0 + commanded_offset_hz, in t_rel_s order.")
    return 0


# ── USRP B210 capture (RX ONLY) ───────────────────────────────────────────────
def cmd_capture(args) -> int:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f0 = require_valid_nominal_center_freq(cfg, context="usrp_capture_ota_iq config")
    lo_off = cfg.get("lo_offset_hz") or 0.0
    fc = f0 + lo_off                     # USRP RX center = F0 + offset (avoid LO on signal)
    fs = cfg["sample_rate_hz"]
    gain = cfg["rx_gain_db"]
    dur = args.duration if args.duration else cfg["replay_window_s"]
    iq_path = out_dir / "capture_iq.npy"

    print("=" * 70)
    print("  usrp_capture_ota_iq — USRP B210 RX-ONLY OTA IQ capture")
    print("=" * 70)
    print(f"  F0 (grid)   : {f0/1e6:.4f} MHz   lo_offset: {lo_off/1e3:.1f} kHz")
    print(f"  rx center   : {fc/1e6:.4f} MHz")
    print(f"  sample rate : {fs/1e6:.3f} MS/s")
    print(f"  rx gain     : {gain} dB  (raise only until just below clipping)")
    print(f"  duration    : {dur} s")
    print("  SAFETY: attach antenna/50-ohm load to LR1121 before any TX; "
          "lowest TX power; USRP RX only.")
    print()

    # Backend preference: (1) prebuilt rx_capture_to_file_cpp, (2) python-uhd.
    # If neither is available, print the operator command and exit non-zero.
    # No backend ever synthesises IQ.
    import subprocess
    if _RX_CPP.exists():
        iq_path = out_dir / "capture_iq.fc32"
        cmd = [str(_RX_CPP), "--freq", f"{fc:.0f}", "--rate", f"{fs:.0f}",
               "--gain", f"{gain:.0f}", "--duration", f"{dur:.0f}",
               "--out", str(iq_path), "--args", args.device_args]
        print(f"[capture] backend=rx_capture_to_file_cpp  args='{args.device_args}'")
        print(f"[capture] {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0 or not iq_path.exists():
            print(f"[capture] ERROR: rx_capture_to_file_cpp failed (rc={r.returncode}). "
                  f"No IQ written; nothing fabricated.", file=sys.stderr)
            return 1
        iq = load_iq(iq_path)
        backend = "rx_capture_to_file_cpp"
    else:
        try:
            import uhd
        except ImportError:
            cmd = (f"uhd_rx_cfile --args={args.device_args} --freq={fc:.0f} --rate={fs:.0f} "
                   f"--gain={gain:.0f} --duration={dur:.0f} --format=fc32 "
                   f"{out_dir / 'capture_iq.fc32'}")
            print("[capture] no rx_capture_to_file_cpp and no python-uhd. Run on the SDR host:")
            print(f"\n    {cmd}\n")
            print("[capture] Then point the analyzers at capture_iq.fc32. "
                  "This script does NOT synthesise IQ.")
            return 2
        usrp = uhd.usrp.MultiUSRP(args.device_args)
        usrp.set_rx_rate(fs)
        usrp.set_rx_freq(uhd.types.TuneRequest(fc))
        usrp.set_rx_gain(gain)
        n_samps = int(fs * dur)
        print(f"[capture] backend=python-uhd  receiving {n_samps} samples ...")
        samps = usrp.recv_num_samps(n_samps, fc, fs, [0], gain)
        iq = np.asarray(samps).reshape(-1).astype(np.complex64)
        np.save(iq_path, iq)
        backend = "python-uhd"

    # clipping guard
    peak = float(np.max(np.abs(iq))) if iq.size else 0.0
    clipping = peak > 0.98

    meta = dict(
        kind="ota_iq_capture",
        nominal_center_freq_hz=f0,
        rx_center_freq_hz=fc,
        lo_offset_hz=lo_off,
        sample_rate_hz=fs,
        rx_gain_db=gain,
        duration_s=dur,
        n_samples=int(iq.size),
        peak_abs=peak,
        clipping_warning=clipping,
        capture_backend=backend,
        iq_file=iq_path.name,
        device_args=args.device_args,
        device="USRP B210 (RX only)",
        antenna="room OTA / near-field probe (uncalibrated)",
        measurement_type="short_range_ota_iq",  # NOT conducted
        validation_scope="short_range_ota_iq_proxy",
        commit=git_commit(),
        capture_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(out_dir / "capture_meta.json", meta)

    print(f"[capture] saved {iq.size} samples → {iq_path}")
    if clipping:
        print(f"[capture] WARNING peak |IQ|={peak:.3f} near 1.0 — REDUCE rx_gain_db and recapture.")
    print(f"[capture] meta → {out_dir / 'capture_meta.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="USRP B210 OTA IQ replay capture (RX only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="build per-burst schedule manifest")
    pp.add_argument("--config", required=True)
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--demo-synthetic", action="store_true",
                    help="synthetic schedule for wiring dry-run ONLY (tagged, not for results)")
    pp.set_defaults(func=cmd_plan)

    pc = sub.add_parser("capture", help="USRP B210 RX-only IQ capture")
    pc.add_argument("--config", required=True)
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--duration", type=float, default=None,
                    help="override capture seconds (default = replay_window_s; use 10 for smoke)")
    pc.add_argument("--device-args", default="type=b200",
                    help="UHD device args, e.g. 'type=b200' or 'serial=8000304'")
    pc.set_defaults(func=cmd_capture)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
