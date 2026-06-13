#!/usr/bin/env python3
"""
replay_driver.py
================
Host-side driver for per-burst open-loop LR-FHSS replay. Reads the burst
schedule produced by `usrp_capture_ota_iq.py plan` and commands the LR1121
(via the host-replay firmware, see hardware/ota_iq/firmware/) one burst at a
time over UART, pacing bursts by their scheduled times so they land inside a
concurrent USRP capture window.

Per burst it sends, over the NUCLEO VCP UART:

    B <burst_index> <rf_freq_hz> <tx_power_dbm> <delay_ms>\\n

and expects the firmware to ack after transmit:

    BURST_DONE <burst_index> <rf_freq_hz> <tx_power_dbm>

rf_freq_hz = nominal_center_freq_hz + commanded_offset_hz (the mode-specific
residual = emulated Doppler minus compensation, taken from the schedule).

This driver ONLY commands transmission and logs what was commanded/acked. It
produces NO CFO/ABLR results. It requires the host-replay firmware to be flashed
(stock SWDM001 ignores these commands and free-runs).

SAFETY:
  * Antenna or 50-ohm load MUST be attached to the LR1121 before running.
  * Use the lowest practical --tx-power-dbm.
  * Use only locally permitted ISM/lab frequencies; bursts are short.

Typical use (two terminals):
  # T1 - start the USRP capture over the full replay window:
  uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \\
      --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \\
      --out-dir hardware/ota_iq/runs/<TS>/pgrl --device-args serial=8000304

  # T2 - immediately drive the bursts on the same schedule:
  uv run python hardware/ota_iq/replay_driver.py \\
      --schedule hardware/ota_iq/runs/<TS>/pgrl/burst_schedule.csv \\
      --uart /dev/cu.usbmodem1303 --tx-power-dbm -9 \\
      --out hardware/ota_iq/runs/<TS>/pgrl/replay_uart_log.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ota_common import require_valid_nominal_center_freq  # noqa: E402


def _read_schedule(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty schedule: {path}")
    return rows


def _drain(ser, deadline: float) -> list[str]:
    """Read available lines until deadline; return decoded non-empty lines."""
    out = []
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("ascii", "replace").strip()
        if s:
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-burst LR-FHSS replay UART driver")
    ap.add_argument("--schedule", required=True, help="burst_schedule.csv from `plan`")
    ap.add_argument("--uart", required=True, help="e.g. /dev/cu.usbmodem1303")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--tx-power-dbm", type=int, required=True,
                    help="lowest practical LR1121 TX power, e.g. -9")
    ap.add_argument("--out", required=True, help="output replay_uart_log.csv")
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="initial wait so the USRP capture is running before burst 0")
    ap.add_argument("--ack-timeout-s", type=float, default=4.0,
                    help="max wait for BURST_DONE per burst")
    ap.add_argument("--no-handshake", action="store_true",
                    help="do not wait for RDY/BURST_DONE; send paced by t_rel_s only")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without opening the UART or transmitting")
    args = ap.parse_args()

    rows = _read_schedule(Path(args.schedule))
    print(f"[replay] {len(rows)} bursts  power={args.tx_power_dbm} dBm  "
          f"uart={args.uart}@{args.baud}  dry_run={args.dry_run}")

    ser = None
    if not args.dry_run:
        import serial
        ser = serial.Serial(args.uart, args.baud, timeout=1)
        # confirm firmware is in host-replay mode (best-effort)
        hello = _drain(ser, time.time() + 2.0)
        if any("REPLAY_READY" in h for h in hello):
            print("[replay] firmware reports REPLAY_READY")
        else:
            print("[replay] WARNING: no REPLAY_READY seen. If the firmware is stock "
                  "SWDM001 it will free-run and IGNORE these commands — flash the "
                  "host-replay firmware (hardware/ota_iq/firmware/). Continuing.")

    log_rows = []
    t0 = time.time() + args.settle_s
    if args.settle_s > 0:
        print(f"[replay] settling {args.settle_s}s (start the USRP capture now if not already)")
        time.sleep(args.settle_s)

    for r in rows:
        idx = int(r["burst_index"])
        f0 = require_valid_nominal_center_freq(r, context=f"replay_driver schedule row {r.get('burst_index', '?')}")
        off = float(r.get("commanded_offset_hz", 0.0) or 0.0)
        rf = int(round(f0 + off))
        # pace to scheduled time (skipped in dry-run, which only prints commands)
        if not args.dry_run:
            target = t0 + float(r["t_rel_s"])
            dt = target - time.time()
            if dt > 0:
                time.sleep(dt)

        cmd = f"B {idx} {rf} {args.tx_power_dbm} 0"
        sent_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ack = ""
        packet_sent = False

        if args.dry_run:
            print(f"[replay] (dry) {cmd}")
        else:
            if not args.no_handshake:
                _drain(ser, time.time() + 0.5)  # consume any RDY prompt
            ser.write((cmd + "\n").encode("ascii"))
            ser.flush()
            if not args.no_handshake:
                lines = _drain(ser, time.time() + args.ack_timeout_s)
                for ln in lines:
                    if ln.startswith("BURST_DONE"):
                        ack = ln
                    if "Packet sent" in ln:
                        packet_sent = True
                if not ack:
                    print(f"[replay] burst {idx}: no BURST_DONE ack within "
                          f"{args.ack_timeout_s}s (freq={rf})")

        log_rows.append(dict(
            burst_index=idx,
            t_rel_s=r["t_rel_s"],
            sent_utc=sent_utc,
            commanded_rf_freq_hz=rf,
            commanded_offset_hz=round(off, 3),
            tx_power_dbm=args.tx_power_dbm,
            ack=ack,
            packet_sent=packet_sent,
        ))

    if ser is not None:
        ser.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    n_ack = sum(1 for r in log_rows if r["ack"])
    print(f"[replay] done. commanded {len(log_rows)} bursts, {n_ack} acked → {out}")
    if not args.dry_run and n_ack == 0 and not args.no_handshake:
        print("[replay] NOTE: 0 acks — verify host-replay firmware is flashed and "
              "the UART port is correct. No replay results should be claimed from this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
