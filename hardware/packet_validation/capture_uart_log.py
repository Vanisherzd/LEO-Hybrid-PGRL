"""Capture a UART serial stream to a log file (pyserial). Capture only --- no
parsing/validation here.

CLI:
  python hardware/packet_validation/capture_uart_log.py \
    --port /dev/cu.usbmodem1303 --baud 115200 \
    --out hardware/rx_logs/real_runA_alpha0_decoded_rx.log --duration 60

Writes UTC start/end metadata comment lines, echoes to terminal, exits cleanly
on Ctrl-C. Conducted/shielded setup only; this tool never configures TX power.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture UART serial to a log file.")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds to capture (0 = until Ctrl-C)")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--raw-bytes", action="store_true",
                    help="write raw bytes (latin-1) if text decode fails")
    args = ap.parse_args(argv)

    try:
        import serial  # type: ignore
    except ImportError:
        print("pyserial is required. Use: uv add pyserial", file=sys.stderr)
        print("Temporary one-off: uv run --with pyserial python "
              "hardware/packet_validation/capture_uart_log.py ...", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    fh = open(args.out, mode, buffering=1)
    fh.write(f"# capture_start_utc={_now()} port={args.port} baud={args.baud}\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as e:  # noqa: BLE001
        print(f"failed to open {args.port}: {e}", file=sys.stderr)
        fh.write(f"# capture_error={e}\n")
        fh.close()
        return 3

    t0 = time.time()
    n_lines = 0
    try:
        while True:
            if args.duration and (time.time() - t0) >= args.duration:
                break
            line = ser.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                if args.raw_bytes:
                    text = line.decode("latin-1").rstrip("\r\n")
                else:
                    continue
            print(text)
            fh.write(text + "\n")
            n_lines += 1
    except KeyboardInterrupt:
        print("\n[capture] interrupted; closing cleanly.")
    finally:
        fh.write(f"# capture_end_utc={_now()} lines={n_lines}\n")
        fh.close()
        try:
            ser.close()
        except Exception:  # noqa: BLE001
            pass
    print(f"[capture] wrote {n_lines} lines to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
