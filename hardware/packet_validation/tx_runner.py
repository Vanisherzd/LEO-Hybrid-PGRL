"""TX backend runner.

Modes:
  mock         : emit UART-like "Packet sent!" lines, no hardware.
  file_replay  : parse an existing UART log for "Packet sent!" lines (TX-side
                 evidence only; counts packets emitted, infers nothing about RX).
  lr1121_uart  : wrap semtech_validation/run_lrfhss_tx.sh on REAL hardware.
                 Conducted/shielded only; requires --i-have-hardware and never
                 changes TX power. Refuses to run without the explicit flag.

Outputs: <out>/tx_records.csv, <out>/tx_uart.log, <out>/tx_summary.json
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation._io import (  # noqa: E402
    ensure_dir, read_csv, write_csv, write_json)
from hardware.packet_validation.schemas import TxPacketRecord  # noqa: E402

_SENT_RE = re.compile(r"packet sent", re.IGNORECASE)


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_mock(payloads, run_id, freq_hz, power_dbm, log_path):
    recs, lines = [], []
    for p in payloads:
        seq = int(p["seq"])
        lines.append(f"Packet to send: {p['payload_hex']}")
        lines.append(f"RF={int(freq_hz)} Hz, PWR={power_dbm} dBm, "
                     f"payload_len={p['payload_len']}")
        lines.append("Packet sent!")
        recs.append(TxPacketRecord(
            run_id=run_id, seq=seq, payload_hex=p["payload_hex"],
            payload_len=int(p["payload_len"]), tx_timestamp_utc=_now(),
            tx_backend="mock", tx_power_dbm=power_dbm,
            center_frequency_hz=freq_hz, lr_fhss_config="mock",
            uart_status="sent", raw_log_path=log_path))
    with open(log_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return recs


def run_file_replay(payloads, run_id, tx_log, log_path):
    """Count 'Packet sent!' lines in an existing UART log (TX-side only)."""
    with open(tx_log) as fh:
        raw = fh.read()
    sent_count = len(_SENT_RE.findall(raw))
    with open(log_path, "w") as fh:
        fh.write(raw)
    recs = []
    # Associate replayed sends with available payload seqs (best effort).
    n = sent_count if not payloads else min(sent_count, len(payloads))
    for i in range(n):
        p = payloads[i] if payloads else {"payload_hex": "", "payload_len": 0}
        recs.append(TxPacketRecord(
            run_id=run_id, seq=i,
            payload_hex=p.get("payload_hex", ""),
            payload_len=int(p.get("payload_len", 0) or 0),
            tx_timestamp_utc=_now(), tx_backend="file_replay",
            uart_status="sent_replayed", raw_log_path=os.path.abspath(tx_log)))
    if not recs:  # log had sends but no payload list
        for i in range(sent_count):
            recs.append(TxPacketRecord(
                run_id=run_id, seq=i, payload_hex="", payload_len=0,
                tx_timestamp_utc=_now(), tx_backend="file_replay",
                uart_status="sent_replayed",
                raw_log_path=os.path.abspath(tx_log)))
    return recs


def run_lr1121_uart(payloads, run_id, log_path, enable_hardware):
    if not enable_hardware:
        raise SystemExit(
            "[tx_runner] lr1121_uart is a REAL-HARDWARE backend. Re-run with "
            "--i-have-hardware after confirming a conducted/shielded setup "
            "(coax + attenuator or dummy load). TX power is NOT modified by "
            "this tool. Aborting (default is dry-run).")
    # Real-hardware path intentionally minimal and non-power-modifying.
    import subprocess
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = os.path.join(root, "semtech_validation", "run_lrfhss_tx.sh")
    out = subprocess.run(["bash", script, "baseline"], capture_output=True,
                         text=True)
    with open(log_path, "w") as fh:
        fh.write(out.stdout + "\n" + out.stderr)
    sent = len(_SENT_RE.findall(out.stdout))
    return [TxPacketRecord(run_id=run_id, seq=i, payload_hex="", payload_len=0,
                           tx_timestamp_utc=_now(), tx_backend="lr1121_uart",
                           uart_status="sent", raw_log_path=log_path)
            for i in range(sent)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="TX backend runner.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backend", choices=["mock", "file_replay", "lr1121_uart"],
                    default="mock")
    ap.add_argument("--payloads", help="path to tx_payloads.csv")
    ap.add_argument("--tx-log", help="UART log for file_replay")
    ap.add_argument("--freq-hz", type=float, default=868_000_000.0)
    ap.add_argument("--power-dbm", type=float, default=10.0)
    ap.add_argument("--i-have-hardware", action="store_true",
                    help="confirm conducted/shielded hardware for lr1121_uart")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    log_path = os.path.join(args.out, "tx_uart.log")
    payloads = read_csv(args.payloads) if args.payloads else []

    if args.backend == "mock":
        recs = run_mock(payloads, args.run_id, args.freq_hz, args.power_dbm,
                        log_path)
    elif args.backend == "file_replay":
        if not args.tx_log:
            raise SystemExit("[tx_runner] file_replay needs --tx-log")
        recs = run_file_replay(payloads, args.run_id, args.tx_log, log_path)
    else:
        recs = run_lr1121_uart(payloads, args.run_id, log_path,
                               args.i_have_hardware)

    write_csv(os.path.join(args.out, "tx_records.csv"),
              [r.to_dict() for r in recs])
    summary = {"run_id": args.run_id, "tx_backend": args.backend,
               "n_tx": len(recs), "uart_log": log_path}
    write_json(os.path.join(args.out, "tx_summary.json"), summary)
    print(f"[tx_runner] backend={args.backend} n_tx={len(recs)}")
    return recs


if __name__ == "__main__":
    main()
