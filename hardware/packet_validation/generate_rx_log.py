"""Synthetic decoded-RX log generator (parser/evaluator TEST FIXTURE).

NOT a hardware result. Consumes tx_payloads.csv/jsonl and emits a synthetic
decoded RX log in a chosen format (A/B/C/JSONL) with configurable loss, CRC
error, duplicate, and false-positive behaviour. Used to exercise the RX log
parser + evaluator before real hardware exists.

CLI:
  python hardware/packet_validation/generate_rx_log.py \
    --tx validation_runs/dryrun_001/tx_payloads.csv \
    --out validation_runs/synthetic_rx/rx.log \
    --format A --loss-rate 0.05 --crc-error-rate 0.01 \
    --duplicate-rate 0.02 --false-positive-rate 0.01 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation._io import ensure_dir, read_csv  # noqa: E402


def _load_tx(path):
    if path.endswith(".jsonl"):
        with open(path) as fh:
            return [json.loads(l) for l in fh if l.strip()]
    return read_csv(path)


def _fmt(fmt, seq, payload, ok, rssi, snr, cfo, ts):
    if fmt == "A":
        return (f"RX seq={seq} payload={payload} crc={'ok' if ok else 'fail'} "
                f"rssi={rssi} snr={snr} cfo={cfo} timestamp={ts}")
    if fmt == "B":
        return (f"[RX] seq:{seq} payload:{payload} {'CRC_OK' if ok else 'CRC_FAIL'} "
                f"RSSI={rssi} SNR={snr} CFO={cfo}")
    if fmt == "C":
        return (f"packet_received seq={seq} len={len(payload)//2} "
                f"payload_hex={payload} crc_ok={'true' if ok else 'false'}")
    if fmt == "JSONL":
        return json.dumps({"seq": seq, "payload_hex": payload, "crc_ok": ok,
                           "rssi_dbm": rssi, "snr_db": snr, "cfo_hz": cfo,
                           "timestamp": ts})
    raise ValueError(f"unknown format {fmt}")


def generate(tx_rows, fmt, loss_rate, crc_error_rate, duplicate_rate,
             false_positive_rate, seed):
    rng = random.Random(f"rxgen:{seed}")
    lines = []
    base_ts = 1_780_000_000.0  # arbitrary deterministic epoch seconds
    for i, r in enumerate(tx_rows):
        if rng.random() < loss_rate:
            continue  # lost packet -> missing seq
        seq = int(r["seq"])
        payload = str(r["payload_hex"])
        ok = rng.random() >= crc_error_rate
        emit_payload = payload if ok else payload[:-2] + "ff"  # corrupt on crc fail
        rssi = round(-70 - 30 * rng.random(), 1)
        snr = round(15 * rng.random(), 1)
        cfo = round(rng.uniform(-500, 500), 1)
        ts = _iso(base_ts + i * 0.05)
        lines.append(_fmt(fmt, seq, emit_payload, ok, rssi, snr, cfo, ts))
        if rng.random() < duplicate_rate:
            lines.append(_fmt(fmt, seq, emit_payload, ok, rssi, snr, cfo, ts))
        if rng.random() < false_positive_rate:
            fp_seq = 90000 + i
            lines.append(_fmt(fmt, fp_seq, "deadbeef", True,
                              round(-95 - 5 * rng.random(), 1), 1.0, 0.0,
                              _iso(base_ts + i * 0.05 + 0.01)))
    return lines


def _iso(epoch_s):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a synthetic decoded RX log.")
    ap.add_argument("--tx", required=True, help="tx_payloads.csv or .jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["A", "B", "C", "JSONL"], default="A")
    ap.add_argument("--loss-rate", type=float, default=0.0)
    ap.add_argument("--crc-error-rate", type=float, default=0.0)
    ap.add_argument("--duplicate-rate", type=float, default=0.0)
    ap.add_argument("--false-positive-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    tx_rows = _load_tx(args.tx)
    lines = generate(tx_rows, args.format, args.loss_rate, args.crc_error_rate,
                     args.duplicate_rate, args.false_positive_rate, args.seed)
    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[generate_rx_log] format={args.format} tx={len(tx_rows)} "
          f"rx_lines={len(lines)} -> {args.out}")
    return lines


if __name__ == "__main__":
    main()
