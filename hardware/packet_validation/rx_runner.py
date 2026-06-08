"""RX backend runner.

Modes:
  mock        : simulate a receiver from tx_payloads.csv with configurable
                loss / crc-error / duplicate / false-positive behaviour.
                Produces DECODED records (decode_status="decoded").
  log_parser  : parse future gateway/LR1121 RX logs ("RX seq=.. payload=..
                crc=ok|fail"). Produces decoded records if any are present.
  iq_only     : attach USRP IQ signal-detection metadata only. Produces NO
                decoded packets: every record is decode_status="not_decoded",
                crc_ok=None, and nothing is counted as delivered.

Outputs: <out>/rx_records.csv, <out>/rx_summary.json
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import random
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation._io import (  # noqa: E402
    ensure_dir, read_csv, read_json, write_csv, write_json)
from hardware.packet_validation.schemas import RxPacketRecord  # noqa: E402


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_mock(payloads, run_id, loss_rate, crc_error_rate, duplicate_rate,
             false_positive_count, seed):
    rng = random.Random(f"{run_id}:rx:{seed}")
    recs = []
    for p in payloads:
        if rng.random() < loss_rate:
            continue  # packet lost -> missing seq
        crc_ok = rng.random() >= crc_error_rate
        rec = RxPacketRecord(
            run_id=run_id, seq=int(p["seq"]),
            payload_hex=p["payload_hex"] if crc_ok else p["payload_hex"][:-2] + "00",
            payload_len=int(p["payload_len"]), rx_timestamp_utc=_now(),
            rx_backend="mock", crc_ok=crc_ok,
            decode_status="decoded" if crc_ok else "crc_fail")
        recs.append(rec)
        if rng.random() < duplicate_rate:
            recs.append(RxPacketRecord(**{**rec.to_dict()}))  # duplicate
    for j in range(false_positive_count):  # spurious RX not in TX set
        recs.append(RxPacketRecord(
            run_id=run_id, seq=10_000 + j, payload_hex=f"dead{j:04x}",
            payload_len=4, rx_timestamp_utc=_now(), rx_backend="mock",
            crc_ok=True, decode_status="decoded"))
    return recs


_RX_RE = re.compile(
    r"RX.*?seq\s*=\s*(\d+).*?payload\s*=\s*([0-9a-fA-F]+).*?crc\s*=\s*(ok|fail)",
    re.IGNORECASE)


def run_log_parser(run_id, rx_log):
    recs = []
    if not rx_log or not os.path.exists(rx_log):
        return recs
    with open(rx_log) as fh:
        for line in fh:
            m = _RX_RE.search(line)
            if not m:
                continue
            seq, payload, crc = m.group(1), m.group(2), m.group(3).lower()
            ok = crc == "ok"
            recs.append(RxPacketRecord(
                run_id=run_id, seq=int(seq), payload_hex=payload,
                payload_len=len(payload) // 2, rx_timestamp_utc=_now(),
                rx_backend="log_parser", crc_ok=ok,
                decode_status="decoded" if ok else "crc_fail",
                raw_log_path=os.path.abspath(rx_log)))
    return recs


def run_iq_only(run_id, iq_metadata):
    """Attach IQ signal-detection metadata. NEVER counts as delivered."""
    meta = {}
    if iq_metadata and os.path.exists(iq_metadata):
        meta = read_json(iq_metadata)
    on = meta.get("tx_on", {}) if isinstance(meta, dict) else {}
    rec = RxPacketRecord(
        run_id=run_id, seq=None, payload_hex=None, payload_len=None,
        rx_timestamp_utc=_now(), rx_backend="iq_only",
        crc_ok=None,
        snr_db=None,
        decode_status="not_decoded",
        raw_log_path=os.path.abspath(iq_metadata) if iq_metadata else None)
    extra = {
        "iq_on_off_delta_db": meta.get("on_off_delta_db"),
        "iq_candidate_score": on.get("lr_fhss_candidate_score"),
        "iq_signal_detected": meta.get("tx_on_stronger_than_off"),
    }
    return [rec], extra


def main(argv=None):
    ap = argparse.ArgumentParser(description="RX backend runner.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backend", choices=["mock", "log_parser", "iq_only"],
                    default="mock")
    ap.add_argument("--payloads", help="tx_payloads.csv for mock")
    ap.add_argument("--rx-log", help="RX/gateway log for log_parser")
    ap.add_argument("--iq-metadata", help="IQ comparison/summary JSON for iq_only")
    ap.add_argument("--mock-loss-rate", type=float, default=0.0)
    ap.add_argument("--mock-crc-error-rate", type=float, default=0.0)
    ap.add_argument("--mock-duplicate-rate", type=float, default=0.0)
    ap.add_argument("--mock-false-positive-count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    extra = {}
    if args.backend == "mock":
        payloads = read_csv(args.payloads) if args.payloads else []
        recs = run_mock(payloads, args.run_id, args.mock_loss_rate,
                        args.mock_crc_error_rate, args.mock_duplicate_rate,
                        args.mock_false_positive_count, args.seed)
    elif args.backend == "log_parser":
        recs = run_log_parser(args.run_id, args.rx_log)
    else:
        recs, extra = run_iq_only(args.run_id, args.iq_metadata)

    write_csv(os.path.join(args.out, "rx_records.csv"),
              [r.to_dict() for r in recs])
    decoded = sum(1 for r in recs if r.decode_status == "decoded")
    summary = {"run_id": args.run_id, "rx_backend": args.backend,
               "n_rx_records": len(recs), "n_decoded": decoded,
               "decoding_available": args.backend != "iq_only" and decoded > 0,
               **extra}
    write_json(os.path.join(args.out, "rx_summary.json"), summary)
    print(f"[rx_runner] backend={args.backend} records={len(recs)} "
          f"decoded={decoded}")
    return recs


if __name__ == "__main__":
    main()
