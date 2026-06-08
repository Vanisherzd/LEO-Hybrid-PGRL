"""Evaluate packet delivery / PER from tx_records.csv + rx_records.csv.

PER is reported ONLY when receiver-side decoding exists. If the RX backend is
iq_only (or no decoded payload is present), the report states:
    "PER unavailable: no receiver-side packet decoding."
and packet_error_rate / packet_delivery_ratio are left null.

Outputs: <out>/packet_validation_summary.{json,csv,md}
"""
from __future__ import annotations

import argparse
import os
import sys
import statistics
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation._io import (  # noqa: E402
    ensure_dir, read_csv, read_json, write_csv, write_json)
from hardware.packet_validation.schemas import (  # noqa: E402
    PacketValidationSummary, REQUIRED_SUMMARY_FIELDS)

PER_UNAVAILABLE_MSG = "PER unavailable: no receiver-side packet decoding."


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def evaluate(tx_rows, rx_rows, run_id, tx_backend=None, rx_backend=None):
    s = PacketValidationSummary(run_id=run_id, tx_backend=tx_backend,
                                rx_backend=rx_backend)
    s.n_tx = len(tx_rows)
    tx_by_seq = {int(r["seq"]): r["payload_hex"] for r in tx_rows if r.get("seq") not in (None, "")}
    tx_seqs = set(tx_by_seq)

    # RX decode attempts = records carrying a decode verdict on a payload.
    attempts = [r for r in rx_rows
                if r.get("decode_status") in ("decoded", "crc_fail")
                and r.get("seq") not in (None, "")]
    decoding_available = any(r.get("decode_status") == "decoded" for r in attempts)
    s.decoding_available = decoding_available
    s.n_rx = len(attempts)

    if not decoding_available:
        s.packet_delivery_ratio = None
        s.packet_error_rate = None
        s.crc_error_rate = None
        s.notes = PER_UNAVAILABLE_MSG
        # still surface IQ evidence if present in rx rows
        return s

    decoded = [r for r in attempts if r.get("decode_status") == "decoded"]
    crc_fail = [r for r in attempts if r.get("decode_status") == "crc_fail"]
    s.n_crc_ok = len(decoded)

    # matched-good = unique tx seq whose decoded rx payload matches tx payload
    matched_seqs = set()
    seen_decoded_seq = {}
    false_pos = 0
    latencies_ms = []
    for r in decoded:
        seq = int(r["seq"])
        seen_decoded_seq[seq] = seen_decoded_seq.get(seq, 0) + 1
        if seq not in tx_seqs:
            false_pos += 1
            continue
        if r.get("payload_hex") == tx_by_seq[seq]:
            matched_seqs.add(seq)
            t_tx = _parse_ts(next((t.get("tx_timestamp_utc") for t in tx_rows
                                   if t.get("seq") not in (None, "")
                                   and int(t["seq"]) == seq), None))
            t_rx = _parse_ts(r.get("rx_timestamp_utc"))
            if t_tx and t_rx:
                latencies_ms.append((t_rx - t_tx).total_seconds() * 1e3)

    s.n_payload_match = len(matched_seqs)
    s.packet_delivery_ratio = s.n_payload_match / s.n_tx if s.n_tx else None
    s.packet_error_rate = (1.0 - s.packet_delivery_ratio
                           if s.packet_delivery_ratio is not None else None)
    s.crc_error_rate = len(crc_fail) / s.n_rx if s.n_rx else 0.0
    # duplicates: decoded records beyond the first per in-set seq
    s.duplicate_rx_count = sum(c - 1 for sq, c in seen_decoded_seq.items()
                               if sq in tx_seqs and c > 1)
    received_inset = {int(r["seq"]) for r in attempts
                      if int(r["seq"]) in tx_seqs}
    s.missing_seq_count = len(tx_seqs - received_inset)
    s.false_positive_count = false_pos
    s.median_latency_ms = (round(statistics.median(latencies_ms), 4)
                           if latencies_ms else None)
    s.notes = "OK: receiver decoding available; PER computed."
    return s


def write_reports(out_dir, summary: PacketValidationSummary, iq_summary=None):
    ensure_dir(out_dir)
    d = summary.to_dict()
    write_json(os.path.join(out_dir, "packet_validation_summary.json"), d)
    write_csv(os.path.join(out_dir, "packet_validation_summary.csv"), [d])

    def fmt(v):
        return "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)

    lines = [f"# Packet Validation Summary: {summary.run_id}", "",
             f"- tx_backend: {summary.tx_backend}",
             f"- rx_backend: {summary.rx_backend}",
             f"- decoding_available: {summary.decoding_available}", ""]
    if not summary.decoding_available:
        lines += [f"**{PER_UNAVAILABLE_MSG}**", "",
                  f"- n_tx: {summary.n_tx}",
                  f"- n_rx (decode attempts): {summary.n_rx}"]
        if iq_summary:
            lines += ["", "## IQ signal-detection evidence (not packet delivery)",
                      f"- on_off_delta_db: {iq_summary.get('iq_on_off_delta_db') or iq_summary.get('on_off_delta_db')}",
                      f"- candidate_score: {iq_summary.get('iq_candidate_score')}",
                      f"- signal_detected: {iq_summary.get('iq_signal_detected')}"]
    else:
        for k in REQUIRED_SUMMARY_FIELDS:
            lines.append(f"- {k}: {fmt(d.get(k))}")
    with open(os.path.join(out_dir, "packet_validation_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate packet delivery / PER.")
    ap.add_argument("--tx-records", required=True)
    ap.add_argument("--rx-records", required=True)
    ap.add_argument("--iq-summary", help="optional rx_summary.json from iq_only")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tx-backend")
    ap.add_argument("--rx-backend")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tx_rows = read_csv(args.tx_records)
    rx_rows = read_csv(args.rx_records)
    # normalise crc/decode strings from CSV
    for r in rx_rows:
        if "decode_status" not in r:
            r["decode_status"] = "not_decoded"
    iq_summary = read_json(args.iq_summary) if args.iq_summary and os.path.exists(args.iq_summary) else None
    summary = evaluate(tx_rows, rx_rows, args.run_id, args.tx_backend,
                       args.rx_backend)
    write_reports(args.out, summary, iq_summary)
    print(f"[evaluate] decoding_available={summary.decoding_available} "
          f"PER={summary.packet_error_rate} PDR={summary.packet_delivery_ratio}")
    return summary


if __name__ == "__main__":
    main()
