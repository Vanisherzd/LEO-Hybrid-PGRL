"""Check whether an RX log is usable for decoded-RX PER evaluation.

Uses hardware.packet_validation.rx_log_parser. usable_for_decoded_rx is true iff
>=1 parsed record has seq + payload_hex + a crc verdict. RSSI/SNR/CFO/timestamp
are optional (coverage reported, not required).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation.rx_log_parser import parse_rx_log  # noqa: E402

ACCEPTED_EXAMPLES = [
    "RX seq=12 payload=abcd1234 crc=ok rssi=-91.5 snr=7.2 cfo=123.4 timestamp=2026-06-08T12:00:01.234Z",
    "[RX] seq:12 payload:abcd1234 CRC_OK RSSI=-91.5 SNR=7.2 CFO=123.4",
    "packet_received seq=12 len=16 payload_hex=abcd1234 crc_ok=true",
    '{"seq":12,"payload_hex":"abcd1234","crc_ok":true,"rssi_dbm":-91.5}',
]


def check(rx_log: str) -> dict:
    n_lines = 0
    if os.path.exists(rx_log):
        with open(rx_log) as fh:
            n_lines = sum(1 for ln in fh if ln.strip())
    recs, stats = parse_rx_log("check", rx_log)
    seqs = [r.seq for r in recs if r.seq is not None]
    uniq = set(seqs)
    dup = len(seqs) - len(uniq)
    payload_present = sum(1 for r in recs if r.payload_hex)
    cov = {
        "rssi_dbm": sum(1 for r in recs if r.rssi_dbm is not None),
        "snr_db": sum(1 for r in recs if r.snr_db is not None),
        "cfo_hz": sum(1 for r in recs if r.cfo_hz is not None),
        "rx_timestamp_utc": sum(1 for r in recs if r.rx_timestamp_utc),
    }
    usable = any(
        (r.seq is not None and r.payload_hex and r.crc_ok is not None)
        for r in recs)
    return {
        "rx_log": rx_log,
        "total_nonblank_lines": n_lines,
        "parsed_records": stats["n_parsed"],
        "unparseable_lines": stats["n_unparseable"],
        "crc_ok": stats["n_crc_ok"],
        "crc_fail": stats["n_crc_fail"],
        "unique_seq": len(uniq),
        "duplicate_seq": dup,
        "payload_hex_present": payload_present,
        "optional_field_coverage": cov,
        "usable_for_decoded_rx": usable,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check an RX log for decoded-RX use.")
    ap.add_argument("--rx-log", required=True)
    args = ap.parse_args(argv)
    rep = check(args.rx_log)
    for k, v in rep.items():
        print(f"{k}: {v}")
    if not rep["usable_for_decoded_rx"]:
        print("\nNOT usable for decoded-RX PER. Accepted line formats:")
        for ex in ACCEPTED_EXAMPLES:
            print(f"  {ex}")
    return rep


if __name__ == "__main__":
    main()
