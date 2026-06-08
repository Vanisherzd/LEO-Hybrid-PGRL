"""Tests for the multi-format RX log parser + decoded-log evaluation.

Runnable with pytest OR plain python:
    python tests/test_rx_log_parser.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hardware.packet_validation import generate_payloads, generate_rx_log, rx_runner
from hardware.packet_validation.rx_log_parser import parse_rx_log
from hardware.packet_validation.evaluate_packet_delivery import evaluate


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def test_parse_format_a():
    p = _write("RX seq=12 payload=abcd1234 crc=ok rssi=-91.5 snr=7.2 cfo=123.4 "
               "timestamp=2026-06-08T12:00:01.234Z\n")
    recs, st = parse_rx_log("t", p)
    assert len(recs) == 1 and st["n_parsed"] == 1
    r = recs[0]
    assert r.seq == 12 and r.payload_hex == "abcd1234" and r.crc_ok is True
    assert r.rssi_dbm == -91.5 and r.snr_db == 7.2 and r.cfo_hz == 123.4
    assert r.rx_timestamp_utc == "2026-06-08T12:00:01.234Z"


def test_parse_format_b():
    p = _write("[RX] seq:12 payload:abcd1234 CRC_OK RSSI=-91.5 SNR=7.2 CFO=123.4\n")
    recs, st = parse_rx_log("t", p)
    assert len(recs) == 1 and recs[0].crc_ok is True
    assert recs[0].seq == 12 and recs[0].rssi_dbm == -91.5


def test_parse_format_c():
    p = _write("packet_received seq=12 len=16 payload_hex=abcd1234 crc_ok=true\n")
    recs, st = parse_rx_log("t", p)
    assert len(recs) == 1 and recs[0].crc_ok is True and recs[0].payload_hex == "abcd1234"


def test_parse_jsonl():
    p = _write('{"seq":12,"payload_hex":"abcd1234","crc_ok":true,"rssi_dbm":-91.5,'
               '"snr_db":7.2,"cfo_hz":123.4,"timestamp":"2026-06-08T12:00:01.234Z"}\n')
    recs, st = parse_rx_log("t", p)
    assert len(recs) == 1 and recs[0].crc_ok is True and recs[0].cfo_hz == 123.4


def test_unparseable_counted():
    p = _write("garbage line\nRX seq=1 payload=aa crc=ok\nnonsense\n")
    recs, st = parse_rx_log("t", p)
    assert st["n_parsed"] == 1 and st["n_unparseable"] == 2


def _tx(n=50, seed=42):
    pls = generate_payloads.build_payloads("t", n, seed)
    rows = [{"seq": p["seq"], "payload_hex": p["payload_hex"],
             "payload_len": p["payload_len"], "tx_timestamp_utc": p["timestamp_utc"]}
            for p in pls]
    return rows, pls


def test_crc_fail_not_delivered():
    tx, pls = _tx(5)
    good = pls[0]["payload_hex"]
    p = _write(f"RX seq=0 payload={good} crc=ok\n"
               f"RX seq=1 payload={pls[1]['payload_hex'][:-2]}ff crc=fail\n")
    recs, _ = parse_rx_log("t", p)
    s = evaluate(tx, [r.to_dict() for r in recs], "t", "ext", "log_parser")
    assert s.n_payload_match == 1          # only the crc=ok matching packet
    assert s.packet_delivery_ratio == 1 / 5


def test_duplicate_detected_by_evaluator():
    tx, pls = _tx(5)
    g0 = pls[0]["payload_hex"]
    p = _write(f"RX seq=0 payload={g0} crc=ok\nRX seq=0 payload={g0} crc=ok\n")
    recs, _ = parse_rx_log("t", p)
    s = evaluate(tx, [r.to_dict() for r in recs], "t", "ext", "log_parser")
    assert s.duplicate_rx_count == 1


def test_false_positive_detected():
    tx, pls = _tx(5)
    g0 = pls[0]["payload_hex"]
    p = _write(f"RX seq=0 payload={g0} crc=ok\nRX seq=9999 payload=deadbeef crc=ok\n")
    recs, _ = parse_rx_log("t", p)
    s = evaluate(tx, [r.to_dict() for r in recs], "t", "ext", "log_parser")
    assert s.false_positive_count == 1


def test_synthetic_rx_log_per_approx():
    tx, pls = _tx(200)
    lines = generate_rx_log.generate(tx, "A", loss_rate=0.05, crc_error_rate=0.01,
                                     duplicate_rate=0.0, false_positive_rate=0.0,
                                     seed=42)
    p = _write("\n".join(lines) + "\n")
    recs, _ = parse_rx_log("t", p)
    s = evaluate(tx, [r.to_dict() for r in recs], "t", "ext", "log_parser")
    assert s.decoding_available
    assert 0.0 <= s.packet_error_rate <= 0.15   # ~loss+crc


def test_iq_only_never_per():
    tx, pls = _tx(10)
    recs, _ = rx_runner.run_iq_only("t", None)
    s = evaluate(tx, [r.to_dict() for r in recs], "t", "file_replay", "iq_only")
    assert s.packet_error_rate is None and "PER unavailable" in s.notes


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
