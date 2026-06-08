"""Tests for the packet-delivery validation framework.

Runnable with pytest *or* plain python (pytest is optional in this env):
    python -m pytest -q tests/test_packet_validation.py
    python tests/test_packet_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hardware.packet_validation import generate_payloads, rx_runner, tx_runner
from hardware.packet_validation.evaluate_packet_delivery import evaluate
from hardware.packet_validation.schemas import REQUIRED_SUMMARY_FIELDS


def _tx_rows(n, run_id="t", seed=42):
    pls = generate_payloads.build_payloads(run_id, n, seed)
    return [{"seq": p["seq"], "payload_hex": p["payload_hex"],
             "payload_len": p["payload_len"], "tx_timestamp_utc": p["timestamp_utc"]}
            for p in pls], pls


def test_mock_per_approx():
    tx, pls = _tx_rows(100)
    rx = [r.to_dict() for r in rx_runner.run_mock(pls, "t", 0.05, 0.01, 0.0, 0, 42)]
    s = evaluate(tx, rx, "t", "mock", "mock")
    assert s.decoding_available
    assert 0.88 <= s.packet_delivery_ratio <= 1.0
    assert abs(s.packet_error_rate - (1 - s.packet_delivery_ratio)) < 1e-9
    assert 0.0 <= s.packet_error_rate <= 0.12


def test_missing_seq_counted():
    tx, pls = _tx_rows(100)
    rx = [r.to_dict() for r in rx_runner.run_mock(pls, "t", 0.20, 0.0, 0.0, 0, 7)]
    s = evaluate(tx, rx, "t", "mock", "mock")
    received = {int(r["seq"]) for r in rx if r["decode_status"] in ("decoded", "crc_fail")}
    assert s.missing_seq_count == 100 - len(received)
    assert s.missing_seq_count > 0


def test_duplicates_detected():
    tx, pls = _tx_rows(50)
    rx = [r.to_dict() for r in rx_runner.run_mock(pls, "t", 0.0, 0.0, 0.5, 0, 1)]
    s = evaluate(tx, rx, "t", "mock", "mock")
    assert s.duplicate_rx_count > 0


def test_false_positives_detected():
    tx, pls = _tx_rows(20)
    rx = [r.to_dict() for r in rx_runner.run_mock(pls, "t", 0.0, 0.0, 0.0, 3, 1)]
    s = evaluate(tx, rx, "t", "mock", "mock")
    assert s.false_positive_count == 3


def test_iq_only_never_reports_per():
    tx, pls = _tx_rows(10)
    recs, _ = rx_runner.run_iq_only("t", None)
    rx = [r.to_dict() for r in recs]
    s = evaluate(tx, rx, "t", "file_replay", "iq_only")
    assert s.decoding_available is False
    assert s.packet_error_rate is None
    assert s.packet_delivery_ratio is None
    assert "PER unavailable" in s.notes


def test_tx_log_parser_no_rx_inference(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    log = os.path.join(d, "uart.log")
    with open(log, "w") as fh:
        fh.write("Packet to send: 0001\nPacket sent!\nPacket to send: 0002\nPacket sent!\n")
    recs = tx_runner.run_file_replay([], "t", log, os.path.join(d, "out.log"))
    assert len(recs) == 2                        # detected two sends
    assert all(r.uart_status == "sent_replayed" for r in recs)
    # pairing with iq_only RX must NOT yield delivery
    iq, _ = rx_runner.run_iq_only("t", None)
    s = evaluate([r.to_dict() for r in recs], [r.to_dict() for r in iq], "t",
                 "file_replay", "iq_only")
    assert s.packet_error_rate is None


def test_summary_schema_fields():
    tx, pls = _tx_rows(10)
    rx = [r.to_dict() for r in rx_runner.run_mock(pls, "t", 0.1, 0.0, 0.0, 0, 1)]
    d = evaluate(tx, rx, "t", "mock", "mock").to_dict()
    for f in REQUIRED_SUMMARY_FIELDS:
        assert f in d, f"missing field {f}"


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
