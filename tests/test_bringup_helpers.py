"""Tests for two-board bring-up helpers (no hardware, no pyserial required).

Runnable with pytest OR plain python:
    python tests/test_bringup_helpers.py
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hardware.packet_validation import list_serial_ports, capture_uart_log
from hardware.packet_validation.check_rx_log_format import check
from hardware.packet_validation.prepare_real_run_config import prepare

SAMPLE = "hardware/packet_validation/examples/sample_decoded_rx_format_a.log"
TEMPLATE = "hardware/packet_validation/real_configs/real_runA_alpha0_decoded_rx.yaml"


def test_check_rx_log_format_usable():
    rep = check(SAMPLE)
    assert rep["usable_for_decoded_rx"] is True
    assert rep["parsed_records"] == 5
    assert rep["crc_ok"] >= 1 and rep["crc_fail"] >= 1
    assert rep["duplicate_seq"] >= 1


def test_prepare_real_run_config_patches():
    out = os.path.join(tempfile.mkdtemp(), "runA_config.yaml")
    cfg = prepare(TEMPLATE, out, rx_log=SAMPLE, operator="Test Op", force=True)
    assert cfg.mode == "decoded_rx"
    assert cfg.rx.get("rx_log_path") == SAMPLE
    txt = open(out).read()
    assert "Test Op" in txt and "REPLACE_WITH_REAL_RX_LOG_PATH" not in txt


def test_prepare_refuses_overwrite_without_force():
    out = os.path.join(tempfile.mkdtemp(), "runA_config.yaml")
    prepare(TEMPLATE, out, rx_log=SAMPLE, operator="Test Op", force=True)
    raised = False
    try:
        prepare(TEMPLATE, out, rx_log=SAMPLE, operator="Test Op")  # no force
    except SystemExit:
        raised = True
    assert raised, "prepare must refuse overwrite without --force"


def test_list_serial_ports_runs():
    ports = list_serial_ports.main([])  # must not crash with/without hardware
    assert isinstance(ports, list)


def test_capture_uart_help_no_pyserial():
    raised = 0
    try:
        capture_uart_log.main(["--help"])
    except SystemExit as e:
        raised = e.code
    assert raised == 0, "--help should exit 0 without needing pyserial/hardware"


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
