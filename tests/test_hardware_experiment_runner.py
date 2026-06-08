"""Tests for the automated conducted-hardware experiment runner.

Runnable with pytest OR plain python:
    python tests/test_hardware_experiment_runner.py

All paths are repo-root-relative; scratch files live under validation_runs/
(gitignored) so nothing is ever committed by the tests.
"""
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hardware.packet_validation import generate_payloads, generate_rx_log
from hardware.packet_validation import run_hardware_experiment as rhe
from hardware.packet_validation import finalize_hardware_manifest as fin
from hardware.packet_validation.hardware_experiment_config import (
    load_config, ConfigError)
from hardware.packet_validation.preflight import run_preflight

SCRATCH = os.path.join(ROOT, "validation_runs", "_test_hw")
IQ_META = "docs/hardware_iq_capture_stage5/cap_868000000_txrx_comparison.json"
TX_LOG = "docs/hardware_iq_capture_stage5/cap_868000000_txrx_on_uart.log"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _dryrun_cfg():
    return _write(os.path.join(SCRATCH, "dryrun.yaml"),
                  "run_id: t_dry\nmode: dryrun\nn_packets: 50\nseed: 42\n"
                  "tx:\n  backend: mock\nrx:\n  backend: mock\n"
                  "mock:\n  loss_rate: 0.05\n  crc_error_rate: 0.0\n")


def _iq_cfg():
    return _write(os.path.join(SCRATCH, "iq.yaml"),
                  "run_id: t_iq\nmode: iq_only\nn_packets: 10\nseed: 42\n"
                  "safety:\n  conducted_or_shielded: true\n  acknowledged_by: tester\n"
                  f"tx:\n  backend: file_replay\n  tx_log: {TX_LOG}\n"
                  f"rx:\n  backend: iq_only\n  iq_metadata_path: {IQ_META}\n")


def test_dryrun_passes_preflight_without_armed():
    cfg = load_config(_dryrun_cfg())
    ok, rep = run_preflight(cfg, armed=False)
    assert ok, rep["fatal"]


def test_hardware_fails_preflight_without_armed():
    cfg = load_config(_iq_cfg())
    ok, rep = run_preflight(cfg, armed=False)
    assert not ok
    assert any("--armed" in f for f in rep["fatal"])


def test_hardware_passes_preflight_with_armed():
    cfg = load_config(_iq_cfg())
    ok, rep = run_preflight(cfg, armed=True)
    assert ok, rep["fatal"]


def test_iq_only_run_reports_per_unavailable():
    out = os.path.join(SCRATCH, "run_iq")
    rc = rhe.main(["--config", _iq_cfg(), "--out", out, "--armed"])
    assert rc == 0
    s = json.load(open(os.path.join(out, "packet_validation_summary.json")))
    assert s["packet_error_rate"] is None
    assert "PER unavailable" in s["notes"]


def test_decoded_rx_synthetic_computes_per():
    run_id, seed, n = "t_dec", 42, 100
    pls = generate_payloads.build_payloads(run_id, n, seed)
    tx_rows = [{"seq": p["seq"], "payload_hex": p["payload_hex"],
                "payload_len": p["payload_len"]} for p in pls]
    lines = generate_rx_log.generate(tx_rows, "A", 0.05, 0.01, 0.0, 0.0, seed)
    rx_log_rel = "validation_runs/_test_hw/dec_rx.log"
    _write(os.path.join(ROOT, rx_log_rel), "\n".join(lines) + "\n")
    cfg = _write(os.path.join(SCRATCH, "dec.yaml"),
                 f"run_id: {run_id}\nmode: decoded_rx\nn_packets: {n}\nseed: {seed}\n"
                 "safety:\n  conducted_or_shielded: true\n  acknowledged_by: tester\n"
                 "tx:\n  backend: mock\n"
                 f"rx:\n  backend: log_parser\n  rx_log_path: {rx_log_rel}\n")
    out = os.path.join(SCRATCH, "run_dec")
    rc = rhe.main(["--config", cfg, "--out", out, "--armed"])
    assert rc == 0
    s = json.load(open(os.path.join(out, "packet_validation_summary.json")))
    assert s["decoding_available"] is True
    assert s["packet_error_rate"] is not None
    assert 0.0 <= s["packet_error_rate"] <= 0.15


def test_finalize_separates_commit_vs_do_not_commit():
    out = os.path.join(SCRATCH, "run_fin")
    rhe.main(["--config", _dryrun_cfg(), "--out", out])
    # drop a fake raw IQ to ensure it is classified do-not-commit
    _write(os.path.join(out, "fake_capture.fc32"), "x" * 16)
    res = fin.finalize(out)
    assert res["n_commit"] > 0
    assert any(p.endswith(".fc32") for p in res["do_not_commit"])
    assert all(not p.endswith(".fc32") for p in res["commit"])


def test_unsafe_hardware_mode_rejected():
    bad = _write(os.path.join(SCRATCH, "bad.yaml"),
                 "run_id: t_bad\nmode: iq_only\nn_packets: 5\n"
                 f"rx:\n  backend: iq_only\n  iq_metadata_path: {IQ_META}\n")
    raised = False
    try:
        load_config(bad)
    except ConfigError:
        raised = True
    assert raised, "iq_only without safety.conducted_or_shielded must be rejected"


def _run_all():
    os.makedirs(SCRATCH, exist_ok=True)
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
    code = 0 if _run_all() else 1
    shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(code)
