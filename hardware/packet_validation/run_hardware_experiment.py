"""Automated conducted-hardware experiment runner (preflight-gated).

Always runs preflight first. Hardware modes require --armed. Maps the YAML
experiment config onto the packet-validation pipeline, then writes manifest.yaml
+ command_log.txt alongside the standard summary files.

  dryrun     : mock TX + mock RX, no hardware.
  iq_only    : reports PER UNAVAILABLE (signal detection only).
  decoded_rx : computes PER only from decoded RX payloads.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation import _io, _yaml  # noqa: E402
from hardware.packet_validation import run_packet_validation as rpv  # noqa: E402
from hardware.packet_validation.hardware_experiment_config import load_config  # noqa: E402
from hardware.packet_validation.preflight import run_preflight  # noqa: E402


def _build_argv(cfg, out):
    argv = ["--run-id", cfg.run_id, "--n-packets", str(cfg.n_packets),
            "--seed", str(cfg.seed), "--out", out]
    tx = cfg.tx.get("backend", "mock")
    argv += ["--tx-backend", tx]
    if "freq_hz" in cfg.tx:
        argv += ["--freq-hz", str(cfg.tx["freq_hz"])]
    if "power_dbm" in cfg.tx:
        argv += ["--power-dbm", str(cfg.tx["power_dbm"])]
    if tx == "file_replay":
        argv += ["--tx-log", cfg.tx["tx_log"]]
    if tx == "lr1121_uart":
        argv += ["--i-have-hardware"]

    if cfg.mode == "dryrun":
        argv += ["--rx-backend", "mock"]
        m = cfg.mock
        argv += ["--mock-loss-rate", str(m.get("loss_rate", 0.0)),
                 "--mock-crc-error-rate", str(m.get("crc_error_rate", 0.0)),
                 "--mock-duplicate-rate", str(m.get("duplicate_rate", 0.0)),
                 "--mock-false-positive-count", str(m.get("false_positive_count", 0))]
    elif cfg.mode == "iq_only":
        argv += ["--rx-backend", "iq_only",
                 "--iq-metadata", cfg.rx["iq_metadata_path"]]
    elif cfg.mode == "decoded_rx":
        argv += ["--rx-backend", "log_parser", "--rx-log", cfg.rx["rx_log_path"]]
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Conducted-hardware experiment runner.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--armed", action="store_true",
                    help="acknowledge conducted/shielded hardware (required for "
                         "iq_only / decoded_rx)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    _io.ensure_dir(args.out)

    ok, report = run_preflight(cfg, args.armed)
    _io.write_json(os.path.join(args.out, "preflight_report.json"), report)
    if not ok:
        print(f"[run_hardware] PREFLIGHT FAILED for mode={cfg.mode}:")
        for f in report["fatal"]:
            print(f"  - {f}")
        return 1

    pv_argv = _build_argv(cfg, args.out)
    cmd_str = "run_packet_validation " + " ".join(pv_argv)
    with open(os.path.join(args.out, "command_log.txt"), "w") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}\n{cmd_str}\n")

    summary = rpv.main(pv_argv)

    manifest = {
        "run_id": cfg.run_id,
        "mode": cfg.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "armed": args.armed,
        "safety": {"conducted_or_shielded": bool(cfg.safety.get("conducted_or_shielded")),
                   "acknowledged_by": cfg.safety.get("acknowledged_by")},
        "decoding_available": summary.decoding_available,
        "packet_error_rate": ("null" if summary.packet_error_rate is None
                              else summary.packet_error_rate),
        "claims": {
            # measured PER / packet delivery only for a real decoded-RX run;
            # dryrun is a software self-test, iq_only is signal-detection only.
            "packet_delivery": cfg.mode == "decoded_rx" and bool(summary.decoding_available),
            "measured_per": cfg.mode == "decoded_rx" and bool(summary.decoding_available),
            "iq_signal_detection_only": cfg.mode == "iq_only",
            "software_selftest_only": cfg.mode == "dryrun",
        },
        "note": ("PER reported only when decoded RX payloads exist; dryrun is a "
                 "software self-test and iq_only is signal-detection only, so "
                 "neither constitutes hardware PER validation."),
    }
    _io.ensure_dir(args.out)
    with open(os.path.join(args.out, "manifest.yaml"), "w") as fh:
        fh.write(_yaml.dumps(manifest) + "\n")
    print(f"[run_hardware] done mode={cfg.mode} out={args.out} "
          f"PER={summary.packet_error_rate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
