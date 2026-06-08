"""One-command packet-delivery validation pipeline (default: dry-run/mock).

generate payloads -> run TX backend -> run RX backend -> evaluate -> manifest.

Examples
--------
Mock end-to-end (can compute PER):
  python hardware/packet_validation/run_packet_validation.py \
    --run-id dryrun_001 --n-packets 100 \
    --tx-backend mock --rx-backend mock \
    --mock-loss-rate 0.05 --mock-crc-error-rate 0.01 \
    --out validation_runs/dryrun_001

IQ-only replay (PER MUST be unavailable):
  python hardware/packet_validation/run_packet_validation.py \
    --run-id iq_only_001 --n-packets 10 \
    --tx-backend file_replay \
    --tx-log docs/hardware_iq_capture_stage5/cap_868000000_txrx_on_uart.log \
    --rx-backend iq_only \
    --iq-metadata docs/hardware_iq_capture_stage5/cap_868000000_txrx_comparison.json \
    --out validation_runs/iq_only_001
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation import generate_payloads, tx_runner, rx_runner  # noqa: E402
from hardware.packet_validation import evaluate_packet_delivery as evalmod  # noqa: E402
from hardware.packet_validation._io import ensure_dir, write_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run packet-delivery validation.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-packets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tx-backend", choices=["mock", "file_replay", "lr1121_uart"],
                    default="mock")
    ap.add_argument("--rx-backend", choices=["mock", "log_parser", "iq_only"],
                    default="mock")
    ap.add_argument("--tx-log")
    ap.add_argument("--rx-log")
    ap.add_argument("--iq-metadata")
    ap.add_argument("--freq-hz", type=float, default=868_000_000.0)
    ap.add_argument("--power-dbm", type=float, default=10.0)
    ap.add_argument("--mock-loss-rate", type=float, default=0.0)
    ap.add_argument("--mock-crc-error-rate", type=float, default=0.0)
    ap.add_argument("--mock-duplicate-rate", type=float, default=0.0)
    ap.add_argument("--mock-false-positive-count", type=int, default=0)
    ap.add_argument("--i-have-hardware", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    payloads_csv = os.path.join(args.out, "tx_payloads.csv")

    generate_payloads.main(["--run-id", args.run_id, "--n-packets",
                            str(args.n_packets), "--seed", str(args.seed),
                            "--out", args.out])

    tx_argv = ["--run-id", args.run_id, "--backend", args.tx_backend,
               "--payloads", payloads_csv, "--freq-hz", str(args.freq_hz),
               "--power-dbm", str(args.power_dbm), "--out", args.out]
    if args.tx_log:
        tx_argv += ["--tx-log", args.tx_log]
    if args.i_have_hardware:
        tx_argv += ["--i-have-hardware"]
    tx_runner.main(tx_argv)

    rx_argv = ["--run-id", args.run_id, "--backend", args.rx_backend,
               "--payloads", payloads_csv, "--seed", str(args.seed),
               "--mock-loss-rate", str(args.mock_loss_rate),
               "--mock-crc-error-rate", str(args.mock_crc_error_rate),
               "--mock-duplicate-rate", str(args.mock_duplicate_rate),
               "--mock-false-positive-count", str(args.mock_false_positive_count),
               "--out", args.out]
    if args.rx_log:
        rx_argv += ["--rx-log", args.rx_log]
    if args.iq_metadata:
        rx_argv += ["--iq-metadata", args.iq_metadata]
    rx_runner.main(rx_argv)

    summary = evalmod.main([
        "--tx-records", os.path.join(args.out, "tx_records.csv"),
        "--rx-records", os.path.join(args.out, "rx_records.csv"),
        "--iq-summary", os.path.join(args.out, "rx_summary.json"),
        "--run-id", args.run_id, "--tx-backend", args.tx_backend,
        "--rx-backend", args.rx_backend, "--out", args.out])

    manifest = {
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tx_backend": args.tx_backend, "rx_backend": args.rx_backend,
        "n_packets": args.n_packets, "seed": args.seed,
        "decoding_available": summary.decoding_available,
        "packet_error_rate": summary.packet_error_rate,
        "outputs": sorted(os.listdir(args.out)),
        "note": ("DRY-RUN / conducted-only framework. iq_only and file_replay "
                 "paths do not claim PER."),
    }
    write_json(os.path.join(args.out, "manifest.json"), manifest)
    print(f"[run] done run_id={args.run_id} out={args.out} "
          f"PER={summary.packet_error_rate}")
    return summary


if __name__ == "__main__":
    main()
