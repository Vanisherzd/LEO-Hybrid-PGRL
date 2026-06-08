"""Pre-flight safety + environment checks for the hardware experiment runner.

Hardware modes (iq_only, decoded_rx) FAIL unless --armed is passed. Also checks
that run outputs and raw IQ are gitignored, required scripts exist, and the
mode's configured input paths exist. Optional: detect uhd_find_devices.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation import _io  # noqa: E402
from hardware.packet_validation.hardware_experiment_config import (  # noqa: E402
    load_config, ExperimentConfig)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REQUIRED_SCRIPTS = [
    "hardware/packet_validation/generate_payloads.py",
    "hardware/packet_validation/tx_runner.py",
    "hardware/packet_validation/rx_runner.py",
    "hardware/packet_validation/rx_log_parser.py",
    "hardware/packet_validation/evaluate_packet_delivery.py",
    "hardware/packet_validation/run_packet_validation.py",
]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO_ROOT, *args],
                              capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        return None


def _is_ignored(relpath: str) -> bool:
    r = _git("check-ignore", "-q", relpath)
    return bool(r) and r.returncode == 0


def run_preflight(cfg: ExperimentConfig, armed: bool) -> tuple[bool, dict]:
    checks = {}
    fatal = []

    br = _git("rev-parse", "--abbrev-ref", "HEAD")
    checks["branch"] = (br.stdout.strip() if br and br.returncode == 0 else "unknown")
    st = _git("status", "--porcelain")
    checks["working_tree_dirty"] = bool(st and st.stdout.strip())

    checks["validation_runs_gitignored"] = _is_ignored("validation_runs/_probe")
    checks["raw_iq_fc32_gitignored"] = _is_ignored("probe.fc32")
    if not checks["validation_runs_gitignored"]:
        fatal.append("validation_runs/ is not gitignored")
    if not checks["raw_iq_fc32_gitignored"]:
        fatal.append("*.fc32 raw IQ is not gitignored")

    missing = [s for s in REQUIRED_SCRIPTS
               if not os.path.exists(os.path.join(REPO_ROOT, s))]
    checks["missing_scripts"] = missing
    if missing:
        fatal.append(f"missing required scripts: {missing}")

    checks["mode"] = cfg.mode
    checks["is_hardware_mode"] = cfg.is_hardware_mode
    checks["armed"] = armed
    if cfg.is_hardware_mode and not armed:
        fatal.append(
            f"mode {cfg.mode} is a hardware mode and requires --armed "
            "(conducted/shielded setup acknowledged). Refusing to proceed.")

    # configured input paths for the mode
    checks["input_paths"] = {}
    if cfg.mode == "iq_only":
        p = cfg.rx.get("iq_metadata_path")
        ok = bool(p) and os.path.exists(os.path.join(REPO_ROOT, p))
        checks["input_paths"]["iq_metadata_path"] = {"path": p, "exists": ok}
        if not ok:
            fatal.append(f"iq_metadata_path missing: {p}")
    elif cfg.mode == "decoded_rx":
        p = cfg.rx.get("rx_log_path")
        ok = bool(p) and os.path.exists(os.path.join(REPO_ROOT, p))
        checks["input_paths"]["rx_log_path"] = {"path": p, "exists": ok}
        if not ok:
            fatal.append(f"rx_log_path missing: {p}")
    if cfg.tx.get("backend") == "file_replay":
        p = cfg.tx.get("tx_log")
        ok = bool(p) and os.path.exists(os.path.join(REPO_ROOT, p))
        checks["input_paths"]["tx_log"] = {"path": p, "exists": ok}
        if not ok:
            fatal.append(f"tx_log missing: {p}")

    # optional SDR tool detection (informational only)
    checks["uhd_find_devices_available"] = shutil.which("uhd_find_devices") is not None
    if cfg.tx.get("backend") == "lr1121_uart":
        checks["note_lr1121_uart"] = "real TX backend also requires --i-have-hardware"

    ok = not fatal
    report = {"ok": ok, "fatal": fatal, "checks": checks}
    return ok, report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hardware experiment pre-flight.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--armed", action="store_true")
    ap.add_argument("--out", help="optional dir to write preflight_report.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ok, report = run_preflight(cfg, args.armed)
    if args.out:
        _io.ensure_dir(args.out)
        _io.write_json(os.path.join(args.out, "preflight_report.json"), report)
    print(f"[preflight] ok={ok}")
    for f in report["fatal"]:
        print(f"  FATAL: {f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
