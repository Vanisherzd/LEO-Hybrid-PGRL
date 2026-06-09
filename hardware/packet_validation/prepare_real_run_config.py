"""Patch a real-run config template into a LOCAL run config (no template edits).

Replaces placeholder values (rx_log_path, operator/acknowledged_by, stress
attenuator) by key name while preserving YAML layout, then validates the result
with hardware_experiment_config.load_config. Refuses to overwrite without --force.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation.hardware_experiment_config import load_config  # noqa: E402


def _patch_key(lines, key, value):
    """Set `key: value` for the first line whose stripped text starts with key:.
    Preserves indentation. Returns (lines, changed)."""
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*.*$")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            lines[i] = f"{m.group(1)}{key}: {value}"
            return lines, True
    return lines, False


def prepare(template, out, rx_log=None, operator=None, attenuator_db=None,
            iq_metadata=None, force=False):
    if not os.path.exists(template):
        raise SystemExit(f"template not found: {template}")
    if os.path.exists(out) and not force:
        raise SystemExit(f"refusing to overwrite {out} (use --force)")
    with open(template) as fh:
        lines = fh.read().splitlines()

    if rx_log is not None:
        lines, _ = _patch_key(lines, "rx_log_path", rx_log)
    if iq_metadata is not None:
        lines, _ = _patch_key(lines, "iq_metadata_path", iq_metadata)
    if operator is not None:
        v = f'"{operator}"' if " " in operator else operator
        lines, _ = _patch_key(lines, "operator", v)
        lines, _ = _patch_key(lines, "acknowledged_by", v)
    if attenuator_db is not None:
        lines, _ = _patch_key(lines, "attenuator_db", attenuator_db)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    cfg = load_config(out)  # validates schema (raises ConfigError on problems)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare a local real-run config.")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rx-log")
    ap.add_argument("--iq-metadata")
    ap.add_argument("--operator")
    ap.add_argument("--attenuator-db")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = prepare(args.template, args.out, rx_log=args.rx_log,
                  operator=args.operator, attenuator_db=args.attenuator_db,
                  iq_metadata=args.iq_metadata, force=args.force)
    out_dir = cfg.raw.get("output_dir") or f"validation_runs/{cfg.run_id}"
    print(f"[prepare] wrote {args.out} (run_id={cfg.run_id}, mode={cfg.mode})")
    print("Run it with:")
    print(f"  python hardware/packet_validation/run_hardware_experiment.py "
          f"--config {args.out} --out {out_dir} --armed")
    return cfg


if __name__ == "__main__":
    main()
