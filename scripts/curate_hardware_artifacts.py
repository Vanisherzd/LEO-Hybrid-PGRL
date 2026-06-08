#!/usr/bin/env python3
"""
curate_hardware_artifacts.py — Copy small SDR evidence artifacts out of a raw
capture sweep directory into a curated, git-committable location.

Raw IQ (.fc32/.dat) and compiled binaries are NEVER copied. Only small evidence
files (.png/.json/.csv/.log/.md) are curated. A README.md and artifact_index.json
are generated from sweep_summary.json + the ON analysis JSON when available.

Usage:
    uv run python scripts/curate_hardware_artifacts.py \
        --source hardware/captures/auto_sweep_20260604_000358 \
        --name lr1121_signal_detected_20260604_000358
"""
import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = REPO_ROOT / "hardware" / "artifacts"

# Small evidence files we curate.
COPY_EXTS = {".png", ".json", ".csv", ".log", ".md"}
# Raw IQ / large binary we must never copy.
NEVER_EXTS = {".fc32", ".dat", ".bin", ".npy", ".h5"}


def is_binary_no_ext(path: Path) -> bool:
    """Treat extensionless files as compiled binaries — never copy."""
    return path.suffix == ""


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def find_detected_row(summary: dict):
    """Return the signal-detected capture row (or first row) from sweep_summary.json."""
    if not summary:
        return {}
    rows = summary.get("captures", []) or []
    for r in rows:
        if r.get("signal_detected") is True:
            return r
    return rows[0] if rows else {}


def find_on_analysis(source: Path):
    """Prefer *_on_analysis.json; fall back to any *_analysis.json."""
    on = sorted(source.glob("*_on_analysis.json"))
    if on:
        return load_json(on[0]) or {}
    any_a = sorted(source.glob("*_analysis.json"))
    return (load_json(any_a[0]) or {}) if any_a else {}


def parse_timestamp(name: str, source: Path) -> str:
    """Extract a YYYYMMDD_HHMMSS stamp from the name/source and format it."""
    m = re.search(r"(\d{8})_(\d{6})", name) or re.search(r"(\d{8})_(\d{6})", source.name)
    if not m:
        return "unknown"
    d, t = m.group(1), m.group(2)
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}:{t[4:6]} local"


def build_readme(name: str, source: Path, row: dict, an: dict,
                 tx_power_dbm, packet_interval_ms) -> str:
    g = lambda d, k, default="--": d.get(k, default) if d.get(k) is not None else default
    freq = g(row, "frequency_hz", g(an, "input_file"))
    when = parse_timestamp(name, source)
    return f"""# LR1121 LR-FHSS SDR signal-detected ON/OFF control

**Experiment:** LR1121 LR-FHSS SDR signal-detected ON/OFF control
**Date/time:** {when}
**Curated from:** `{source.relative_to(REPO_ROOT)}`

## Hardware
- Semtech LR1121 + NUCLEO-L476RG
- USRP B210
- SWDM001 firmware, target `lr1121_xtal`

## Firmware settings
- RF = {freq} Hz
- TX power = {tx_power_dbm} dBm
- packet interval = {packet_interval_ms} ms

## SDR settings
- antenna = {g(row, "antenna")}
- sample rate = {g(row, "rate")} sps (1 Msps)
- gain = {g(row, "gain")} dB
- duration = {g(row, "duration")} s

## Validation results
- validation_status = {g(row, "validation_status", g(an, "validation_status"))}
- signal_detected = {g(row, "signal_detected", g(an, "signal_detected"))}
- UART packet count = {g(row, "uart_packet_sent_count", g(an, "uart_packet_sent_count"))}
- ON/OFF delta = {g(row, "on_off_delta_db")} dB
- TX-ON stronger than TX-OFF = {g(row, "tx_on_stronger_than_off")}
- LR-FHSS candidate score = {g(an, "lr_fhss_candidate_score")}
- hop-like segments = {g(an, "hop_like_segment_count")}
- occupied frequency bins = {g(an, "occupied_frequency_bins")}
- max-hold peaks = {g(an, "maxhold_peak_count_excluding_dc")}

## Claim boundary
- This is **IQ-level SDR signal detection** only.
- This is **NOT** standard-compliant LR-FHSS decoding.
- This is **NOT** a PER (packet-error-rate) measurement.
- This is **NOT** a full LR-FHSS gateway receiver.

Raw `.fc32` IQ is intentionally NOT included here (git-ignored, kept under
`hardware/captures/`). Only small evidence artifacts are curated.
"""


def main():
    ap = argparse.ArgumentParser(description="Curate small SDR evidence artifacts.")
    ap.add_argument("--source", required=True, help="Source raw capture directory.")
    ap.add_argument("--name", required=True, help="Curated artifact name (subdir under hardware/artifacts/).")
    ap.add_argument("--tx-power-dbm", default="10", help="Firmware TX power for README (default 10).")
    ap.add_argument("--packet-interval-ms", default="1000", help="Firmware packet interval for README (default 1000).")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()
    if not source.is_dir():
        print(f"ERROR: source directory not found: {source}", file=sys.stderr)
        sys.exit(1)

    curated = ARTIFACTS_ROOT / args.name
    curated.mkdir(parents=True, exist_ok=True)

    copied, excluded = [], []
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        ext = item.suffix.lower()
        if ext in NEVER_EXTS or is_binary_no_ext(item):
            excluded.append({"file": item.name, "reason": "raw_iq_or_binary", "size_bytes": item.stat().st_size})
            continue
        if ext in COPY_EXTS:
            shutil.copy2(item, curated / item.name)
            copied.append(item.name)
        else:
            excluded.append({"file": item.name, "reason": "not_in_copy_whitelist", "size_bytes": item.stat().st_size})

    summary = load_json(source / "sweep_summary.json") or {}
    row = find_detected_row(summary)
    an = find_on_analysis(source)

    # README (generated; do not copy a stale one over it).
    readme = build_readme(args.name, source, row, an, args.tx_power_dbm, args.packet_interval_ms)
    (curated / "README.md").write_text(readme, encoding="utf-8")
    if "README.md" not in copied:
        copied.append("README.md")

    index = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_directory": str(source.relative_to(REPO_ROOT)) if str(source).startswith(str(REPO_ROOT)) else str(source),
        "curated_directory": str(curated.relative_to(REPO_ROOT)),
        "copied_files": sorted(copied),
        "excluded_raw_files": excluded,
        "validation_status": row.get("validation_status") or an.get("validation_status"),
        "signal_detected": row.get("signal_detected") if row.get("signal_detected") is not None else an.get("signal_detected"),
        "key_metrics": {
            "frequency_hz": row.get("frequency_hz"),
            "antenna": row.get("antenna"),
            "uart_packet_sent_count": row.get("uart_packet_sent_count"),
            "on_off_delta_db": row.get("on_off_delta_db"),
            "tx_on_stronger_than_off": row.get("tx_on_stronger_than_off"),
            "lr_fhss_candidate_score": an.get("lr_fhss_candidate_score"),
            "hop_like_segment_count": an.get("hop_like_segment_count"),
            "occupied_frequency_bins": an.get("occupied_frequency_bins"),
            "maxhold_peak_count_excluding_dc": an.get("maxhold_peak_count_excluding_dc"),
        },
    }
    (curated / "artifact_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[curate] Curated -> {curated.relative_to(REPO_ROOT)}")
    print(f"[curate] Copied {len(copied)} small files; excluded {len(excluded)} raw/binary files.")
    for e in excluded:
        print(f"[curate]   excluded {e['file']} ({e['reason']}, {e['size_bytes']} bytes)")
    print(f"[curate] validation_status={index['validation_status']} signal_detected={index['signal_detected']}")


if __name__ == "__main__":
    main()
