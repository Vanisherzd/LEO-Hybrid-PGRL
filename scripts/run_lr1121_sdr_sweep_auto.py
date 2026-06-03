#!/usr/bin/env python3
"""Automated USRP B210 LR-FHSS sweep for LR1121 hardware bring-up.

Starts the rx_capture_to_file_cpp capture, optionally resets the NUCLEO board
(so its TX burst lands inside the capture window), logs UART, then runs the
analyzers. Conservative: signal_detected is ONLY ever set true when the
analyzer JSON explicitly reports signal_detected == true. Otherwise the run
stays in hardware-bringup / pending.

Build constraints: stdlib only, plus optional lazily-imported pyserial.

Typical use:
    uv run python scripts/run_lr1121_sdr_sweep_auto.py \
        --serial 8000304 --freqs "868e6,915e6,923e6" --antennas "RX2,TX/RX" \
        --rate 1e6 --gain 45 --duration 10 \
        --reset-method stlink --uart /dev/tty.usbmodem1303 --uart-baud 115200
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# Repository root = parent of this script's directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

CAPTURE_BIN = os.path.join(REPO_ROOT, "hardware", "usrp_scripts", "rx_capture_to_file_cpp")
ANALYZE_PY = os.path.join(REPO_ROOT, "hardware", "usrp_scripts", "analyze_capture.py")
QUICK_MAXHOLD_PY = os.path.join(REPO_ROOT, "hardware", "usrp_scripts", "quick_maxhold.py")

# Markers in capture stdout/stderr that indicate a UHD overflow / metadata error.
OVERFLOW_MARKERS = (
    "RX metadata error",
    "ERROR_CODE_OVERFLOW",
    "overflow",
)

CSV_HEADER = [
    "capture_file",
    "frequency_hz",
    "antenna",
    "rate",
    "gain",
    "duration",
    "actual_duration_s",
    "capture_status",
    "validation_status",
    "signal_detected",
    "peak_to_median_db",
    "burst_energy_excess_db",
    "maxhold_excess_db",
    "peak_frequency_offset_hz",
    "uart_log",
    "uart_packet_sent_count",
    "uart_seen_init",
    "note",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[AUTO-SWEEP] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[AUTO-SWEEP][WARN] {msg}", flush=True)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_int_hz(value: str) -> int:
    """Convert a possibly sci-notation rate string ('1e6') to integer Hz."""
    return int(float(value))


def sanitize_antenna(antenna: str) -> str:
    a = antenna.strip().upper()
    if a == "TX/RX":
        return "txrx"
    if a == "RX2":
        return "rx2"
    # Generic fallback: keep only filename-safe chars.
    return "".join(c if c.isalnum() else "_" for c in a.lower())


def sanitize_freq(freq: str) -> str:
    """Turn a freq string like '868e6' into '868000000' for filenames."""
    try:
        return str(int(float(freq)))
    except (ValueError, TypeError):
        return "".join(c if c.isalnum() else "_" for c in str(freq))


def python_runner() -> list:
    """Return the command prefix used to run the analyzer scripts.

    Prefer `uv run python`; fall back to the current interpreter if uv is
    not on PATH.
    """
    if shutil.which("uv"):
        return ["uv", "run", "python"]
    return [sys.executable]


# --------------------------------------------------------------------------- #
# pyserial (lazy / optional)
# --------------------------------------------------------------------------- #
def try_import_serial():
    """Lazily import pyserial. Return module or None (with a WARN)."""
    try:
        import serial  # type: ignore

        return serial
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Reset handling
# --------------------------------------------------------------------------- #
def stlink_available() -> bool:
    """Detect st-flash availability (probe only; not used to reset)."""
    if shutil.which("st-flash") is None:
        return False
    return True


def do_reset(method: str, uart_port, serial_mod) -> None:
    """Perform the post-capture-start reset for the given method.

    'manual' is handled by the caller BEFORE the capture starts, so it is a
    no-op here. 'none' is a no-op. 'stlink' / 'serial-dtr' are quick and run
    while the capture is already streaming.
    """
    if method in ("none", "manual"):
        return

    if method == "stlink":
        if not stlink_available():
            warn(
                "st-flash not found; install with `brew install stlink`; "
                "falling back to no reset for this run"
            )
            return
        try:
            subprocess.run(
                ["st-flash", "reset"],
                check=False,
                timeout=15,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log("st-flash reset issued")
        except Exception as exc:  # noqa: BLE001
            warn(f"st-flash reset failed: {exc}; continuing without reset")
        return

    if method == "serial-dtr":
        if uart_port is None:
            warn("serial-dtr reset requested but no --uart given; skipping reset")
            return
        if serial_mod is None:
            warn(
                "pyserial not installed; run `uv add pyserial`; "
                "skipping serial-dtr reset"
            )
            return
        # Best-effort: toggling DTR/RTS may not reset every NUCLEO board.
        try:
            ser = serial_mod.Serial(uart_port)
            try:
                ser.dtr = False
                ser.rts = True
                time.sleep(0.1)
                ser.dtr = True
                ser.rts = False
                time.sleep(0.1)
                ser.dtr = False
                ser.rts = False
            finally:
                ser.close()
            log("serial-dtr reset toggled (best-effort; may not reset all boards)")
        except Exception as exc:  # noqa: BLE001
            warn(f"serial-dtr reset failed: {exc}; continuing without reset")
        return


# --------------------------------------------------------------------------- #
# UART logging
# --------------------------------------------------------------------------- #
class UartLogger:
    """Background UART reader. Robust to read/decode errors."""

    def __init__(self, serial_mod, port: str, baud: int, log_path: str, read_seconds: float):
        self._serial_mod = serial_mod
        self._port = port
        self._baud = baud
        self._log_path = log_path
        self._read_seconds = read_seconds
        self._thread = None
        self.packet_sent_count = 0
        self.seen_init = False

    def _run(self) -> None:
        deadline = time.time() + self._read_seconds
        ser = None
        try:
            ser = self._serial_mod.Serial(self._port, self._baud, timeout=1)
        except Exception as exc:  # noqa: BLE001
            warn(f"UART open failed on {self._port}: {exc}; skipping UART logging")
            # Still create an (empty) log file so the path is consistent.
            try:
                with open(self._log_path, "w", encoding="utf-8"):
                    pass
            except Exception:  # noqa: BLE001
                pass
            return

        try:
            with open(self._log_path, "w", encoding="utf-8") as fh:
                while time.time() < deadline:
                    try:
                        raw = ser.readline()
                    except Exception as exc:  # noqa: BLE001
                        warn(f"UART read error: {exc}; continuing")
                        continue
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        line = repr(raw) + "\n"
                    fh.write(line)
                    fh.flush()
                    if "Packet sent!" in line:
                        self.packet_sent_count += 1
                    if "LR11XX-LR-FHSS Ping Init" in line:
                        self.seen_init = True
        except Exception as exc:  # noqa: BLE001
            warn(f"UART logging error: {exc}; continuing")
        finally:
            try:
                if ser is not None:
                    ser.close()
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# --------------------------------------------------------------------------- #
# Capture + analysis for a single (freq, antenna)
# --------------------------------------------------------------------------- #
def build_capture_cmd(args, freq: str, antenna: str, out_path: str) -> list:
    cmd = [
        CAPTURE_BIN,
        "--freq", str(freq),
        "--rate", str(args.rate),
        "--gain", str(args.gain),
        "--duration", str(args.duration),
        "--out", out_path,
        "--antenna", antenna,
        "--channel", str(args.channel),
    ]
    # There is NO --serial flag; serial goes inside --args. Only pass --args
    # when a serial was given, so the binary's default type=b200 applies.
    if args.serial:
        cmd += ["--args", f"serial={args.serial}"]
    return cmd


def detect_overflow(text: str) -> bool:
    low = text.lower()
    for marker in OVERFLOW_MARKERS:
        if marker.lower() in low:
            return True
    # Standalone runs of 'O' chars (UHD overflow indicator). Look for a line
    # that is only O's (length >= 2) to avoid false positives on normal words.
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if len(stripped) >= 2 and set(stripped) == {"O"}:
            return True
    return False


def run_analyzer(args, fc32_path: str, base: str) -> tuple:
    """Run analyze_capture.py. Return (analysis_dict_or_None, ok_bool)."""
    analysis_json = base + "_analysis.json"
    waterfall_png = base + "_waterfall.png"
    maxhold_png = base + "_maxhold.png"
    sample_rate = to_int_hz(str(args.rate))

    cmd = python_runner() + [
        ANALYZE_PY,
        fc32_path,
        "--sample-rate", str(sample_rate),
        "--output-json", analysis_json,
        "--plot", waterfall_png,
        "--maxhold-plot", maxhold_png,
        "--signal-threshold-db", str(args.signal_threshold_db),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=300,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"analyze_capture.py failed to run: {exc}")
        return None, False

    if proc.returncode != 0:
        warn(
            "analyze_capture.py exited "
            f"{proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
        return None, False

    if not os.path.isfile(analysis_json):
        warn(f"analyze_capture.py produced no JSON at {analysis_json}")
        return None, False

    try:
        with open(analysis_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not read analysis JSON {analysis_json}: {exc}")
        return None, False

    return data, True


def run_quick_maxhold(args, fc32_path: str, base: str) -> None:
    """Run quick_maxhold.py (best-effort; failures are non-fatal)."""
    out_png = base + "_maxhold_quick.png"
    out_json = base + "_maxhold.json"
    sample_rate = to_int_hz(str(args.rate))
    cmd = python_runner() + [
        QUICK_MAXHOLD_PY,
        fc32_path,
        "--sample-rate", str(sample_rate),
        "--out", out_png,
        "--json", out_json,
    ]
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=300,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"quick_maxhold.py failed (non-fatal): {exc}")


def empty_row(args, freq: str, antenna: str, capture_file: str) -> dict:
    """A row template with all numeric/analysis fields blank."""
    return {
        "capture_file": capture_file,
        "frequency_hz": to_int_hz(str(freq)),
        "antenna": antenna,
        "rate": to_int_hz(str(args.rate)),
        "gain": args.gain,
        "duration": args.duration,
        "actual_duration_s": "",
        "capture_status": "capture_failed",
        "validation_status": "",
        "signal_detected": False,
        "peak_to_median_db": "",
        "burst_energy_excess_db": "",
        "maxhold_excess_db": "",
        "peak_frequency_offset_hz": "",
        "uart_log": "",
        "uart_packet_sent_count": 0,
        "uart_seen_init": False,
        "note": "",
    }


def process_capture(args, freq: str, antenna: str, serial_mod) -> dict:
    """Run one (freq, antenna) capture+analysis. Always returns a row dict."""
    ant_tag = sanitize_antenna(antenna)
    freq_tag = sanitize_freq(freq)
    cap_name = f"cap_{freq_tag}_{ant_tag}"
    fc32_path = os.path.join(args.outdir, cap_name + ".fc32")
    base = os.path.join(args.outdir, cap_name)

    row = empty_row(args, freq, antenna, os.path.basename(fc32_path))

    # 'manual' reset: prompt BEFORE starting the capture so the TX burst lands
    # inside the window.
    if args.reset_method == "manual":
        try:
            input(
                f"[AUTO-SWEEP] Press Enter to reset NUCLEO then capture "
                f"({freq} Hz, {antenna})... "
            )
        except EOFError:
            warn("No interactive stdin for manual reset; continuing")

    # UART logging thread (reads for duration + 2s).
    uart_logger = None
    uart_log_rel = ""
    if args.uart:
        if serial_mod is None:
            warn(
                "pyserial not installed; run `uv add pyserial`; "
                "skipping UART logging"
            )
        else:
            uart_log_path = base + "_uart.log"
            uart_log_rel = os.path.basename(uart_log_path)
            uart_logger = UartLogger(
                serial_mod,
                args.uart,
                args.uart_baud,
                uart_log_path,
                read_seconds=float(args.duration) + 2.0,
            )
            uart_logger.start()

    # 1. Start capture (non-blocking).
    cmd = build_capture_cmd(args, freq, antenna, fc32_path)
    log(f"starting capture: freq={freq} antenna={antenna} -> {os.path.basename(fc32_path)}")
    start_wall = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"Failed to launch capture binary: {exc}")
        row["capture_status"] = "capture_failed"
        row["note"] = f"popen_failed: {exc}"
        if uart_logger is not None:
            uart_logger.join(timeout=float(args.duration) + 5.0)
            row["uart_log"] = uart_log_rel
            row["uart_packet_sent_count"] = uart_logger.packet_sent_count
            row["uart_seen_init"] = uart_logger.seen_init
        return row

    # 2. Wait reset-delay so the capture is streaming before we reset.
    time.sleep(float(args.reset_delay_s))

    # 3. Trigger reset (stlink / serial-dtr) while capture runs.
    do_reset(args.reset_method, args.uart, serial_mod)

    # 4. Wait for capture to exit (generous timeout).
    wait_timeout = float(args.duration) * 2 + 15
    timed_out = False
    try:
        stdout_b, stderr_b = proc.communicate(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            stdout_b, stderr_b = proc.communicate(timeout=10)
        except Exception:  # noqa: BLE001
            stdout_b, stderr_b = b"", b""
    actual_duration = time.time() - start_wall
    row["actual_duration_s"] = round(actual_duration, 3)

    stdout_txt = stdout_b.decode("utf-8", "replace") if stdout_b else ""
    stderr_txt = stderr_b.decode("utf-8", "replace") if stderr_b else ""
    combined = stdout_txt + "\n" + stderr_txt

    # Join UART thread now that capture is done.
    if uart_logger is not None:
        uart_logger.join(timeout=float(args.duration) + 5.0)
        row["uart_log"] = uart_log_rel
        row["uart_packet_sent_count"] = uart_logger.packet_sent_count
        row["uart_seen_init"] = uart_logger.seen_init

    # Determine usable file.
    file_ok = os.path.isfile(fc32_path) and os.path.getsize(fc32_path) > 0

    is_exception = "EXCEPTION:" in combined
    is_timeout_msg = "Timeout while streaming" in combined
    is_overflow = detect_overflow(combined)

    # 5. Classify capture_status.
    if timed_out and not file_ok:
        row["capture_status"] = "capture_failed"
        row["note"] = "wait_timeout_killed_no_file"
        return row

    if not file_ok:
        row["capture_status"] = "capture_failed"
        if is_exception:
            row["note"] = "capture_exception_no_file"
        elif is_timeout_msg:
            row["note"] = "stream_timeout_no_file"
        elif proc.returncode not in (0, None):
            row["note"] = f"nonzero_exit_{proc.returncode}_no_file"
        else:
            row["note"] = "output_file_missing_or_empty"
        return row

    # We have a usable file. Decide ok vs overflow (overflow still analyzes).
    if is_overflow:
        row["capture_status"] = "overflow"
    elif proc.returncode not in (0, None):
        # Nonzero exit but a usable partial file exists -> treat like overflow
        # (partial data); still analyze.
        row["capture_status"] = "overflow"
    else:
        row["capture_status"] = "ok"

    # 6. Analyze.
    analysis, ok = run_analyzer(args, fc32_path, base)
    run_quick_maxhold(args, fc32_path, base)

    if not ok or analysis is None:
        # File existed but analysis failed.
        if row["capture_status"] == "overflow":
            row["note"] = "overflow_then_analysis_failed"
        else:
            row["capture_status"] = "analysis_failed"
            row["note"] = "analysis_failed"
        return row

    # Populate analysis fields. signal_detected ONLY from analyzer JSON.
    sig = analysis.get("signal_detected")
    row["signal_detected"] = bool(sig) if sig is not None else False
    row["validation_status"] = analysis.get("validation_status", "")
    row["peak_to_median_db"] = analysis.get("peak_to_median_db", "")
    row["burst_energy_excess_db"] = analysis.get("burst_energy_excess_db", "")
    row["maxhold_excess_db"] = analysis.get("maxhold_excess_db", "")
    row["peak_frequency_offset_hz"] = analysis.get("peak_frequency_offset_hz", "")

    # 7. Note logic.
    if row["uart_packet_sent_count"] and row["validation_status"] == "noise_floor_only":
        row["note"] = "firmware_tx_reported_but_no_rf_detected"
    elif not row["note"]:
        vs = row["validation_status"] or "unknown"
        row["note"] = f"{row['capture_status']}; {vs}"

    return row


# --------------------------------------------------------------------------- #
# Summary writers
# --------------------------------------------------------------------------- #
def write_csv(outdir: str, rows: list) -> str:
    path = os.path.join(outdir, "sweep_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in CSV_HEADER}
            # Booleans as lowercase strings for readability.
            out["signal_detected"] = "true" if row.get("signal_detected") else "false"
            out["uart_seen_init"] = "true" if row.get("uart_seen_init") else "false"
            writer.writerow(out)
    return path


def _json_num(value):
    """Return None for empty/missing numeric values, else the value."""
    if value == "" or value is None:
        return None
    return value


def write_json(outdir: str, args, rows: list) -> str:
    path = os.path.join(outdir, "sweep_summary.json")
    n_signal = sum(1 for r in rows if r.get("signal_detected"))
    captures = []
    for r in rows:
        captures.append(
            {
                "capture_file": r.get("capture_file", ""),
                "frequency_hz": _json_num(r.get("frequency_hz")),
                "antenna": r.get("antenna", ""),
                "rate": _json_num(r.get("rate")),
                "gain": _json_num(r.get("gain")),
                "duration": _json_num(r.get("duration")),
                "actual_duration_s": _json_num(r.get("actual_duration_s")),
                "capture_status": r.get("capture_status", ""),
                "validation_status": r.get("validation_status", ""),
                "signal_detected": bool(r.get("signal_detected")),
                "peak_to_median_db": _json_num(r.get("peak_to_median_db")),
                "burst_energy_excess_db": _json_num(r.get("burst_energy_excess_db")),
                "maxhold_excess_db": _json_num(r.get("maxhold_excess_db")),
                "peak_frequency_offset_hz": _json_num(r.get("peak_frequency_offset_hz")),
                "uart_log": r.get("uart_log", ""),
                "uart_packet_sent_count": r.get("uart_packet_sent_count", 0),
                "uart_seen_init": bool(r.get("uart_seen_init")),
                "note": r.get("note", ""),
            }
        )
    doc = {
        "serial": args.serial or "",
        "generated_at": utc_now_iso(),
        "reset_method": args.reset_method,
        "rate": to_int_hz(str(args.rate)),
        "gain": args.gain,
        "duration": args.duration,
        "n_captures": len(rows),
        "n_signal_detected": n_signal,
        "captures": captures,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_lr1121_sdr_sweep_auto.py",
        description=(
            "Automated USRP B210 LR-FHSS sweep for LR1121 hardware bring-up. "
            "Captures IQ, optionally resets the NUCLEO so its TX burst lands in "
            "the capture window, logs UART, and runs the analyzers. Conservative: "
            "signal_detected is set only when the analyzer JSON says so."
        ),
    )
    p.add_argument("--serial", default=None, help="USRP serial (passed as --args serial=...). Optional.")
    p.add_argument("--freqs", default="868e6,915e6,923e6", help="Comma-separated center freqs in Hz.")
    p.add_argument("--antennas", default="RX2,TX/RX", help="Comma-separated RX antennas (RX2,TX/RX).")
    p.add_argument("--rate", default="1e6", help="Sample rate in Hz (default 1e6; safe on Mac).")
    p.add_argument("--gain", default="45", help="RX gain in dB (default 45).")
    p.add_argument("--duration", default="10", help="Capture duration in seconds (default 10).")
    p.add_argument("--channel", default="0", help="RX channel index (default 0).")
    p.add_argument(
        "--reset-method",
        choices=["none", "stlink", "serial-dtr", "manual"],
        default="none",
        help="How to reset the NUCLEO before each capture (default none).",
    )
    p.add_argument("--reset-delay-s", type=float, default=1.0, help="Delay after capture start before reset (default 1.0).")
    p.add_argument("--uart", default=None, help="UART tty for logging / serial-dtr reset (optional).")
    p.add_argument("--uart-baud", type=int, default=115200, help="UART baud (default 115200).")
    p.add_argument("--signal-threshold-db", default="8", help="Detector threshold dB passed to analyzer (default 8).")
    p.add_argument("--outdir", default=None, help="Output directory (default hardware/captures/auto_sweep_<ts>).")
    return p.parse_args(argv)


def resolve_outdir(raw: str | None) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default = os.path.join(REPO_ROOT, "hardware", "captures", f"auto_sweep_{ts}")
    if raw is None:
        return default
    # Replace the literal placeholder if the caller passed it verbatim.
    if "auto_sweep_YYYYMMDD_HHMMSS" in raw:
        raw = raw.replace("auto_sweep_YYYYMMDD_HHMMSS", f"auto_sweep_{ts}")
    if not os.path.isabs(raw):
        raw = os.path.join(REPO_ROOT, raw)
    return raw


def parse_list(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


def status_phrase(row: dict) -> str:
    if row.get("signal_detected"):
        return "SIGNAL DETECTED"
    if row.get("capture_status") in ("capture_failed", "analysis_failed"):
        return f"{row.get('capture_status')} ({row.get('capture_status')})"
    vs = row.get("validation_status") or ""
    if vs in ("noise_floor_only", "weak_signal_candidate"):
        return vs
    # Fall back to validation_status if present, else capture_status.
    return vs or row.get("capture_status", "unknown")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.outdir = resolve_outdir(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    freqs = parse_list(args.freqs)
    antennas = parse_list(args.antennas)

    # Probe pyserial once (only if needed for UART logging or serial-dtr reset).
    serial_mod = None
    if args.uart or args.reset_method == "serial-dtr":
        serial_mod = try_import_serial()
        if serial_mod is None:
            warn(
                "pyserial not installed; run `uv add pyserial`; "
                "UART logging / serial-dtr reset will be skipped"
            )

    # Probe capture binary presence (non-fatal; captures will be marked failed).
    if not os.path.isfile(CAPTURE_BIN):
        warn(f"capture binary not found at {CAPTURE_BIN}; all captures will fail")

    log(f"output directory: {args.outdir}")
    log(f"freqs={freqs} antennas={antennas} rate={args.rate} gain={args.gain} duration={args.duration}")
    log(f"reset-method={args.reset_method} reset-delay-s={args.reset_delay_s} uart={args.uart or 'none'}")

    rows = []
    for freq in freqs:
        for antenna in antennas:
            # Wrap each capture so one failure does not stop the sweep.
            try:
                row = process_capture(args, freq, antenna, serial_mod)
            except Exception as exc:  # noqa: BLE001
                warn(f"unhandled exception during capture ({freq}, {antenna}): {exc}")
                ant_tag = sanitize_antenna(antenna)
                freq_tag = sanitize_freq(freq)
                cap_file = f"cap_{freq_tag}_{ant_tag}.fc32"
                row = empty_row(args, freq, antenna, cap_file)
                row["note"] = f"unhandled_exception: {exc}"
            rows.append(row)
            log(
                f"{freq} {antenna} -> {status_phrase(row)} "
                f"[capture_status={row.get('capture_status')}]"
            )

    # ALWAYS write summaries, even if every capture failed.
    csv_path = write_csv(args.outdir, rows)
    json_path = write_json(args.outdir, args, rows)

    n_signal = sum(1 for r in rows if r.get("signal_detected"))
    n_total = len(rows)
    n_failed = sum(1 for r in rows if r.get("capture_status") in ("capture_failed", "analysis_failed"))

    print("", flush=True)
    log(f"captures: {n_total} total, {n_signal} signal_detected, {n_failed} failed/analysis_failed")
    log(f"summary CSV : {csv_path}")
    log(f"summary JSON: {json_path}")
    if n_signal > 0:
        log("HARDWARE STATUS: signal-detected")
    else:
        log("HARDWARE STATUS: pending (hardware-bringup; no RF above noise floor)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
