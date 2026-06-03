#!/usr/bin/env python3
"""Automated USRP B210 LR-FHSS sweep for LR1121 hardware bring-up.

Starts the rx_capture_to_file_cpp capture, optionally resets the NUCLEO board
(so its TX burst lands inside the capture window), logs UART, then runs the
analyzers. Conservative: signal_detected is ONLY ever set true when the
analyzer JSON explicitly reports signal_detected == true. Otherwise the run
stays in hardware-bringup / pending.

Build constraints: stdlib only, plus optional lazily-imported pyserial.

Two modes:
  --mode normal  (default): one capture per (freq, antenna), as before.
  --mode on-off : for each (freq, antenna) take a TX-ON capture, then prompt
                  the operator to power off the LR1121 and take a TX-OFF
                  reference capture, then run compare_tx_on_off.py to produce
                  an on/off delta. This is corroborating evidence only; the
                  analyzer JSON remains the sole authority for signal_detected.

Typical use:
    uv run python scripts/run_lr1121_sdr_sweep_auto.py \
        --serial 8000304 --freqs "868e6,915e6,923e6" --antennas "RX2,TX/RX" \
        --rate 1e6 --gain 45 --duration 10 \
        --reset-method stlink --uart /dev/tty.usbmodem1303 --uart-baud 115200

    uv run python scripts/run_lr1121_sdr_sweep_auto.py \
        --serial 8000304 --freqs "923e6" --antennas "RX2" --mode on-off
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
COMPARE_PY = os.path.join(REPO_ROOT, "hardware", "usrp_scripts", "compare_tx_on_off.py")

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
    # --- on/off differential mode (appended; empty/false in normal mode) ---
    "on_off_delta_db",
    "tx_on_stronger_than_off",
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


def prompt_operator(message: str) -> None:
    """Blocking prompt for a manual operator action. Robust to no stdin."""
    try:
        input(f"[AUTO-SWEEP] {message}")
    except EOFError:
        warn("No interactive stdin available for operator prompt; continuing")


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


def run_analyzer(args, fc32_path: str, base: str, uart_packet_count=None) -> tuple:
    """Run analyze_capture.py in LR-FHSS mode. Return (analysis_dict_or_None, ok_bool).

    uart_packet_count: if not None, forwarded as --uart-packet-sent-count so the
    analyzer applies the UART corroboration gate (b). Pass None when no UART log
    was collected so the gate is skipped rather than failed.
    """
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
        "--lr-fhss-mode",
    ]
    if uart_packet_count is not None:
        cmd += ["--uart-packet-sent-count", str(int(uart_packet_count))]
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


def run_compare_on_off(args, on_fc32: str, off_fc32: str, base: str) -> tuple:
    """Run compare_tx_on_off.py for an ON/OFF pair.

    Returns (on_off_delta_db_or_None, tx_on_stronger_or_None, ok_bool). Parses
    the produced comparison.json with json.load (never grep). All failures are
    non-fatal: they return (None, None, False) so the sweep continues.
    """
    out_json = base + "_comparison.json"
    out_png = base + "_comparison.png"
    sample_rate = to_int_hz(str(args.rate))

    if not os.path.isfile(COMPARE_PY):
        warn(f"compare_tx_on_off.py not found at {COMPARE_PY}; skipping comparison")
        return None, None, False

    cmd = python_runner() + [
        COMPARE_PY,
        "--tx-on", on_fc32,
        "--tx-off", off_fc32,
        "--sample-rate", str(sample_rate),
        "--out-json", out_json,
        "--out-plot", out_png,
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
        warn(f"compare_tx_on_off.py failed to run: {exc}")
        return None, None, False

    if proc.returncode != 0:
        warn(
            "compare_tx_on_off.py exited "
            f"{proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
        return None, None, False

    if not os.path.isfile(out_json):
        warn(f"compare_tx_on_off.py produced no JSON at {out_json}")
        return None, None, False

    try:
        with open(out_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not read comparison JSON {out_json}: {exc}")
        return None, None, False

    delta = data.get("on_off_delta_db")
    stronger = data.get("tx_on_stronger_than_off")
    return delta, bool(stronger) if stronger is not None else None, True


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
        # on/off differential fields (blank/false unless on-off mode populates).
        "on_off_delta_db": "",
        "tx_on_stronger_than_off": False,
    }


def _do_capture(args, freq: str, antenna: str, fc32_path: str, serial_mod,
                do_reset_actions: bool, uart_active: bool):
    """Run a single capture subprocess (start, optional reset, wait, classify).

    Returns (capture_status, note, actual_duration, file_ok, returncode,
    uart_logger_or_None). Does NOT analyze. Shared by normal and on-off modes
    so all the existing capture_status classification is preserved verbatim.

    do_reset_actions: when False (e.g. a TX-OFF reference), the reset is NOT
    issued -- we want the transmitter to stay off.
    uart_active: whether to spin up the background UART logger for this capture.
    """
    base = os.path.splitext(fc32_path)[0]

    # UART logging thread (reads for duration + 2s).
    uart_logger = None
    if uart_active and args.uart and serial_mod is not None:
        uart_log_path = base + "_uart.log"
        uart_logger = UartLogger(
            serial_mod,
            args.uart,
            args.uart_baud,
            uart_log_path,
            read_seconds=float(args.duration) + 2.0,
        )
        uart_logger.start()

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
        if uart_logger is not None:
            uart_logger.join(timeout=float(args.duration) + 5.0)
        return "capture_failed", f"popen_failed: {exc}", "", False, None, uart_logger

    # Wait reset-delay so the capture is streaming before we reset.
    time.sleep(float(args.reset_delay_s))

    # Trigger reset (stlink / serial-dtr) while capture runs -- only when asked.
    if do_reset_actions:
        do_reset(args.reset_method, args.uart, serial_mod)

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
    actual_duration = round(time.time() - start_wall, 3)

    stdout_txt = stdout_b.decode("utf-8", "replace") if stdout_b else ""
    stderr_txt = stderr_b.decode("utf-8", "replace") if stderr_b else ""
    combined = stdout_txt + "\n" + stderr_txt

    if uart_logger is not None:
        uart_logger.join(timeout=float(args.duration) + 5.0)

    file_ok = os.path.isfile(fc32_path) and os.path.getsize(fc32_path) > 0

    is_exception = "EXCEPTION:" in combined
    is_timeout_msg = "Timeout while streaming" in combined
    is_overflow = detect_overflow(combined)

    if timed_out and not file_ok:
        return "capture_failed", "wait_timeout_killed_no_file", actual_duration, False, proc.returncode, uart_logger

    if not file_ok:
        if is_exception:
            note = "capture_exception_no_file"
        elif is_timeout_msg:
            note = "stream_timeout_no_file"
        elif proc.returncode not in (0, None):
            note = f"nonzero_exit_{proc.returncode}_no_file"
        else:
            note = "output_file_missing_or_empty"
        return "capture_failed", note, actual_duration, False, proc.returncode, uart_logger

    if is_overflow:
        status = "overflow"
    elif proc.returncode not in (0, None):
        status = "overflow"
    else:
        status = "ok"

    return status, "", actual_duration, True, proc.returncode, uart_logger


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

    uart_log_rel = ""
    if args.uart and serial_mod is None:
        warn(
            "pyserial not installed; run `uv add pyserial`; "
            "skipping UART logging"
        )

    status, note, actual_duration, file_ok, returncode, uart_logger = _do_capture(
        args, freq, antenna, fc32_path, serial_mod,
        do_reset_actions=True, uart_active=True,
    )
    row["capture_status"] = status
    row["note"] = note
    row["actual_duration_s"] = actual_duration

    if uart_logger is not None:
        uart_log_rel = os.path.basename(base + "_uart.log")
        row["uart_log"] = uart_log_rel
        row["uart_packet_sent_count"] = uart_logger.packet_sent_count
        row["uart_seen_init"] = uart_logger.seen_init

    if not file_ok:
        return row

    # Analyze. Forward UART count only when a UART log was collected (gate b);
    # otherwise None so the analyzer skips the gate rather than failing it.
    uart_count = row["uart_packet_sent_count"] if args.uart else None
    analysis, ok = run_analyzer(args, fc32_path, base, uart_packet_count=uart_count)
    run_quick_maxhold(args, fc32_path, base)

    if not ok or analysis is None:
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

    # Note logic.
    if row["uart_packet_sent_count"] and row["validation_status"] == "noise_floor_only":
        row["note"] = "firmware_tx_reported_but_no_rf_detected"
    elif not row["note"]:
        vs = row["validation_status"] or "unknown"
        row["note"] = f"{row['capture_status']}; {vs}"

    return row


def process_capture_on_off(args, freq: str, antenna: str, serial_mod,
                           more_pairs_remain: bool) -> dict:
    """TX ON/OFF differential capture for one (freq, antenna).

    Captures a TX-ON file (with reset so the LR1121 transmits), analyzes it as
    the authoritative row, then prompts the operator to power the TX off, takes
    a TX-OFF reference, and runs compare_tx_on_off.py for the on/off delta. The
    delta is corroboration only; signal_detected stays analyzer-driven.
    """
    ant_tag = sanitize_antenna(antenna)
    freq_tag = sanitize_freq(freq)
    on_name = f"cap_{freq_tag}_{ant_tag}_on"
    off_name = f"cap_{freq_tag}_{ant_tag}_off"
    pair_name = f"cap_{freq_tag}_{ant_tag}"

    on_fc32 = os.path.join(args.outdir, on_name + ".fc32")
    off_fc32 = os.path.join(args.outdir, off_name + ".fc32")
    on_base = os.path.join(args.outdir, on_name)
    off_base = os.path.join(args.outdir, off_name)
    pair_base = os.path.join(args.outdir, pair_name)

    row = empty_row(args, freq, antenna, os.path.basename(on_fc32))

    if args.uart and serial_mod is None:
        warn(
            "pyserial not installed; run `uv add pyserial`; "
            "skipping UART logging"
        )

    # ----- Step 1: TX ON capture -----
    log("=" * 60)
    log(f"TX ON capture  (freq={freq} Hz, antenna={antenna})")
    log("=" * 60)

    # 'manual' reset is interactive and must precede the ON capture.
    if args.reset_method == "manual":
        prompt_operator(
            f"Press Enter to reset NUCLEO then TX-ON capture "
            f"({freq} Hz, {antenna})... "
        )

    status, note, actual_duration, file_ok, returncode, uart_logger = _do_capture(
        args, freq, antenna, on_fc32, serial_mod,
        do_reset_actions=True, uart_active=True,
    )
    row["capture_status"] = status
    row["note"] = note
    row["actual_duration_s"] = actual_duration

    if uart_logger is not None:
        row["uart_log"] = os.path.basename(on_base + "_uart.log")
        row["uart_packet_sent_count"] = uart_logger.packet_sent_count
        row["uart_seen_init"] = uart_logger.seen_init

    on_analysis_ok = False
    if file_ok:
        uart_count = row["uart_packet_sent_count"] if args.uart else None
        analysis, ok = run_analyzer(args, on_fc32, on_base, uart_packet_count=uart_count)
        run_quick_maxhold(args, on_fc32, on_base)
        if not ok or analysis is None:
            if row["capture_status"] == "overflow":
                row["note"] = "overflow_then_analysis_failed"
            else:
                row["capture_status"] = "analysis_failed"
                row["note"] = "analysis_failed"
        else:
            on_analysis_ok = True
            sig = analysis.get("signal_detected")
            row["signal_detected"] = bool(sig) if sig is not None else False
            row["validation_status"] = analysis.get("validation_status", "")
            row["peak_to_median_db"] = analysis.get("peak_to_median_db", "")
            row["burst_energy_excess_db"] = analysis.get("burst_energy_excess_db", "")
            row["maxhold_excess_db"] = analysis.get("maxhold_excess_db", "")
            row["peak_frequency_offset_hz"] = analysis.get("peak_frequency_offset_hz", "")
            if row["uart_packet_sent_count"] and row["validation_status"] == "noise_floor_only":
                row["note"] = "firmware_tx_reported_but_no_rf_detected"
            elif not row["note"]:
                vs = row["validation_status"] or "unknown"
                row["note"] = f"{row['capture_status']}; {vs}"

    # ----- Step 2: prompt operator to power TX off -----
    prompt_operator(
        "ACTION: power off / disconnect the LR1121 TX (or stop SWDM001 "
        "firmware) for the OFF reference, then press Enter. "
    )

    # ----- Step 3: TX OFF reference capture -----
    log(f"TX OFF reference capture  (freq={freq} Hz, antenna={antenna})")
    off_status, off_note, _off_dur, off_file_ok, _off_rc, _off_uart = _do_capture(
        args, freq, antenna, off_fc32, serial_mod,
        do_reset_actions=False, uart_active=False,
    )
    if off_file_ok:
        # Analyze the OFF reference into its own _analysis.json (best-effort).
        run_analyzer(args, off_fc32, off_base)
        run_quick_maxhold(args, off_fc32, off_base)
    else:
        warn(f"TX-OFF reference capture failed ({off_status}: {off_note})")

    # ----- Step 4: compare ON vs OFF -----
    if file_ok and off_file_ok:
        delta, stronger, cmp_ok = run_compare_on_off(args, on_fc32, off_fc32, pair_base)
        if cmp_ok:
            row["on_off_delta_db"] = delta if delta is not None else ""
            row["tx_on_stronger_than_off"] = bool(stronger) if stronger is not None else False
            log(
                f"on/off delta_db={row['on_off_delta_db']} "
                f"tx_on_stronger_than_off={row['tx_on_stronger_than_off']}"
            )
        else:
            row["on_off_delta_db"] = ""
            row["tx_on_stronger_than_off"] = False
            row["note"] = (row["note"] + "; " if row["note"] else "") + "on_off_compare_failed"
    else:
        row["on_off_delta_db"] = ""
        row["tx_on_stronger_than_off"] = False
        reason = "on_off_off_capture_failed" if not off_file_ok else "on_off_on_capture_failed"
        row["note"] = (row["note"] + "; " if row["note"] else "") + reason

    # ----- Step 5: prompt operator to power TX back on for the next pair -----
    if more_pairs_remain:
        prompt_operator(
            "ACTION: power the LR1121 TX back on before the next frequency, "
            "then press Enter. "
        )

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
            out["tx_on_stronger_than_off"] = "true" if row.get("tx_on_stronger_than_off") else "false"
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
                "on_off_delta_db": _json_num(r.get("on_off_delta_db")),
                "tx_on_stronger_than_off": bool(r.get("tx_on_stronger_than_off")),
            }
        )
    doc = {
        "serial": args.serial or "",
        "generated_at": utc_now_iso(),
        "mode": args.mode,
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
            "signal_detected is set only when the analyzer JSON says so. With "
            "--mode on-off, also takes a TX-off reference and reports an on/off "
            "delta as corroborating evidence."
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
        "--mode",
        choices=["normal", "on-off"],
        default="normal",
        help=(
            "normal: one capture per (freq, antenna). on-off: TX-ON capture + "
            "operator-prompted TX-OFF reference + compare_tx_on_off.py delta "
            "(corroboration only; analyzer remains the signal_detected authority)."
        ),
    )
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

    if args.mode == "on-off" and not os.path.isfile(COMPARE_PY):
        warn(
            f"compare_tx_on_off.py not found at {COMPARE_PY}; "
            "on/off deltas will be blank but the sweep will still run"
        )

    log(f"output directory: {args.outdir}")
    log(f"mode={args.mode}")
    log(f"freqs={freqs} antennas={antennas} rate={args.rate} gain={args.gain} duration={args.duration}")
    log(f"reset-method={args.reset_method} reset-delay-s={args.reset_delay_s} uart={args.uart or 'none'}")

    # Pre-compute the (freq, antenna) pairs so on-off mode knows when to stop
    # prompting for the next-pair power-on.
    pairs = [(f, a) for f in freqs for a in antennas]

    rows = []
    for idx, (freq, antenna) in enumerate(pairs):
        more_pairs_remain = idx < len(pairs) - 1
        # Wrap each capture so one failure does not stop the sweep.
        try:
            if args.mode == "on-off":
                row = process_capture_on_off(
                    args, freq, antenna, serial_mod, more_pairs_remain
                )
            else:
                row = process_capture(args, freq, antenna, serial_mod)
        except Exception as exc:  # noqa: BLE001
            warn(f"unhandled exception during capture ({freq}, {antenna}): {exc}")
            ant_tag = sanitize_antenna(antenna)
            freq_tag = sanitize_freq(freq)
            suffix = "_on" if args.mode == "on-off" else ""
            cap_file = f"cap_{freq_tag}_{ant_tag}{suffix}.fc32"
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

    # on/off corroboration: only an analyzer-confirmed signal that ALSO shows
    # TX-on stronger than the TX-off reference earns "signal-detected" in on-off
    # mode. The on/off delta never elevates a row on its own.
    corroborated = any(
        r.get("signal_detected") and r.get("tx_on_stronger_than_off")
        for r in rows
    )
    if args.mode == "on-off" and corroborated:
        log("HARDWARE STATUS: signal-detected")
    elif n_signal > 0 and args.mode != "on-off":
        log("HARDWARE STATUS: signal-detected")
    else:
        log("HARDWARE STATUS: pending (hardware-bringup; no RF above noise floor)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
