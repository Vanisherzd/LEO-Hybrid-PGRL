#!/usr/bin/env python3
"""
rx_capture_to_file.py
Capture IQ samples from USRP B210 and save to .fc32 file.

Usage:
    uv run python hardware/usrp_scripts/rx_capture_to_file.py \
        --freq 915e6 --rate 1e6 --gain 30 --duration 2 \
        --out hardware/captures/baseline.fc32

Output: binary .fc32 file (complex float32) readable by analyze_capture.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Capture IQ from USRP B210 to .fc32 file")
    parser.add_argument("--freq", type=float, required=True, help="Center freq in Hz (e.g. 915e6)")
    parser.add_argument("--rate", type=float, default=1e6, help="Sample rate in Hz")
    parser.add_argument("--gain", type=float, default=30.0, help="RX gain in dB")
    parser.add_argument("--duration", type=float, default=2.0, help="Capture duration in seconds")
    parser.add_argument("--out", type=str, required=True, help="Output .fc32 file path")
    parser.add_argument("--args", type=str, default="type=b200", help="UHD device args")
    parser.add_argument("--format", type=str, default="fc32", choices=["fc32", "sc16"],
                        help="Sample format (fc32 for Python analysis)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try native Python UHD first, fall back to CLI
    try:
        import uhd
        print(f"[rx_capture] Using python-uhd ({uhd.__version__})")
        usrp = uhd.Usrp.MultiBaordUSSP(args.args)
        usrp.set_rx_freq(args.freq)
        usrp.set_rx_rate(args.rate)
        usrp.set_rx_gain(args.gain)
        streamer_args = ""
        md = uhd.types.TXMetadata()
        buffer = uhd.types.StreamArgs("FC32", "sc16")
        rx_streamer = usrp.get_rx_stream(streamer_args)
        n_samples = int(args.rate * args.duration)
        rx_buffer = rx_streamer.recv(num_samps=n_samples, stream_cmd=uhd.types.StreamCMD("start"))
        with open(out_path, "wb") as f:
            f.write(rx_buffer.tobytes())
        print(f"[rx_capture] Captured {len(rx_buffer)} samples → {out_path} ({out_path.stat().st_size // 1024} KB)")
    except ImportError:
        print("[rx_capture] python-uhd not available, using uhd_rx_cfile CLI")
        import subprocess
        cmd = [
            "uhd_rx_cfile",
            f"--args={args.args}",
            f"--freq={args.freq}",
            f"--rate={args.rate}",
            f"--gain={args.gain}",
            f"--duration={args.duration}",
            f"--output={args.out}",
        ]
        print(f"[rx_capture] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[rx_capture] ERROR:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        size_kb = out_path.stat().st_size // 1024
        print(f"[rx_capture] Captured → {out_path} ({size_kb} KB)")
        print(f"[rx_capture] stdout: {result.stdout}")

    # Write metadata alongside the capture
    import json
    meta_path = out_path.with_suffix(".json")
    meta = {
        "center_freq_hz": args.freq,
        "sample_rate_hz": args.rate,
        "rx_gain_db": args.gain,
        "duration_s": args.duration,
        "capture_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_type": "hardware",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[rx_capture] Metadata → {meta_path}")

    # Write UHD command log
    log_path = out_path.with_suffix(".uhd_command.log")
    with open(log_path, "w") as f:
        f.write(f"Command: uhd_rx_cfile --args={args.args} --freq={args.freq} "
                f"--rate={args.rate} --gain={args.gain} --duration={args.duration} "
                f"--output={args.out}\n")
        f.write(f"Timestamp: {meta['capture_timestamp_utc']}\n")
        f.write(f"Python: {'uhd' if 'uhd' in dir() else 'uhd_rx_cfile CLI'}\n")
    print(f"[rx_capture] UHD command log → {log_path}")

if __name__ == "__main__":
    main()