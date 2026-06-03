#!/usr/bin/env python3
"""
compare_tx_on_off.py
TX-ON vs TX-OFF comparison for LR-FHSS hardware validation.

Loads two .fc32 captures (TX transmitting vs TX off / antenna idle) and compares
their spectrogram-derived metrics. The point: a defensible, A/B-controlled claim
that the radiated energy is REAL (ON significantly stronger than OFF) rather than
an artifact of the receive chain / ambient RF.

All metric math is REUSED from analyze_capture.py (no duplication): the shared
spectrogram, max-hold, hot-tile, and LR-FHSS-score functions.

Usage:
    uv run python hardware/usrp_scripts/compare_tx_on_off.py \
        --tx-on on.fc32 --tx-off off.fc32 \
        --sample-rate 1000000 \
        --out-json comparison.json \
        --out-plot comparison.png \
        --signal-threshold-db 8 \
        --spectrogram-nfft 4096 \
        --lr-fhss-score-threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse ALL metric math from analyze_capture (no duplication).
from analyze_capture import (  # type: ignore
    detect_signal,
    load_samples,
    _ONOFF_FREQ_OCC_MARGIN,
    _ONOFF_MAXHOLD_MARGIN_DB,
)


def _metrics_for(samples: np.ndarray, fs: float, nfft: int, threshold_db: float) -> dict:
    """Compute the canonical comparison metrics for one capture via detect_signal."""
    det = detect_signal(samples, fs, nfft, threshold_db)
    return {
        "maxhold_excess_db": det["maxhold_excess_db"],
        "hot_bin_count": det["time_frequency_hot_bin_count"],
        "occupied_frequency_bins": det["occupied_frequency_bins"],
        "occupied_time_bins": det["occupied_time_bins"],
        "lr_fhss_candidate_score": det["lr_fhss_candidate_score"],
        # kept private for plotting only (stripped before JSON write)
        "_mh_freqs": det["_mh_freqs"],
        "_mh_db": det["_mh_db"],
    }


def write_plot(on: dict, off: dict, out_path: Path):
    """Side-by-side max-hold (ON vs OFF) + occupancy bars. matplotlib optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available — skipping plot (JSON still written)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: overlaid max-hold spectra.
    ax1.plot(on["_mh_freqs"], on["_mh_db"], linewidth=0.8, color="crimson", label="TX-ON")
    ax1.plot(off["_mh_freqs"], off["_mh_db"], linewidth=0.8, color="navy", alpha=0.7,
             label="TX-OFF")
    ax1.set_xlabel("Frequency offset (Hz)")
    ax1.set_ylabel("Max-hold power (dB)")
    ax1.set_title("Max-hold: TX-ON vs TX-OFF")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: occupancy / metric bars.
    labels = ["occ_freq", "occ_time", "hot_bins"]
    on_vals = [on["occupied_frequency_bins"], on["occupied_time_bins"], on["hot_bin_count"]]
    off_vals = [off["occupied_frequency_bins"], off["occupied_time_bins"], off["hot_bin_count"]]
    x = np.arange(len(labels))
    w = 0.38
    ax2.bar(x - w / 2, on_vals, w, color="crimson", label="TX-ON")
    ax2.bar(x + w / 2, off_vals, w, color="navy", alpha=0.7, label="TX-OFF")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("count")
    ax2.set_title("Occupancy / hot-tile counts")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] Plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare TX-ON vs TX-OFF .fc32 captures")
    parser.add_argument("--tx-on", required=True, help="TX-ON .fc32 capture")
    parser.add_argument("--tx-off", required=True, help="TX-OFF .fc32 capture")
    parser.add_argument("--sample-rate", type=float, default=1e6, help="Sample rate in Hz")
    parser.add_argument("--out-json", required=True, help="Output comparison.json path")
    parser.add_argument("--out-plot", default=None, help="Output comparison PNG path")
    parser.add_argument("--signal-threshold-db", type=float, default=8.0,
                        help="hot-tile / max-hold threshold (dB), default 8")
    parser.add_argument("--spectrogram-nfft", type=int, default=4096,
                        help="FFT size (default 4096)")
    parser.add_argument("--lr-fhss-score-threshold", type=float, default=0.5,
                        help="lr_fhss_candidate_score threshold (default 0.5)")
    args = parser.parse_args()

    on_path = Path(args.tx_on)
    off_path = Path(args.tx_off)
    for p in (on_path, off_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    fs = float(args.sample_rate)
    nfft = int(args.spectrogram_nfft)
    thr = float(args.signal_threshold_db)

    print(f"[compare] Loading TX-ON {on_path} ...")
    on_samples = load_samples(on_path)
    print(f"[compare] Loading TX-OFF {off_path} ...")
    off_samples = load_samples(off_path)
    if len(on_samples) == 0 or len(off_samples) == 0:
        print("ERROR: one of the captures is empty (0 samples)", file=sys.stderr)
        sys.exit(1)

    on_m = _metrics_for(on_samples, fs, nfft, thr)
    off_m = _metrics_for(off_samples, fs, nfft, thr)

    on_off_delta_db = float(on_m["maxhold_excess_db"] - off_m["maxhold_excess_db"])
    hot_bin_delta = int(on_m["hot_bin_count"] - off_m["hot_bin_count"])
    freq_occupancy_delta = int(on_m["occupied_frequency_bins"] - off_m["occupied_frequency_bins"])

    # ON significantly stronger than OFF iff BOTH occupancy AND max-hold deltas
    # exceed the documented margins (shared with analyze_capture gate (c)).
    tx_on_stronger_than_off = bool(
        freq_occupancy_delta >= _ONOFF_FREQ_OCC_MARGIN
        and on_off_delta_db >= _ONOFF_MAXHOLD_MARGIN_DB
    )

    def _public(m):
        return {
            "maxhold_excess_db": m["maxhold_excess_db"],
            "hot_bin_count": m["hot_bin_count"],
            "occupied_frequency_bins": m["occupied_frequency_bins"],
            "occupied_time_bins": m["occupied_time_bins"],
            "lr_fhss_candidate_score": m["lr_fhss_candidate_score"],
        }

    note = (
        f"ON vs OFF A/B comparison. tx_on_stronger_than_off requires "
        f"freq_occupancy_delta >= {_ONOFF_FREQ_OCC_MARGIN} AND on_off_delta_db "
        f">= {_ONOFF_MAXHOLD_MARGIN_DB} dB. Metrics computed via "
        f"analyze_capture.detect_signal (shared math). DC/LO guard band excluded "
        f"from all counts."
    )

    comparison = {
        "tx_on": _public(on_m),
        "tx_off": _public(off_m),
        "on_off_delta_db": round(on_off_delta_db, 4),
        "hot_bin_delta": hot_bin_delta,
        "freq_occupancy_delta": freq_occupancy_delta,
        "tx_on_stronger_than_off": tx_on_stronger_than_off,
        "sample_rate_hz": fs,
        "threshold_db": thr,
        "note": note,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[compare] JSON → {out_json}")

    if args.out_plot:
        write_plot(on_m, off_m, Path(args.out_plot))

    print("\n=== TX-ON vs TX-OFF ===")
    print(f"  ON  : maxhold_excess={on_m['maxhold_excess_db']:.2f} dB  "
          f"occ_freq={on_m['occupied_frequency_bins']}  hot={on_m['hot_bin_count']}  "
          f"score={on_m['lr_fhss_candidate_score']:.3f}")
    print(f"  OFF : maxhold_excess={off_m['maxhold_excess_db']:.2f} dB  "
          f"occ_freq={off_m['occupied_frequency_bins']}  hot={off_m['hot_bin_count']}  "
          f"score={off_m['lr_fhss_candidate_score']:.3f}")
    print(f"  on_off_delta_db        = {on_off_delta_db:.2f} dB")
    print(f"  freq_occupancy_delta   = {freq_occupancy_delta}")
    print(f"  hot_bin_delta          = {hot_bin_delta}")
    print(f"  tx_on_stronger_than_off= {tx_on_stronger_than_off}")
    print("=======================")


if __name__ == "__main__":
    main()
