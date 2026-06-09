#!/usr/bin/env python3
"""Plot decoded packet-delivery metrics from packet_validation_summary.json file(s).

Consumes ONLY validation-run outputs (summary JSON + optional rx_records.csv /
tx_records.csv in the same run dir). Runs with PER unavailable (iq_only / no
decoding) are SKIPPED. Does not fabricate hardware results.

Usage:
  python scripts/plot_packet_delivery_metrics.py RUN_DIR [RUN_DIR ...] \
         --out paper/figures/fig6_packet_delivery.pdf
"""
import argparse
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _load_csv(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    return list(csv.DictReader(open(p, newline="")))


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v not in (None, "", "None"):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="run directories")
    ap.add_argument("--summaries", nargs="+", default=[],
                    help="packet_validation_summary.json paths (dir is derived)")
    ap.add_argument("--out", default="paper/figures/fig6_packet_delivery.pdf")
    args = ap.parse_args(argv)

    dirs = list(args.run_dirs) + [os.path.dirname(os.path.abspath(p))
                                  for p in args.summaries]
    if not dirs:
        ap.error("provide run_dirs and/or --summaries")

    runs = []
    for d in dirs:
        s = _load_json(os.path.join(d, "packet_validation_summary.json"))
        if not s or not s.get("decoding_available") or s.get("packet_error_rate") is None:
            print(f"[skip] {d}: PER unavailable")
            continue
        s["_dir"] = d
        runs.append(s)
    if not runs:
        print("No decoded runs to plot. Nothing written.")
        return

    # collect optional RX field data from the richest run
    rich = max(runs, key=lambda s: len(_load_csv(os.path.join(s["_dir"], "rx_records.csv"))))
    rx = _load_csv(os.path.join(rich["_dir"], "rx_records.csv"))
    tx = _load_csv(os.path.join(rich["_dir"], "tx_records.csv"))
    rssi, snr, cfo = _floats(rx, "rssi_dbm"), _floats(rx, "snr_db"), _floats(rx, "cfo_hz")

    # latency CDF (rx_ts - tx_ts per matched seq)
    lat = []
    if tx and rx:
        tx_ts = {}
        for r in tx:
            try:
                tx_ts[int(r["seq"])] = datetime.fromisoformat(r["tx_timestamp_utc"])
            except Exception:
                pass
        for r in rx:
            if r.get("decode_status") != "decoded":
                continue
            try:
                seq = int(r["seq"]); t1 = datetime.fromisoformat(r["rx_timestamp_utc"])
                if seq in tx_ts:
                    lat.append((t1 - tx_ts[seq]).total_seconds() * 1e3)
            except Exception:
                pass

    panels = [("pdrper", True), ("rssi", bool(rssi)), ("snr", bool(snr)),
              ("cfo", bool(cfo)), ("lat", bool(lat))]
    active = [p for p, on in panels if on]
    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.5))
    if n == 1:
        axes = [axes]
    ax = dict(zip(active, axes))

    ids = [s["run_id"] for s in runs]
    x = range(len(runs))
    a = ax["pdrper"]
    a.bar([i - 0.2 for i in x], [s["packet_delivery_ratio"] for s in runs], 0.4,
          label="PDR", color="#2ca02c")
    a.bar([i + 0.2 for i in x], [s["packet_error_rate"] for s in runs], 0.4,
          label="PER", color="#d62728")
    a.set_xticks(list(x)); a.set_xticklabels(ids, rotation=30, ha="right", fontsize=6)
    a.set_ylim(0, 1.05); a.legend(fontsize=7); a.set_title("PDR / PER", fontsize=8)

    for key, data, xl in [("rssi", rssi, "RSSI (dBm)"), ("snr", snr, "SNR (dB)"),
                          ("cfo", cfo, "CFO (Hz)")]:
        if key in ax:
            ax[key].hist(data, bins=12, color="#1f77b4")
            ax[key].set_xlabel(xl, fontsize=7); ax[key].set_title(xl, fontsize=8)
    if "lat" in ax:
        sl = sorted(lat)
        ax["lat"].plot(sl, [i / len(sl) for i in range(1, len(sl) + 1)], color="#9467bd")
        ax["lat"].set_xlabel("Latency (ms)", fontsize=7); ax["lat"].set_title("Latency CDF", fontsize=8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
