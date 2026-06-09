#!/usr/bin/env python3
"""Conservative LR-FHSS IQ-structure analysis (NOT a decoder).

IQ-structure analysis only; no payload decode; no CRC; no PER.

Reads a .fc32 (interleaved complex float32) USRP capture and computes structural
evidence of an LR-FHSS-like hopping signal: PSD/max-hold, spectrogram, noise
floor, occupied frequency/time bins, candidate narrowband tones/bursts, and a
bounded heuristic structure_score in [0,1]. Nothing here is a decoded packet.

CLI:
  uv run python scripts/analyze_lrfhss_iq_structure.py \
    --iq hardware/captures/<cap>.fc32 --sample-rate 1000000 \
    --center-frequency 868000000 --run-id stage5_structure \
    --metadata docs/hardware_iq_capture_stage5/cap_868000000_txrx_comparison.json \
    --out validation_runs/lrfhss_iq_structure_stage5
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

LABEL = "IQ-structure analysis only; no payload decode; no CRC; no PER."
DC_GUARD_HZ = 5000.0
THRESH_DB = 8.0          # PSD max-hold display threshold (above global floor)
# Detection margin for a bin/frame to count as a narrowband tone. Must exceed
# the natural peak-to-median of complex-Gaussian noise over ~NPERSEG bins
# (~8-9 dB), so noise-only frames are NOT flagged as bursts.
DET_DB = 14.0
NPERSEG = 1024


def load_fc32(path, fs, max_seconds=None):
    iq = np.fromfile(path, dtype=np.complex64)
    if max_seconds:
        n = int(max_seconds * fs)
        iq = iq[:n]
    return iq


def analyze(iq, fs):
    f, t, Z = signal.stft(iq, fs=fs, nperseg=NPERSEG, noverlap=NPERSEG // 2,
                          return_onesided=False)
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    P = np.abs(Z) ** 2                       # (freq, time)
    Pdb = 10.0 * np.log10(np.maximum(P, 1e-20))
    maxhold_db = Pdb.max(axis=1)             # per freq
    noise_floor_db = float(np.median(Pdb))   # robust floor (reporting)
    thr_db = noise_floor_db + THRESH_DB

    guard = np.abs(f) < DC_GUARD_HZ
    ng = (~guard)
    n_time = Pdb.shape[1]

    # frame-relative detection: a bin is active if it stands out above the
    # per-frame median by THRESH_DB (robust to overall gain / global floor).
    frame_med = np.median(Pdb[ng, :], axis=0)                 # (time,)
    active = (Pdb > (frame_med[None, :] + DET_DB)) & ng[:, None]

    # a time frame is "on" iff it contains a dominant narrowband tone
    frame_peak = Pdb[ng, :].max(axis=0)
    frame_on = (frame_peak - frame_med) > DET_DB

    # frequency-bin occupancy (only frames that are on)
    active_on = active & frame_on[None, :]
    freq_active_frames = active_on.sum(axis=1)
    n_occupied_freq = int((freq_active_frames > 0).sum())
    # time-bin occupancy
    time_active_bins = active_on.sum(axis=0)
    time_occupancy = float(frame_on.mean())

    # candidate bursts: contiguous on-frames; per burst, dominant freq
    bursts = []
    i = 0
    while i < n_time:
        if frame_on[i]:
            j = i
            while j < n_time and frame_on[j]:
                j += 1
            seg = Pdb[:, i:j].copy()
            seg[guard, :] = -300
            dom_idx = int(np.unravel_index(np.argmax(seg), seg.shape)[0])
            bursts.append({
                "start_s": float(t[i]), "end_s": float(t[j - 1]),
                "duration_s": float(t[j - 1] - t[i]) if j - 1 > i else float(t[1] - t[0]),
                "dominant_freq_offset_hz": float(f[dom_idx]),
                "peak_db": float(seg.max()),
            })
            i = j
        else:
            i += 1

    # dominant offsets (top freq bins by maxhold, excluding DC)
    mh = maxhold_db.copy(); mh[guard] = -300
    top = np.argsort(mh)[-8:][::-1]
    dom_offsets = [{"freq_offset_hz": float(f[k]), "maxhold_db": float(maxhold_db[k])}
                   for k in top]

    durs = [b["duration_s"] for b in bursts]
    peak_mh = float(maxhold_db[~guard].max()) if (~guard).any() else float(maxhold_db.max())

    # bounded heuristic structure_score in [0,1]
    s_freq = min(1.0, n_occupied_freq / 20.0)
    s_burst = min(1.0, len(bursts) / 10.0)
    s_snr = min(1.0, max(0.0, (peak_mh - thr_db) / 20.0))
    structure_score = round(0.4 * s_freq + 0.3 * s_burst + 0.3 * s_snr, 3)

    summary = {
        "label": LABEL,
        "n_samples": int(iq.size),
        "duration_s": float(iq.size / fs),
        "noise_floor_db": round(noise_floor_db, 3),
        "threshold_db": THRESH_DB,
        "peak_maxhold_db": round(peak_mh, 3),
        "dc_guard_hz": DC_GUARD_HZ,
        "n_occupied_freq_bins": n_occupied_freq,
        "total_freq_bins": int(f.size),
        "time_occupancy_fraction": round(time_occupancy, 4),
        "n_candidate_bursts": len(bursts),
        "burst_duration_s": {
            "min": round(min(durs), 6) if durs else None,
            "median": round(float(np.median(durs)), 6) if durs else None,
            "max": round(max(durs), 6) if durs else None,
        },
        "dominant_freq_offsets_hz": dom_offsets,
        "structure_score": structure_score,
        "interpretation": ("higher structure_score = more LR-FHSS-like hopping "
                           "structure (candidate evidence only, not decode)"),
    }
    arrays = {"f": f, "t": t, "Pdb": Pdb, "maxhold_db": maxhold_db,
              "freq_active_frames": freq_active_frames, "bursts": bursts,
              "n_time": n_time, "thr_db": thr_db, "noise_floor_db": noise_floor_db}
    return summary, arrays


def write_outputs(out, summary, A, metadata):
    os.makedirs(out, exist_ok=True)
    f, t, Pdb = A["f"], A["t"], A["Pdb"]

    if metadata:
        summary["source_metadata"] = metadata

    with open(os.path.join(out, "iq_structure_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(out, "psd_maxhold.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["freq_offset_hz", "maxhold_db"])
        for fk, mk in zip(f, A["maxhold_db"]):
            w.writerow([f"{fk:.1f}", f"{mk:.3f}"])

    with open(os.path.join(out, "tone_occupancy.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["freq_offset_hz", "active_frames", "occupancy_fraction"])
        for fk, af in zip(f, A["freq_active_frames"]):
            w.writerow([f"{fk:.1f}", int(af), f"{af / max(A['n_time'],1):.4f}"])

    with open(os.path.join(out, "burst_candidates.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["burst_id", "start_s", "end_s", "duration_s",
                    "dominant_freq_offset_hz", "peak_db"])
        for i, b in enumerate(A["bursts"]):
            w.writerow([i, f"{b['start_s']:.6f}", f"{b['end_s']:.6f}",
                        f"{b['duration_s']:.6f}", f"{b['dominant_freq_offset_hz']:.1f}",
                        f"{b['peak_db']:.3f}"])

    # PSD max-hold
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(f / 1e3, A["maxhold_db"], lw=0.6, color="#1f77b4")
    ax.axhline(A["thr_db"], ls="--", color="#d62728", lw=0.8, label="threshold")
    ax.axhline(A["noise_floor_db"], ls=":", color="gray", lw=0.8, label="noise floor")
    ax.set_xlabel("Freq offset (kHz)"); ax.set_ylabel("Max-hold (dB)")
    ax.set_title("LR-FHSS-like PSD max-hold (candidate)", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lrfhss_psd.png"), dpi=130)
    plt.close(fig)

    # spectrogram
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.pcolormesh(t, f / 1e3, Pdb, shading="auto", cmap="viridis")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq offset (kHz)")
    ax.set_title("Spectrogram (candidate hops)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lrfhss_spectrogram.png"), dpi=130)
    plt.close(fig)

    # burst timeline
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for b in A["bursts"]:
        ax.hlines(b["dominant_freq_offset_hz"] / 1e3, b["start_s"], b["end_s"],
                  color="#ff7f0e", lw=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Dominant offset (kHz)")
    ax.set_title(f"Candidate burst timeline (n={len(A['bursts'])})", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lrfhss_burst_timeline.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write(LABEL + "\n\n")
        fh.write("Outputs are structural/heuristic evidence of an LR-FHSS-like "
                 "hopping signal. They are NOT decoded packets: no header "
                 "recovery, no fragment reconstruction, no CRC, no payload, no "
                 "PER. Terms used: candidate burst, candidate hop/tone, occupied "
                 "bin, LR-FHSS-like structure, IQ evidence.\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="LR-FHSS IQ-structure analysis (no decode).")
    ap.add_argument("--iq", required=True)
    ap.add_argument("--sample-rate", type=float, required=True)
    ap.add_argument("--center-frequency", type=float, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--metadata")
    args = ap.parse_args(argv)

    if not os.path.exists(args.iq):
        raise SystemExit(f"IQ file not found (raw IQ is local-only): {args.iq}")
    meta = None
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata) as fh:
            meta = json.load(fh)

    iq = load_fc32(args.iq, args.sample_rate, args.max_seconds)
    summary, A = analyze(iq, args.sample_rate)
    summary["run_id"] = args.run_id
    summary["center_frequency_hz"] = args.center_frequency
    summary["sample_rate_hz"] = args.sample_rate
    write_outputs(args.out, summary, A, meta)
    print(f"[lrfhss_iq] {LABEL}")
    print(f"[lrfhss_iq] run={args.run_id} bursts={summary['n_candidate_bursts']} "
          f"occ_freq={summary['n_occupied_freq_bins']} "
          f"structure_score={summary['structure_score']} -> {args.out}")
    return summary


if __name__ == "__main__":
    main()
