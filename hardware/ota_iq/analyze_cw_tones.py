#!/usr/bin/env python3
"""Analyze CW (C-command) tones: per-tone carrier vs commanded, de-biased.
Outputs per-mode residual CFO. grid_ref = nominal_center F0 (no LR-FHSS grid)."""
import sys, json, csv, time, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, "hardware/ota_iq")
from ota_common import load_iq, detect_bursts

run = Path(sys.argv[1]); modes = sys.argv[2:]
fs, lo, f0 = 1_000_000.0, 200_000.0, 868_000_000.0

def tone_offsets(mode_dir):
    iq = load_iq(mode_dir / "capture_iq.fc32")
    bursts = detect_bursts(iq, fs, nfft=1024, hop=256, snr_gate_db=6.0)
    sched = list(csv.DictReader(open(mode_dir / "burst_schedule.csv")))
    out = []
    for b in bursts:
        # high-res carrier estimate over the burst span
        i0 = int(b.t_start_s * fs); i1 = int(b.t_end_s * fs)
        seg = iq[i0:i1]
        if seg.size < 4096: continue
        nfft = 1 << 20
        w = np.hanning(seg.size)
        sp = np.abs(np.fft.fftshift(np.fft.fft(seg * w, nfft)))
        fr = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
        m = np.abs(fr) > 5e3   # exclude DC/LO baseband
        k = int(np.argmax(sp * m))
        meas_from_f0 = fr[k] + lo            # measured carrier offset from F0
        out.append((b.index, b.t_start_s, meas_from_f0))
    return out, sched

# pass 1: gather measured-vs-commanded across ALL modes to estimate common bias
allpairs = []; per_mode = {}
for mode in modes:
    md = run / mode
    tones, sched = tone_offsets(md)
    rows = []
    for n, (idx, t, meas) in enumerate(tones):
        cmd = float(sched[n]["commanded_offset_hz"]) if n < len(sched) else 0.0
        rows.append([idx, t, meas, cmd])
        allpairs.append(meas - cmd)
    per_mode[mode] = rows

bias = float(np.median(allpairs)) if allpairs else 0.0  # common TX+RX oscillator offset
print(f"global_oscillator_bias_hz = {bias:.1f}  (from {len(allpairs)} tones across modes)")

summary = {"global_oscillator_bias_hz": round(bias, 2), "grid_reference": "nominal_center_F0",
           "schedule_source": "demo_synthetic", "modes": {}}
for mode, rows in per_mode.items():
    md = run / mode
    csv_rows = []; res_grid = []; real_err = []
    for idx, t, meas, cmd in rows:
        rg = meas - bias                  # residual vs intended grid F0 (de-biased) ~ commanded
        re = meas - cmd - bias            # realization error (de-biased) ~ LR1121 accuracy
        csv_rows.append(dict(burst_index=idx, t_start_s=round(t,4),
            measured_offset_from_f0_hz=round(meas,2), commanded_offset_hz=round(cmd,2),
            residual_to_grid_hz=round(rg,2), realization_error_hz=round(re,2)))
        res_grid.append(abs(rg)); real_err.append(abs(re))
    with open(md / "cw_cfo_per_tone.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    rg = np.array(res_grid); re = np.array(real_err)
    s = dict(n_tones=len(rows),
        median_abs_residual_to_grid_hz=round(float(np.median(rg)),2),
        p95_abs_residual_to_grid_hz=round(float(np.percentile(rg,95)),2),
        max_abs_residual_to_grid_hz=round(float(np.max(rg)),2),
        median_abs_realization_error_hz=round(float(np.median(re)),2),
        p95_abs_realization_error_hz=round(float(np.percentile(re,95)),2),
        max_abs_realization_error_hz=round(float(np.max(re)),2))
    summary["modes"][mode] = s
    print(f"{mode}: n={s['n_tones']}  |residual_to_grid| med/p95/max = "
          f"{s['median_abs_residual_to_grid_hz']}/{s['p95_abs_residual_to_grid_hz']}/{s['max_abs_residual_to_grid_hz']} Hz  | "
          f"|realization_err| med/p95 = {s['median_abs_realization_error_hz']}/{s['p95_abs_realization_error_hz']} Hz")

commit = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
summary["commit"] = commit; summary["analyzed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
(run / "cw_cfo_summary.json").write_text(json.dumps(summary, indent=2))
print(f"summary -> {run/'cw_cfo_summary.json'}")
