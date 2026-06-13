# Local OTA-IQ Run Artifact Audit

Read-only provenance/safety/usability audit of `hardware/ota_iq/runs/` (~2.1 G
local). **No hardware, no capture/replay, no UART, no raw-IQ re-analysis, no
paper edit.** Metadata/JSON/CSV-header/log-only inspection.

- **Date:** 2026-06-13
- **Repo:** `/Users/laizhendong/Desktop/LEO-Hybrid-PGRL`
- **Branch:** `paper-crosslayer-rewrite`
- **Commit (HEAD):** `5a81f5d` — *docs(review): add replay data gap closure plan*

## 1. Tracking status (git)
- **Tracked** (curated, small): `capture_meta.json`, `schedule_meta.json`,
  `burst_schedule.csv`, `*_summary.json`, `cfo_residual_*.csv`, `ablr_*.csv`,
  `replay_uart_log.csv`, figures (`.png/.pdf`), `uart_log.txt`, `short_*.yaml`.
- **Git-ignored** (`!!`): every `capture_iq.fc32` (14 files, ~2.1 G),
  `capture_stdout.log`, `orchestrate.log`, and the whole dir
  `20260611_154227/`.
- No raw IQ is committed. Curated metadata/figures are committed.

## 2. Run directory inventory
| Run dir | Size | Group / class | Notes |
|---|---|---|---|
| `20260611_131516` | 76 M | noise floor / RX-only (TX off) | peak 8.24e-4 |
| `20260611_134233` | 153 M | TX-on smoke + paired TX-off | paired max-hold Δ=38.79 dB |
| `20260611_150700` | 229 M | 3-mode LR-FHSS (FAILED — no RF) | only `no_compensation` captured, peak 8.08e-4 = noise; sgp4/pgrl planned-only |
| `20260611_151529` | 76 M | TX-on RF diagnostic (LR-FHSS, B-cmd) | peak 3.30e-3, signal_detected 0.851 |
| `20260611_152017` | 687 M | 3-mode LR-FHSS CFO/ABLR (NEGATIVE) | RF present but no mode separation; stale-format summaries |
| `20260611_153751` | 76 M | CW smoke (FAILED — no radiate) | peak 8.42e-4 = noise (transmitDirect not radiating) |
| `20260611_154034` | 76 M | CW smoke iteration | fc32+uart only, **no curated meta** |
| `20260611_154227` | 76 M | CW smoke orphan | fc32+log only, **no meta/uart** — incomplete, do-not-use |
| `20260611_154458` | 687 M | 3-mode CW CFO diagnostic (clean) | mode separation present; **synthetic schedule** |

## 3. Available artifacts per run (summary)
- `131516`: capture_meta (tracked), fc32 (ignored).
- `134233`: smoke_txon + smoke_txoff capture_meta, `paired_on_off_summary.json`,
  `signal_detection_summary.json`, maxhold/waterfall figs, uart_log; fc32 ignored.
- `150700`: per-mode burst_schedule + schedule_meta + capture_meta (no_compensation
  only) + replay_uart_log; sgp4/pgrl schedule-only; fc32+logs ignored.
- `151529/rf_diag`: capture_meta + signal_detection_summary + maxhold fig; fc32 ignored.
- `152017`: per-mode capture_meta + schedule_meta + burst_schedule +
  cfo_residual_{timeseries.csv,summary.json} + ablr_{per_burst.csv,summary.json}
  + figures; fc32+logs ignored.
- `153751/cw_smoke`: capture_meta; fc32+log ignored.
- `154034/cw_smoke`: fc32 + capture_stdout.log + replay_uart_log.txt (no meta).
- `154227/cw_smoke`: fc32 + capture_stdout.log only (no meta/uart).
- `154458`: per-mode capture_meta (`waveform=true_CW_setTxCw`) + schedule_meta +
  burst_schedule + `cw_cfo_per_tone.csv` + `cw_cfo_summary.json` + figures; fc32 ignored.

## 4. Key metric summaries (from JSON)
- **134233 paired:** max-hold Δ=38.79 dB; TX-on peak 2.83e-3, TX-off 7.96e-4; no
  clipping. TX-on signal_detected=True, score 0.884, p2m 41.2 dB.
- **151529 rf_diag:** signal_detected=True, score 0.851, p2m 47.3 dB; peak 3.30e-3.
- **152017 (LR-FHSS 3-mode, NEGATIVE):** median |CFO| no_comp 4151, sgp4 10306,
  pgrl 9856 Hz (no ordering); ABLR no_comp −32.2, sgp4 −31.0, pgrl −31.1 dB
  (≤1 dB spread). 20–26 bursts detected vs 10 commanded. *Summaries are
  stale-format (`schedule_source/status/median_norm_abs_residual_cfo_grid/
  intended_bin_energy_fraction` = null; written before the c0d516d robust-metric
  fields).*
- **154458 (CW 3-mode, clean):** oscillator bias −2750 Hz; median residual-to-grid
  no_comp 12583, sgp4 635, pgrl 98 Hz (clean no_comp≫sgp4>pgrl); realization
  error ~17–28 Hz all modes.

## 5. Schedule / config provenance
- **Every** captured 3-mode run (`150700`, `152017`, `154458`) has
  `schedule_meta.demo_synthetic = True` and `capture_meta.schedule_source =
  demo_synthetic`. **No run used a real model-derived schedule.**
- Smoke/diagnostic runs (`131516`, `134233`, `151529`, `153751`, `154034`,
  `154227`) are fixed-config smokes, not schedule-driven model comparisons.

## 6. RF frequency / safety risk findings
Risk-term scan over `*.json/*.yaml/*.csv/*.log/*.txt` under `runs/`:
- `868000000` → **42 files**. **All runs were taken at the 868 MHz lab diagnostic
  frequency** (`nominal_center_freq_hz=868000000`, `rx_center=868200000`). Per
  current config policy this is a **lab diagnostic only, not a deployment /
  Taiwan-plan frequency**. Treat as historical lab artifacts; do not reuse the
  value as an approved carrier.
- `923` → 4 files: **benign** — substring inside CSV decimal values (e.g.
  `42042.923`), **not** a 923 MHz frequency.
- `PER` → 3 files: **benign** — appears only inside the disclaimer
  `"NOT LR-FHSS/PER/decoding"`, i.e. a negative statement, not a PER claim.
- `868 MHz`, `EU868`, `923` (as MHz), `true_doppler_hz`, `measured_doppler`,
  `truth`, `BER`, `CRC`, `gateway`, `live satellite` → **0 files**.
- No clipping in any run (`clipping_warning=false` throughout). TX power 10 dBm.

## 7. Usability classification per run
Legend: **A** = internal debugging only · **B** = conducted RF diagnostic
evidence · **C** = paper-grade PGRL-vs-SGP4 evidence.

| Run | A | B | C | Reason |
|---|---|---|---|---|
| 131516 noise floor | ✅ | ❌ | ❌ | RX-only noise reference; OTA, not conducted |
| 134233 TX-on/off paired | ✅ | ❌ | ❌ | real OTA RF presence; **room OTA, not conducted**; no schedule |
| 150700 3-mode (failed) | ✅ | ❌ | ❌ | no RF (noise); negative diagnostic; synthetic |
| 151529 rf_diag | ✅ | ❌ | ❌ | OTA RF presence; not conducted; no schedule |
| 152017 3-mode LR-FHSS | ✅ | ❌ | ❌ | no mode separation (negative); synthetic; not conducted |
| 153751 CW smoke (failed) | ✅ | ❌ | ❌ | no radiate |
| 154034 CW smoke | ✅ | ❌ | ❌ | incomplete metadata |
| 154227 CW smoke orphan | ✅ | ❌ | ❌ | incomplete; do-not-use |
| 154458 3-mode CW (clean) | ✅ | ❌ | ❌ | clean method demo, but **synthetic schedule** + **room OTA, not conducted** |

**No run qualifies for B or C.** All are OTA / room (uncalibrated), none conducted
(no coax/attenuator), and every schedule is `demo_synthetic`.

## 8. Explicit limitations (apply to ALL runs)
- **Not** live-satellite validation.
- **Not** measured Doppler truth — all schedules are synthetic; modeled/reference
  Doppler only, never measured.
- **Not** PER / BER / CRC / gateway-ACK evidence.
- **Not** regulatory-compliance evidence; 868 MHz here is a lab diagnostic, not a
  deployment frequency.
- **Not** conducted measurements — short-range room OTA / near-field, uncalibrated.

## 9. Recommended next step
- Keep `runs/` as an **internal debugging archive only**; do not cite any run in
  the paper. The ~2.1 G of raw `.fc32` is git-ignored (safe; not pushed).
- `154458` (clean CW 3-mode separation) and `134233` (paired Δ=38.79 dB) are the
  only methodology demonstrators worth keeping handy — for **debugging/method
  illustration only**, not evidence.
- Incomplete dirs `154227` (no meta) and `154034` (no meta) add no curated value;
  candidates for local cleanup **with user approval** (not done here).
- Paper-grade evidence still requires (separately): a **real model-derived
  schedule** (PGRL checkpoint + real TLE → predictions CSV) **and** a **conducted
  RF path** at an **approved** frequency — none of which exist in these runs.

> Do not commit (audit only).
