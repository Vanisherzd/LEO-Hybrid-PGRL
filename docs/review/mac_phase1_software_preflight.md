# Mac Phase 1 — Software-Only Preflight

Software-only preflight for the OTA-IQ replay pipeline. **No RF capture, no TX,
no UART command** was performed. All checks are read-only or `/tmp`-scoped
dry-runs.

- **Date:** 2026-06-13
- **Repo:** `/Users/laizhendong/Desktop/LEO-Hybrid-PGRL`
- **Branch:** `paper-crosslayer-rewrite`
- **Commit (HEAD):** `c0d516d` — *metrics: add robust OTA-IQ diagnostic summaries*
- **Working tree:** clean (no modified tracked files at preflight start)

## Python dependency status
| Package | Version | Status |
|---|---|---|
| numpy | 2.4.6 | OK |
| pyyaml | 6.0.3 | OK |
| pyserial | 3.5 | OK |

`hardware/ota_iq` scripts compile OK (confirmed prior).

## Device discovery (from previously provided results — NOT re-run here)
- **UHD / USRP B210:** detected by UHD on this host; device serial `8000304`
  (B200/B210, USB 3.0). *No `uhd_find_devices` or RX stream was started in this
  preflight.*
- **Serial port:** `/dev/cu.usbmodem1303` = STM32 NUCLEO-L476RG ST-LINK VCP
  (host-replay LR1121 board). *No UART write/handshake performed in this preflight.*

## Schema / guard dry-run results

### Safe reference schema — PASS
`generate_real_replay_schedule.py --predictions-csv <safe.csv> --nominal-center-hz 1 --dry-run`
- Input `/tmp/phase1_reference_doppler.csv` (**synthetic preflight fixture, NOT
  real model output**), columns `t_s, reference_doppler_hz,
  sgp4_model_doppler_hz, pgrl_model_doppler_hz`.
- `--nominal-center-hz 1` is a **schema/guard placeholder only** — it is not an
  experiment frequency.
- Dry-run printed 10 schedule rows; each row carries `reference_is_measured_truth=false`.
- Per-mode offset stats + md5 (from this synthetic fixture):

  | mode | min | median | max | md5 |
  |---|---|---|---|---|
  | no_compensation | 0.00 | 12000.00 | 18000.00 | `a437188c61dce1631a65c48c0abd6262` |
  | sgp4_only | 0.00 | 2150.00 | 3000.00 | `b7efbe9a8003f2e974915926f80ab1fe` |
  | pgrl_corrected | 0.00 | 75.00 | 100.00 | `e853059472f6ff54a4c91a782f3f5524` |
- **DRY-RUN: nothing written** (no `--write`). No repo files created.

### Unsafe schema (measured-truth naming) — REFUSED
`--predictions-csv` with a `true_doppler_hz` column was **rejected**:
> REFUSED: misleading Doppler-truth column(s) present: `['true_doppler_hz']` …
> This pipeline does not accept `true_doppler_hz` or measured-truth naming …

Prohibited input columns enforced: `true_doppler_hz`, `truth_doppler_hz`,
`measured_doppler_hz`, `measured_doppler_truth_hz`.

### F0 = 0 guard — BLOCKS plan
`usrp_capture_ota_iq.py plan` on a repo config (`nominal_center_freq_hz: 0`)
**halted before any output**:
> nominal_center_freq_hz must be explicitly set to a positive carrier frequency
> after NCC/local/gateway frequency-plan confirmation. No default carrier is
> used, and OTA must remain disabled until confirmed.

`/tmp/phase1_plan_guard` was **not created** (guard fired before write).

## Config guard state (all three replay configs)
`replay_no_compensation.yaml`, `replay_sgp4_only.yaml`, `replay_pgrl_corrected.yaml`:
- `nominal_center_freq_hz: 0`
- `frequency_plan: "TAIWAN_AS923_REVIEW_REQUIRED"`
- `ota_transmission_allowed: false`

## Analyzer robustness
- `analyze_cfo_residual.py`: emits `status="no_bursts_detected"`,
  `n_bursts_detected=0` summary on empty/noise input (no crash).
- `analyze_adjacent_bin_leakage.py`: emits `status="no_bursts_detected"` summary.
- Not exercised against real USRP capture (none taken in this preflight).

## Current allowed next step
Software-only iteration only: schema/guard dry-runs, analyzer logic, schedule
generation **dry-runs**. No capture, no TX, no UART command.

## Explicit blockers before any capture / TX
1. **Confirmed local / NCC / gateway channel plan** (Taiwan AS923 review pending;
   `frequency_plan = TAIWAN_AS923_REVIEW_REQUIRED`).
2. **Explicit positive `nominal_center_freq_hz`** set in configs **only after**
   approval (currently 0 → all RF paths blocked).
3. **Conducted/coax RF path** with attenuator (≥30 dB) or 50-ohm load — preferred
   over OTA; room OTA remains uncalibrated diagnostic only.
4. **No antenna TX** until the RF path above is in place and approved.
5. **User approval before `usrp_capture_ota_iq.py capture`**.
6. **User approval before `replay_driver.py` without `--dry-run`**.
7. **LR1121 host-replay firmware / handshake confirmation** (REPLAY_READY / RDY /
   CARRIER_DONE) on `/dev/cu.usbmodem1303`.

## Real model-derived data status (unchanged blocker)
No trained PGRL checkpoint, no per-sample SGP4/PGRL Doppler trace, and no real
TLE/pass exist in the repo. The safe dry-run above used a **synthetic** fixture.
A real schedule still requires the trained checkpoint + real prediction CSV.
