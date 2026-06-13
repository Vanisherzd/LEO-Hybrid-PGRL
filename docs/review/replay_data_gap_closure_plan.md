# Replay Data Gap Closure Plan (software-only)

Plan to close the data gap that blocks a **real** model-derived OTA-IQ replay
schedule. Software-only; **no hardware, no capture, no TX, no UART**. No paper edit.

- **Date:** 2026-06-13
- **Repo:** `/Users/laizhendong/Desktop/LEO-Hybrid-PGRL`
- **Branch:** `paper-crosslayer-rewrite`
- **Commit (HEAD):** `8cbb700` — *docs(review): add replay data provenance inventory*

> Scope: orbital/model inputs only. No live-satellite validation, no measured
> Doppler truth, no PER/BER/CRC, no gateway ACK, no regulatory-compliance claim.

## 1. Current blocker summary
From `docs/review/data_provenance_inventory_for_replay.md` (MISSING DATA):
- **No trained PGRL checkpoint/weights** in repo (only model code).
- **No real TLE / pass** (`sample_tle.txt` = placeholder NORAD 99999; `data_raw/tle/`
  git-ignored + absent).
- **No real replay CSV** with `reference_doppler_hz, sgp4_model_doppler_hz,
  pgrl_model_doppler_hz`.
- Configs ship `doppler_profile_csv: null`, `compensation_csv: null`,
  `nominal_center_freq_hz: 0`, `ota_transmission_allowed: false`.
- Hardware discovery passed, but **capture/TX remains blocked**.

## 2. Required final replay CSV schema
`generate_real_replay_schedule.py` consumes:

```
t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz
```

Per-mode commanded offsets:
- `no_compensation_offset_hz = reference_doppler_hz`
- `sgp4_only_offset_hz       = reference_doppler_hz − sgp4_model_doppler_hz`
- `pgrl_corrected_offset_hz  = reference_doppler_hz − pgrl_model_doppler_hz`

Generator emits `schedule_source = model_reference_derived` and
`reference_is_measured_truth = false`. Prohibited column names (refused):
`true_doppler_hz, truth_doppler_hz, measured_doppler_hz, measured_doppler_truth_hz`.

## 3. Safe meaning of each column
- **`reference_doppler_hz`** — model/reference-derived replay Doppler profile
  (e.g. from TLE propagation). **NOT measured on-air Doppler truth.**
- **`sgp4_model_doppler_hz`** — SGP4/open-loop **baseline model** Doppler (the
  uncorrected predictor the terminal would use without learning).
- **`pgrl_model_doppler_hz`** — PGRL-corrected model Doppler, **populated only if
  a trained PGRL checkpoint with provenance exists**. Never fabricated.

> Implementation note: the repo's `models/orbital_physics.py` propagator is an
> analytic **Kepler + J2** model (`propagate_j2`), **not** the standard SGP4/SDP4
> library. A baseline produced from it must be labeled "physics/J2 analytic
> baseline (SGP4-class)", not raw SGP4, unless a true SGP4 propagator is added.

## 4. Three paths

### Path A — Real-TLE SGP4/physics-only software baseline
- **Required files:** a real TLE (under git-ignored `data_raw/tle/`),
  `models/orbital_physics.py`, a small software generator script.
- **Missing files:** the **real TLE** (only placeholder exists).
- **Scripts needed:** new `orbital_physics`-driven reference generator →
  emits `t_s, reference_doppler_hz, sgp4_model_doppler_hz`; then
  `generate_real_replay_schedule.py` (consumes them). `controller/doppler_precomp.py`
  provides `residual_doppler_after_precomp` / `doppler_rate_from_series`.
- **`pgrl_model_doppler_hz`:** absent → only `no_compensation` + `sgp4_only`
  modes are populated; `pgrl_corrected` cannot be filled.
- **Allowed claim level:** SGP4/physics-class **open-loop baseline diagnostic**
  (model-derived). NOT a PGRL-vs-SGP4 result. NOT measured truth.
- **Supports hardware capture?** Only after RF gates (§6) — and only the two
  baseline modes; not the PGRL comparison.

### Path B — Real-TLE + real PGRL checkpoint
- **Required files:** real TLE **and** trained PGRL checkpoint (+ provenance),
  `models/orbital_physics.py`, PGRL inference runner, generator.
- **Missing files:** **PGRL checkpoint** and **real TLE** (both absent).
- **Scripts needed:** Path A generator **plus** a PGRL inference step emitting
  `pgrl_model_doppler_hz`; then `generate_real_replay_schedule.py --write`.
- **Allowed claim level:** full **model-derived PGRL-vs-SGP4** replay comparison
  (still model/reference Doppler, not measured truth).
- **Supports hardware capture?** Yes, once **both** data gates and RF gates are
  cleared. This is the only path that yields a paper-grade PGRL claim.

### Path C — Synthetic-only dry-run
- **Required files:** a synthetic `/tmp` fixture (no repo write).
- **Missing files:** none.
- **Scripts needed:** `generate_real_replay_schedule.py --dry-run` (already works).
- **Allowed claim level:** **NONE** — schema/wiring validation only. **Explicitly
  NOT paper evidence.** Must never be transmitted as real.
- **Supports hardware capture?** **No.** Synthetic schedules must not drive a real
  TX run; dry-run plumbing checks only.

## 5. Recommended safest next software-only action
Build, as **software-only** prep, a guarded `orbital_physics`-driven reference
generator for **Path A** that:
- accepts a **real** TLE path, **refuses** placeholder/synthetic markers
  (NORAD 99999, "example", "synthetic"),
- emits `t_s, reference_doppler_hz, sgp4_model_doppler_hz` with provenance,
- is **validated only via a synthetic `/tmp` dry-run** (Path C) — never run on a
  real or placeholder TLE in this step, never `--write` into the repo,
- leaves `pgrl_model_doppler_hz` empty until a checkpoint exists (Path B).

This advances readiness with zero RF risk and zero fabrication. Do **not**
populate PGRL output and do **not** treat any TLE-derived series as measured truth.

## 6. Hardware capture/TX gate (unchanged)
Hardware capture/TX **remains BLOCKED** until **both** gate sets clear:

- **Data gates:** real TLE selected (Path A/B) **and**, for any PGRL claim, a
  trained PGRL checkpoint (Path B). Path C never authorizes TX.
- **RF gates:** user approval · RX-only confirmed for capture · confirmed
  NCC/local/gateway channel plan · explicit positive `nominal_center_freq_hz`
  after approval · conducted/coax + attenuator (≥30 dB) or 50-Ω load · no antenna
  TX · LR1121 firmware/handshake confirmed · lowest practical TX power · user
  approval before any non-dry-run replay.

## 7. Conclusion
Only **Path C (synthetic dry-run)** is possible right now, and it is **not** paper
evidence. **Path A** is one real TLE away from a baseline-only diagnostic;
**Path B** (the only paper-grade PGRL path) additionally needs the trained PGRL
checkpoint. No path authorizes hardware capture/TX until §6 gates clear.
