# BLACK KITE-1 Target-Specific TLE-History Residual Doppler Experiment — Path B
*Generated: 2026-06-13 15:50:19 UTC  |  `tools/bk1_target_specific_residual_experiment.py`*

## Formulation (target-specific)

| Parameter | Value |
|---|---|
| Target | BLACK KITE-1  NORAD 66741 (single satellite) |
| `reference_doppler_hz` | Later/newer-TLE SGP4-derived Doppler |
| `sgp4_model_doppler_hz` | Stale/older-TLE SGP4 open-loop Doppler at same UTC |
| `pgrl_model_doppler_hz` | `sgp4_model_doppler_hz + predicted_residual` (trained only on BK1 pre-test history) |
| `reference_is_measured_truth` | **False** |
| Carrier | 868.0 MHz (LR-FHSS band) |
| Ground station | 24.0°N 121.0°E 100.0 m (Taiwan representative) |
| SGP4 propagator | python-sgp4 (Brandon Rhodes) |
| Compute | cuda (torch GPU) + scikit-learn |
| Hardware / RF | **None — software-only** |

## Data summary

| Satellite | NORAD | Records | Epoch start | Epoch end |
|---|---|---|---|---|
| BLACK KITE-1 | 66741 | 415 | 2025-12-18 | 2026-06-12 |

**Inter-TLE gap (h):** median 6.3  p10 0.0  p90 17.4  max 493.7

## Chronological split (by reference epoch — zero future-TLE leakage)

Two global UTC boundaries from the 60%/20% time-quantile of the BK1 record-epoch span gate **every** staleness window. A TLE pair is assigned by its reference (newer) epoch; its stale (older) TLE is always earlier, so training never observes a future TLE.

| Segment | From (UTC) | To (UTC) | Records |
|---|---|---|---|
| Train (early 60%) | 2025-12-18T03:26:34.220832+00:00 | 2026-04-03T01:54:07.412026+00:00 | 173 |
| Validation (mid 20%) | 2026-04-03T01:54:07.412026+00:00 | 2026-05-08T09:23:18.475757+00:00 | 132 |
| Held-out test (late 20%) | 2026-05-08T09:23:18.475757+00:00 | 2026-06-12T16:52:29.539488+00:00 | 110 |

## Features (10)

`t_age_s, t_gap_s, stale_doppler_hz, sin_phase, cos_phase, elevation_deg, range_km, stale_mean_motion, stale_bstar, stale_ecc`

Target: `residual_hz = reference_doppler_hz - sgp4_model_doppler_hz`.

## Dataset construction

- **Staleness pairing (operational):** for each reference (newer) TLE, the *older* TLE whose epoch gap is inside the window band and closest to the target staleness is selected — **not** restricted to consecutive TLEs. This is the genuine "hold a TLE that is N h old, propagate open-loop, compare to a later TLE" scenario, and is what populates the long-staleness windows (BK1 refreshes ~6 h, so a 48 h-stale pair is an older TLE 48 h back, not a 48 h consecutive gap).
- 24 UTC samples per pair over one orbital period (~95 min).
- **Outlier rejection:** a pair is discarded if any sample's |residual| > 1500 Hz (manoeuvre / bad-OD epoch). Counts reported below.

## Models

zero-residual baseline · median (constant-bias) baseline · ridge (alpha tuned on val) · random forest (sklearn) · gradient boosting (sklearn HistGBR) · MLP (torch, CUDA GPU).
The two baselines are references, **not** residual correctors. Selection is among the **learned** models (ridge/RF/GBR/MLP) by **validation** MAE; the selected learned model's **held-out test** MAE/RMSE is the decision metric. A window counts as supported only if that learned model beats the zero baseline by ≥1% relative MAE **and** improves RMSE — sub-noise constant-offset wins do not qualify.

## Per-window held-out TEST MAE [Hz @ 868 MHz]

| Window | n(tr/va/te) | outliers | zero | median | ridge | RF | GBR | MLP | selected | test MAE | Δ% | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8h | 2712/2520/1968 | 0 | 0.2430 | 0.2438 | 2.4941 | 1.8807 | 4.3976 | 0.3501 | mlp_gpu | 0.3501 | -44.1 | ✗ blocked |
| 24h | 3240/3144/2376 | 6 | 0.8161 | 0.8250 | 4.0449 | 15.8441 | 20.6578 | 0.9109 | mlp_gpu | 0.9109 | -11.6 | ✗ blocked |
| 48h | 2928/3144/2520 | 5 | 1.9433 | 1.9576 | 14.1854 | 21.3059 | 19.9189 | 2.8608 | mlp_gpu | 2.8608 | -47.2 | ✗ blocked |
| 72h | 2736/3168/2640 | 11 | 4.8947 | 4.9404 | 5.9092 | 12.7209 | 15.2894 | 9.3962 | ridge | 5.9092 | -20.7 | ✗ blocked |
| 96h | 2688/3168/2616 | 13 | 10.1153 | 10.1320 | 11.7663 | 120.5161 | 156.5905 | 22.0370 | ridge | 11.7663 | -16.3 | ✗ blocked |
| 168h | 2448/3168/2640 | 19 | 26.9243 | 30.2474 | 45.2629 | 62.6705 | 227.7139 | 89.3023 | ridge | 45.2629 | -68.1 | ✗ blocked |

*Columns zero…MLP are each model's held-out test MAE. "selected" = best **learned** model by validation MAE; Δ% = its improvement over the zero baseline.*

## Held-out test residual distribution [Hz]

| Window | n | mean | std | p50(|r|) | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| 8h | 1968 | +0.004 | 0.419 | 0.136 | 0.575 | 1.658 | 3.217 |
| 24h | 2376 | -0.006 | 1.413 | 0.458 | 1.815 | 5.401 | 12.961 |
| 48h | 2520 | -0.003 | 3.252 | 1.219 | 4.323 | 12.563 | 41.941 |
| 72h | 2640 | +0.018 | 8.473 | 2.982 | 11.020 | 33.250 | 98.635 |
| 96h | 2616 | -0.081 | 19.455 | 5.606 | 22.434 | 79.681 | 257.543 |
| 168h | 2640 | +0.035 | 45.383 | 15.874 | 57.607 | 191.260 | 519.419 |

## Outlier (manoeuvre / bad-OD) pairs removed

| Window | removed pairs | top |max residual| [Hz] |
|---|---|---|
| 8h | 0 | — |
| 24h | 6 | 2326, 2194, 2194, 2072, 2072 |
| 48h | 5 | 14457, 3028, 2208, 1549, 1549 |
| 72h | 11 | 26210, 11386, 8388, 8388, 6855 |
| 96h | 13 | 14755, 14078, 14078, 12500, 6890 |
| 168h | 19 | 30574, 28081, 22768, 19454, 19263 |

## Verdict

> **Path B remains blocked even for BLACK KITE-1.**

Decision rule: Path B is supported for a window iff the validation-selected **learned** model's held-out test MAE is below the zero-residual stale-TLE baseline test MAE by ≥1% **and** its test RMSE also improves. The median (constant-bias) baseline is reported but never triggers support.

## Output files

| File | Path | SHA256 |
|---|---|---|
| Replay CSV | `N/A (no improvement — CSV not exported)` | N/A |
| This report | `docs/review/black_kite_1_target_specific_residual_experiment.md` | (self) |
| Evidence gate | `docs/review/black_kite_residual_evidence_gate.md` | (see file) |

Replay schema: `t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz` (+ `# reference_is_measured_truth=false` metadata header).

## Exact limitations

1. `reference_doppler_hz` is a later TLE's SGP4 propagation — **not measured Doppler truth**.
2. Ground station is representative (24°N 121°E 100 m); no real GS coordinates used.
3. No hardware, UART, replay, TX/RX, or RF signal was involved.
4. No live satellite contact, PER/BER/CRC, or gateway ACK.
5. No synthetic `sat_*.npz` data; no old synthetic PGRL checkpoint used.
6. Doppler residuals encode mean-element + drag-term + numerical-fit differences between successive TLE solutions; they are a proxy for stale-TLE open-loop error, not an absolute frequency error.
7. Split is chronological by reference epoch; the stale TLE in any test pair is always available before the test reference epoch.
8. Only the software-defined TLE-history residual correction approach is evaluated.
