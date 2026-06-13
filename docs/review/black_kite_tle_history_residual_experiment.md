# BLACK KITE TLE-History Residual Doppler Experiment — Path B
*Generated: 2026-06-13 15:20:57 UTC  |  `tools/bk_tle_residual_experiment.py`*

## Experiment formulation

| Parameter | Value |
|---|---|
| `reference_doppler_hz` | Newer-TLE SGP4-derived Doppler |
| `sgp4_model_doppler_hz` | Stale/older-TLE SGP4-derived Doppler at same UTC |
| `pgrl_model_doppler_hz` | `sgp4_model_doppler_hz + predicted_residual` |
| `reference_is_measured_truth` | **False** |
| Carrier | 868.0 MHz (LR-FHSS band) |
| Ground station | 24.0°N 121.0°E 100.0 m (Taiwan representative) |
| SGP4 propagator | python-sgp4 (Brandon Rhodes) v2.25 |
| Hardware / RF | **None — software-only** |

## Data summary

| Satellite | NORAD | Records | Epoch start | Epoch end | Role |
|---|---|---|---|---|---|
| BLACK KITE-1 | 66741 | 415 | 2025-12-18 | 2026-06-12 | Train |
| BLACK KITE-2 | 68474 | 184 | 2026-04-13 | 2026-06-13 | **Held-out test** |

**BK1 inter-TLE gap (h):** median 6.3  p10 0.0  p90 17.4  max 493.7
**BK2 inter-TLE gap (h):** median 6.4  p10 0.0  p90 16.1  max 70.8

## Dataset construction

- Each TLE pair (stale, reference) with gap in a staleness window is used.
- Pairs containing any sample with |residual| > 150 Hz are discarded (manoeuvre / bad-fit epoch).
- 24 UTC time samples per pair over one orbital period (95 min).
- **Split rule:** BK1 entirely used for training; BK2 entirely held-out for testing. Chronological within each satellite. Zero leakage.

## Features

| # | Name | Description |
|---|---|---|
| 0 | `t_age_s` | Age of stale TLE at sample time [s] |
| 1 | `t_gap_s` | Epoch gap between stale and reference TLE [s] |
| 2 | `stale_doppler_hz` | Stale-TLE predicted Doppler [Hz] |
| 3 | `sin_phase` | sin(mean anomaly at t, from stale TLE) |
| 4 | `cos_phase` | cos(mean anomaly at t) |
| 5 | `elevation_deg` | Satellite elevation from GS (stale TLE) [°] |
| 6 | `range_km` | Slant range GS→sat (stale TLE) [km] |

## Multi-staleness results (BK1→BK2 cross-satellite)

| Stale window | BK1 MAE [Hz] | BK2 MAE [Hz] | Baseline | Ridge | MLP | Verdict |
|---|---|---|---|---|---|---|
| 8h | 0.669 | 0.188 | 0.1877 | 0.3738 | 0.3261 | ✗ blocked (MLP -73.7%) |
| 24h | 3.007 | 0.497 | 0.4969 | 2.5413 | 1.8639 | ✗ blocked (MLP -275.1%) |
| 48h | 2.172 | 2.409 | 2.4092 | 2.8458 | 2.9593 | ✗ blocked (Ridge -18.1%) |

*Baseline = zero-residual (predict 0 correction, use stale Doppler as-is).
All MAE values in Hz at 868 MHz.*

## Verdict

> **Path B remains blocked.**

### Root-cause analysis

1. **BK2 residuals are negligibly small at short staleness (≤14 h):**
   The Space-Track TLE refresh cadence for BK2 is ~6 h median,
   producing consecutive-TLE Doppler residuals with MAE < 0.25 Hz at 868 MHz.
   There is essentially no signal to correct in the short-staleness regime.

2. **BK1 training distribution severely mismatched from BK2 test distribution:**
   BK1 residuals are 4×
   larger than BK2 residuals at the 8h staleness window.
   BK1 exhibits extreme outlier residuals (up to 6930 Hz) consistent with
   post-launch orbit determination instability or possible manoeuvre epochs.
   Models trained on BK1 import this large-residual bias onto BK2, increasing
   error rather than reducing it.

3. **No staleness window yielded improvement:**
   Even at 24 h and 48 h staleness the cross-satellite model (BK1→BK2)
   fails to improve over the zero-residual baseline.

## Output files

| File | SHA256 |
|---|---|
| Replay CSV | N/A |
| This report | (self) |

*No CSV exported (no improvement).*

Replay schema: `t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz`

## Exact limitations

1. `reference_doppler_hz` is derived from a later TLE's SGP4 propagation — **not measured Doppler truth**.
2. Ground station position is representative (24°N 121°E 100 m); no real GS coordinates used.
3. No hardware, UART, replay, TX/RX, or RF signal was involved.
4. No live satellite contact, PER/BER/CRC, or gateway ACK.
5. No synthetic `sat_*.npz` data used. No old synthetic PGRL checkpoint used.
6. BK1 TLE history exhibits post-launch orbit determination instability and probable manoeuvre epochs not present in BK2, making cross-satellite residual transfer unreliable.
7. Doppler residuals encode mean-element update + drag parameter changes + numerical fitting differences between successive TLE solutions.
8. The experiment covers only the software-defined TLE-history residual correction approach; no alternative path-B formulations were evaluated.
