# Real BLACK KITE Residual — Compact Negative Result

> **SOFTWARE-ONLY DERIVED SUMMARY.** Re-summarised from existing experiment
> reports. No new experiment was run, no model was trained, no hardware was used
> to produce this file. `reference_is_measured_truth=false`. Data: `bk_negative_result_compact.csv`.

*Generated: 2026-06-14 UTC. Sources:
`docs/review/black_kite_1_target_specific_residual_experiment.md` (BK1
target-specific, 8–168 h) and
`docs/review/black_kite_tle_history_residual_experiment.md` (BK1→BK2
cross-satellite). All MAE in Hz at 868 MHz; `zero` = zero-residual stale-TLE
baseline.*

## BK1 target-specific (held-out test MAE)

| Staleness | Zero baseline MAE | Selected learned | Learned MAE | Δ% vs zero | Verdict |
|---|---|---|---|---|---|
| 8 h | 0.2430 | mlp_gpu | 0.3501 | −44.1 | ✗ blocked |
| 24 h | 0.8161 | mlp_gpu | 0.9109 | −11.6 | ✗ blocked |
| 48 h | 1.9433 | mlp_gpu | 2.8608 | −47.2 | ✗ blocked |
| 72 h | 4.8947 | ridge | 5.9092 | −20.7 | ✗ blocked |
| 96 h | 10.1153 | ridge | 11.7663 | −16.3 | ✗ blocked |
| 168 h | 26.9243 | ridge | 45.2629 | −68.1 | ✗ blocked |

## BK1→BK2 cross-satellite (held-out test MAE)

| Staleness | Zero baseline MAE | Selected learned | Learned MAE | Δ% vs zero | Verdict |
|---|---|---|---|---|---|
| 8 h | 0.1877 | mlp | 0.3261 | −73.7 | ✗ blocked |
| 24 h | 0.4969 | mlp | 1.8639 | −275.1 | ✗ blocked |
| 48 h | 2.4092 | ridge | 2.8458 | −18.1 | ✗ blocked |

## One-line result

On real Space-Track TLE history for two BLACK KITE LEO satellites (NORAD 66741,
68474), with strict chronological leakage-free splits, a learned Doppler residual
model did **not** beat the open-loop SGP4 / stale-TLE zero baseline at **any**
tested staleness (8–168 h), target-specific or cross-satellite. Δ% is negative
everywhere. The inter-TLE residual is near zero-mean and unpredictable; the
evidence gate therefore **closes** on real data.

## What this does NOT say

- ❌ not "learning improves real BLACK KITE Doppler" (the opposite holds)
- ❌ not measured Doppler truth (`reference_is_measured_truth=false`)
- ❌ not a live-satellite, RF, PER/BER/CRC/PDR, or gateway-ACK result
- ❌ not a universal claim for all satellites or all staleness regimes
