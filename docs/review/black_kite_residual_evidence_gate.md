# BLACK KITE Residual-Correction Evidence Gate
*Generated: 2026-06-13 15:50:19 UTC*

This note records the data-driven gate that decides, **per target satellite and
staleness regime**, whether a learned TLE-history Doppler residual correction is
enabled or whether the system falls back to open-loop SGP4 / stale-TLE
compensation. `reference_is_measured_truth = false` in all cases (the reference
Doppler is a later-TLE SGP4 propagation, not a measured signal).

## Gate logic

```text
if heldout_residual_model_MAE < stale_baseline_MAE:
    enable residual correction          # learned model beats open-loop
else:
    disable residual correction         # open-loop SGP4 / stale-TLE is better/equal
```

The model is selected on a chronological **validation** segment and the gate is
evaluated **once** on a chronological **held-out test** segment. Splits are by
reference epoch with no future-TLE leakage.

## Case 1 — BLACK KITE-1 (NORAD 66741): high-residual target

BK1 (NORAD 66741) target-specific: **BLOCKED**

No staleness window improved held-out late-BK1 test MAE over the zero-residual stale-TLE baseline. Residual correction DISABLED; fall back to open-loop SGP4 / stale-TLE compensation.

- Experiment: `tools/bk1_target_specific_residual_experiment.py`
- Report: `docs/review/black_kite_1_target_specific_residual_experiment.md` (SHA256 `31a08f064a58b07662fefc68b89e1aadda9614fb4258d863b2d9f6c2332e5a54`)
- Replay CSV: not exported (correction not enabled).

## Case 2 — BLACK KITE-2 (NORAD 68474): negative control

BK2 is a **negative-control / negative-transfer** case, not the main target.
Its Space-Track TLE refresh cadence (~6 h median) makes consecutive-TLE Doppler
residuals negligible (held-out MAE < 0.25 Hz at 868 MHz in the short-staleness
regime). The zero-residual baseline is already excellent, so the gate
**disables** residual correction for BK2. A cross-satellite BK1→BK2 model
*increases* error (distribution mismatch + BK1 manoeuvre outliers), confirming
that residual correction must be evidence-gated rather than always-on.

- Experiment: `tools/bk_tle_residual_experiment.py`
- Report: `docs/review/black_kite_tle_history_residual_experiment.md`
  (SHA256 `fdd8b19582d1623541ee5cb6baeb73824dbeb545da94ab1d70bb4f399d2d2a93`)

## System rule

The deployed compensator MUST consult this gate:

1. If a target/staleness regime has a validated residual model with
   `heldout_MAE < baseline_MAE`, apply `pgrl_model_doppler_hz` (stale + residual).
2. Otherwise, apply open-loop `sgp4_model_doppler_hz` (stale-TLE Doppler) and do
   **not** claim a residual-correction benefit.
3. Never treat `reference_doppler_hz` as measured truth; it is model-derived.

## Limitations

No hardware, RF, UART, replay, TX/RX, PER/BER/CRC, or gateway ACK was involved.
No synthetic `sat_*.npz` or old PGRL checkpoint was used as BLACK KITE evidence.
Raw `dataraw/` Space-Track files are inputs only and are not committed.
