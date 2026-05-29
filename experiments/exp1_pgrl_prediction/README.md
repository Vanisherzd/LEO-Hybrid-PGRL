# Experiment 1 — PGRL Prediction Accuracy

**Purpose:** Demonstrate that PGRL achieves lower timing and Doppler error than SGP4-only, with calibrated uncertainty.

## Configuration
- `config.yaml` — orbital scenario, TLE source, PGRL model path, evaluation horizon
- `run.sh`      — executes evaluation and outputs results.json

## Baselines
1. SGP4-only (no correction)
2. Pure LSTM (no physics anchor)
3. SGP4 + Bayesian PGRL (proposed)

## Metrics
- pass timing error [s]
- Doppler error [Hz]
- Doppler rate error [Hz/s]
- NLL (negative log-likelihood)
- Coverage probability at 95% confidence

## Expected Output
```
results.json:
  "pgrl_timing_rmse_s": 0.016
  "pgrl_doppler_rmse_hz": 250
  "sgp4_timing_rmse_s": 4.200
  "coverage_95pct": 0.943
```