# Experiment 3 — LR-FHSS-Inspired Frequency-Grid Proxy

**Purpose:** Evaluate residual Doppler impact on frequency-bin alignment and collision proxy using a grid-based LR-FHSS proxy metric. This is NOT a full LR-FHSS PHY decoder — it is an RF-quality proxy for hopping-bin orthogonality.

## Baselines
1. No compensation
2. SGP4 compensation
3. PGRL compensation
4. Oracle (true Doppler)

## Metrics
- frequency-bin alignment error
- orthogonality score
- collision proxy (probability)
- hopping-bin energy consistency

> **Note on naming:** The grid proxy measures how residual Doppler shifts the received hopping bins relative to the nominal grid. It does not perform LR-FHSS demodulation or compute standard PER. Naming follows the "LR-FHSS-inspired proxy" convention from the paper scope.