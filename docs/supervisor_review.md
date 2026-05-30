# Supervisor Review Package

> **Project:** PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT
> **Target:** IEEE GLOBECOM 2026
> **Status:** Trace-driven evaluation complete; hardware measurement planned
> **Commit:** `6ee154e` (https://github.com/Vanisherzd/LEO-Hybrid-PGRL/tree/globecom-prehw-2026-05)

---

## Problem Statement

Direct-to-Satellite (D2S) IoT using LEO satellites enables connectivity for power-constrained devices in remote areas, but high relative velocity (up to 8 km/s) induces Doppler shifts exceeding ±30 kHz and rapid Doppler-rate changes that stress the physical layer. Long-Range Frequency-Hopping Spread Spectrum (LR-FHSS), as standardized by Semtech (AN1200.64), achieves robust uplink by distributing transmit energy across a grid of narrow frequency bins. However, LR-FHSS guard bands must accommodate worst-case timing and frequency uncertainty, imposing idle-time overhead that erodes the energy budget of battery-powered IoT nodes.

Existing approaches treat orbital prediction and physical-layer control separately: TLE-based propagators (SGP4/SDP4) provide deterministic orbital state but accumulate errors over prediction horizons beyond a few minutes, while LR-FHSS systems typically rely on wide guard bands or continuous ground-station feedback.

---

## Key Idea

Use **Physics-Guided Residual Learning (PGRL)** — a Bayesian neural network anchored on SGP4 — to produce calibrated uncertainty estimates alongside point predictions of pass timing and Doppler. These outputs drive three physical-layer control decisions without continuous ground-station feedback.

---

## Contributions

1. **PGRL Orbital Predictor:** SGP4/SDP4-anchored Bayesian residual learner. Produces calibrated one-sigma uncertainty estimates alongside pass timing, Doppler residual, and Doppler-rate. Achieves 16 ms timing RMSE (>99.6% improvement vs SGP4) and <300 Hz residual Doppler (>95.5% improvement).

2. **Uncertainty-Aware LR-FHSS Uplink Controller:** Converts PGRL outputs into adaptive guard-time, transmission-time selection, and Doppler pre-compensation — without continuous ground-station feedback.

3. **Reproducible Evaluation Pipeline:** Trace-driven evaluation with explicit validation-type labeling (trace-driven / simulation / proxy / planned hardware), together with a Semtech LR1121/LR11xx and SDR-ready hardware validation path for future IQ-level measurement.

---

## Current Results

| Component | Metric | Baseline | Proposed | Validation |
|-----------|--------|----------|----------|------------|
| PGRL predictor | Pass timing RMSE | 4.1 s | 16.4 ms | Trace-driven |
| PGRL predictor | Residual Doppler | >5 kHz | <300 Hz | Trace-driven |
| Guard-band policy | Guard overhead | 2.41% | 0.23% | Simulation |
| Guard-band policy | Missed-opp. rate | 0.0005 | 0.0005 | Simulation |
| Doppler pre-comp | Residual Doppler | 2728 Hz | 334 Hz | Simulation |
| Doppler pre-comp | QPSK EVM (SNR=40 dB) | >208% | 67% | Proxy simulation |
| Doppler pre-comp | Oracle EVM (upper bound) | 0.95% | — | Upper bound |
| LR-FHSS grid proxy | Grid orthogonality | 0.978 | 0.706 | LR-FHSS-inspired proxy |

> **Note on EVM:** The QPSK EVM is an RF-quality proxy under controlled SNR=40 dB, not a standard LR-FHSS PER. PGRL compensation (67%) substantially reduces impairment relative to SGP4-only (>208%) but remains far from the oracle upper bound (0.95%). Production LR-FHSS systems require additional hardware CFO tracking beyond PGRL pre-compensation.

---

## What Is NOT Claimed (Limitations)

The paper explicitly states in the *Validation Scope and Limitations* section:

- No full standard-compliant LR-FHSS gateway is implemented
- No LR-FHSS packet error rate (PER) is reported
- QPSK EVM is a physical-layer RF-quality proxy only
- Semtech LR-FHSS TX and SDR IQ capture are **planned hardware stages**

This is intentional: the paper is positioned as a **trace-driven + hardware preparation** contribution, and the hardware limitations are disclosed proactively.

---

## Paper Structure

```
1. Introduction (problem, gap, contributions)
2. Background and System Model (D2S channel, SGP4, PGRL architecture)
3. PGRL Orbital Predictor (PGRLOutput schema, training loss, trace-driven results)
4. LR-FHSS Uplink Control (guard-band policy, Doppler pre-comp, TX timing)
5. Evaluation (experimental setup, trace-driven results, limitations)
6. Hardware Validation Plan (Semtech TX path, SDR IQ testbed — planned)
7. Conclusion
```

Figures: architecture diagram, uncertainty calibration, guard-band comparison, LR-FHSS grid orthogonality, SDR synthetic pipeline (hardware results pending).

Tables: main results, PGRL ablation, guard-band policy comparison — auto-generated from `experiments/summary_table.py`.

---

## Pending Hardware

| Hardware | Purpose | Status |
|----------|---------|--------|
| Semtech LR1121/LR11xx + SWDM001 | LR-FHSS TX, waterfall characterization | Planned |
| USRP B210 + 30 dB attenuator | IQ capture, CFO estimation, EVM measurement | Planned |

Once hardware is connected, the dry-run pipeline (`semtech_validation/tx_config_from_pgrl.py`, `sdr_hwil/`) is ready to generate real measurement outputs. Fig. 5 will be replaced with hardware IQ capture results.

---

## What's Needed from Supervisor

1. **Review paper structure and writing** — particularly the introduction framing and the contribution claims
2. **Advise on scope management** — the paper currently uses trace-driven and proxy results; confirm this is appropriate for GLOBECOM submission without hardware measurements
3. **Hardware access** — if Semtech LR1121 and/or USRP B210 are available, the measurement stage can be executed in parallel with paper submission

---

## How to Run the Experiments

```bash
cd LEO-Hybrid-PGRL
uv sync

# Generate paper tables from experiment results
uv run python experiments/summary_table.py

# Generate paper figures (matplotlib)
uv run python experiments/generate_figures.py

# Individual experiments
uv run python physics_ml/evaluate_prediction.py
bash experiments/exp2_guard_band_energy/run.sh
bash experiments/exp3_lrfhss_grid_proxy/run.sh
```