# GLOBECOM 2026 Pre-Hardware Review Package

**Project:** PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for D2S IoT
**Version:** Pre-hardware draft — trace-driven + proxy validation
**Date:** 2026-05-30
**Commit:** b6a2429 (main branch, https://github.com/Vanisherzd/LEO-Hybrid-PGRL)

---

## Package Contents

```
review_package/
├── globecom_main.pdf       ← Compiled GLOBECOM paper (196 KB, 9 sections)
├── supervisor_review.md    ← 1-page supervisor summary
└── README_snapshot.md     ← This file
```

---

## What This Version Is

This is a **trace-driven + proxy-simulation draft**. It is NOT a hardware-validated submission.

The paper establishes a complete framework from orbital prediction to uplink control,
with all numerical results derived from TLE traces, Bayesian predictor evaluation,
energy modeling, and LR-FHSS grid proxy analysis.

Hardware validation (Semtech LR1121 TX + USRP B210 IQ capture) is **planned** and
documented, but not yet executed.

---

## Key Claims in This Draft

| Claim | Validation Type | Status |
|-------|----------------|--------|
| PGRL 16 ms timing RMSE vs SGP4 4.1 s | Trace-driven | ✅ Complete |
| PGRL residual Doppler <300 Hz vs SGP4 >5 kHz | Trace-driven | ✅ Complete |
| Guard overhead reduction 60% | Simulation | ✅ Complete |
| LR-FHSS grid orthogonality (proxy) | LR-FHSS-inspired proxy | ✅ Complete |
| Semtech TX / SDR HWIL CFO + EVM | Planned hardware | ⏳ Pending |

---

## What Needs to Be Done Before Submission

1. **Supervisor review** of this draft — wording, contributions, scope
2. **Paper Polish** — tighten to 6-page IEEE limit, refine figures
3. **Hardware** — connect USRP B210 + Semtech LR1121:
   - Replace Fig. 5 with real IQ capture
   - Add hardware CFO/EVM table
   - Update abstract and table validation labels

---

## How to Run the Experiment Pipeline

```bash
git clone https://github.com/Vanisherzd/LEO-Hybrid-PGRL.git
cd LEO-Hybrid-PGRL
uv sync

# Trace-driven evaluation
uv run python physics_ml/evaluate_prediction.py

# Generate paper tables from results
uv run python experiments/summary_table.py

# Generate paper figures
uv run python experiments/generate_figures.py

# Compile paper PDF (requires tectonic or texlive)
cd paper && tectonic globecom_main.tex
```

---

## What NOT to Claim Yet

- Any hardware measurement result (CFO, EVM, waterfall)
- Full LR-FHSS gateway compliance
- Packet error rate (PER) under standard LR-FHSS decoding
- USRP/Semtech validation

---

## Paper Structure

1. Introduction — Problem, Gap, Contributions (3 items)
2. Background & System Model — D2S IoT, LEO Doppler, LR-FHSS
3. PGRL Orbital Predictor — SGP4 anchor, Bayesian residual, loss functions
4. LR-FHSS Uplink Controller — Guard-band, TX timing, Doppler pre-comp
5. Evaluation — Setup, trace-driven results (Table 1), ablation (Table 2)
6. Guard-Band Policy Analysis — Energy model, policy comparison (Table 3)
7. Validation Scope and Limitations — Explicit scope boundaries
8. Hardware Validation Plan — Semtech TX path, SDR IQ testbed (planned)
9. Conclusion

Figures: Architecture, Uncertainty ECE, Guard Energy, Grid Orthogonality, SDR Pipeline
Tables: Main results, Ablation, Guard-Band Policy

---

## After Hardware Arrives

```bash
# Test USRP connection
uhd_find_devices

# Run SDR HWIL script
uv run python hardware/usrp_scripts/dual_mode_trx.py --hw

# Regenerate tables + figures
uv run python experiments/summary_table.py
uv run python experiments/generate_figures.py
```

Then update paper:
- Replace Fig. 5
- Update Table 1 validation labels: `Hardware (planned)` → `Hardware (validated)`
- Update abstract final sentence
- Add hardware results subsection