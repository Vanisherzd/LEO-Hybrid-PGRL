> **OBSOLETE legacy note — archived; not current claims.**
> For the current scope and claims see `paper/icc_main.tex` and `README.md`.

# Legacy Submission Scope — PGRL-Assisted LR-FHSS Uplink Control

## Research Question
How can a Direct-to-Satellite IoT terminal use PGRL-predicted timing, Doppler, and uncertainty estimates to improve LR-FHSS uplink control decisions?

## Included in the submission
- SGP4-anchored PGRL residual predictor with Gaussian NLL loss and uncertainty calibration
- Uncertainty-aware adaptive guard-band scheduling
- Doppler pre-compensation using PGRL mean prediction
- LR-FHSS-inspired frequency-grid proxy evaluation
- Semtech LR1121 / LR11xx LR-FHSS TX validation path (standards-aligned)
- SDR-based IQ-level D2S-like measurement (CFO, EVM proxy, waterfall)

## Excluded from the submission
These modules are reserved for journal or thesis extension:
- **PPO / GRPO MAC optimization** — valuable but adds scope; not required for core LR-FHSS uplink argument
- **ISAC closed-loop correction** — residual CFO feedback for online PGRL calibration
- **Semantic communication** — separate research thread
- **Multi-carrier water-filling** — beyond LR-FHSS single-carrier scope
- **Full LR-FHSS receiver** — SDR serves as proxy; real PER requires standards decoder
- **Multi-node scheduling** — network topology beyond single-terminal focus

## Motivation for narrow scope
The target venue's initial submission limit is **6 printed pages** (10-point font, English). The paper focuses on one defensible contribution: prediction-driven uplink-control for D2S IoT LR-FHSS. Peripheral contributions that could expand the scope are explicitly deferred to avoid reviewer scatter.
