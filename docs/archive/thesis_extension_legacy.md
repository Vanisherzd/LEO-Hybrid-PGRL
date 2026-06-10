> **OBSOLETE legacy note — archived; not current claims.**
> For the current scope and claims see `paper/icc_main.tex` and `README.md`.

# Thesis Extension Roadmap (legacy)

This document lists modules developed in the broader LEO-PINN project that are **outside the conference submission scope** but intended for thesis or journal extension.

## Reserved for Thesis / Journal
| Module | Status | Notes |
|--------|--------|-------|
| PPO / GRPO MAC scheduler | Implemented (scripts/, protocols/) | MAC optimization is a full contribution; keep for thesis |
| ISAC RF-driven self-healing | Implemented (src/sdr/isac_rf_orbit_healing.py) | Powerful narrative but adds complexity; journal paper |
| Residual CFO feedback online calibration | Concept | Use Costas Loop CFO as gradient signal for PGRL fine-tuning |
| Multi-node collision scheduling | Concept | Multiple D2S terminals; network-level optimization |
| Full LR-FHSS gateway | Planned | Standards-compliant decoder from Semtech; separate project |
| Long-term autonomous operation (>7 days) | Concept | Requires hardware + field deployment |

## Relationship to the conference paper
The conference paper establishes the **PGRL prediction core** and the **LR-FHSS uplink-control** framework. The thesis extends this to:
1. MAC-layer scheduling optimization (PPO/GRPO)
2. Cyber-physical closed-loop ISAC (residual CFO → model update)
3. Network-level multi-terminal access

These are **forward references** in the conference paper's Conclusion / Future Work section.