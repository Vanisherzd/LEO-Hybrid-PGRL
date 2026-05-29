# LEO-PGRL-LRFHSS: PGRL-Assisted LR-FHSS Uplink Control for Direct-to-Satellite IoT

---

## Project Overview

This repository implements a **prediction-driven uplink-control framework** for Direct-to-Satellite (D2S) IoT terminals using LR-FHSS modulation. The core is a Physics-Guided Residual Learning (PGRL) predictor that uses SGP4/SDP4 as a physical anchor and learns residual timing and Doppler corrections with **calibrated uncertainty estimates**.

PGRL predictions drive three uplink-control decisions:
1. **Adaptive guard-band scheduling** — k-sigma timing uncertainty widens or tightens guard intervals dynamically
2. **TX timing selection** — link score, Doppler rate, and idle energy costs balanced via utility maximization
3. **Doppler pre-compensation** — carrier frequency adjusted before TX using PGRL-predicted mean Doppler

The current release includes:
- PGRL prediction core (SGP4 anchor + Bayesian residual corrector)
- Uncertainty-aware guard-band policy
- Doppler pre-compensation controller
- LR-FHSS-inspired frequency-grid proxy evaluation
- Semtech LR1121 / LR11xx LR-FHSS TX validation path (hardware-in-the-loop)
- SDR-based IQ-level D2S-like testbed (CFO, EVM proxy, waterfall)

> This is a **research prototype** for trace-driven evaluation and RF-quality validation. The SDR components serve as an IQ-level measurement platform, not a full standard-compliant LR-FHSS gateway.

---

## Repository Structure

```
leo-pinn/                    # Project root
├── controller/               # PGRL → uplink-control interface
│   ├── pgrl_output_schema.py    # PGRLOutput dataclass
│   ├── guard_band_policy.py     # adaptive_guard_time()
│   ├── doppler_precomp.py       # compensated_tx_frequency()
│   ├── tx_timing_policy.py      # select_tx_time()
│   └── energy_model.py          # receiver / TX energy accounting
├── physics_ml/              # PGRL core
│   ├── pinn_core.py              # Siren/PINN architecture
│   ├── orbital_physics.py        # SGP4 anchor, Keplerian propagation
│   ├── losses.py                 # Gaussian NLL, physics residuals
│   └── grpo_agent.py             # GRPO online training (extension)
├── semtech_validation/       # LR1121/LR11xx hardware validation
│   ├── README_bringup.md        # Bring-up guide
│   ├── tx_config_from_pgrl.py   # PGRL → TX config JSON
│   ├── lr1121_tx_config_template.json
│   └── run_lrfhss_tx.sh
├── sdr_hwil/                # SDR / HWIL IQ-level validation
│   ├── README.md
│   ├── estimate_cfo.py           # Phase-derivative CFO estimation
│   ├── evm_proxy.py              # QPSK EVM as RF-quality proxy
│   ├── packet_detection_proxy.py # Energy-based detection
│   ├── capture_iq.py             # IQ capture script
│   └── plot_waterfall.py         # Waterfall visualization
├── experiments/             # Reproducible experiment folders
│   ├── exp1_pgrl_prediction/    # Prediction accuracy
│   ├── exp2_guard_band_energy/   # Guard overhead / energy tradeoff
│   ├── exp3_lrfhss_grid_proxy/   # LR-FHSS grid proxy evaluation
│   ├── exp4_semtech_lrfhss_tx/   # Semtech hardware bring-up
│   └── exp5_sdr_doppler_precomp/ # SDR HWIL Doppler pre-comp
├── paper/                    # GLOBECOM submission
│   ├── globecom_main.tex         # IEEEtran skeleton
│   └── refs.bib
├── docs/
│   ├── globecom_scope.md         # GLOBECOM paper scope
│   ├── thesis_extension.md       # Deferred modules roadmap
│   └── MAC_DEPLOYMENT_GUIDE.md   # MacBook + USRP B210 setup
├── hardware/usrp_scripts/    # USRP SDR scripts (legacy/HWIL)
├── data/tle/                 # TLE datasets
├── plots/paper_final/        # Figure outputs
└── README.md                 # this file
```

---

## Status Table

| Component | Status | Type |
|-----------|--------|------|
| PGRL prediction core | ✅ Complete | Trace-driven |
| Guard-band policy | ✅ Complete | Simulation |
| Doppler pre-compensation | ✅ Complete | Simulation |
| LR-FHSS grid proxy | ✅ Complete | Simulation |
| Semtech LR-FHSS TX bring-up | 🟡 Planned | Hardware |
| SDR HWIL validation | 🟡 Planned | Hardware |
| GLOBECOM paper skeleton | ✅ Complete | Write-up |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run PGRL prediction evaluation
uv run python physics_ml/evaluate_prediction.py

# Run guard-band energy experiment
uv run python experiments/exp2_guard_band_energy/run.py

# Run LR-FHSS grid proxy
uv run python experiments/exp3_lrfhss_grid_proxy/run.py

# Optional: uncertainty calibration and ablation (thesis extension, not GLOBECOM scope)
# uv run python src/sdr/isac_rf_orbit_healing.py --output Fig8_ISAC_Self_Healing.png
```

---

## Key Results

All trace-driven and simulation results. Hardware validation (Semtech + SDR) is the next stage.

| Component           | Metric                    | SGP4-only  | PGRL (this work) | Validation Type      |
|---------------------|--------------------------:|----------:|----------------:|----------------------|
| PGRL predictor      | Pass timing RMSE          |    4.2 s  |     **16 ms**   | Trace-driven         |
| PGRL predictor      | Residual Doppler          |  > 5 kHz  |    **< 300 Hz** | Trace-driven         |
| Guard-band policy   | Guard overhead            |      64 % |        **< 5 %** | Simulation           |
| RF quality proxy    | QPSK EVM proxy (40 dB SNR) |   208 %  |      **0.95 %** | Proxy simulation     |
| LR-FHSS grid proxy  | Grid orthogonality        |    0.587  |      **0.979**  | LR-FHSS-inspired proxy |
| Semtech TX          | Waterfall / CFO           |         — |     **Planned** | Hardware (pending)    |
| SDR HWIL            | Residual CFO / EVM        |         — |     **Planned** | Hardware (pending)    |

> All completed values are from trace-driven simulation or proxy evaluation. The Semtech LR-FHSS TX bring-up and SDR HWIL validation stages are defined in `experiments/exp4/` and `experiments/exp5/`, pending hardware acquisition.

---

## What This Is Not

This project does **not** claim:
- Commercial-ready autonomous operation
- Perfect LEO tracking with infinite blind-flight horizon
- Full standard-compliant LR-FHSS gateway implementation
- Catastrophic link recovery or world-first autonomy

The SDR and LR-FHSS grid components are **RF-quality proxies** for a trace-driven evaluation of physical-layer signal quality. Real-world PER requires a standards-compliant LR-FHSS decoder on the satellite side.

---

## Citation

```bibtex
@article{vanisherz2025leopgrllrfhss,
  title   = {PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT},
  author  = {Vanisherz, Z. D.},
  year    = {2025},
  note    = {GitHub: https://github.com/Vanisherzd/LEO-PGRL-LRFHSS}
}
```