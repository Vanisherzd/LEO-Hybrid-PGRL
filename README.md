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

> This is a **research prototype** for simulation/offline evaluation and RF-quality proxy measurement. The SDR components serve as an IQ-level measurement platform, not a full standard-compliant LR-FHSS gateway.

---

## Repository Structure

```
LEO-Hybrid-PGRL/              # Project root
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
│   ├── exp1_pgrl_prediction/     # Prediction accuracy + ablation
│   ├── exp2_guard_band_energy/   # Guard overhead / energy tradeoff
│   ├── exp3_lrfhss_grid_proxy/   # LR-FHSS grid proxy evaluation
│   ├── exp4_semtech_lrfhss_tx/   # Semtech hardware bring-up
│   └── exp5_sdr_doppler_precomp/ # SDR HWIL Doppler pre-comp
├── paper/                    # ICC submission
│   ├── icc_main.tex         # IEEEtran skeleton
│   ├── tables/main_results.tex   # Auto-generated tables
│   └── refs.bib
├── docs/
│   ├── globecom_scope.md         # ICC paper scope
│   ├── hardware_claim_checklist.md
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
| PGRL prediction core | ✅ Complete | Offline trained-model |
| Guard-band policy | ✅ Complete | Simulation |
| Doppler pre-compensation | ✅ Complete | Simulation |
| LR-FHSS grid proxy | ✅ Complete | Simulation |
| Semtech LR1121 TX bring-up | ✅ Complete | hardware-bringup |
| USRP B210 IQ capture | ✅ Complete | hardware-signal-detected |
| LR-FHSS decoding / PER | ⛔ Not claimed | Out of scope |
| ICC paper skeleton | ✅ Complete | Write-up |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run PGRL prediction evaluation (offline)
uv run python physics_ml/evaluate_prediction.py

# Run guard-band energy experiment (simulation)
uv run bash experiments/exp2_guard_band_energy/run.sh

# Run LR-FHSS grid proxy (proxy simulation)
uv run bash experiments/exp3_lrfhss_grid_proxy/run.sh

# Generate Semtech LR-FHSS TX config from PGRL output (dry-run, no hardware required)
python semtech_validation/tx_config_from_pgrl.py --output-json /tmp/pgrl_tx_config.json

# SDR HWIL synthetic IQ pipeline (dry-run, no hardware required)
uv run bash experiments/exp5_sdr_doppler_precomp/run.sh dry-run

# Build the active paper (ICC)
tectonic paper/icc_main.tex
```

---

## Data registry

`data/` is a lightweight registry/catalog (schemas, manifests, and tiny examples
only) describing how TLE and SGP4-derived data are organized for the PGRL pipeline.

- **TLE records** are the orbital source; **SGP4/SDP4-derived states** are generated
  features; PGRL consumes normalized SGP4-derived state/elements plus a query time.
- Committed: `data/schemas/`, `data/manifests/`, `data/examples/`, `data/registry.yaml`.
- Raw datasets are **local-only** under `data_raw/` or `local_archive/`; processed
  features go to `data_processed/` (both git-ignored).
- **No raw IQ or large validation outputs are committed.** Hardware IQ stays under
  `hardware/captures/` or `local_archive/raw_iq/`.

See [`data/README.md`](data/README.md) for the full catalog.

---

## Key Results

All results are simulation, offline trained-model, or conducted IQ-level signal detection. No measured PER, packet delivery, or receiver decoding is claimed.

| Stage | Result | Scope |
|-------|--------|-------|
| Deterministic predictor | Position RMSE **5.35 m**, unchanged after uncertainty-head training (mean head frozen) | offline trained-model |
| Stage 3F calibration | **T=1.0**, Cov68/Cov95 = **0.713 / 0.947** (no post-hoc rescaling needed) | offline trained-model |
| Stage 4 risk-aware guard | Outage proxy **5.0% → 1.7%** at **+13.8%** guard overhead; reward **0.9500 → 0.9690** (best α=0.25) | control proxy (not PER) |
| Stage 5 conducted IQ capture | LR1121 → USRP B210, TX-ON/OFF margin **9.82 dB**, `signal_detected` | IQ-level RF detection only |
| Stage 5C IQ-structure analysis | 101 candidate bursts, 19 occupied freq bins, ~12% time occupancy, structure score ~0.845 | IQ-structure only (no decode) |
| Stage 6 PER harness | End-to-end packet/PER harness prepared; **no measured PER** without a decoded receiver log | harness only; PER unavailable |

> Outage, success, and reward are control proxies, **not** measured packet-error rates; the deterministic RMSE is **not** improved by the uncertainty stages (mean head frozen). The conducted LR1121→USRP capture is **IQ-level RF signal detection only** and does not imply LR-FHSS packet decoding, CRC validation, or PER. Raw IQ is held locally and not committed.

---

## What This Is Not

This project does **not** claim:
- Commercial-ready autonomous operation
- Perfect LEO tracking with infinite blind-flight horizon
- Full standard-compliant LR-FHSS gateway implementation
- Catastrophic link recovery or world-first autonomy

The SDR and LR-FHSS grid components are **RF-quality proxies** for a simulation/proxy evaluation of physical-layer signal quality. Hardware results are **IQ-level signal detection only**; real-world PER requires a standards-compliant LR-FHSS decoder on the satellite side.

> **Scope note:** GRPO/PPO online refinement and ISAC / self-healing directions are **thesis extensions**, not part of the ICC paper scope. The ICC submission covers PGRL prediction, uncertainty-aware uplink control, and preliminary IQ-level hardware signal detection.

---

## Citation

```bibtex
@misc{lai2026leopgrllrfhss,
  title   = {{PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT}},
  author  = {Lai, Zhen-Dong},
  year    = {2026},
  howpublished = {\url{https://github.com/Vanisherzd/LEO-Hybrid-PGRL}},
  note    = {Research prototype; simulation/offline results with preliminary IQ-level hardware signal detection}
}
```