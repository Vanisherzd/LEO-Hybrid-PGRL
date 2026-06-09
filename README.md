# PGRL-LRFHSS-D2S: Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT

This repository contains a research prototype for **PGRL-assisted, uncertainty-aware LR-FHSS uplink control** in direct-to-satellite (D2S) IoT. The submission-scope contribution is narrow: convert SGP4-anchored prediction uncertainty into terminal-side LR-FHSS guard, timing, and Doppler pre-compensation decisions.

> Recommended GitHub repository name: `PGRL-LRFHSS-D2S` or `PGRL-LRFHSS-Uplink-Control`.
>
> Current GitHub repository settings still show the old repository name until renamed manually in GitHub Settings. The code/package metadata has been retargeted to `pgrl-lrfhss-d2s`.

---

## Submission Scope

**Included in the workshop paper:**

- SGP4/SDP4-anchored PGRL residual prediction
- Calibrated heteroscedastic uncertainty head trained with Gaussian NLL
- Risk-aware LR-FHSS guard adaptation
- TX timing and Doppler pre-compensation control logic
- LR-FHSS-inspired frequency-grid and RF-quality proxy evaluation
- Conducted LR1121 to USRP B210 IQ-level signal detection
- Offline IQ-structure analysis of sparse-hop-like RF energy

**Not claimed:**

- No LR-FHSS packet decoding
- No packet-error-rate (PER) result
- No standards-compliant LR-FHSS gateway
- No full MAC / TDMA protocol contribution
- No online GRPO/PPO learning claim
- No ISAC/self-healing claim
- No semantic-communication claim
- No production-ready satellite modem

The hardware evidence is **IQ-level RF signal detection only**. It supports the existence of conducted sparse-hop RF energy under the lab setup; it does not prove decoded packet delivery.

---

## Repository Structure

```text
PGRL-LRFHSS-D2S/
├── controller/                 # submission-scope uplink-control logic
│   ├── pgrl_output_schema.py
│   ├── guard_band_policy.py
│   ├── doppler_precomp.py
│   ├── tx_timing_policy.py
│   └── energy_model.py
├── physics_ml/                 # PGRL predictor and uncertainty-related model code
│   ├── pinn_core.py
│   ├── orbital_physics.py
│   ├── losses.py
│   └── grpo_agent.py           # legacy/thesis extension, not paper scope
├── semtech_validation/          # LR1121/LR11xx TX configuration and bring-up helpers
├── sdr_hwil/                    # IQ-level SDR measurement/proxy scripts
├── experiments/                 # reproducible evaluation folders
├── hardware/                    # hardware helpers and IQ/packet-validation tooling
├── paper/                       # IEEE-style manuscript source
│   ├── icc_main.tex
│   ├── figures/
│   ├── tables/
│   └── refs.bib
├── docs/                        # scope, claim-safety, and hardware notes
├── scripts/                     # evaluation and plotting utilities
├── tests/                       # regression tests, including legacy modules
└── README.md
```

---

## Submission-Scope Status

| Component | Status | Validation Type | Submission Claim |
|---|---:|---|---|
| PGRL deterministic predictor | Complete | offline / trace-driven | Yes |
| Uncertainty calibration | Complete | offline evaluation | Yes |
| Risk-aware guard policy | Complete | control proxy | Yes |
| Doppler pre-compensation | Complete | simulation / proxy | Yes, as proxy |
| LR-FHSS grid analysis | Complete | LR-FHSS-inspired proxy | Yes, as proxy |
| LR1121 + USRP B210 capture | Complete | hardware-signal-detected | Yes, IQ-level only |
| IQ sparse-hop structure analysis | Complete | offline IQ analysis | Yes, non-decoding only |
| LR-FHSS decoder / PER | Not available | out of scope | No |
| TDMA / MAC / GRPO online learning | Legacy extension | simulation only | No |
| ISAC self-healing | Legacy extension | prototype only | No |

---

## Key Results Used in the Paper

| Result | Reported Meaning | Claim Boundary |
|---|---|---|
| Position RMSE remains 5.35 m | Frozen mean head unchanged by uncertainty training | Not a new point-accuracy improvement |
| Cov68/Cov95 = 0.713 / 0.947 at T = 1.0 | Learned uncertainty is near-calibrated | Position-domain uncertainty only |
| Outage proxy 5.0% → 1.7% | Risk-aware guard reduces a control proxy | Not measured PER |
| Guard overhead +13.8% | Energy/control tradeoff at best reward point | Proxy metric |
| TX-ON/OFF margin 9.82 dB | Conducted IQ signal detection | Not decoding / not PER |
| 101 candidate bursts / 19 occupied bins / score ≈ 0.845 | Offline sparse-hop-like IQ structure | Not packet validation |

---

## Quick Start

```bash
uv sync

# PGRL prediction evaluation
uv run python physics_ml/evaluate_prediction.py

# Guard-band energy/control experiment
uv run bash experiments/exp2_guard_band_energy/run.sh

# LR-FHSS grid proxy
uv run bash experiments/exp3_lrfhss_grid_proxy/run.sh

# Generate Semtech LR-FHSS TX config from PGRL output
uv run python semtech_validation/tx_config_from_pgrl.py --output-json /tmp/pgrl_tx_config.json

# SDR HWIL synthetic IQ pipeline
uv run bash experiments/exp5_sdr_doppler_precomp/run.sh dry-run
```

---

## Legacy / Thesis-Extension Modules

The following code is useful for thesis or journal extension, but should not be presented as part of the workshop paper contribution:

- `protocols/mac_tdma.py`
- `scripts/train_online.py`
- `scripts/benchmark_tdma.py`
- `models/grpo_agent.py`
- `physics_ml/grpo_agent.py`
- `configs/online_grpo.yaml`
- `src/sdr/isac_rf_orbit_healing.py`
- `hardware/packet_validation/`
- `docs/run_real_hardware_per_experiment.md`
- `docs/real_hardware_per_experiment_plan.md`
- `docs/packet_delivery_validation_plan.md`
- `docs/THE_D2S_REVOLUTION.md`

These files may remain in the repository for reproducibility, but the clean submission branch should treat them as **non-paper artifacts** unless a later journal version adds the missing receiver/PER or online-learning evidence.

---

## Citation

```bibtex
@misc{lai2026pgrllrfhssd2s,
  title   = {{PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT}},
  author  = {Lai, Zhen-Dong},
  year    = {2026},
  howpublished = {GitHub repository},
  note    = {Research prototype; trace-driven/proxy evaluation with IQ-level hardware signal detection only}
}
```
