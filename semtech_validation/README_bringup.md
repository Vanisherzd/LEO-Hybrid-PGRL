# Semtech LR-FHSS TX Validation

## Goal
Validate that the PGRL controller can drive a standards-aligned LR-FHSS transmitter using Semtech LR1121 / LR11xx / SX126x hardware.

> **Note:** The SDR is used as an IQ-level measurement receiver — it is NOT a full standard-compliant LR-FHSS gateway. This validation path establishes hardware-ground-truth for RF quality metrics.

## Hardware
- Semtech LR1121, LR1110, or SX126x evaluation board
- Host MCU (STM32 or Linux SBC) running Semtech drivers
- Coaxial cable + 30 dB attenuator (for direct RF loopback)
- USRP B210 or equivalent SDR for IQ capture

## Software
- [Semtech SWDM001](https://github.com/Lora-net/SWDM001) — official LR-FHSS TX demonstration firmware
- Semtech LR11xx driver library
- Python 3.10+ with `numpy`, `scipy`, `matplotlib`

## Validation Steps

### 1 — Baseline LR-FHSS TX (no compensation)
```bash
# Build and flash SWDM001 onto LR1121 eval board
# Then capture with SDR:
python sdr_hwil/capture_iq.py --freq 915e6 --duration 10 --output baseline.cfile
```
Verify: LR-FHSS hopping pattern visible in waterfall, hopping bins identifiable.

### 2 — Export PGRL TX configuration
```bash
python semtech_validation/tx_config_from_pgrl.py
# Outputs: lr1121_tx_config_example.json
```
The JSON encodes frequency, guard time, TX time, spreading factor, and metadata.

### 3 — PGRL-compensated TX
Apply `lr1121_tx_config_example.json` to SWDM001, re-run capture:
```bash
python sdr_hwil/capture_iq.py --freq 914.99875e6 --duration 10 --output compensated.cfile
```

### 4 — Compare RF quality
```bash
python sdr_hwil/estimate_cfo.py baseline.cfile compensated.cfile
python sdr_hwil/plot_waterfall.py baseline.cfile compensated.cfile
```

### 5 — Expected hardware outcomes (target)
| Metric | Baseline TX | PGRL-Compensated TX |
|--------|-------------|----------------------|
| Residual CFO | > 200 Hz | < 50 Hz |
| Waterfall alignment | Degraded bins | Cleaner spectral bins |
| EVM proxy | > 15 % | < 5 % |

## File Manifest
```
semtech_validation/
├── README_bringup.md          ← this file
├── tx_config_from_pgrl.py     ← generates LR1121 JSON config from PGRLOutput
├── lr1121_tx_config_template.json  ← annotated template
└── run_lrfhss_tx.sh           ← baseline / compensated / compare runner
```

## References
- [SWDM001 — Semtech LR-FHSS TX demo](https://github.com/Lora-net/SWDM001)
- [LR1121 Datasheet](https://files.waveshare.com/wiki/Core1121/LR1121_H2_DS_v2_0.pdf)
- [AN1200.64 — LR-FHSS System Performance](https://www.mouser.com/pdfDocs/AN1200-64_LR-FHSS_system_performance_V1_2.pdf)