# Experiment 4 — Semtech LR-FHSS TX Bring-Up

**Purpose:** Hardware validation using Semtech LR1121 / LR11xx / SX126x running SWDM001 LR-FHSS firmware.

## Hardware Setup
- Semtech LR1121 eval board + antenna
- Coax cable + 30 dB attenuator
- USRP B210 for IQ capture

## Validation Sequence
1. Run baseline SWDM001 TX → capture IQ
2. Apply PGRL TX config → capture compensated IQ
3. Compare: residual CFO, EVM proxy, waterfall alignment

## Metrics
- Residual CFO [Hz]
- EVM proxy [%]
- Occupied bandwidth
- Spectral centroid offset
- Packet detection proxy (energy-based)

## Minimum Success Criteria
- SDR sees LR-FHSS hopping pattern in waterfall
- PGRL-compensated TX shows measurably lower residual CFO than baseline