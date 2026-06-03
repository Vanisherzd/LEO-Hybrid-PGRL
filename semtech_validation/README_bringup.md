# Semtech LR-FHSS TX Validation — Bring-Up

## Goal
Validate that the PGRL controller can drive a standards-aligned LR-FHSS
transmitter using Semtech LR1121 / LR11xx / SX126x hardware, measured at the
IQ level by a USRP B210 SDR.

> **Note:** The SDR is used as an IQ-level measurement receiver — it is **NOT**
> a full standard-compliant LR-FHSS gateway. This path establishes
> hardware-ground-truth for RF-quality metrics only. Do **not** claim LR-FHSS
> PER, standard decoding, or successful RF capture unless the signal detector
> passes (`validation_status == "signal_detected"`).

For the full procedure, schema, and troubleshooting checklist see
[`docs/lr1121_sdr_validation_workflow.md`](../docs/lr1121_sdr_validation_workflow.md).
This file is the short bring-up companion.

## Hardware
- Semtech LR1121, LR1110, or SX126x evaluation board (default rig:
  NUCLEO-L476RG + LR1121)
- Host MCU running Semtech drivers
- Coaxial cable + **30 dB attenuator** for any conducted (cabled) RF loopback
- USRP B210 (or equivalent SDR) for IQ capture

## Software
- [Semtech SWDM001](https://github.com/Lora-net/SWDM001) — official LR-FHSS TX
  demonstration firmware
- Semtech LR11xx driver library
- Python 3.10+ with `numpy`, `scipy`, `matplotlib` (run via `uv`)

## Firmware quick facts (SWDM001)
| Item | Value |
|------|-------|
| Build target | `lr1121_xtal` |
| Keil project | `project/keil_polling_STM32L476/STM32L476.uvprojx` |
| Generated binary | `project/keil_polling_STM32L476/STM32L476_lr11xx.bin` |
| Flash target | `NOD_L476RG` |
| UART baud | 115200 |
| Expected UART output | `LR11XX-LR-FHSS Ping Init` then `Packet sent!` |
| **Default RF frequency** | **868 MHz** (`RF_FREQUENCY ( 868000000 )` in `src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`) |

> **Important:** The SWDM001 LR11xx demo transmits at **868 MHz by default**.
> 915 MHz is **optional** and only valid **after** you edit `RF_FREQUENCY` in
> the firmware (and/or align the PGRL config) and reflash. Always set the SDR
> capture centre frequency to whatever the firmware actually transmits.

## Validation Steps

### 1 — Baseline LR-FHSS TX (no compensation)
Build and flash SWDM001 onto the LR1121 eval board, confirm
`LR11XX-LR-FHSS Ping Init` / `Packet sent!` on UART, then capture IQ with the
SDR at the firmware's RF frequency (**868 MHz** by default):

```bash
# Capture .fc32 IQ at 868 MHz (default firmware frequency), then analyze:
uv run python hardware/usrp_scripts/analyze_capture.py baseline.fc32 \
    --sample-rate 1000000 \
    --output-json baseline.json \
    --plot baseline.png \
    --signal-threshold-db 8 \
    --maxhold-plot baseline_maxhold.png
```

Verify: the analyzer reports `validation_status == "signal_detected"` and the
waterfall / max-hold shows the LR-FHSS hopping. If you only see a DC line, the
SDR did **not** receive a usable RF signal — see the troubleshooting checklist
in the workflow doc.

### 2 — Export PGRL TX configuration
```bash
python semtech_validation/tx_config_from_pgrl.py
# Outputs: lr1121_tx_config_example.json
```
The JSON encodes frequency, guard time, TX time, spreading factor, and metadata.

### 3 — PGRL-compensated TX
Apply `lr1121_tx_config_example.json` to SWDM001, reflash, re-capture, and
re-run `analyze_capture.py` (use the same centre frequency the firmware is now
transmitting on).

### 4 — Compare RF quality
Compare the canonical analyzer JSON from the baseline vs. compensated captures
(see schema below).

### 5 — Expected hardware outcomes (target, not yet achieved)
| Metric | Baseline TX | PGRL-Compensated TX |
|--------|-------------|----------------------|
| Residual CFO (`rx_cfo_hz`) | > 200 Hz | < 50 Hz |
| Waterfall alignment | Degraded bins | Cleaner spectral bins |
| EVM proxy (`rx_evm_percent`) | > 15 % | < 5 % |

> These are target outcomes. As of this branch **no LR-FHSS waveform has been
> observed yet** — captures at 868 MHz still look like noise / DC-only.

## Canonical analyzer JSON schema
`validation_status` (`"signal_detected"` | `"noise_floor_only"` |
`"weak_signal_candidate"`), `signal_detected` (bool), `peak_to_median_db`,
`burst_energy_excess_db`, `peak_frequency_offset_hz`, `rx_cfo_hz`,
`rx_cfo_std_hz`, `rx_evm_percent`, `rx_snr_db`. The analyzer excludes the
DC/LO spike (±2% of sample rate), so a DC-only capture is **not** flagged as a
signal.

## Status ladder
Advance a status only when its evidence bar is met — never skip a rung.

| Status | Evidence required |
|--------|-------------------|
| `firmware_running` | SWDM001 boots and prints `LR11XX-LR-FHSS Ping Init` / `Packet sent!` on UART. Proves firmware, **not** RF. |
| `iq_capture_done` | A valid `.fc32` IQ file recorded at the correct centre frequency / sample rate. Proves capture, not signal content. |
| `signal_detected` | Analyzer reports `validation_status == "signal_detected"` (DC excluded) **and** a waveform figure shows the energy. |
| `hardware_validated` | `signal_detected` plus CFO / EVM / SNR metrics consistent with LR-FHSS, documented with figures. Highest, most defensible claim. |

> **Warning:** Do **not** change any status from `pending` to `validated` until
> the signal detector passes (`validation_status == "signal_detected"`) **AND**
> a waveform figure (waterfall / max-hold) has been generated.

## File Manifest
```
semtech_validation/
├── README_bringup.md               ← this file
├── tx_config_from_pgrl.py          ← generates LR1121 JSON config from PGRLOutput
├── lr1121_tx_config_template.json  ← annotated template
└── run_lrfhss_tx.sh                ← baseline / compensated / compare runner
```

## References
- [SWDM001 — Semtech LR-FHSS TX demo](https://github.com/Lora-net/SWDM001)
- [LR1121 Datasheet](https://files.waveshare.com/wiki/Core1121/LR1121_H2_DS_v2_0.pdf)
- [AN1200.64 — LR-FHSS System Performance](https://www.mouser.com/pdfDocs/AN1200-64_LR-FHSS_system_performance_V1_2.pdf)
- [`docs/lr1121_sdr_validation_workflow.md`](../docs/lr1121_sdr_validation_workflow.md)
