# LR1121 LR-FHSS SDR signal-detected ON/OFF control

**Experiment:** LR1121 LR-FHSS SDR signal-detected ON/OFF control
**Date/time:** 2026-06-04 00:03:58 local
**Curated from:** `hardware/captures/auto_sweep_20260604_000358`

## Hardware
- Semtech LR1121 + NUCLEO-L476RG
- USRP B210
- SWDM001 firmware, target `lr1121_xtal`

## Firmware settings
- RF = 868000000 Hz
- TX power = 10 dBm
- packet interval = 1000 ms

## SDR settings
- antenna = TX/RX
- sample rate = 1000000 sps (1 Msps)
- gain = 45 dB
- duration = 10 s

## Validation results
- validation_status = signal_detected
- signal_detected = True
- UART packet count = 7
- ON/OFF delta = 8.88 dB
- TX-ON stronger than TX-OFF = True
- LR-FHSS candidate score = 0.7534
- hop-like segments = 173
- occupied frequency bins = 91
- max-hold peaks = 16

## Claim boundary
- This is **IQ-level SDR signal detection** only.
- This is **NOT** standard-compliant LR-FHSS decoding.
- This is **NOT** a PER (packet-error-rate) measurement.
- This is **NOT** a full LR-FHSS gateway receiver.

Raw `.fc32` IQ is intentionally NOT included here (git-ignored, kept under
`hardware/captures/`). Only small evidence artifacts are curated.
