# LR1121 LR-FHSS SDR ON/OFF repeatability

Three successful TX ON/OFF trials, all IQ-level `signal_detected`.

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
- sample rate = 1 Msps
- gain = 45 dB
- duration = 10 s

## Results
- Run 1 (auto_sweep_20260604_000358): 8.88 dB ON/OFF delta
- Run 2 (auto_sweep_20260604_011203): 11.87 dB ON/OFF delta
- Run 3 (auto_sweep_20260604_011519): 9.82 dB ON/OFF delta
- All three runs: `signal_detected = true`, `tx_on_stronger_than_off = true`

See `repeatability_summary.csv` / `.json` for the full per-run metrics, and
`run{1,2,3}_*/` for the curated per-run evidence.

## Claim boundary
- This is **IQ-level SDR RF signal detection**.
- It validates LR1121 **RF signal presence** under TX ON/OFF control.
- It is **NOT** LR-FHSS packet decoding.
- It is **NOT** a PER (packet-error-rate) measurement.
- It is **NOT** a full LR-FHSS gateway receiver.

Raw `.fc32` IQ is intentionally excluded (git-ignored under `hardware/captures/`).
