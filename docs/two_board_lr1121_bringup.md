# Two-Board LR1121/Nucleo Decoded-RX Bring-Up

## Python environment
This repo uses **uv**. Run project scripts via uv; do not install packages into
the global system Python.
- Run a script: `uv run python hardware/packet_validation/<script>.py ...`
- Persistent dependency: `uv add pyserial`
- One-off serial capture (no persistent install): `uv run --with pyserial python hardware/packet_validation/capture_uart_log.py ...`

## 1. Goal
Move from IQ-level signal detection to **decoded-RX packet-delivery evidence**:
obtain a receiver-side log with `seq` / `payload` / `crc`, enabling PDR/PER.

## 2. Physical topology
```
Mac USB  ->  TX Nucleo-L476RG + LR1121   (Board 1, TX)
Mac USB  ->  RX Nucleo-L476RG + LR1121   (Board 2, RX)
TX RF  ->  coax + attenuator / shielded path  ->  RX RF
(optional) USRP B210  ->  IQ monitor only
```

## 3. Safety (mandatory)
- **Conducted or shielded only.** No over-the-air (OTA).
- **Do not increase TX power automatically** — use the configured value.
- Confirm regional RF settings **manually** on each device.

## 4. Serial port discovery
```
uv run python hardware/packet_validation/list_serial_ports.py
ls /dev/cu.usbmodem*
```

## 5. Board role labeling
- Plug **TX only** -> record its `/dev/cu.usbmodem*` port.
- Plug **RX only** -> record its port.
- Plug **both** -> confirm two distinct `/dev/cu.*` ports.

## 6. Firmware requirement
- TX firmware emits/accepts deterministic payload seqs (see `generate_payloads`).
- RX firmware prints one decoded packet line per received frame.
- Minimum RX line: `RX seq=0 payload=<hex> crc=ok`

## 7. Accepted RX log formats
```
A: RX seq=12 payload=abcd1234 crc=ok rssi=-91.5 snr=7.2 cfo=123.4 timestamp=2026-06-08T12:00:01.234Z
B: [RX] seq:12 payload:abcd1234 CRC_OK RSSI=-91.5 SNR=7.2 CFO=123.4
C: packet_received seq=12 len=16 payload_hex=abcd1234 crc_ok=true
JSONL: {"seq":12,"payload_hex":"abcd1234","crc_ok":true,"rssi_dbm":-91.5,"snr_db":7.2,"cfo_hz":123.4,"timestamp":"..."}
```
seq + payload + crc are required; RSSI/SNR/CFO/timestamp optional.

## 8. Capture RX UART
```
uv run python hardware/packet_validation/capture_uart_log.py \
  --port /dev/cu.usbmodem<RX> --baud 115200 \
  --out hardware/rx_logs/real_runA_alpha0_decoded_rx.log --duration 60
```

## 9. Check RX log
```
uv run python hardware/packet_validation/check_rx_log_format.py \
  --rx-log hardware/rx_logs/real_runA_alpha0_decoded_rx.log
```
Proceed only if `usable_for_decoded_rx: True`.

## 10. Prepare local config (do not edit templates)
```
uv run python hardware/packet_validation/prepare_real_run_config.py \
  --template hardware/packet_validation/real_configs/real_runA_alpha0_decoded_rx.yaml \
  --out validation_runs/real_runA_alpha0_config.yaml \
  --rx-log hardware/rx_logs/real_runA_alpha0_decoded_rx.log \
  --operator "Dong Zheng"
```

## 11. Run real_runA / real_runB / real_runC (conducted, --armed)
```
uv run python hardware/packet_validation/run_hardware_experiment.py \
  --config validation_runs/real_runA_alpha0_config.yaml \
  --out validation_runs/real_runA_alpha0 --armed
# repeat for run B (alpha=0.25) and run C (stress: prepare with --attenuator-db)
```

## 12. Finalize
```
uv run python hardware/packet_validation/finalize_hardware_manifest.py \
  --run-dir validation_runs/real_runA_alpha0
```

## 13. What to commit later (copy out of gitignored validation_runs/)
- `manifest_final.yaml`
- `packet_validation_summary.json` / `.csv` / `.md`
- decoded RX log text (if small)
- small metadata (`artifact_index.csv`, `sha256sums.txt`)

## 14. What NOT to commit
- raw IQ `*.fc32`
- the whole `validation_runs/` directory (gitignored)
- build artifacts (`*.pdf`/`*.log`/`*.blg`)

## 15. Claim gating
- No decoded RX log -> **no PER**.
- Decoded RX log with payload/CRC -> **PDR/PER allowed**.
- Repeated controlled runs -> **stronger conducted validation**.
- Gateway interoperability -> only if a gateway decoder produced the log.
