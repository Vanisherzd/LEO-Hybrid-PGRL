# LR1121 RX Firmware Gap (Decoded-RX / PER Blocker)

Date: 2026-06-09. Status: **decoded_rx not achievable with current assets.**

## Feasibility class: C
No LR1121 firmware source exists in this repository (no `.c`/`.uvprojx`/`.bin`,
no PlatformIO/Cube/Arduino project). The repo holds only a config/dry-run
wrapper and a config generator; all firmware is external (Semtech SWDM001).

## Current TX board evidence (working)
- UART (`/dev/cu.usbmodem1303`): `Packet to send: ...`, `RF=868000000 Hz, PWR=10 dBm`, `Packet sent!`
- Source: **external** SWDM001 demo `src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`
  (github.com/Lora-net/SWDM001), built in **Keil** (`project/keil_polling_STM32L476/STM32L476.uvprojx`,
  binary `STM32L476_lr11xx.bin`), flashed to **NUCLEO-L476RG / NOD_L476RG** via ST-LINK.
- Baud 115200. Default RF 868 MHz. This demo is **LR-FHSS transmit (ping) only.**
- Repo references: `semtech_validation/run_lrfhss_tx.sh`, `semtech_validation/README_bringup.md`,
  `experiments/exp4_semtech_lrfhss_tx/`. No firmware sources are vendored here.

## Board 1403 evidence (wrong firmware)
- UART after reset: `7Semi BME280 I2C Example`, `BME280 init failed!`
- This is an **unrelated 3rd-party 7Semi BME280 I2C example**, not LR1121 RX
  firmware. Its source is **not** in this repo. Board 1403 must be reflashed.

## Why PER is unavailable
- No receiver-side decoded packet log (`RX seq=.. payload=.. crc=..`) exists.
- The TX UART "Packet sent!" is transmitter-side only — not RX decode.
- The USRP path is explicitly IQ-level signal detection only, **not** a
  standard-compliant LR-FHSS gateway decoder (see `README_bringup.md`).

## Critical RF constraint (read before buying time on board 1403)
**LR-FHSS is an uplink modulation with no symmetric peer receiver.** Decoding
LR-FHSS frames requires a gateway-class demodulator (e.g. SX1302/SX1303). A
**second LR1121 cannot decode LR-FHSS**, and SWDM001 ships **no LR-FHSS RX
demo**. Therefore:
- Flashing an RX firmware onto board 1403 will **not** yield LR-FHSS decoded PER.
- A second LR1121 board is sufficient **only** for a standard **LoRa** (not
  LR-FHSS) ping-pong, which would produce *LoRa* PER — a different modulation
  than the paper's LR-FHSS focus.

## What RX firmware must output (any one accepted format)
```
RX seq=<n> payload=<hex> crc=ok rssi=<dbm> snr=<db>
RX seq=<n> payload=<hex> crc=fail            (on CRC failure)
```
(Also accepted: Format B `[RX] seq:.. payload:.. CRC_OK ..`, Format C
`packet_received seq=.. payload_hex=.. crc_ok=true`, or JSONL.) seq + payload +
crc are required; rssi/snr/cfo/timestamp optional.

## Recommended next action (pick one)
1. **LR-FHSS PER (correct for the paper's modulation):** obtain an LR-FHSS
   gateway (SX1302/SX1303 concentrator) and run its packet-forwarder/decoder;
   format its output to the accepted RX-log schema. (Second LR1121 is NOT enough.)
2. **LoRa peer-decode PER (different modulation, clearly labeled):** flash a
   standard LoRa RX example (SWDM001/LR11xx LoRa ping-pong RX role) onto board
   1403, print decoded `RX seq/payload/crc`, and report it explicitly as LoRa
   PER, not LR-FHSS PER.
3. **Stay at current evidence:** keep IQ-level signal detection only; PER remains
   unavailable; paper unchanged.

Until option 1 or 2 produces a decoded RX log, `decoded_rx` cannot run and no PER
is claimed.
