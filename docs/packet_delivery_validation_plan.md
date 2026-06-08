# Packet-Delivery / PER Validation Plan

## Purpose
Provide a reproducible framework to move from IQ-level RF signal detection
(Stage 5) toward measured packet delivery, packet-error rate (PER), and
receiver-side decoding — **without claiming results that do not yet exist**.

## Current status
- Supported today: **IQ-level signal detection only** (conducted LR1121→USRP
  capture, ON/OFF delta 9.82 dB, `signal_detected`). See
  `docs/hardware_iq_capture_stage5_results.txt`.
- **Not yet supported:** packet delivery, measured PER, receiver decoding,
  full hardware validation.

## What is needed for measured PER
1. A receiver that **decodes** LR-FHSS payloads (LR1121 RX or an LR-FHSS
   gateway), producing per-packet payload + CRC verdicts.
2. RX logs aligned to the deterministic TX payloads (seq + CRC).
3. Repeated conducted trials with committed summary artifacts (raw IQ may stay
   local; small metadata + summaries committed).

## RF safety / scope
- **Conducted or shielded only**: coax + attenuator, dummy load, or shielded
  enclosure. **No over-the-air tests.**
- TX power is **never raised automatically**; defaults are used as-is.
- No guidance for bypassing regional RF limits.
- The pipeline **defaults to dry-run/mock**; real-hardware backends require an
  explicit `--i-have-hardware` flag.

## Run commands

### Mock (software pipeline only — can compute PER as a self-test)
```
python hardware/packet_validation/run_packet_validation.py \
  --run-id dryrun_001 --n-packets 100 \
  --tx-backend mock --rx-backend mock \
  --mock-loss-rate 0.05 --mock-crc-error-rate 0.01 \
  --out validation_runs/dryrun_001
```

### IQ-only replay (TX UART + IQ evidence; PER MUST be unavailable)
```
python hardware/packet_validation/run_packet_validation.py \
  --run-id iq_only_stage5_replay --n-packets 10 \
  --tx-backend file_replay \
  --tx-log docs/hardware_iq_capture_stage5/cap_868000000_txrx_on_uart.log \
  --rx-backend iq_only \
  --iq-metadata docs/hardware_iq_capture_stage5/cap_868000000_txrx_comparison.json \
  --out validation_runs/iq_only_stage5_replay
```

### Future real-hardware procedure (conducted/shielded)
1. Connect LR1121 TX → attenuator/dummy load → USRP/gateway RX by coax.
2. `generate_payloads` → deterministic TX payload set.
3. Apply TX config (no power change); run `tx-backend lr1121_uart
   --i-have-hardware`.
4. Capture RX decode log; run `rx-backend log_parser --rx-log <log>`.
5. `evaluate_packet_delivery` computes PER from decoded payloads.
6. Commit small summaries; keep raw IQ local.

## Claims supported by each mode

| TX / RX mode | Evidence | Claim supported |
|---|---|---|
| mock / mock | software only | none (self-test of the pipeline; no hardware claim) |
| file_replay / iq_only | TX UART "Packet sent!" + IQ signal detection | TX activity + IQ-level RF signal detection; **no PER** |
| lr1121_uart / log_parser (decoded RX payloads) | conducted decode | packet delivery / PER |
| lr1121_uart + gateway decode + repeated trials | repeated conducted decode | stronger hardware validation |

## Guarantees
- `iq_only` always sets `decode_status="not_decoded"`, never counts a packet as
  delivered, and the evaluator emits
  `"PER unavailable: no receiver-side packet decoding."`
- PER/PDR are reported only when at least one decoded RX payload exists.
