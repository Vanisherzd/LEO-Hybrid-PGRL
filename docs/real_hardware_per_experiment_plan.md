# Real-Hardware Packet-Delivery / PER Experiment Plan

## Purpose
Upgrade from **IQ-level RF signal detection** (Stage 5) to **receiver-side
decoded packet delivery** and measured **PER**, using the packet-validation
harness. No PER is claimed until a decoded RX log with CRC/payload matching
exists.

## Setup (mandatory)
- **Conducted or shielded only**: coax + attenuator, dummy load, or shielded
  enclosure. **No over-the-air transmission.**
- TX power is **never raised automatically**; configure the device explicitly.
- Respect the regional RF profile; record it in the manifest `region_note`.
- Real-hardware backends require explicit flags (`--i-have-hardware`).

## Paper build note
This repo is not a Tectonic "project", so use a direct input file:
```
tectonic paper/icc_main.tex      # or: cd paper && tectonic icc_main.tex
```
(`tectonic -X compile` requires a `Tectonic.toml` project, which this repo lacks.)

## Required artifacts (commit small ones; never commit raw IQ)
1. `tx_payloads.csv` / `tx_payloads.jsonl`
2. `tx_records.csv`
3. TX UART log
4. **RX decoded packet log** (formats A/B/C/JSONL; see parser)
5. optional IQ metadata (small JSON; raw `.fc32` stays local)
6. `packet_validation_summary.json` / `.csv` / `.md`
7. `real_hardware_run_manifest.yaml`

## Minimum experiment matrix
| Run | Guard policy | Purpose |
|---|---|---|
| A | `alpha=0.00` deterministic guard | baseline outage/PER |
| B | `alpha=0.25` uncertainty-aware guard | risk-aware improvement |
| C | stress: lower SNR / higher attenuation / injected CFO | robustness check |

## Metrics
PDR, PER, CRC error rate, missing-seq count, duplicate count, false positives,
latency CDF, RSSI/SNR/CFO distributions.

## Claim gating
- **IQ-only**: no PER, no packet delivery (signal detection only).
- **decoded RX log + payload match**: packet delivery and PER supported.
- **repeated trials, controlled conducted setup**: stronger conducted hardware
  validation.
- **gateway interoperability**: only if an actual LR-FHSS gateway performs the
  decode.

## Commands

### Mock parser validation (no hardware; self-test of the pipeline)
```
python hardware/packet_validation/run_packet_validation.py \
  --run-id parser_e2e_tx --n-packets 100 \
  --tx-backend mock --rx-backend mock \
  --mock-loss-rate 0.0 --mock-crc-error-rate 0.0 \
  --out validation_runs/parser_e2e_tx

python hardware/packet_validation/generate_rx_log.py \
  --tx validation_runs/parser_e2e_tx/tx_payloads.csv \
  --out validation_runs/parser_e2e_rx/rx_format_a.log \
  --format A --loss-rate 0.05 --crc-error-rate 0.01 \
  --duplicate-rate 0.02 --false-positive-rate 0.01 --seed 42

python hardware/packet_validation/run_packet_validation.py \
  --run-id parser_e2e_eval \
  --tx-records validation_runs/parser_e2e_tx/tx_records.csv --skip-tx \
  --rx-backend log_parser \
  --rx-log validation_runs/parser_e2e_rx/rx_format_a.log \
  --out validation_runs/parser_e2e_eval
```

### Future real RX-log evaluation (conducted/shielded)
```
# 1. generate deterministic payloads + apply TX config (no power change)
# 2. TX on real LR1121 (records UART log):
python hardware/packet_validation/run_packet_validation.py \
  --run-id real_runA --n-packets 200 \
  --tx-backend lr1121_uart --i-have-hardware \
  --rx-backend log_parser --rx-log <decoded_rx.log> \
  --out validation_runs/real_runA
# 3. fill hardware/packet_validation/templates/real_hardware_run_manifest.yaml
# 4. commit manifest + summaries + small IQ metadata (NOT raw .fc32)
```

## Status
Current repo supports IQ-level signal detection only. Decoded RX path is
implemented and tested with synthetic fixtures; **measured PER awaits a real
decoded RX log.** The paper makes no PER / packet-delivery claim.
