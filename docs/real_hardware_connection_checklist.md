# Real Hardware Connection Checklist (Conducted/Shielded PER Run)

Complete EVERY item before an `--armed` decoded-RX run.

## RF safety (mandatory)
- [ ] **Conducted or shielded setup only** — coax + attenuator, dummy load, or
      shielded enclosure.
- [ ] **No over-the-air test.** No antenna radiating into open air.
- [ ] Regional RF settings confirmed **manually** on the device.
- [ ] **TX power is NOT increased automatically** — use the configured value.

## Signal path
- [ ] LR1121 UART path confirmed (TX command + "Packet sent!" log reachable).
- [ ] USRP / gateway / RX-decoder path confirmed.
- [ ] Decoded RX log format confirmed (A / B / C / JSONL — see rx_log_parser).

## Artifacts
- [ ] `tx_payloads.csv` saved for the run (deterministic seq + CRC).
- [ ] Decoded RX log captured and path set in the run config.
- [ ] Raw IQ kept **local-only** (never committed; `*.fc32` is gitignored).
- [ ] Manifest saved (`manifest.yaml` / `manifest_final.yaml`).
- [ ] `validation_runs/` outputs NOT committed (gitignored); copy small
      summaries out if they must be committed.

## Claim gating
- **IQ-only**: no PER, no packet delivery (signal detection only).
- **Decoded RX log + payload/CRC matching**: PER / PDR allowed.
- **Repeated runs (A/B/C), controlled conducted setup**: stronger conducted
  hardware validation.
- **Gateway interoperability**: only if an actual gateway decoder produced the
  decoded RX log.

## Do NOT
- [ ] Do not modify paper claims based on a single unverified run.
- [ ] Do not claim measured PER until a decoded RX log with payload/CRC matching
      exists and `packet_validation_summary.json` reports it.
