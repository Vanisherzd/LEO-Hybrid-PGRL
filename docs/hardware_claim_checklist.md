# Hardware Claim Checklist

> A result may be labeled "hardware-validated" or "hardware-measured" **only** when all artifacts listed below exist in the repository. Until then, the result must be labeled using one of the allowed validation-type tags.

## Allowed Validation-Type Tags

| Tag | Meaning |
|-----|---------|
| `trace-driven` | Evaluated against recorded TLE/ephemeris data — no hardware |
| `simulation` | Physics/communication simulation in Python — no hardware |
| `proxy_simulation` | Physical proxy metric (e.g., QPSK EVM) demonstrating RF chain quality — no hardware |
| `lrfhss_inspired_proxy` | LR-FHSS grid proxy — not full PER from standards-compliant decoder |
| `planned` | Hardware acquisition and measurement are planned but not yet executed |
| `hardware-bringup` | Capture/transmit workflow runs on real hardware (USRP B210 RX path, SWDM001 LR1121 firmware), but no signal detected yet — workflow validation only |
| `hardware-signal-detected` | Automated detector found sparse-hop RF energy above the noise floor (DC/LO excluded), corroborated where available by UART logs + TX ON/OFF reference. **IQ-level RF presence only — NOT decoding/PER** |
| `hardware (acquired)` | Hardware is in hand; measurement script has been run |
| `hardware (validated)` | Full artifact chain completed (see checklist below) |

---

## Full Hardware-Validation Artifact Chain

Every hardware-validated result **must** have all of the following in the repository:

### Raw Data
- [ ] Raw IQ capture file (`.cfile`, `.sigmf`, or equivalent)
- [ ] Hardware log / command log showing TX configuration applied
- [ ] SDR session configuration (center frequency, sample rate, gain)

### Configuration
- [ ] TX configuration JSON corresponding to the capture
- [ ] PGRL prediction JSON (or `results.json`) at time of capture

### Analysis
- [ ] Analysis script with commit hash recorded in results.json
- [ ] `results.json` with `commit`, `timestamp`, and per-metric values
- [ ] Generated figure (`.png` / `.pdf`) reproducible from the capture

### Provenance
- [ ] `commit` field in `results.json` referencing the analysis script version
- [ ] `validation_type` field set to `hardware (validated)`

---

## `hardware-signal-detected` Requirements

A result may carry `hardware-signal-detected` when **all** of the following hold:

- [ ] Analyzer reports `validation_status == "signal_detected"` and `signal_detected == true` (sparse-hop occupancy above per-bin noise floor, DC/LO spike excluded).
- [ ] UART packet evidence (`uart_packet_sent_count > 0`) when a UART log is available.
- [ ] TX ON/OFF control (or an equivalent transmitter-off reference) showing `tx_on_stronger_than_off == true`.
- [ ] Curated analysis JSON + figures committed under `hardware/artifacts/` (raw `.fc32` excluded).

**What it does NOT mean:** it does **not** mean standard-compliant LR-FHSS decoding, packet-error-rate (PER) measurement, or a full gateway receiver. It is IQ-level RF signal presence only.

**Current status (2026-06-04):** three repeated LR1121 + USRP B210 TX ON/OFF trials at 868 MHz reached `hardware-signal-detected` (ON/OFF deltas 8.88 / 11.87 / 9.82 dB). Curated under `hardware/artifacts/lr1121_signal_detected_repeatability_20260604/`.

---

## Premature Claim Examples (Do Not Use)

| ❌ Claim | Reason | ✅ Correct Label |
|---------|--------|-----------------|
| "validated with USRP B210" | No IQ capture file | `planned` |
| "EVM < 3% on hardware" | No raw results.json | `planned` |
| "SDR measurement confirms" | No capture or log | `planned` |
| "Semtech LR-FHSS TX tested" | No hardware in hand | `planned` |
| "proves LR-FHSS PER improvement" | Not a standards-compliant PER | `proxy_simulation` |

---

## GitHub Issue Template for Hardware Validation

When creating the issue to track hardware acquisition and validation, use this template:

```markdown
## Hardware Validation: [Exp name]

### Status
- [ ] Hardware acquired
- [ ] Baseline TX captured
- [ ] PGRL-compensated TX captured
- [ ] IQ analysis script run
- [ ] results.json written
- [ ] Figure generated
- [ ] Reviewer-ready label applied

### Commit hash of analysis script
TBD

### Validation type to use in README until all items checked
`planned`
```