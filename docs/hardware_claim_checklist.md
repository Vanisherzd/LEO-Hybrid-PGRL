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