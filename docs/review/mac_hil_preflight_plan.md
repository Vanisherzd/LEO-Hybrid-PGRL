# Mac Conducted-HIL Preflight Plan

*Planning / audit only. No hardware is connected yet. Created on the Windows
side before moving back to Mac to attach hardware. Nothing in this document is a
results claim. `reference_is_measured_truth=false`.*

---

## 1. Purpose of the future Mac HIL run

Provide **conducted (cabled) IQ-level evidence** that the prediction-driven
transmit configuration (LR1121 terminal, 868 MHz, LR-FHSS-like waveform) emits an
observable, hop-structured RF signal over a wired path into an SDR receiver
(USRP B210). The run corroborates the *physical-layer signal-presence* leg of the
paper (the conducted capture already referenced in `paper/icc_main.tex` Validation
Scope) under a clean, reproducible bench setup — and nothing more.

This is engineering corroboration of signal presence, **not** a satellite link,
**not** a decoder, **not** a performance measurement.

## 2. Scope guardrails (read before connecting anything)

- **Conducted / cabled diagnostic only.** TX → attenuator → (optional shielded
  enclosure) → SDR over coax. No antenna radiating to a satellite, no over-the-air
  satellite path.
- **No live-satellite claim.** No satellite is contacted, tracked, or received.
- **No measured-Doppler-truth claim.** Any Doppler value remains SGP4-from-TLE
  model-derived; the bench does not measure true satellite Doppler.
- **No PER / BER / CRC / gateway-ACK claim.** No standards-compliant LR-FHSS
  decoder is in the loop; packet-error/bit-error/CRC/PDR/ACK are out of scope and
  unavailable from an IQ-only capture.
- **IQ-level only.** Evidence ceiling is RF signal presence + hop-like spectral
  structure + TX-ON/OFF margin. Raw IQ is held locally and not committed.

## 3. Expected hardware artifacts

| Artifact | What it is |
|---|---|
| LR1121 terminal (board id recorded, e.g. `1403`) | Transmitter under prediction-driven config |
| USRP B210 SDR | Conducted receiver / IQ capture device |
| Coax + fixed attenuator (value recorded, dB) | Cabled path; protects SDR front end |
| Optional shielded enclosure / RF box | Isolation to exclude ambient 868 MHz |
| UART TX log | Terminal-side ground truth of what/when was transmitted |
| Raw IQ capture (local only, not committed) | Receiver samples for offline detection |
| Detection summary (JSON) | Margins, detection booleans, frequencies (committable) |
| Repeatability figure (PNG/JSON) | TX-ON/OFF across N trials (committable) |
| Run manifest (JSON) | Provenance: hardware ids, RF params, git commit, SHA256s |

(Artifact filenames + required fields are specified in
`docs/review/hil_artifact_schema.md`.)

## 4. Success criteria (conducted IQ-level)

A run is a **success** (signal-presence corroborated) iff ALL hold:

1. **TX-ON/OFF margin** ≥ a pre-registered threshold (target ≥ 6 dB; the existing
   reference capture reported 9.82 dB) in the configured 868 MHz band.
2. **Hop-like sparse structure** detected (energy across multiple narrow bins
   consistent with LR-FHSS-like hopping), not a single CW tone.
3. **DC/LO and image bins excluded** — detected energy is not a receiver artifact.
4. **UART TX log agrees** with detected ON intervals (timing correlation).
5. **Reproducible** across N ≥ 3 independent TX-ON/OFF trials.

Success wording ceiling: "conducted IQ-level RF signal presence with hop-like
structure and an X dB TX-ON/OFF margin." Nothing stronger.

## 5. Failure criteria

Any of the following → **failure / inconclusive** (report honestly, do not
upgrade the claim):

- TX-ON/OFF margin below threshold, or ON/OFF indistinguishable.
- Detected energy confined to DC/LO/image bins (receiver artifact).
- No multi-bin hop structure (only CW or noise).
- UART TX log does not correlate with detected ON intervals.
- Not reproducible across trials (intermittent / single-shot only).
- Any attempt to read decoding/PER from the IQ → automatically a scope failure,
  not a result.

A failure is a valid, publishable outcome ("conducted bench did not establish
signal presence under config X"). It must never be relabelled as a different,
stronger claim.

## 6. Exact ALLOWED wording (post-run, if success)

- "A **conducted (cabled)** LR1121→USRP B210 capture shows **IQ-level** RF signal
  presence with **LR-FHSS-like hop structure** and a **<X> dB TX-ON/OFF margin**."
- "Physical-layer **signal-presence proxy** under a conducted bench setup."
- "UART TX log **corroborates** the detected ON intervals."
- "Receiver-side decoding and PER remain **future gateway-level work**."

## 7. Exact FORBIDDEN wording

- ❌ "live satellite", "satellite link", "downlink/uplink to orbit", "OTA to satellite"
- ❌ "measured Doppler", "Doppler truth", "true satellite Doppler"
- ❌ "PER", "BER", "CRC", "PDR", "packet delivery", "decoded packet", "gateway ACK"
- ❌ "hardware validates the model", "hardware-validated", "validated on hardware"
- ❌ "guarantee", "guaranteed improvement", "worst-case bound"
- ❌ "RF capture proves LR-FHSS", "successful LR-FHSS reception", "link demonstrated"
- ❌ any wording implying radiated transmission or a standards-compliant receiver

---

*Cross-references:* `docs/review/hil_artifact_schema.md` (artifact schema),
`docs/review/claim_evidence_matrix.md` (claim discipline, row 7 conducted-only),
`docs/review/overclaim_audit_before_paper_rewrite.md` (pre-rewrite audit).
