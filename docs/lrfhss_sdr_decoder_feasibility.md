# LR-FHSS SDR Receiver Feasibility (Structure Scaffold)

## Current status
- LR1121 TX board works (emits LR-FHSS TX demo; UART "Packet sent!").
- USRP B210 IQ capture works (conducted/shielded).
- **No LR-FHSS payload decode. No CRC. No PER.**
- A second LR1121 cannot decode LR-FHSS; full decode needs a gateway-class
  receiver (see `docs/lr1121_rx_firmware_gap.md`).

## What this scaffold does
`scripts/analyze_lrfhss_iq_structure.py` performs **IQ-structure analysis only**
on an existing `.fc32` capture:
- PSD / max-hold spectrum, time-frequency spectrogram, noise-floor estimate
- frame-relative narrowband detection (margin above noise peak-to-median)
- candidate burst / candidate hop-tone detection, occupied frequency/time bins
- dominant frequency offsets, burst-duration statistics
- a bounded heuristic `structure_score` in [0,1]

It strengthens the existing conducted IQ-level evidence: the Stage-5 ON capture
shows **sparse, multi-frequency, short candidate bursts** — i.e. LR-FHSS-*like*
hopping structure — without any decode claim.

Representative result (Stage-5 ON capture, 868 MHz, 1 MHz, first 3 s):
~101 candidate bursts across ~19 occupied frequency bins, ~12% time occupancy,
structure_score ≈ 0.85.

## What it does NOT do
- No header recovery
- No fragment reconstruction
- No deinterleaving / FEC
- No payload reconstruction
- No CRC
- No PER

## Milestones toward a full SDR decoder
1. header replica detection
2. hop timing recovery
3. fragment extraction
4. deinterleaving / FEC (if required)
5. payload reconstruction
6. CRC validation
7. PER measurement

## Why this is enough for the current paper
It strengthens the hardware/RF evidence (LR-FHSS-like signal structure under a
conducted setup) **without overclaiming** decode/PER. The paper continues to
state IQ-level signal detection only; PER remains future work pending a
gateway-class decoder.

## Allowed vs forbidden wording
- Allowed: "candidate burst", "candidate hop/tone", "occupied bin",
  "LR-FHSS-like structure", "IQ-level evidence", "structure score".
- Forbidden: "decoded packet", "payload", "CRC", "packet delivery", "PER",
  "hardware-validated" (except in explicit negations).
