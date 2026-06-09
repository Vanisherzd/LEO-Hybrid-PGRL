# Submission Scope — PGRL-Assisted LR-FHSS Uplink Control

This document defines the clean workshop-submission scope for the first paper. It intentionally avoids venue-specific wording because the target may be ICC/GLOBECOM/VTC/other top-tier workshops, and China-based venues are excluded by author preference.

## Research Question

How can a direct-to-satellite IoT terminal use SGP4-anchored PGRL predictions and calibrated uncertainty estimates to make safer LR-FHSS uplink-control decisions under timing and Doppler uncertainty?

## Included in the Submission

- SGP4/SDP4-anchored PGRL residual predictor
- Frozen-mean uncertainty-head fine-tuning with Gaussian NLL
- Post-hoc calibration check showing no scalar rescaling is needed in the reported setup
- Risk-aware guard adaptation driven by calibrated position-domain uncertainty
- TX timing selection and Doppler pre-compensation as controller components
- LR-FHSS-inspired frequency-grid / RF-quality proxy analysis
- Conducted LR1121 to USRP B210 IQ-level signal detection
- Offline sparse-hop-like IQ-structure analysis

## Explicitly Excluded

These are not paper contributions and should not appear as central claims:

- **PER / packet delivery:** no standards-compliant LR-FHSS receiver or gateway is available, so PER is unavailable.
- **Full LR-FHSS decoder:** future work only.
- **MAC / TDMA protocol:** legacy simulation and thesis extension only.
- **PPO / GRPO online learning:** useful extension, but not part of the workshop paper.
- **ISAC closed-loop correction:** journal/thesis extension.
- **Semantic communication:** separate research thread.
- **Multi-node network scheduling:** outside the single-terminal uplink-control focus.
- **Production deployment:** no autonomous field deployment claim.

## Claim Language to Use

Use:

- "control proxy"
- "LR-FHSS-inspired proxy"
- "IQ-level signal detection"
- "conducted LR1121-to-USRP capture"
- "receiver decoding and PER remain future work"
- "position-domain uncertainty mapped to guard-control proxy"

Avoid:

- "PER improvement"
- "packet-delivery validation"
- "hardware-validated LR-FHSS link"
- "full gateway receiver"
- "protocol contribution"
- "online learning deployment"
- "autonomous satellite modem"

## Why the Scope Must Stay Narrow

The paper is strongest as a cross-layer workshop paper: **calibrated orbital-prediction uncertainty → risk-aware LR-FHSS uplink control**. PER, TDMA MAC, online GRPO, and ISAC each require enough evidence to become separate papers. Including them now would reduce clarity and increase reviewer attack surface.
