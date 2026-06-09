# Submission Scope — PGRL-Assisted LR-FHSS Uplink Control

This document defines the clean repository scope for the first paper. It intentionally avoids venue-specific wording because the paper target may change, and manuscript text is being edited locally.

## Research Question

How can a direct-to-satellite IoT terminal use SGP4-anchored PGRL predictions and calibrated uncertainty estimates to make safer LR-FHSS uplink-control decisions under timing and Doppler uncertainty?

## Included in the Submission Scope

- SGP4/SDP4-anchored PGRL residual predictor
- Frozen-mean uncertainty-head fine-tuning with Gaussian NLL
- Post-hoc calibration check for uncertainty scaling
- Risk-aware guard adaptation driven by calibrated position-domain uncertainty
- TX timing selection and Doppler pre-compensation as controller components
- LR-FHSS-inspired frequency-grid / RF-quality proxy analysis
- Conducted LR1121 to USRP B210 IQ-level signal detection
- Offline sparse-hop-like IQ-structure analysis

## Explicitly Excluded from the Clean Repo Scope

These are not first-paper contributions and should not appear as root-level active modules:

- **PER / packet delivery:** no standards-compliant LR-FHSS receiver or gateway is available, so PER is unavailable.
- **Full LR-FHSS decoder:** future work only.
- **MAC / TDMA protocol:** legacy simulation and thesis extension only.
- **PPO / GRPO online learning:** useful extension, but not part of the first paper.
- **ISAC closed-loop correction:** journal/thesis extension.
- **Semantic communication:** separate research thread.
- **Multi-node network scheduling:** outside the single-terminal uplink-control focus.
- **Production deployment:** no autonomous field deployment claim.

## Repo-Cleanup Rule

Repository cleanup should not edit the active manuscript source or author placeholder. The manuscript and author list are handled locally by the author until the advisor/coauthor list is confirmed.

Use the repository cleanup only to remove or quarantine files whose names imply unavailable evidence or a different paper scope.

## Claim Language to Keep in Repo Docs

Use:

- "control proxy"
- "LR-FHSS-inspired proxy"
- "IQ-level signal detection"
- "conducted LR1121-to-USRP capture"
- "receiver decoding and PER are outside the present scope"
- "position-domain uncertainty mapped to a guard-control proxy"

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
