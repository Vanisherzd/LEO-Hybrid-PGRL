# OTA IQ validation scope (CFO residual + adjacent-bin leakage)

Scope discipline for the `hardware/ota_iq/` experiments. Read this before writing
any paper text, README claim, or results label from these captures.

## What this is

Short-range **room OTA / near-field** IQ captures from a USRP B210 (RX only) of a
Semtech LR1121 emitting repeated short LR-FHSS bursts under a programmed-Doppler
replay. From the raw IQ we estimate two **IQ-level physical-layer proxies**:

1. **Residual CFO** — per-burst carrier-frequency offset from the LR-FHSS grid.
2. **Adjacent-bin leakage ratio (ABLR)** — per-burst power in the adjacent grid
   bins relative to the target grid bin.

These proxies compare three open-loop feedforward modes under an identical
Doppler profile: no compensation, SGP4-only, and PGRL-corrected.

## What this is NOT

This setup has **no SMA coax and no calibrated attenuator**, so it is **not a
conducted measurement**. It is **not** a standard-compliant LR-FHSS gateway and
does **not**:

- decode packets,
- measure packet-error rate (PER),
- check CRC,
- validate a receiver or gateway,
- measure absolute (calibrated) power.

Do **not** use the words *conducted*, *PER*, *decoded packet*, *CRC*, *gateway
validation*, or *receiver validation* for anything produced here.

## Measurement-type and validation labels

| Field | Value | Meaning |
|---|---|---|
| `measurement_type` | `short_range_ota_iq` | room OTA / near-field, uncalibrated, **not conducted** |
| `validation_scope` | `short_range_ota_iq_proxy` | IQ-level physical-layer proxy, **not PER / not decoding** |

These extend the existing ladder in
[`docs/hardware_claim_checklist.md`](hardware_claim_checklist.md). The OTA IQ
proxies sit **above** `hardware-signal-detected` (they quantify grid confinement,
not just RF presence) but **below** any decoding/PER claim, which remains future
work.

## Why these proxies are meaningful

The PGRL contribution is tighter open-loop Doppler pre-compensation. Its physical
consequence is that an emitted burst lands closer to its intended orthogonal
LR-FHSS grid bin. Residual CFO measures that distance directly; ABLR measures the
spectral spill into neighbouring grid bins that off-grid emission causes. Both
are observable at the IQ level without decoding, which is exactly why they are
the honest evidence available with this hardware.

### Known limitations (state these alongside results)

- **Uncalibrated amplitude.** ABLR is a *ratio* of in-capture powers, robust to
  the unknown absolute gain, but no absolute power level is claimed.
- **Room multipath / near-field coupling.** Short-range OTA adds reflections and
  coupling not present in a conducted path; treat absolute leakage floors as
  upper bounds, comparisons (B vs C) as the signal.
- **No receiver.** Grid confinement is a *necessary* condition for orthogonal
  reception, not a *sufficient* one. PER is not implied.
- **Replay, not a live pass.** Doppler is emulated via feedforward; the residual
  reflects the feedforward error model fed in from real SGP4/PGRL predictions.
- **Grid spacing is verified.** ABLR uses `grid_spacing_hz = 25391` Hz, read from
  the SWDM001 firmware (`LR_FHSS_V1_GRID_25391_HZ`), not assumed.
- **Replay needs host-replay firmware.** The three modes are only distinguishable
  if the LR1121 accepts a per-burst frequency/power command (see
  `hardware/ota_iq/firmware/`). With stock free-running firmware every mode emits
  identical bursts — any "mode comparison" from that would be invalid. The driver
  requiring `BURST_DONE` acks is the guard: 0 acks ⇒ no replay result.

## Allowed paper claim

> The OTA IQ replay shows that the PGRL-corrected feedforward schedule reduces
> measured residual CFO and adjacent-bin leakage relative to SGP4-only replay,
> providing physical-layer evidence for improved grid confinement. Receiver-side
> packet decoding and PER remain future work.

Two-node waterfalls (Experiment 3) are **qualitative OTA demonstrations only**:
with no shared timing/trigger, **no measured collision probability** may be
reported.

## Artifact chain required before citing a number

For each cited OTA IQ proxy number, the repository must contain:

- the run dir with `capture_meta.json` (`measurement_type == short_range_ota_iq`,
  `clipping_warning == false`),
- the analysis `*_summary.json` carrying a `commit` field and
  `validation_scope == short_range_ota_iq_proxy`,
- the per-burst CSV the summary was computed from,
- the figure reproducible from that CSV.

Raw `.npy` / `.fc32` IQ may be archived locally and excluded from the repo, but
the curated JSON + CSV + figure must be committed.
