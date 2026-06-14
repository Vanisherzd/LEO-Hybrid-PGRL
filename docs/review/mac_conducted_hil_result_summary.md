# Mac Conducted-HIL Result Summary — INCONCLUSIVE / INVALID SETUP

> **HONEST NEGATIVE/INCONCLUSIVE RECORD. NO signal presence, NO hardware
> validation, NO paper claim is made from this run.** The bench was **not** a
> valid conducted setup: there was **no SMA coax cable and no fixed attenuator**
> between the LR1121 RF output and the B210 RX input — the two boards were only
> placed near each other. Therefore this run **cannot** support IQ-level conducted
> RF signal-presence evidence. `reference_is_measured_truth=false`.

*Run: 2026-06-14 05:00 UTC. Output dir `outputs/conducted_hil/20260614_050039/`.
Repo HEAD at run time `1f25ec5`. B210 serial `8000304` (RX-only). Stopped by
operator (`STOP_CONDUCTED_HIL_RUN`) after trial 1.*

---

## 1. What happened

- Physical-setup approval was given, but the setup was **later found invalid**:
  **no coax, no fixed attenuator** between LR1121 TX and B210 RX. Boards were
  merely near each other → **no conducted RF path**.
- 3× TX-OFF baseline captures and 1× TX-ON capture were taken (B210 receive-only,
  10 s, 868 MHz, TX/RX port, gain 45). The LR1121 ping was to be triggered by
  NUCLEO reset during the ON window.
- Trials 2–3 were **not** run. No retry was performed. The run was stopped.

## 2. Trial 1 result — inconclusive (ON not stronger than OFF)

ON vs OFF comparison (`cmp_trial1.json`):

| Metric | TX-ON | TX-OFF |
|---|---|---|
| maxhold_excess_db | 3.19 | 4.28 |
| occupied_frequency_bins | 4 | 2 |
| hot_bin_count | 4 | 2 |
| lr_fhss_candidate_score | 0.200 | 0.120 |

- `on_off_delta_db = −1.09 dB` (criterion needs **≥ +3.0 dB**)
- `freq_occupancy_delta = 2` (criterion needs **≥ 3**)
- **`tx_on_stronger_than_off = FALSE`** — ON is not distinguishable from OFF; OFF
  even shows marginally higher max-hold excess (noise).

Per-capture detector (`analyze_capture.py`, DC/LO guard excluded):

| Capture | validation_status | signal_detected | maxhold_peak_count (excl. DC) | CFO/EVM |
|---|---|---|---|---|
| `cap_868_on1.fc32` | **noise_floor_only** | **False** | 0 | skipped (noise floor only) |
| `cap_868_off1.fc32` | **noise_floor_only** | **False** | 0 | skipped (noise floor only) |

The ±22 kHz "peak offsets" reported are noise artifacts (peak-to-median ≈ 1.9 dB,
zero non-DC max-hold peaks). No hop-like sparse structure. This is exactly what a
**no-RF-path** bench produces.

## 3. Root cause

**Missing conducted RF path.** No coax cable and no fixed attenuator connected the
LR1121 RF output to the B210 RX input; the boards were only positioned near each
other. With no cabled path (and no antenna, correctly), essentially no LR1121
energy reached the B210 RX, so ON ≈ OFF ≈ noise floor. The negative result is a
property of the **invalid setup**, not evidence about the LR1121 waveform.

## 4. Artifacts

Output dir: `outputs/conducted_hil/20260614_050039/` (**git-ignored**).

Raw IQ — **LOCAL-ONLY, NOT staged, NOT committed** (76.3 MB each):

| File | SHA256 |
|---|---|
| `cap_868_off1.fc32` | `c4b39aed33a1fb6b5931ba8cb9b6f592834573bcbceca15c43ab5d770c67a33c` |
| `cap_868_off2.fc32` | `0f2bd7715ce082c96e902337af151d6029a8fb230bb7bb08cf3b5d06c165d4fc` |
| `cap_868_off3.fc32` | `07e3fdcb3bdddcc48fdc7c4ac3d25add2278d80b2074b9c6927b2e644d451969` |
| `cap_868_on1.fc32` | `15d5eeb74b72511f3ad837582294cbf8f53c208109396605d3d996f2dadba1bf` |

Small derived summaries (committable only after human review):

| File | SHA256 |
|---|---|
| `cmp_trial1.json` | `a9765bd40b3a5e6b1ecbbf857ebf2b8280c3096e8931c85814a18870f5658eca` |
| `on1_analysis.json` | `b547869787ddd34f21d6414020105ff3a49921a13cef265ef9fd960099b43029` |
| `off1_analysis.json` | `9900678eb986e3aedd1557f64763a787b6f49651cbb0c71c95992770f7e487ec` |

## 5. What this run supports / does NOT support

**Supports (only):**
- The software/RX pipeline functions end-to-end: B210 RX capture
  (`rx_capture_to_file_cpp`) + offline `analyze_capture.py` / `compare_tx_on_off.py`
  ran and correctly reported `noise_floor_only` (a true negative under no RF path).
  This is a **tooling sanity check**, not RF evidence.

**Does NOT support (must not be claimed):**
- ❌ conducted IQ-level RF signal presence
- ❌ hop-like / LR-FHSS-like spectral structure
- ❌ any TX-ON/OFF margin (ON was not above OFF)
- ❌ CFO / occupancy proxy (skipped; noise floor only)
- ❌ live satellite · measured Doppler truth · LR-FHSS decoding
- ❌ PER / BER / CRC / PDR · gateway ACK · hardware-validated link
- ❌ any paper claim whatsoever from this run

## 6. Required next physical setup (before any future conducted-HIL run)

1. **Cable the path:** LR1121 RF output → **coax (SMA)** → **fixed attenuator
   (known dB)** → B210 RX input. No air gap, no antenna.
2. Size the attenuator so post-attenuation power is well below the B210 RX damage
   threshold (≈ −15 dBm) — protect the SDR front end before enabling TX.
3. Confirm the coax is on the B210 SMA port passed to `--antenna` (TX/RX or RX2).
4. Verify the LR1121 actually transmits (board prints `Packet sent!` / TX LED) and
   that the ping lands inside the capture window (or use continuous-TX firmware).
5. Only then re-run the supervised TX-ON/OFF repeatability diagnostic
   (`docs/review/mac_conducted_hil_supervised_runbook.md`).

Until a proper cabled+attenuated path exists, **no conducted-HIL signal-presence
claim is possible.**

---

*Cross-references:* `docs/review/mac_conducted_hil_supervised_runbook.md`,
`docs/review/hil_artifact_schema.md`, `docs/review/mac_hil_preflight_plan.md`,
`docs/review/claim_evidence_matrix.md` (row 7 conducted-only).
