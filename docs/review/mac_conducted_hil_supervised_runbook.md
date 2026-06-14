# Mac Conducted-HIL Supervised Runbook (DO NOT EXECUTE AUTONOMOUSLY)

> **RUNBOOK ONLY.** This document describes a human-supervised conducted (cabled)
> hardware-in-the-loop run. **No step here is executed by any autonomous pass.**
> Conducted/cabled only — no antenna, no radiation, no satellite. No transmit
> occurs until a human performs §3 by hand. `reference_is_measured_truth=false`.

*Generated: 2026-06-13 20:12 UTC. Detected hardware (read-only):
USRP B210 `serial=8000304`; LR1121 board UART `/dev/cu.usbmodem1303` (ST-Link
VCP). See `docs/review/mac_hardware_readiness_inventory.md`.*

*Governs:* `docs/review/mac_hil_preflight_plan.md` (scope/success criteria),
`docs/review/hil_artifact_schema.md` (artifact fields), `docs/review/claim_evidence_matrix.md`
(wording).

---

## 0. CONFIRMED COMMAND SEQUENCE (interface-verified this session)

Verified against the actual script interfaces. **B210 = RX-only; LR1121 NUCLEO =
conducted transmitter (human-triggered).** Output dir: `outputs/conducted_hil/<UTCSTAMP>/`.
Raw `.fc32` stays **local** (not staged). Run only after `APPROVE_CONDUCTED_HIL_RUN`.

```bash
TS=$(date -u +%Y%m%d_%H%M%S); OUT=outputs/conducted_hil/$TS; mkdir -p "$OUT"
CAP=hardware/usrp_scripts/rx_capture_to_file_cpp        # RX-only, writes .fc32
SER=8000304; FREQ=868e6; RATE=1e6; GAIN=45; ANT="TX/RX"; DUR=10   # ANT = the port the coax is plugged into

# (1) TX-OFF baseline — human keeps LR1121 NOT transmitting:
$CAP --args serial=$SER --freq $FREQ --rate $RATE --gain $GAIN --antenna "$ANT" --duration $DUR --out "$OUT/cap_868_off.fc32"

# (2) TX-ON — human triggers LR1121 ping (NUCLEO reset button) / continuous-TX firmware, then:
$CAP --args serial=$SER --freq $FREQ --rate $RATE --gain $GAIN --antenna "$ANT" --duration $DUR --out "$OUT/cap_868_on.fc32"
# (repeat (1)+(2) for N>=3 trials for repeatability)

# (3) Offline analysis (no hardware):
uv run python hardware/usrp_scripts/analyze_capture.py "$OUT/cap_868_on.fc32"  --sample-rate $RATE --output-json "$OUT/on_analysis.json"  --plot "$OUT/on_waterfall.png"  --signal-threshold-db 8 --maxhold-plot "$OUT/on_maxhold.png"
uv run python hardware/usrp_scripts/analyze_capture.py "$OUT/cap_868_off.fc32" --sample-rate $RATE --output-json "$OUT/off_analysis.json" --plot "$OUT/off_waterfall.png" --signal-threshold-db 8 --maxhold-plot "$OUT/off_maxhold.png"
uv run python hardware/usrp_scripts/compare_tx_on_off.py --tx-on "$OUT/cap_868_on.fc32" --tx-off "$OUT/cap_868_off.fc32" --sample-rate $RATE --out-json "$OUT/on_off_comparison.json" --out-plot "$OUT/on_off_comparison.png" --signal-threshold-db 8
```

**Constraints honoured:** B210 never transmits; no `--reset-method stlink` (tooling
absent); no UART monitor opened by host; raw IQ local-only; no `dataraw/` write.
`analyze_capture.py` emits `validation_status ∈ {signal_detected,
weak_signal_candidate, no_signal}` + CFO/EVM **proxy** + SNR (IQ-level only, not PER).

---

## 1. Pre-run PHYSICAL checklist (human, before powering TX)

- [ ] **Cabled / conducted path only.** TX (LR1121 SMA) → **fixed attenuator** →
      (optional shielded RF box) → coax → USRP B210 RX. **No antenna on either
      end.** Nothing radiating to a satellite or to air.
- [ ] **Attenuator present and rated.** Value recorded in dB; sized so the TX power
      after attenuation is **well below** the B210 RX damage threshold
      (B210 RX max ≈ −15 dBm; keep a comfortable margin). If unsure, add more
      attenuation, not less.
- [ ] **No antenna / no radiation.** Visually confirm no antenna is attached to the
      LR1121 TX port. A conducted run with an antenna would be radiated — not
      allowed.
- [ ] **RF box / shielding** (if used) closed, to exclude ambient 868 MHz.
- [ ] **SDR front-end protection.** Coax fully seated; no open SMA on the powered
      TX; attenuator between TX and RX **before** any TX is enabled.
- [ ] **Power sanity.** B210 on a powered USB3 port (its own hub leg); LR1121 board
      powered via its ST-Link USB. No brown-out / shared underpowered hub for the
      B210.
- [ ] **Frequency authorization.** A carrier is set only after NCC / local /
      gateway frequency-plan confirmation. Until then `nominal_center_freq_hz`
      stays unset and the schedule generator refuses to run
      (`docs/review/frequency_schema_f0_guard_verification.md`).

## 2. SOFTWARE checklist (human, read-only first)

- [ ] `uhd_find_devices` → expect `serial 8000304` (already confirmed read-only).
- [ ] `uhd_usrp_probe --args "serial=8000304"` → confirm clock/daughterboard
      (loads FPGA; **no TX**). Human-supervised.
- [ ] Confirm `/dev/cu.usbmodem1303` is the LR1121 UART (manual monitor, read-only,
      observe boot banner). Note baud rate.
- [ ] If the capture path uses the Python UHD API, confirm `import uhd` works;
      otherwise use the CLI/script path. (Python `uhd` was **not** importable at
      readiness time.)
- [ ] Record `git_commit` for the run manifest.
- [ ] Pre-register the TX-ON/OFF margin threshold (target ≥ 6 dB; prior reference
      capture reported 9.82 dB) **before** looking at results.

## 3. Run sequence (human-supervised; TX happens here — NOT autonomous)

1. Generate schedule with an **explicit** carrier (refuses without it):
   `python3 hardware/ota_iq/generate_real_replay_schedule.py --nominal-center-hz <HZ> ...`
2. Start the conducted capture (`hardware/ota_iq/usrp_capture_ota_iq.py` /
   `hardware/usrp_scripts/rx_capture_to_file.py`) for the TX-OFF baseline first.
3. Enable TX via the replay driver (`hardware/ota_iq/replay_driver.py`) for the
   TX-ON capture, with the UART TX log (`/dev/cu.usbmodem1303`) recording.
4. Repeat ON/OFF for **N ≥ 3** trials (repeatability).
5. Power down TX before disconnecting anything.

## 4. Artifact paths (per `hil_artifact_schema.md`)

Write under a run directory, e.g. `hardware/ota_iq/runs/<run_id>/`:

```
run_manifest.json            schedule_meta.json          tx_config.json
uart_tx_log.txt              usrp_capture_meta.json      raw_iq_capture.sigmf-data  (LOCAL ONLY)
raw_iq_capture.sigmf-meta    on_analysis.json            off_analysis.json
on_off_comparison.json       repeatability_summary.json  cfo_residual_summary.json
paper_candidate_fig_hw_conducted_iq.png                  hardware_claim_summary.md
```

- Raw IQ (`*.sigmf-data`) is **local-only**, never committed; pinned by
  `usrp_capture_meta.json::iq_file_sha256`.
- Offline analysis helpers: `hardware/ota_iq/analyze_cfo_residual.py`,
  `hardware/ota_iq/analyze_adjacent_bin_leakage.py`,
  `hardware/usrp_scripts/compare_tx_on_off.py`.

## 5. Stop conditions (abort immediately if any occur)

- B210 RX overload / clipping warning, or any thermal/power warning → stop TX,
  add attenuation.
- Any uncertainty that an antenna is attached or the path is radiating → stop.
- Carrier not authorized / `nominal_center_freq_hz` unset → do not transmit.
- Capture writing to `dataraw/` → stop (never write `dataraw/`).
- Any temptation to read decoding/PER from the IQ → stop; that is a scope failure.

## 6. Expected logs

- UART TX log lines with timestamps + `TX start/end` aligned to scheduled ON
  intervals.
- UHD capture stdout: device args `serial=8000304`, sample rate, center freq, gain.
- Per-trial `on_analysis.json` / `off_analysis.json` with `validation_status` in
  {`signal_detected`, `weak_signal_candidate`, `no_signal`}, DC/LO + image bins
  excluded.

## 7. Exact ALLOWED claim ceiling (only if success criteria met)

- "A **conducted (cabled)** LR1121→USRP B210 capture shows **IQ-level** RF signal
  presence with **LR-FHSS-like hop structure** and a **<X> dB TX-ON/OFF margin**,
  reproducible across N≥3 trials."
- "Residual-CFO / hop-orthogonality reported as **RF-quality proxies**."
- "UART TX log **corroborates** the detected ON intervals (timing only)."
- "Receiver-side decoding and PER remain **future gateway-level work**."
- "All Doppler is **model-derived (SGP4 from real TLE)**;
  `reference_is_measured_truth=false`."

## 8. Exact FORBIDDEN claims

- ❌ "live satellite", "satellite link", "satellite validation", "OTA to satellite"
- ❌ "measured Doppler", "Doppler truth", "measured link"
- ❌ "PER", "BER", "CRC", "PDR", "packet delivery", "decoded packet", "gateway ACK"
  (except as explicit future-work non-claims)
- ❌ "hardware validates the model", "hardware-validated", "validated on hardware"
- ❌ "guarantee", "worst-case bound", "can only help"
- ❌ "successful LR-FHSS reception", "standards-compliant LR-FHSS decoding",
  "link demonstrated"
- ❌ any wording implying radiated transmission or a standards-compliant receiver

## 9. If USRP not detected

1. Re-seat the B210 on a powered USB3 port (own hub leg); avoid an underpowered
   shared hub.
2. `uhd_find_devices` again; if firmware-load errors, re-run (FX3 load is volatile).
3. Confirm UHD images exist (`usrp_b210_fpga.bin` present at readiness time).
4. Try a different USB3 cable/port. Do not proceed to TX until discovery is clean.

## 10. If LR1121 UART (`/dev/cu.usbmodem1303`) not detected

1. Re-seat the ST-Link USB cable; re-check `ls /dev/cu.usbmodem*`.
2. The ST-Link/V2-1 may enumerate without exposing the VCP if the board firmware
   does not open the CDC port — confirm board power and firmware state.
3. ST-Link CLI tools were not installed at readiness; install `stlink` only if a
   (separate, out-of-scope) MCU operation is genuinely required. Flashing/reset is
   **not** part of an IQ-level capture.
4. Without a UART log, a capture can still proceed but loses the timing-corroboration
   leg — note this limitation; do not upgrade the claim to compensate.

## 11. If signal absent (TX-ON ≈ TX-OFF)

- Report **failure / inconclusive** honestly ("conducted bench did not establish
  signal presence under config X"). A failure is a valid, publishable outcome.
- Check: TX actually enabled? carrier correct? attenuation too high (signal below
  noise)? wrong center freq / band? DC-LO masking?
- **Never** relabel a null result as a different, stronger claim, and never read
  PER/decoding from the IQ.

## 12. After the run

- [ ] Power down TX before disconnecting.
- [ ] Compute SHA256 of every committable artifact; fill
      `run_manifest.json::artifact_sha256` and `input_sha256` (incl. local IQ hash).
- [ ] Write `hardware_claim_summary.md` using only §7 allowed wording + a *what
      this is NOT* list.
- [ ] Keep raw IQ local; commit only small summaries/figures/manifests (when a
      human approves a commit).
- [ ] Update `docs/review/claim_evidence_matrix.md` row 7 and
      `docs/review/paper_figure_table_plan.md` Fig 3 with the actual result.

---

**This runbook is documentation. It is not executed by any autonomous pass. The
first transmit occurs only when a human runs §3 by hand under supervision.**
