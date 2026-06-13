# Mac Conducted-HIL Artifact Schema

*Planning / audit only. No hardware was run to produce this document. It defines
the artifacts that a **future** human-supervised Mac conducted-HIL run is expected
to emit, the metadata each must carry, and the exact claim ceiling each artifact
can and cannot support. `reference_is_measured_truth = false` throughout.*

*Cross-references:* `docs/review/mac_hil_preflight_plan.md` (run plan),
`docs/review/claim_evidence_matrix.md` (row 7 = conducted-only),
`docs/review/overclaim_audit_before_paper_rewrite.md` (pre-rewrite audit),
`docs/hardware_claim_checklist.md` and `docs/ota_iq_validation_scope.md`
(pre-existing IQ-level scope definitions this schema is consistent with).

---

## 0. Global rules (apply to every artifact below)

1. **Conducted / cabled only.** Every artifact describes a wired TX→attenuator→SDR
   path. No artifact may be captioned as radiated, over-the-air-to-satellite, or
   live-satellite.
2. **`reference_is_measured_truth = false`.** Any Doppler/frequency value that
   appears is SGP4-from-TLE model-derived; no artifact measures true satellite
   Doppler.
3. **IQ-level ceiling.** The strongest claim any artifact set supports is
   *conducted IQ-level RF signal presence with LR-FHSS-like hop structure and an
   X dB TX-ON/OFF margin*. Nothing stronger.
4. **PER/BER/CRC/PDR/gateway-ACK require a decoded receiver log.** None of the
   artifacts below is a decoded RX log. Until a `decoded_rx` artifact with
   `RX seq=.. payload=.. crc=..` lines exists (see
   `docs/lr1121_rx_firmware_gap.md`), no link-layer performance number may be
   reported. This schema deliberately does **not** define a decoded-RX artifact —
   that is a separate, currently-blocked workstream.
5. **Provenance is mandatory.** Every committable artifact must carry, or be
   accompanied by, a `run_manifest.json` reference, a git commit hash, and a
   SHA256 of every raw input it derives from.
6. **Local-only vs committable.** Raw IQ is local-only. Small summaries,
   manifests, figures, SHA256 sidecars, and the claim summary are committable.

---

## 1. Artifact catalogue

For each artifact: **filename · committable? · required fields · supports ·
does NOT support.**

### 1.1 `run_manifest.json` — provenance root

- **Committable:** ✅ yes (small JSON).
- **Required fields:**
  - `run_id` (string, unique, e.g. `mac_conducted_hil_20260615_001`)
  - `timestamp_utc` (ISO-8601)
  - `git_commit` (full SHA of the repo at run time)
  - `operator` (human who supervised; HIL is human-supervised, never unattended)
  - `mode` = `"conducted"` (literal; reject `"radiated"`/`"ota_satellite"`)
  - `tx_board` (e.g. LR1121 board id `1403`)
  - `rx_device` (e.g. `USRP B210`, serial)
  - `cable_path` (free text: coax, attenuator value in dB, optional shielded box)
  - `nominal_center_freq_hz` (integer; must be explicitly set, never a silent
    868 MHz default — see `docs/review/frequency_schema_f0_guard_verification.md`)
  - `frequency_plan` (e.g. `TAIWAN_AS923_REVIEW_REQUIRED` until NCC-confirmed)
  - `ota_transmission_allowed` = `false`
  - `reference_is_measured_truth` = `false`
  - `input_sha256` (map of every input file → SHA256)
  - `artifact_sha256` (map of every emitted committable artifact → SHA256)
- **Supports:** reproducibility, provenance, audit trail.
- **Does NOT support:** any physical result on its own.

### 1.2 `schedule_meta.json` — what was scheduled

- **Committable:** ✅ yes.
- **Required fields:** `run_id`, `n_trials` (≥3 for repeatability), per-trial
  `tx_on_intervals` / `tx_off_intervals` (UTC), `waveform` (e.g.
  `lrfhss_like`), `hop_grid` description, `nominal_center_freq_hz`,
  `ota_transmission_allowed=false`, `reference_is_measured_truth=false`.
- **Supports:** the plan-vs-observed timing correlation; reproducibility.
- **Does NOT support:** that anything was actually emitted (that needs the
  capture analyses).

### 1.3 `tx_config.json` — transmitter configuration

- **Committable:** ✅ yes.
- **Required fields:** `run_id`, `tx_board`, `tx_power_setting` (raw register /
  dBm-as-configured, **not** a calibrated EIRP claim), `waveform`, `hop_pattern`,
  `nominal_center_freq_hz`, `prediction_driven_config_source` (which
  prediction/precomp produced these params), `reference_is_measured_truth=false`.
- **Supports:** that the transmit path was configured by the prediction-driven
  pipeline.
- **Does NOT support:** radiated power / EIRP / link-budget claims.

### 1.4 `uart_tx_log.txt` — terminal-side ground truth

- **Committable:** 🟡 small log, committable after a quick scan for stray
  secrets; otherwise summarise. Prefer committing a curated excerpt.
- **Required content:** per-line timestamp + TX event (`TX start`, `TX end`,
  hop/frame ids if printed). Must align in time with `schedule_meta.json`.
- **Supports:** corroboration that the terminal commanded TX during the detected
  ON intervals (timing correlation only).
- **Does NOT support:** that a packet was *received* or *decoded*; UART TX log is
  transmit-side only, never an RX/PER source.

### 1.5 `usrp_capture_meta.json` — receiver capture metadata

- **Committable:** ✅ yes (metadata only; the IQ itself is separate and local).
- **Required fields:** `run_id`, `sample_rate_hz`, `center_freq_hz`,
  `rx_gain_db`, `capture_start_utc`, `capture_duration_s`, `n_samples`,
  `iq_format` (e.g. `sigmf` / `cf32`), `iq_file_sha256` (hash of the local raw
  IQ so the committed metadata pins the exact uncommitted capture),
  `mode="conducted"`, `reference_is_measured_truth=false`.
- **Supports:** that a capture of stated parameters exists and pins its hash.
- **Does NOT support:** signal presence by itself (needs the analyses below).

### 1.6 `raw_iq_capture.sigmf-data` (or `*.cf32` raw IQ) — raw samples

- **Committable:** ⛔ **NO. Local-only** unless explicitly curated and approved
  later (size + provenance). Treated like `dataraw/` inputs: never committed in
  this pass.
- **Required:** pinned by `iq_file_sha256` in `usrp_capture_meta.json` and the
  sigmf-meta sidecar.
- **Supports:** the underlying evidence for all detection analyses (offline).
- **Does NOT support:** anything until analysed; raw IQ alone is not a result.

### 1.7 `raw_iq_capture.sigmf-meta` — IQ sidecar

- **Committable:** ✅ the small meta sidecar is committable even though the
  `.sigmf-data` is not.
- **Required fields:** SigMF global (`core:datatype`, `core:sample_rate`,
  `core:version`), captures (`core:frequency`, `core:datetime`), and a custom
  block mirroring `usrp_capture_meta.json` provenance + `mode="conducted"` +
  `reference_is_measured_truth=false`.
- **Supports:** provenance/format of the local IQ.
- **Does NOT support:** signal presence by itself.

### 1.8 `on_analysis.json` — TX-ON detection

- **Committable:** ✅ yes.
- **Required fields:** `run_id`, `trial_id`, detection statistic(s)
  (band power, sparse-hop occupancy count, per-bin noise floor), `dc_lo_excluded=true`,
  `image_bins_excluded=true`, `detected_bins`, `validation_status`
  (`signal_detected` / `weak_signal_candidate` / `no_signal`), `center_freq_hz`,
  `reference_is_measured_truth=false`.
- **Supports:** ON-state RF energy / hop-like structure (IQ-level only).
- **Does NOT support:** PER/BER/CRC, decoding, or link quality.

### 1.9 `off_analysis.json` — TX-OFF baseline

- **Committable:** ✅ yes. Same field shape as `on_analysis.json`.
- **Supports:** the noise-floor reference for the ON/OFF margin.
- **Does NOT support:** anything on its own; only meaningful paired with ON.

### 1.10 `on_off_comparison.json` — margin

- **Committable:** ✅ yes.
- **Required fields:** `run_id`, `trial_id`, `on_band_power_db`,
  `off_band_power_db`, `margin_db`, `margin_threshold_db` (pre-registered, e.g.
  6 dB), `pass` (bool), `dc_lo_excluded`, `image_bins_excluded`,
  `reference_is_measured_truth=false`.
- **Supports:** the **X dB TX-ON/OFF margin** statement (e.g. existing reference
  capture reported 9.82 dB).
- **Does NOT support:** any decoding/PER/link claim; margin ≠ link quality.

### 1.11 `repeatability_summary.json` — N≥3 trials

- **Committable:** ✅ yes.
- **Required fields:** `run_id`, `n_trials`, per-trial `margin_db` +
  `validation_status`, aggregate (`min/median/max margin_db`,
  `n_signal_detected`), `reproducible` (bool: all N detected above threshold),
  `reference_is_measured_truth=false`.
- **Supports:** that the IQ-level detection is reproducible (criterion 5 of the
  preflight plan).
- **Does NOT support:** statistical link performance / PER.

### 1.12 `cfo_residual_summary.json` — hop-structure / CFO diagnostic

- **Committable:** ✅ yes (it is a derived summary, not raw IQ).
- **Required fields:** `run_id`, per-burst residual CFO stats (median, p99 of
  |CFO| in Hz), hop-grid orthogonality / adjacent-bin-leakage proxy,
  `is_proxy=true`, `reference_is_measured_truth=false`, explicit note that CFO is
  an RF-quality **proxy**, not PER (consistent with
  `docs/ota_iq_validation_scope.md`).
- **Supports:** hop-structure / residual-CFO **proxy** diagnostic.
- **Does NOT support:** PER/BER/CRC; CFO/EVM proxies are RF-quality only.

### 1.13 `paper_candidate_fig_hw_conducted_iq.png` — candidate figure

- **Committable:** ✅ the figure PNG is committable; the raw IQ behind it is not.
- **Required:** caption must state *conducted, IQ-level, signal-presence, X dB
  margin, not PER, not decoding, not satellite*. Must carry a provenance line
  (run_id + git commit) and an accompanying `.sha256` sidecar.
- **Supports:** a single paper figure for the conducted-HIL diagnostic leg.
- **Does NOT support:** any claim its caption does not literally state; the
  caption is the claim ceiling.

### 1.14 `hardware_claim_summary.md` — human-readable claim ceiling

- **Committable:** ✅ yes.
- **Required content:** one paragraph stating exactly what the run established
  (conducted IQ-level signal presence, margin, reproducibility), an explicit
  *what this is NOT* list (no satellite, no Doppler truth, no PER/BER/CRC/PDR/ACK,
  no decoding), the `run_manifest.json` reference, and the SHA256 table of every
  committable artifact.
- **Supports:** the audited, paste-ready wording for the paper's validation leg.
- **Does NOT support:** anything beyond the literal allowed-wording block.

---

## 2. Committable vs local-only summary table

| Artifact | Committable | Notes |
|---|---|---|
| `run_manifest.json` | ✅ | provenance root |
| `schedule_meta.json` | ✅ | |
| `tx_config.json` | ✅ | configured power only, not EIRP |
| `uart_tx_log.txt` | 🟡 | commit curated excerpt; TX-side only |
| `usrp_capture_meta.json` | ✅ | pins local IQ by SHA256 |
| `raw_iq_capture.sigmf-data` / raw IQ | ⛔ | **local-only** unless curated later |
| `raw_iq_capture.sigmf-meta` | ✅ | small sidecar |
| `on_analysis.json` | ✅ | |
| `off_analysis.json` | ✅ | |
| `on_off_comparison.json` | ✅ | margin |
| `repeatability_summary.json` | ✅ | N≥3 |
| `cfo_residual_summary.json` | ✅ | proxy, not PER |
| `paper_candidate_fig_hw_conducted_iq.png` | ✅ | caption = claim ceiling |
| `hardware_claim_summary.md` | ✅ | + SHA256 table |

---

## 3. SHA256 / provenance requirement

- Every committable artifact ships with its SHA256 either inside
  `run_manifest.json::artifact_sha256` or as a `<artifact>.sha256` sidecar.
- `run_manifest.json::input_sha256` hashes every input (schedule, tx_config, and
  the local raw IQ).
- The committed `usrp_capture_meta.json::iq_file_sha256` pins the uncommitted
  raw IQ so its identity is auditable without committing it.
- No artifact is considered evidence unless its provenance chain terminates at a
  `run_manifest.json` with a git commit hash.

---

## 4. Exact ALLOWED wording (post-run, only if success criteria met)

- "A **conducted (cabled)** LR1121→USRP B210 capture shows **IQ-level** RF signal
  presence with **LR-FHSS-like hop structure** and a **<X> dB TX-ON/OFF margin**,
  reproducible across N≥3 trials."
- "Residual-CFO and hop-orthogonality are reported as **RF-quality proxies**."
- "The UART TX log **corroborates** the detected ON intervals (timing only)."
- "Receiver-side decoding and PER remain **future gateway-level work**."
- "All Doppler is **model-derived (SGP4 from real TLE)**;
  `reference_is_measured_truth=false`."

## 5. Exact FORBIDDEN wording

- ❌ "live satellite", "satellite link", "downlink/uplink to orbit", "OTA to
  satellite", "satellite validation", "satellite link demonstrated"
- ❌ "measured Doppler", "Doppler truth", "true satellite Doppler", "measured link"
- ❌ "PER", "BER", "CRC", "PDR", "packet delivery", "decoded packet",
  "gateway ACK" (except as an explicit *future-work / not-measured* non-claim)
- ❌ "hardware validates the model", "hardware-validated", "validated on hardware"
- ❌ "guarantee", "guaranteed improvement", "worst-case bound", "can only help"
- ❌ "successful RF capture proves LR-FHSS", "successful LR-FHSS reception",
  "standards-compliant LR-FHSS decoding"
- ❌ any wording implying radiated transmission, a standards-compliant receiver,
  or that the learned residual improves real BLACK KITE Doppler

---

## 6. What NO artifact in this schema can ever support

Even with a fully successful conducted-HIL run, the artifacts above **cannot**
support, and the paper must not claim:

1. Live-satellite contact, tracking, or reception.
2. Measured Doppler truth (reference stays SGP4-from-TLE, model-derived).
3. PER / BER / CRC / PDR / gateway ACK / packet delivery (no decoded RX log).
4. Standards-compliant LR-FHSS decoding.
5. That a learned residual correction improves real BLACK KITE Doppler (the real
   data shows the opposite — the evidence gate closes; see
   `docs/review/black_kite_residual_evidence_gate.md`).
6. Any net-improvement guarantee or worst-case bound.

These require a separate decoded-RX / gateway workstream that is currently
**blocked** (`docs/lr1121_rx_firmware_gap.md`).
