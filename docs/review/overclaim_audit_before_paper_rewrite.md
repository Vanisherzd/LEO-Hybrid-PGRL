# Overclaim Audit — Before Paper Rewrite

*Audit only. **No source file was edited to produce this document.** It greps
`README.md`, `docs/`, and `paper/` for dangerous phrases and reports each match's
risk and a suggested replacement. The paper (`paper/icc_main.tex`) is **not**
edited in this pass. `reference_is_measured_truth=false`.*

*Generated: 2026-06-14 UTC. Method: case-insensitive `grep -niE` over the three
trees for the phrase set below. Cross-reference: `docs/review/claim_evidence_matrix.md`.*

---

## 0. Method and phrase set

Searched (case-insensitive, word-bounded where shown):

```
guarantee · worst-case · measured Doppler · live satellite · satellite validation
hardware validates · hardware-validated · successful RF capture · \bPER\b · \bBER\b
\bCRC\b · \bPDR\b · gateway ACK · can only help · real BLACK KITE improves
measured link · packet delivery · decoded packet · hardware validation
satellite link demonstrated · live RF
```

**Grep artefact to know about:** with `-i`, `\bPER\b` also matches the ordinary
English token **"per"** (as in *per-axis*, *per-pass*, *per-burst*, *per-sample*).
Those are **false positives**, not packet-error-rate claims, and are tagged
`FALSE-POSITIVE` below.

**Risk scale:** `low` = already an explicit non-claim / hedged proxy / legitimate
engineering term · `medium` = currently safe but wording could be misread out of
context and should be tightened during rewrite · `high` = an un-hedged dangerous
claim that must be fixed (none found).

**Headline result: no `high`-risk un-hedged claim was found.** Every PER/BER/CRC/
satellite/hardware-validated occurrence in `README.md`, `docs/`, and
`paper/icc_main.tex` is either an explicit *not-claimed / out-of-scope / future-
work* statement, a labelled proxy, or a legitimate engineering use of
"worst-case margin". A small number of `medium` items are flagged for tightening.

---

## 1. `README.md`

| Line | Matched phrase | Risk | Why | Suggested replacement | Action |
|---|---|---|---|---|---|
| 40 | "hardware validation" (dir comment `semtech_validation/`) | low | Directory description, not a result claim. | leave, or rename comment to "hardware signal-detection harness" | leave as limitation context |
| 84 | "PER … ⛔ Not claimed" | low | Explicit non-claim row in scope table. | none | leave (good) |
| 135 | "No measured PER, packet delivery, or receiver decoding is claimed." | low | Explicit non-claim. | none | leave (good) |
| 141 | "Outage proxy 5.0%→1.7% … (not PER)" | low | Labelled control proxy. | none | leave (good) |
| 144 | "Stage 6 PER harness … no measured PER without a decoded receiver log" | low | Harness-only, explicit blocker. | none | leave (good) |
| 146 | "IQ-level RF signal detection only … does not imply … PER" | low | Explicit ceiling. | none | leave (good) |
| 158 | "RF-quality proxies … real-world PER requires a standards-compliant decoder" | low | Explicit ceiling. | none | leave (good) |

**README verdict:** claim-disciplined. No fix required before rewrite. README is
**not** edited in this pass (out of allowed-edit scope anyway).

---

## 2. `paper/icc_main.tex` (NOT EDITED — audit only)

| Line | Matched phrase | Risk | Why | Suggested replacement | Action |
|---|---|---|---|---|---|
| 49 | "worst-case margins" (abstract) | low | Engineering term ("fixed worst-case margins waste energy"), not a guarantee claim. | none | leave |
| 49 | "conducted LR1121-to-USRP B210 capture provides IQ-level evidence … 9.82 dB TX-ON/OFF margin … decoding and PER … future" | low | Already conducted + IQ-level + future-work hedged. | none | leave; keep "conducted" + "IQ-level" adjacent to every capture mention |
| 60 | "worst-case timing and frequency uncertainty" | low | Engineering term. | none | leave |
| 62 | "fixed worst-case margins" | low | Engineering term. | none | leave |
| 162 | "fixed worst-case margin" | low | Engineering term. | none | leave |
| 206,215,217,223 | "per-axis / per-pass" (matched as "per") | FALSE-POSITIVE | The token "per", not PER. | none | ignore |
| 236 | "deterministic worst-case margin"; "outage proxy" | low | Engineering term + labelled proxy. | none | leave |
| 252 | "we report only proxy RF-quality indicators … and do not measure PER" | low | Explicit non-claim. | none | leave (good) |
| 285 | "RF-quality indicators, not LR-FHSS PER" | low | Explicit non-claim. | none | leave (good) |
| 306,310 | "outage proxy"; per-sample (matched "per") | low / FALSE-POSITIVE | Labelled proxy / "per" token. | none | leave |
| 357 | "Online adaptation, packet decoding, and PER measurement are future gateway-level work." | low | Explicit future-work. | none | leave (good) |
| 385 | "conducted … capture confirms … emits observable LR-FHSS-like structure (9.82 dB margin) … future work will close … with measured PER" | medium | "confirms" is acceptable for IQ-level signal presence, but during rewrite ensure it reads "confirms the configured transmit path emits observable LR-FHSS-like structure" (signal presence), **not** "confirms the link/decoding". Keep "conducted" + the future-PER hedge. | keep verb scoped to *signal presence*; never "confirms the link" | watch during rewrite |

**`paper/icc_main.tex` verdict:** no un-hedged dangerous claim. The only items to
watch on rewrite are (a) keep "conducted" + "IQ-level" glued to every capture
mention, and (b) keep the verb "confirms/provides evidence" scoped to *signal
presence*, never to a link/decode/PER. **File untouched this pass** (baseline
SHA256 `1819aeac7ce12fc7d9a179b54272d2e7447eaa1089980ca46252c2bb00525d84`).

---

## 3. `paper/tables/` and `paper/figures/`

| File:line | Matched phrase | Risk | Why | Action |
|---|---|---|---|---|
| `main_results.tex:45` | "control proxies (not PER)" | low | Labelled proxy. | leave (good) |
| `main_results.json:88` | "UART-confirmed packets per ON capture" | medium | "packets" near hardware can be misread as RX/packet-delivery; it is a TX-side / UART-side count. | rename metric to "UART-confirmed TX bursts per ON capture" during a future tables pass (not edited now) |
| `main_results.json:94` | "IQ-level RF signal detection only; no LR-FHSS decoding or PER" | low | Explicit ceiling. | leave (good) |
| `main_results.json:194` | "NOT a standards-compliant PER" | low | Explicit non-claim. | leave (good) |
| `ablation_baselines.tex:20` | "not a PER result" | low | Explicit non-claim. | leave (good) |
| `figures/hardware/README.md:30` | "not PER" | low | Explicit non-claim. | leave (good) |
| `figures/hardware/fig_hw_lrfhss_repeatability.json:3` | "IQ-level RF signal detection only; no … PER" | low | Explicit ceiling. | leave (good) |
| Binary figure PDFs/PNGs matching "PER" | (inside rasterised captions) | low | Caption text in image; matches the disciplined wording. | leave |

**Watch item (`medium`):** `main_results.json:88` "UART-confirmed packets per ON
capture". Not edited now. Flag for the future tables/figures pass.

---

## 4. `docs/` (excluding this audit's own new files)

All `docs/` matches fall into the explicit-non-claim / scope-definition / blocker
category and are `low` risk. Representative (not exhaustive):

| File | Nature | Risk |
|---|---|---|
| `docs/lr1121_rx_firmware_gap.md` | Defines the decoded-RX / PER **blocker**; states PER is unavailable without a decoded RX log. | low (this is the guardrail) |
| `docs/hardware_claim_checklist.md` | Defines when "hardware-validated" may be used (only with full artifact chain) + forbidden mappings. | low (guardrail) |
| `docs/ota_iq_validation_scope.md` | IQ-level proxy scope; "not PER / not decoding / not CRC". | low (guardrail) |
| `docs/lr1121_lrfhss_signal_detection.md` | "not PER", "not hardware validation", detection-only. | low (guardrail) |
| `docs/mac_m2_hardware_setup.md` | "Do NOT label as hardware-validated until all items checked." | low (guardrail) |
| `docs/real_hardware_connection_checklist.md` | IQ-only → no PER; decoded RX log → PER allowed. | low (guardrail) |
| `docs/risk_aware_control_stage4_results.txt` | "does not show … real hardware packet delivery." | low (non-claim) |

**`docs/` verdict:** the guardrail documents *contain* the dangerous phrases by
design (they define what must not be claimed). No fix needed.

---

## 5. Phrases searched that produced ZERO matches anywhere

These dangerous phrases do **not** appear (good — confirms discipline):

- "live satellite" / "satellite validation" / "satellite link demonstrated" /
  "live RF" — **0** assertive matches
- "measured Doppler" / "measured link" — **0**
- "hardware validates" (assertive) — **0** (only "hardware-validated" inside
  guardrail "do not label as" contexts)
- "successful RF capture" — **0**
- "can only help" — **0**
- "real BLACK KITE improves" — **0**
- "guarantee" / "guaranteed improvement" — **0** assertive (only "worst-case
  margin" engineering usage)
- "\bBER\b" / "gateway ACK" assertive — **0**

---

## 6. Summary and pre-rewrite gate

| Severity | Count | Disposition |
|---|---|---|
| high (un-hedged dangerous claim) | **0** | none to fix |
| medium (tighten on rewrite) | **2** | `icc_main.tex:385` verb scope; `main_results.json:88` "packets"→"TX bursts" |
| low (already disciplined / engineering term / explicit non-claim) | many | leave |

**Gate decision:** the corpus is claim-safe enough that the paper rewrite can
proceed against `docs/review/claim_evidence_matrix.md` without first editing
source files. The two `medium` items are wording-tightening tasks for the rewrite
itself, not blockers. **No source file was edited in this audit.**
