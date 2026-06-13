# Paper Figure / Table Plan — Physics-First Evidence-Gated D2S LR-FHSS

*Planning only. The paper (`paper/icc_main.tex`) is **not** edited by this
document. Each item is bound to a source artifact and to what it can / cannot
support. `reference_is_measured_truth=false`. Wording is governed by
`docs/review/claim_evidence_matrix.md`.*

*Generated: 2026-06-14 UTC.*

---

## Readiness legend

- **READY NOW** — source artifact exists; figure/table can be drafted from it now.
- **WAIT (Mac HIL)** — needs the future conducted-HIL run
  (`docs/review/mac_hil_preflight_plan.md`).
- **WAIT (decoder)** — needs a decoded RX log; currently blocked
  (`docs/lr1121_rx_firmware_gap.md`).
- **Enter main paper before HW run?** — whether this item can go into the rewrite
  *now* (yes) or must wait.

---

## Figure 1 — Physics-first evidence-gated architecture

- **Source artifact:** conceptual; formalised in
  `docs/review/paper_reframing_blueprint.md` §5 and the gate equation
  `G = 1[MAE_ml(V) < γ·MAE_phys(V)]`; integration point
  `controller/doppler_precomp.py`.
- **Claim supported:** the architecture — SGP4 physics baseline as default,
  learned residual as a *gated option*, gate fed by chronological held-out
  validation.
- **Claim NOT supported:** that the learned branch helps on real data (it does
  not); any measured/RF/satellite claim.
- **Readiness:** READY NOW (schematic). **Enter main paper before HW run:** ✅ yes.
- **Missing result:** none (diagram only).

## Table 1 — Real BK1/BK2 negative residual result

- **Source artifact:** `docs/review/black_kite_1_target_specific_residual_experiment.md`
  (BK1 target, 8–168 h) + `docs/review/black_kite_tle_history_residual_experiment.md`
  (BK1→BK2 cross-sat); compact form in
  `docs/review/bk_negative_result_compact.{md,csv}` (this pass).
- **Claim supported:** learned residual does **not** beat the SGP4/stale-TLE zero
  baseline at any tested staleness; residual is near zero-mean and unpredictable.
- **Claim NOT supported:** learning improves real BLACK KITE Doppler; measured
  Doppler truth; any net-improvement.
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes (this is the
  paper's central real-data result).
- **Missing result:** none.

## Table 2 — Synthetic stress gate behaviour

- **Source artifact:** `docs/review/evidence_gate_stress_experiment.md`
  (`tools/evidence_gate_stress_experiment.py`); compact form in
  `docs/review/gate_stress_compact.{md,csv}` (this pass).
- **Claim supported:** in a controlled synthetic regime, the gate opens under
  dominant systematic drift (ML reduces simulated MAE/guard/outage) and stays
  closed in a noise-dominated regime.
- **Claim NOT supported:** real-satellite benefit; that real BLACK KITE has such a
  systematic regime (it does not); PER/BER/CRC.
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes, **iff**
  labelled CONTROLLED SYNTHETIC SIMULATION.
- **Missing result:** none; must keep the synthetic label.

## Figure 2 — Gate close/open behaviour

- **Source artifact:** the γ-sweep and val-window sweep tables in
  `docs/review/evidence_gate_stress_experiment.md`; interpretation in
  `docs/review/gate_threshold_interpretation.md` and
  `docs/review/validation_window_sensitivity.md` (this pass).
- **Claim supported:** gate decision vs γ and vs validation-window size; closed on
  noise-dominated (real-like), open on systematic.
- **Claim NOT supported:** that opening implies real benefit; guarantee/bound.
- **Readiness:** READY NOW (plot from existing sweep numbers). **Enter main paper
  before HW run:** ✅ yes.
- **Missing result:** none (a raw per-point CSV export from the tool would make
  the plot reproducible without re-running; optional).

## Figure 3 — Mac conducted-HIL diagnostic (TO BE FILLED AFTER HARDWARE RUN)

- **Source artifact:** future `paper_candidate_fig_hw_conducted_iq.png` +
  `on_off_comparison.json` + `repeatability_summary.json`
  (`docs/review/hil_artifact_schema.md`).
- **Claim supported (after run):** conducted IQ-level RF signal presence,
  LR-FHSS-like hop structure, X dB TX-ON/OFF margin, reproducible N≥3.
- **Claim NOT supported:** satellite link; measured Doppler; PER/BER/CRC/PDR/ACK;
  decoding.
- **Readiness:** WAIT (Mac HIL). **Enter main paper before HW run:** ❌ no — leave
  a captioned placeholder; an existing reference capture (9.82 dB) may be cited as
  prior conducted evidence but the *new* figure waits.
- **Missing result:** the conducted-HIL capture itself.

## Table 3 — Compact claim-evidence matrix

- **Source artifact:** `docs/review/claim_evidence_matrix.md` (condense to a
  paper-sized table); claim-to-artifact index in
  `docs/review/claim_to_artifact_index.md` (this pass).
- **Claim supported:** the discipline mapping (claim → artifact → allowed/forbidden
  wording).
- **Claim NOT supported:** anything beyond what each row states.
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes (appendix or
  methods).
- **Missing result:** none.

## Table 4 — Limitations table

- **Source artifact:** `docs/review/paper_reframing_blueprint.md` §7 +
  `docs/review/claim_evidence_matrix.md` standing rules.
- **Claim supported:** explicit scope/limitation statements (model-derived
  reference, negative real result, synthetic-only positive, proxies only,
  validation-window-only gate property, no live-satellite/RF/PER).
- **Claim NOT supported:** n/a (it is the limitations).
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes.
- **Missing result:** none.

## Appendix Table A — Artifact provenance / SHA256

- **Source artifact:** SHA256 blocks self-reported by each `docs/review/*.md`,
  the data inventory `docs/review/black_kite_family_spacetrack_inventory.md`, and
  (post-run) `run_manifest.json::artifact_sha256`.
- **Claim supported:** reproducibility / provenance.
- **Claim NOT supported:** any physical result.
- **Readiness:** READY NOW for software artifacts; the hardware rows are WAIT
  (Mac HIL). **Enter main paper before HW run:** ✅ partial (software rows now;
  hardware rows after run).
- **Missing result:** hardware-artifact hashes (post-run).

## Optional Table B — Gate threshold (γ) / validation-window sensitivity

- **Source artifact:** γ-sweep + val-window sweep in
  `docs/review/evidence_gate_stress_experiment.md`;
  `docs/review/gate_threshold_interpretation.md`,
  `docs/review/validation_window_sensitivity.md` (this pass).
- **Claim supported:** gate stability across γ and val-window size; tiny windows
  (n≈100) can mis-decide near threshold.
- **Claim NOT supported:** that any γ guarantees net improvement; real-data
  applicability of the open regime.
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes (optional /
  appendix).
- **Missing result:** none; an explicit per-row CSV from the tool would tighten
  reproducibility (optional, software-only).

## Optional Table C — Missing evidence and claim status

- **Source artifact:** `docs/review/missing_experiment_gap_analysis.md` (this pass).
- **Claim supported:** an honest "what is and is not yet evidenced" map.
- **Claim NOT supported:** n/a.
- **Readiness:** READY NOW. **Enter main paper before HW run:** ✅ yes (strengthens
  the limitations / honesty of the paper).
- **Missing result:** none.

---

## Roll-up: what can enter the main paper *now* vs after the Mac HIL run

| Item | Ready now? | In paper before HW run? |
|---|---|---|
| Fig 1 architecture | ✅ | ✅ |
| Table 1 real BK negative result | ✅ | ✅ |
| Table 2 synthetic stress | ✅ | ✅ (synthetic-labelled) |
| Fig 2 gate close/open | ✅ | ✅ |
| Fig 3 conducted-HIL diagnostic | ❌ | ❌ (placeholder; cite prior 9.82 dB only) |
| Table 3 claim-evidence matrix | ✅ | ✅ |
| Table 4 limitations | ✅ | ✅ |
| Appendix A provenance/SHA256 | ✅ (sw) / ❌ (hw) | ✅ partial |
| Opt B γ / val-window sensitivity | ✅ | ✅ |
| Opt C missing-evidence status | ✅ | ✅ |

**Bottom line:** the physics-first / evidence-gate / negative-real-result paper
can be fully drafted now; only the *new* conducted-HIL figure (Fig 3) and the
hardware provenance rows wait for the human-supervised Mac run. Nothing in this
plan requires a live satellite, a decoder, or any forbidden claim.
