# Paper Rewrite Checklist

> Built from `docs/review/claim_evidence_matrix.md`,
> `docs/review/overclaim_audit_before_paper_rewrite.md`, and
> `docs/review/paper_figure_table_plan.md`. This is a checklist for the **future,
> human-supervised** rewrite of `paper/icc_main.tex`. The paper is **NOT** edited
> in this pass. `reference_is_measured_truth=false`.

*Generated: 2026-06-14 UTC.*

## Before touching `paper/icc_main.tex`

- [ ] Re-read `docs/review/claim_evidence_matrix.md` — it governs all wording.
- [ ] Confirm `overclaim_audit_before_paper_rewrite.md` still shows **0 high-risk**
      items (re-run the grep if source changed).
- [ ] Decide narrative order: (B) real negative result → gate → (A)+(C) synthetic
      characterisation → (D) conducted-HIL as a separate engineering check.

## Claims to ADD / strengthen (all backed, ready now)

- [ ] Rigorous **negative result on real data** (Table 1; `bk_negative_result_compact.md`).
- [ ] **Evidence Gate** definition + validation-window property, explicitly NOT a
      guarantee (`paper_reframing_blueprint.md` §5).
- [ ] **Synthetic stress** characterisation, labelled CONTROLLED SYNTHETIC
      (Table 2; `gate_stress_compact.md`).
- [ ] **γ and validation-window sensitivity** (Opt Table B;
      `gate_threshold_interpretation.md`, `validation_window_sensitivity.md`).
- [ ] **Claim-evidence matrix** as an appendix/methods table (Table 3).
- [ ] **Limitations** table (Table 4) and **missing-evidence status** (Opt Table C).
- [ ] **Provenance / SHA256** appendix for software artifacts (Appendix A).

## Wording watch-items (from the audit)

- [ ] Keep "**conducted**" + "**IQ-level**" glued to every capture mention.
- [ ] Keep verbs ("confirms / provides evidence") scoped to **signal presence**,
      never to a link / decode / PER (watch `icc_main.tex:385`).
- [ ] If editing tables later, rename `main_results.json:88` "UART-confirmed
      packets per ON capture" → "...TX bursts per ON capture".
- [ ] Every "reference"/"truth" Doppler paired with
      `reference_is_measured_truth=false`.

## Must NOT appear (hard non-claims)

- [ ] No "measured Doppler truth", "live satellite", "satellite validation".
- [ ] No "PER/BER/CRC/PDR/gateway ACK" except as explicit future-work non-claims.
- [ ] No "guarantee", "worst-case bound" (engineering "worst-case margin" is OK),
      "can only help", "hardware-validated satellite link".
- [ ] No claim that a learned residual improves real BLACK KITE Doppler.

## Hold for the human-supervised Mac HIL run

- [ ] Fig 3 conducted-HIL diagnostic — placeholder only until the run
      (`mac_hil_preflight_plan.md`, `hil_artifact_schema.md`).
- [ ] Hardware provenance rows in Appendix A — fill from `run_manifest.json` post-run.
- [ ] The paper may cite the **prior** 9.82 dB conducted reference capture as
      existing conducted evidence; do not present a *new* hardware result early.

## Blocked (do not attempt for this paper)

- [ ] Decoded RX / PER (`lr1121_rx_firmware_gap.md`) — future gateway/decoder work.
- [ ] Live-satellite validation — out of scope.
