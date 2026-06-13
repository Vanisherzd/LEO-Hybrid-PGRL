# Autonomous Pre-HIL Preparation Summary

*Autonomous software-only pass. No hardware/RF command was run, no firmware
touched, no `dataraw/` written, no model trained, no GPU used. `paper/icc_main.tex`
was NOT edited. Nothing was committed or staged. `reference_is_measured_truth=false`.*

*Generated: 2026-06-14 UTC. Branch: `experiment-bk2-tle-residual` (HEAD `2a8b6f4`).*

---

## 1. What was completed

Created the missing readiness + audit documentation and compact derived summaries
under `docs/review/` (all software-only, all claim-disciplined):

- `hil_artifact_schema.md` — full Mac conducted-HIL artifact schema (filenames,
  required fields, SHA256/provenance, committable vs local-only, per-artifact
  can/cannot-support, allowed/forbidden wording).
- `overclaim_audit_before_paper_rewrite.md` — grep audit of `README.md`, `docs/`,
  `paper/` for dangerous phrases; **0 high-risk**, 2 medium watch-items; source
  files NOT edited.
- `paper_figure_table_plan.md` — figure/table plan with source artifact, claim
  supported/not-supported, readiness, and before/after-HW status.
- `missing_experiment_gap_analysis.md` — every remaining gap classified
  (`READY_NOW` / `SAFE_SOFTWARE_ONLY` / `WAIT_MAC_HIL` / `WAIT_DECODER` /
  `OUT_OF_SCOPE`); runbooks where unsafe to run.
- `bk_negative_result_compact.{md,csv}` — compact real BK1/BK2 negative result.
- `gate_stress_compact.{md,csv}` — compact synthetic gate-behaviour summary.
- `claim_to_artifact_index.md` — flat claim→artifact lookup.
- `gate_threshold_interpretation.md` — γ-sweep interpretation (γ=0.95 default).
- `validation_window_sensitivity.md` — val-window stability interpretation.
- `paper_rewrite_checklist.md` — paste-ready rewrite checklist.

## 2. What was checked (software-only)

- `python3 -m py_compile` on the three experiment tools → **COMPILE_OK**.
- Stale path `tools/evidence_gate_stress_eperiment.py` → **0 matches** (no stale ref).
- Typos `peformance` / `pape` / `pper` / mis-spelled "guarante" / "eperiment" in
  `docs/review/` → **0 matches**.
- Overclaim grep across `README.md` + `docs/` + `paper/` → **0 high-risk
  un-hedged claims**; all PER/satellite/hardware-validated occurrences are
  explicit non-claims, labelled proxies, or engineering "worst-case margin" usage.
- Cross-reference check: every `docs/review/*.md|csv` filename referenced by a
  review doc **exists** (no broken readiness-doc references).
- `paper/icc_main.tex` SHA256 unchanged
  (`1819aeac7ce12fc7d9a179b54272d2e7447eaa1089980ca46252c2bb00525d84`) → untouched.
- `git diff --cached` empty; no `dataraw/` path staged.

## 3. What was NOT touched

- `paper/icc_main.tex` (verified by hash) and the rest of `paper/`.
- `README.md`, hardware scripts, firmware, `dataraw/`.
- No file was staged or committed.
- No hardware/RF/UART/USRP/UHD/GNU Radio/ST-Link/OpenOCD/flash/TX/replay/capture
  command was issued.
- No model training; no GPU use.

## 4. What remains before the paper rewrite

- All software-evidence figures/tables (Fig 1, Tables 1–4, Opt B/C, Appendix A
  software rows) are **ready now** — see `paper_figure_table_plan.md`.
- During rewrite, address the 2 medium audit watch-items: keep capture verbs
  scoped to *signal presence* (`icc_main.tex:385`); rename
  `main_results.json:88` "packets"→"TX bursts".
- The rewrite of `paper/icc_main.tex` itself is intentionally deferred to a
  human-supervised session (out of scope for this pass).

## 5. What remains before the Mac conducted-HIL run

- Execute the run per `mac_hil_preflight_plan.md` (human-supervised, conducted/
  cabled only).
- Emit artifacts per `hil_artifact_schema.md`; keep raw IQ local-only.
- Fill Fig 3 + hardware provenance rows afterward.
- Decoded RX / PER stays blocked (`docs/lr1121_rx_firmware_gap.md`) — do not claim.

## 6. Human-supervised next steps

1. Review the 12 new `docs/review/` files.
2. (Optional) Add a CPU/quick mode to `tools/evidence_gate_stress_experiment.py`
   so a per-row γ/val-window CSV can be exported without GPU (runbook R1 in
   `missing_experiment_gap_analysis.md`).
3. When ready, attach hardware on the Mac and run the conducted-HIL pass under
   supervision.
4. Begin the `paper/icc_main.tex` rewrite using `paper_rewrite_checklist.md` and
   `claim_evidence_matrix.md`.

## 7. Commit recommendation (DO NOT COMMIT in this pass)

When a human approves, the 13 new `docs/review/` files (12 listed above +
`autonomous_pre_hil_summary.md`) are safe to commit together — they are
software-only, contain no `dataraw/` content, and do not modify the paper.
Suggested message:

```
docs(review): add HIL artifact schema, overclaim audit, gap analysis, and
pre-HIL summaries
```

No staging or commit was performed by this autonomous pass.
