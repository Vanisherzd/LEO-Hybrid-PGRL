# Missing-Experiment Gap Analysis — Before Paper Rewrite

*Planning / audit only. No experiment was run unattended, no hardware was used,
no model was trained to produce this document. Each gap is classified and, where
it cannot be safely closed now, a runbook/TODO is given instead of running it.
`reference_is_measured_truth=false`.*

*Generated: 2026-06-14 UTC.*

---

## Classification legend

- `READY_NOW_FROM_EXISTING_ARTIFACTS` — evidence already exists; only needs
  summarising/plotting.
- `SAFE_SOFTWARE_ONLY_CHECK_POSSIBLE_NOW` — a bounded software-only check could
  strengthen it without hardware/training; runbook given.
- `WAIT_FOR_MAC_CONDUCTED_HIL` — needs the human-supervised conducted run.
- `WAIT_FOR_DECODER_OR_GATEWAY` — needs a decoded RX log / gateway; currently
  blocked.
- `OUT_OF_SCOPE_DO_NOT_CLAIM` — must never be claimed by this paper.

---

## Gap table

| # | Gap | Class | Status / what exists | Action |
|---|---|---|---|---|
| 1 | **BK1/BK2 real TLE negative residual evidence** | READY_NOW_FROM_EXISTING_ARTIFACTS | `black_kite_1_target_specific_residual_experiment.md` (8–168 h) + `black_kite_tle_history_residual_experiment.md` (BK1→BK2). Learned residual loses at every staleness. | Summarised in `bk_negative_result_compact.{md,csv}` (this pass). No new run. |
| 2 | **Evidence gate closes on real data** | READY_NOW_FROM_EXISTING_ARTIFACTS | `black_kite_residual_evidence_gate.md` records the gate closing for BK1 (BLOCKED) and BK2 (negative control). | Cited in claim-to-artifact index. No new run. |
| 3 | **Synthetic stress gate opens under systematic drift** | READY_NOW_FROM_EXISTING_ARTIFACTS | `evidence_gate_stress_experiment.md`: closed on `fresh_low_residual`, open on `extreme_systematic`. | Summarised in `gate_stress_compact.{md,csv}` (this pass). No new run. |
| 4 | **Gate threshold (γ) sensitivity** | READY_NOW_FROM_EXISTING_ARTIFACTS | γ-sweep table (0.90–1.00) already in `evidence_gate_stress_experiment.md`. | Interpreted in `gate_threshold_interpretation.md` (this pass). Optional: tool could emit a per-row CSV (software-only). |
| 5 | **Validation-window sensitivity** | READY_NOW_FROM_EXISTING_ARTIFACTS | val-n sweep (100/300/1000/3000) already in `evidence_gate_stress_experiment.md`. | Interpreted in `validation_window_sensitivity.md` (this pass). |
| 6 | **LR-FHSS guard/outage/energy proxy** | READY_NOW_FROM_EXISTING_ARTIFACTS (proxy) | Guard = 2·p99\|err\|, outage = P(\|err\|>500 Hz), energy ∝ guard, in stress experiment. Already labelled PROXY. | Keep proxy label. No new run. **Not** PER/BER/CRC. |
| 7 | **Conducted IQ signal-presence diagnostic** | WAIT_FOR_MAC_CONDUCTED_HIL | A prior reference capture reported 9.82 dB ON/OFF margin (cited in paper). New conducted run pending. | Run per `mac_hil_preflight_plan.md`, emit artifacts per `hil_artifact_schema.md`. Human-supervised. |
| 8 | **CFO residual / hop-structure diagnostic** | WAIT_FOR_MAC_CONDUCTED_HIL | Scope/proxy defined in `docs/ota_iq_validation_scope.md`. Needs the new capture. | Emit `cfo_residual_summary.json` (proxy) from the conducted run. **Not** PER. |
| 9 | **Decoded RX / PER** | WAIT_FOR_DECODER_OR_GATEWAY | Blocked: no decoded RX log exists. `docs/lr1121_rx_firmware_gap.md`. | Do **not** claim PER/BER/CRC/PDR. Future gateway/decoder workstream only. |
| 10 | **Live-satellite validation** | OUT_OF_SCOPE_DO_NOT_CLAIM | No satellite contacted/tracked/received. | Never claim live satellite, satellite link, measured Doppler truth. |
| 11 | **Artifact provenance** | READY_NOW_FROM_EXISTING_ARTIFACTS (sw) / WAIT_FOR_MAC_CONDUCTED_HIL (hw) | Each `docs/review/*.md` self-reports SHA256; data inventory exists. Hardware hashes pending the run. | Software provenance table now; hardware rows post-run via `run_manifest.json`. |
| 12 | **Figure/table readiness** | READY_NOW_FROM_EXISTING_ARTIFACTS (mostly) | `paper_figure_table_plan.md` maps each item. Only Fig 3 (conducted-HIL) waits. | Draft all but Fig 3 now. |
| 13 | **Paper old-narrative risk** | SAFE_SOFTWARE_ONLY_CHECK_POSSIBLE_NOW | `overclaim_audit_before_paper_rewrite.md` found 0 high-risk, 2 medium watch-items. | Address the 2 medium items during rewrite. No source edited now. |
| 14 | **Whether `paper/icc_main.tex` must wait** | (decision) | See §"Does the paper need hardware first?" below. | Rewrite the software-evidence paper now; hold only the conducted-HIL figure. |

---

## Safe-software-only checks that *could* be run now (runbooks, not executed)

These are bounded, hardware-free, training-free, and write only small derived
summaries. **None was run unattended in this pass beyond what is explicitly
listed in the completion report.** Listed as runbooks for the human-supervised
session.

### R1 — Per-row CSV export of the γ / val-window sweep (software-only)

- **Why:** makes Fig 2 / Opt-Table-B reproducible without re-deriving numbers
  from prose.
- **Constraint:** must read only the *existing* stress-experiment outputs or
  re-run `tools/evidence_gate_stress_experiment.py` **only if** it can be forced
  to CPU and a small sample count and < 5 min, writing CSV under `docs/review/`.
  The tool currently defaults to CUDA/GPU and 20000 samples/regime → **do not run
  as-is** (violates no-GPU / training constraints). Treat as a TODO requiring a
  CPU/quick flag.
- **Status:** runbook only; not run (would need a CPU/quick mode added first).

### R2 — Recompute SHA256 of all `docs/review/*.md` for the provenance appendix

- **Why:** Appendix A provenance table.
- **Constraint:** `shasum -a 256` only; no hardware, no training.
- **Status:** **safe to run now** — included in this pass's completion report.

### R3 — Static consistency check of cross-references between review docs

- **Why:** catch broken `docs/review/...` links / stale filenames before rewrite.
- **Constraint:** grep/find only.
- **Status:** **safe to run now** — included in this pass.

---

## Does the paper (`paper/icc_main.tex`) need hardware first?

**No — for the software-evidence core.** The physics-first / evidence-gate /
real-data-negative-result narrative is fully supported by existing software
artifacts (gaps 1–6, 11-sw, 12, 13). The rewrite can proceed against
`docs/review/claim_evidence_matrix.md`.

**Yes — only for the *new* conducted-HIL figure (Fig 3) and hardware provenance
rows (gap 7, 8, 11-hw).** Those wait for the human-supervised Mac run. The paper
may cite the *prior* conducted reference capture (9.82 dB) as existing conducted
evidence, but must not present a new hardware result before the run.

**Per the task constraint, `paper/icc_main.tex` is NOT edited in this pass.** The
decision above is a recommendation for the later, human-supervised rewrite.

---

## Hard non-claims (restate)

- No live-satellite contact, no measured Doppler truth, no radiated transmission.
- No PER/BER/CRC/PDR/gateway-ACK (blocked: `docs/lr1121_rx_firmware_gap.md`).
- No claim that a learned residual improves real BLACK KITE Doppler (the gate
  closes on real data).
- No guarantee / worst-case bound / "can only help" / hardware-validated
  satellite link.
