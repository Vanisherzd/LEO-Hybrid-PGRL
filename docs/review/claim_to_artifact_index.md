# Claim → Artifact Index

> **SOFTWARE-ONLY DERIVED INDEX.** A flat lookup from each claim the paper may
> make to the concrete artifact that backs it, the support level, and the
> readiness stage. Condensed from `docs/review/claim_evidence_matrix.md`. No new
> result is introduced here. `reference_is_measured_truth=false`.

*Generated: 2026-06-14 UTC.*

| Claim | Backing artifact(s) | Support | Stage |
|---|---|---|---|
| Real Space-Track BLACK KITE TLE provenance (NORAD 66741, 68474) | `docs/review/black_kite_family_spacetrack_inventory.md`; raw under `dataraw/` (not committed) | ✅ | ready now |
| Model-derived SGP4 Doppler replay from real TLE | `docs/review/black_kite_2_real_data_evidence_chain.md` | ✅ (model-derived) | ready now |
| Learned residual does NOT beat SGP4/stale-TLE baseline (BK1 target-specific 8–168 h) | `docs/review/black_kite_1_target_specific_residual_experiment.md`; `bk_negative_result_compact.{md,csv}` | ✅ (negative) | ready now |
| Learned residual does NOT transfer BK1→BK2 | `docs/review/black_kite_tle_history_residual_experiment.md`; `bk_negative_result_compact.{md,csv}` | ✅ (negative) | ready now |
| Evidence gate closes on real data | `docs/review/black_kite_residual_evidence_gate.md` | ✅ | ready now |
| Gate opens only under synthetic systematic drift | `docs/review/evidence_gate_stress_experiment.md`; `gate_stress_compact.{md,csv}` | 🟡 (synthetic) | ready now |
| Gate γ-threshold behaviour / stability | `docs/review/evidence_gate_stress_experiment.md`; `gate_threshold_interpretation.md` | 🟡 (synthetic) | ready now |
| Gate validation-window stability | `docs/review/evidence_gate_stress_experiment.md`; `validation_window_sensitivity.md` | 🟡 (synthetic) | ready now |
| Guard-band / outage / energy PROXIES | `docs/review/evidence_gate_stress_experiment.md` | 🟡 (proxy) | ready now |
| Gate validation-window inequality property (not a guarantee) | `docs/review/paper_reframing_blueprint.md` §5 | ✅ (scoped) | ready now |
| Conducted IQ-level RF signal presence + hop structure + X dB margin | prior reference capture (9.82 dB); future `on_off_comparison.json`, `repeatability_summary.json` (`hil_artifact_schema.md`) | 🟡 (conducted bench) | prior cite now / new = WAIT Mac HIL |
| Residual-CFO / hop-orthogonality proxy | future `cfo_residual_summary.json`; scope `docs/ota_iq_validation_scope.md` | 🟡 (proxy) | WAIT Mac HIL |
| PER / BER / CRC / PDR / gateway ACK | none (blocked) `docs/lr1121_rx_firmware_gap.md` | ⛔ not claimed | WAIT decoder/gateway |
| Live-satellite validation / measured Doppler truth | none | ⛔ not claimed | OUT OF SCOPE |
| Net-improvement guarantee / worst-case bound / "can only help" | none | ⛔ not claimed | OUT OF SCOPE |

**Support legend:** ✅ supported · 🟡 proxy/synthetic/conducted-only · ⛔ explicit
non-claim. **Stage legend:** ready now · WAIT Mac HIL · WAIT decoder/gateway ·
OUT OF SCOPE.

*Every "reference" Doppler above is SGP4-from-real-TLE, model-derived, never an RF
measurement.*
