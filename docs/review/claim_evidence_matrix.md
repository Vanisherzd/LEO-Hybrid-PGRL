# Claim–Evidence Matrix — Physics-First Evidence-Gated D2S LR-FHSS

*Generated: 2026-06-14 UTC. Governs wording for the reframed paper. Every claim
below is bound to a concrete artifact and to explicit allowed / forbidden
phrasings. `reference_is_measured_truth = false` everywhere: all "reference"
Doppler is model-derived (SGP4 from real TLE), never an RF measurement.*

## Legend

- **Supported?** — ✅ supported by an artifact · 🟡 supported only as a *proxy* or
  *controlled simulation* · ⛔ explicitly NOT claimed.
- Artifact SHA256 values are listed in the run's final status block and each
  report self-reports its own hash.

## Matrix

| # | Claim | Evidence artifact | Supported? | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|
| 1 | **Real Space-Track BLACK KITE TLE provenance** — real GP/TLE history for NORAD 66741 (BK1) and 68474 (BK2) was retrieved and inventoried | `dataraw/spacetrack/black_kite_1_66741/gp_history_66741.json` (415 recs), `…/black_kite_2_68474/gp_history_68474.json` (184 recs); `docs/review/black_kite_family_spacetrack_inventory.md` | ✅ | "Real Space-Track GP/TLE history for NORAD 66741 / 68474 was retrieved, counted, and epoch-ranged." | "measured Doppler", "live satellite contact", "downlink capture" |
| 2 | **Path A model-derived Doppler replay chain** — open-loop SGP4 Doppler computed at 868 MHz from real TLE, exported as replay CSV | `docs/review/black_kite_2_real_data_evidence_chain.md`; `dataraw/pgrl/black_kite_2_68474_replay_doppler.csv` | ✅ (as model-derived) | "SGP4-derived open-loop Doppler replay computed from real TLE; `reference_is_measured_truth=false`." | "measured Doppler truth", "live downlink Doppler", "868 MHz approved for satellite TX" |
| 3 | **BK1/BK2 residual-learning blockers** — learned TLE-history residual correction does NOT beat the SGP4/stale-TLE baseline on chronological held-out real data | `tools/bk_tle_residual_experiment.py` + `docs/review/black_kite_tle_history_residual_experiment.md` (BK1→BK2); `tools/bk1_target_specific_residual_experiment.py` + `docs/review/black_kite_1_target_specific_residual_experiment.md` (BK1 target, 8–168 h) | ✅ (negative result) | "On chronological held-out real BK1/BK2 TLE history, a learned Doppler residual model did not improve over the open-loop SGP4/stale-TLE baseline at any tested staleness; the residual is near zero-mean and unpredictable." | "learning improves real BLACK KITE Doppler", "guarantees net improvement", "residual correction works" |
| 4 | **Evidence gate disables harmful ML** — the gate closes on real data, defaulting to the physics baseline and preventing a worse-than-baseline learner | `docs/review/black_kite_residual_evidence_gate.md`; gate `G=1[MAE_ML_val < γ·MAE_SGP4_val]` evaluated in artifacts of row 3 | ✅ | "The evidence gate disables the learned correction when chronological held-out validation shows no improvement, falling back to open-loop SGP4/stale-TLE compensation." | "guarantees net improvement", "worst-case bound" (unless proven), "optimal control" |
| 5 | **Controlled stress case where the gate may enable ML** — in a synthetic regime with a dominant systematic drift, the gate opens and ML reduces simulated error | `tools/evidence_gate_stress_experiment.py` + `docs/review/evidence_gate_stress_experiment.md` (synthetic only) | 🟡 (controlled simulation) | "In a controlled synthetic simulation, a dominant-systematic regime causes the gate to open and the learned corrector to reduce simulated MAE/guard-band/outage; a noise-dominated regime keeps the gate closed." | "real satellite", "measured", "BLACK KITE evidence", presenting the simulation as real-data validation |
| 6 | **LR-FHSS guard-band / energy / outage proxy** — guard-band width, frequency-miss outage, and energy/overhead trends from the controlled simulation | `docs/review/evidence_gate_stress_experiment.md` (proxies: guard = 2·p99\|err\|, outage = P(\|err\|>500 Hz)) | 🟡 (proxy only) | "guard-band / outage / energy PROXIES derived from a controlled simulation." | "PER", "BER", "CRC", "measured link performance", "gateway ACK", "link-budget validated" |
| 7 | **Conducted hardware diagnostic only** — any RF bench result is conducted (cabled), not radiated and not to a satellite; treated as a separate later step | (separate conducted-mode bench diagnostic; not produced or relied on in this software evidence set) | 🟡 (conducted bench only, separate) | "A conducted (cabled) RF bench diagnostic, if reported, is over-the-wire only and separate from the orbital-software evidence." | "live satellite RF", "radiated over-the-air to satellite", "satellite link PER/BER", "gateway-confirmed reception" |
| 8 | **No live-satellite / PER / BER / CRC claim** — explicitly NOT claimed anywhere | (none — explicit non-claim across all reports) | ⛔ (not claimed) | "No live-satellite contact, link-layer performance, or gateway acknowledgement is claimed or evaluated." | "live satellite validation", "PER/BER/CRC", "gateway ACK", "end-to-end link demonstrated" |

## Standing wording rules (apply to all of the above)

1. Never write **"measured Doppler truth"** — the reference is SGP4-from-real-TLE,
   model-derived; always pair with `reference_is_measured_truth=false`.
2. Never write **"guarantees net improvement"** or **"worst-case bound"** unless a
   proof is included. The gate gives a *validation-window* inequality only, not a
   held-out or distribution-shift guarantee.
3. Never write **"live satellite validation"**, **"PER/BER/CRC"**, or
   **"gateway ACK"** — none were performed.
4. The controlled simulation (rows 5–6) must always be labelled synthetic and may
   never be cited as real BLACK KITE evidence.
5. Real BLACK KITE evidence (rows 1–4) is the *negative* result + the gate that
   acts on it. The positive ML benefit exists only in simulation (row 5).
