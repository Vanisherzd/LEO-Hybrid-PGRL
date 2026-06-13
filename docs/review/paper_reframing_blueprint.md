# Paper Reframing Blueprint — Physics-First Evidence-Gated D2S LR-FHSS

*Draft narrative only. The main paper is NOT edited by this document. Wording is
constrained by `docs/review/claim_evidence_matrix.md`. `reference_is_measured_truth=false`
throughout; no live-satellite, RF, PER/BER/CRC, or gateway-ACK claim is made.*

---

## 1. Title options

1. **Physics-First Evidence-Gated Uplink Control for LR-FHSS Direct-to-Satellite IoT**
2. **When Not to Learn: An Evidence Gate for Doppler Pre-Compensation in D2S LR-FHSS IoT**
3. **Open-Loop by Default — Conditionally Enabling Learned Doppler Correction with a Held-Out Evidence Gate**
4. **Safe-by-Default Doppler Compensation: Physics Baseline with Evidence-Gated Learning for LEO IoT Uplinks**

Primary recommendation: **#1** (states the architecture); **#2** as a provocative alt.

---

## 2. Revised abstract (draft)

> Direct-to-satellite (D2S) LR-FHSS IoT uplinks must pre-compensate large LEO
> Doppler shifts from a satellite ephemeris that is, in practice, a *stale* TLE.
> A natural idea is to learn a data-driven residual correction on top of the
> open-loop SGP4 prediction. We test this rigorously on **real Space-Track TLE
> history** for two BLACK KITE LEO satellites (NORAD 66741, 68474) using strict
> chronological, leakage-free splits, and find that a learned residual model does
> **not** improve over the open-loop SGP4 / stale-TLE baseline at any tested
> staleness (8–168 h): the inter-TLE Doppler residual is essentially zero-mean
> and unpredictable. Rather than discard learning, we make it **conditional**. We
> introduce an **Evidence Gate** that enables the learned corrector only when a
> chronological held-out validation window shows it beats the physics baseline by
> a margin γ, and otherwise falls back to open-loop SGP4/stale-TLE compensation.
> On the real BLACK KITE data the gate correctly *closes* (learning disabled). In
> a controlled synthetic stress study we characterise the regime in which the gate
> *opens*: when a dominant systematic drift exists, the gated corrector reduces
> simulated frequency error, guard-band, and outage proxies, while a noise-
> dominated regime keeps the gate shut. The result is a safe-by-default uplink
> controller whose learned component is only enabled after chronological
> validation evidence shows improvement over the physics baseline. All Doppler references are model-derived (SGP4 from real TLE),
> not measured; no live-satellite, link-layer, or hardware-radiated results are
> claimed.

Key abstract guardrails: "does not improve", "conditional", "correctly closes",
"controlled synthetic", "proxies", "model-derived, not measured". No "guarantee",
no "worst-case bound", no "validation on satellite".

---

## 3. Revised introduction — problem statement

- D2S LR-FHSS uplinks need accurate Doppler pre-compensation; narrow LR-FHSS sub-
  channels make residual frequency error costly (wider guard bands, missed
  channels, energy/airtime overhead).
- The on-board/edge device only has a **stale TLE**; open-loop SGP4 propagation of
  a stale TLE is the deployable physics baseline.
- Tempting hypothesis: a learned model can predict the stale-TLE→truth Doppler
  residual and shrink the error.
- **Gap we expose:** on real TLE history this hypothesis fails — the residual is
  unpredictable. Naïvely deploying an always-on learner would *increase* error.
- **Our stance:** treat learning as an *option to be earned*, gated on held-out
  evidence, with the physics model as the safe default.

---

## 4. Revised contributions

1. **A rigorous negative result on real data.** Using real Space-Track TLE history
   for two BLACK KITE satellites and strict chronological splits, we show learned
   Doppler residual correction does not beat the open-loop SGP4/stale-TLE baseline
   at 8–168 h staleness (target-specific and cross-satellite).
2. **The Evidence Gate.** A simple, auditable decision rule
   `G = 1[MAE_ML_val < γ·MAE_SGP4_val]` that enables learning only on proven
   held-out improvement and defaults to physics otherwise. We integrate it at the
   uplink Doppler-precompensation point (`controller/doppler_precomp.py`).
3. **A controlled characterisation of when the gate opens.** A synthetic stress
   study spanning noise-dominated → systematic regimes, with γ and validation-
   window sweeps, showing the gate closes on unlearnable residuals and opens on
   dominant systematic drift, with guard-band/outage/energy proxies.
4. **A claim–evidence discipline.** An explicit matrix binding every claim to an
   artifact and to allowed/forbidden wording, keeping model-derived results
   strictly separated from any measured/RF claim (of which there are none).

---

## 5. Evidence Gate — mathematical formulation

Let `D_ref(t)` be the later/newer-TLE SGP4 Doppler (the *reference*; model-derived,
`reference_is_measured_truth=false`). Let the open-loop physics prediction from the
stale TLE be `f_phys(t)` and the learned correction be
`f_ml(t) = f_phys(t) + r̂(x_t)`, where `r̂` is the residual model and `x_t` the
feature vector.

On a chronological **validation** window `V` (strictly later than train, strictly
earlier than test):

```
MAE_phys(V) = (1/|V|) Σ_{t∈V} | D_ref(t) − f_phys(t) |
MAE_ml(V)   = (1/|V|) Σ_{t∈V} | D_ref(t) − f_ml(t)   |
```

**Gate:**
```
G = 1   if   MAE_ml(V) < γ · MAE_phys(V),     γ ∈ (0, 1]
G = 0   otherwise
```

**Deployed corrector:**
```
f̂(t) = G · f_ml(t) + (1 − G) · f_phys(t)
```

**Property (validation-window only, honest scope):** if `G=1` then by definition
`MAE_ml(V) < γ·MAE_phys(V) ≤ MAE_phys(V)`, so the gated corrector's validation MAE
does not exceed the physics baseline's; if `G=0` it equals it. Therefore on the
validation window `MAE_gated(V) ≤ MAE_phys(V)`.

**What this is NOT:** this is a property *on the validation window*. It is **not** a
held-out guarantee and **not** a worst-case bound — distribution shift between
validation and deployment can still make a gate-opened learner underperform on
unseen data. γ trades safety (small γ → conservative, rarely opens) against
opportunism (γ→1 → opens on any apparent gain, including noise). The paper must
state this scope explicitly and must not claim a net-improvement guarantee.

---

## 6. Evaluation structure

| Stage | Question | Artifact | Outcome |
|---|---|---|---|
| (A) Simulation potential benefit | Can a learned residual *ever* help, and under what structure? | `tools/evidence_gate_stress_experiment.py` | Yes, only in a dominant-systematic synthetic regime (controlled, not real) |
| (B) Real BLACK KITE negative evidence | Does learning help on real TLE history? | `tools/bk1_target_specific_residual_experiment.py`, `tools/bk_tle_residual_experiment.py` + reports | No — baseline wins at all staleness; residual is zero-mean/unpredictable |
| (C) Evidence-gate stress experiment | Does the gate open/close correctly across regimes, γ, val-window? | `docs/review/evidence_gate_stress_experiment.md` | Closes on noise-dominated (real-like), opens on systematic; stable above tiny val windows |
| (D) Conducted RF diagnostic (later) | Does the precompensator behave on a cabled bench? | *(separate conducted-mode bench step; not in this software evidence set)* | Conducted/cabled only — **never** live-satellite or radiated-to-orbit |

Narrative order in the paper: motivate (B) the real negative result first, then
present the gate as the principled response, then (A)+(C) to characterise its
behaviour in a controlled setting, and finally (D) as a separate conducted-mode
engineering check — explicitly not a satellite-link validation.

---

## 7. Limitations wording (paste-ready)

- All "reference"/"truth" Doppler is **model-derived** (SGP4 from real TLE):
  `reference_is_measured_truth=false`. No RF measurement of Doppler was made.
- The real-data finding is a **negative** result for learned residual correction
  on two BLACK KITE satellites; it is not a universal claim for all satellites or
  all staleness/feature regimes.
- The positive benefit of learning is shown **only in a controlled synthetic
  simulation** and must not be read as real-satellite evidence.
- Guard-band, outage, and energy figures are **proxies** (`guard = 2·p99|err|`,
  `outage = P(|err|>F_TOL)`); they are not link-budget, PER, BER, CRC, or
  gateway-ACK measurements.
- The Evidence Gate guarantees only a **validation-window** inequality, not a
  held-out or worst-case bound; deployment under distribution shift is not
  guaranteed to improve.
- No live-satellite contact, no radiated transmission, no hardware-in-the-loop
  satellite link. Any RF bench work referred to elsewhere is **conducted/cabled**
  and separate.
