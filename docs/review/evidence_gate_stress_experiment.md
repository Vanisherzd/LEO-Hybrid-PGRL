# Evidence-Gate Stress Experiment — CONTROLLED SIMULATION
*Generated: 2026-06-13 16:28:30 UTC  |  `tools/evidence_gate_stress_experiment.py`*

> **THIS IS A CONTROLLED SYNTHETIC SIMULATION. IT IS NOT REAL BLACK KITE
> EVIDENCE.** It does not use any Space-Track TLE, any measured signal, or any
> hardware. `reference_is_measured_truth = false`. Its sole purpose is to
> characterise the *evidence-gate mechanism* — i.e. the residual-structure
> regime under which a learned correction is correctly enabled or disabled.

## Relationship to the real BLACK KITE evidence

| Source | Result | Gate outcome |
|---|---|---|
| **Real BK2** (`tools/bk_tle_residual_experiment.py`) cross-satellite | learned residual worse than zero baseline | gate **closes** (learning disabled) |
| **Real BK1** (`tools/bk1_target_specific_residual_experiment.py`) target-specific, 8–168 h | learned residual worse than zero baseline at every staleness | gate **closes** (learning disabled) |
| **This file** controlled stress simulation | demonstrates the *condition* under which the gate would open | mechanism only — no real-satellite claim |

The real-data residual is zero-mean and unpredictable, which is exactly the
`fresh_low_residual` (noise-dominated) regime below — and the gate correctly
closes there. The gate opens only in a strongly systematic regime that the real
BLACK KITE TLE history does **not** exhibit.

## Setup

- Compute: cuda (torch GPU MLP); 20000 synthetic samples/regime, 6 features.
- Residual model: `r = A·systematic(x) + N(0, sigma)`, `systematic` a fixed nonlinear function of the features (learnable). A, sigma chosen per regime to bracket gate behaviour; not fitted to any satellite.
- Chronological split per regime: train 12000 / val 4000 / test 4000.
- Correctors: SGP4-only baseline (predict 0) · always-on ML (MLP) · evidence-gated · oracle (perfect = 0 error, upper bound).
- Gate: `G = 1 if MAE_ML_val < gamma · MAE_SGP4_val else 0`.
- Frequency-miss tolerance proxy `F_TOL = 500` Hz; guard-band proxy = 2·p99(|error|); energy/overhead proxy ∝ guard-band width.

## Per-regime held-out test (always-on ML vs baseline)

| Regime | A [Hz] | sigma [Hz] | baseline MAE/RMSE | ML MAE/RMSE | winner | guard base→ML [Hz] | outage base→ML |
|---|---|---|---|---|---|---|---|
| fresh_low_residual | 0 | 12 | 9.558 / 11.984 | 9.608 / 12.052 | baseline better | 62 -> 61 | 0.000 -> 0.000 |
| moderate_staleness | 22 | 75 | 63.174 / 79.453 | 60.176 / 75.459 | ML better | 412 -> 387 | 0.000 -> 0.000 |
| extreme_systematic | 2000 | 90 | 1859.296 / 2440.950 | 99.748 / 151.157 | ML better | 14120 -> 950 | 0.814 -> 0.008 |

*Oracle (later-reference Doppler) = 0 error in all regimes by construction (upper bound on achievable correction).*

## Gate decision (OPEN = enable ML) — sweep over gamma (val n = largest)

| Regime | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
| fresh_low_residual (val n=3000) | closed | closed | closed | closed |
| moderate_staleness (val n=3000) | closed | closed | OPEN | OPEN |
| extreme_systematic (val n=3000) | OPEN | OPEN | OPEN | OPEN |

## Gate stability vs validation sample count

**fresh_low_residual** (expected: closed):

| val n | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
| 100 | closed | closed | closed | closed |
| 300 | closed | closed | closed | closed |
| 1000 | closed | closed | closed | closed |
| 3000 | closed | closed | closed | closed |

**extreme_systematic** (expected: open):

| val n | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
| 100 | OPEN | OPEN | OPEN | OPEN |
| 300 | OPEN | OPEN | OPEN | OPEN |
| 1000 | OPEN | OPEN | OPEN | OPEN |
| 3000 | OPEN | OPEN | OPEN | OPEN |

## Evidence-gated held-out test (γ=0.95, val n = largest)

| Regime | SGP4-only MAE | always-on ML MAE | **gated MAE** | gate | gated guard [Hz] | gated outage |
|---|---|---|---|---|---|---|
| fresh_low_residual | 9.558 | 9.608 | 9.558 | closed | 62 | 0.000 |
| moderate_staleness | 63.174 | 60.176 | 63.174 | closed | 412 | 0.000 |
| extreme_systematic | 1859.296 | 99.748 | 99.748 | OPEN | 950 | 0.008 |

The gated controller defaults to the physics baseline and adopts ML only when
the validation evidence clears the gamma margin (`MAE_ML_val < gamma·MAE_SGP4_val`).
With gamma < 1 it is deliberately conservative: in the moderate regime ML is
marginally better on test (60.2 vs 63.2) yet the gate at gamma=0.95 keeps the
baseline, accepting a small missed opportunity in exchange for not enabling a
learner that has not decisively proven itself. In the extreme regime the margin
is enormous, so the gate opens and the gated error collapses to the ML error.
This is a safety-first / opportunism trade-off controlled by gamma, not a net-
improvement guarantee.

## Interpretation

1. **Noise-dominated (fresh):** ML cannot beat the SGP4-only baseline; the gate
   closes for all sensible gamma. This matches the real BLACK KITE TLE result.
2. **Systematic (extreme):** a large learnable drift exists; ML reduces test MAE,
   guard-band, and outage substantially; the gate opens.
3. **Moderate:** outcome depends on gamma — a stricter gamma (0.90) is more
   conservative (keeps baseline), a lenient gamma (1.00) enables ML. This is the
   tuning knob between safety and opportunism.
4. The gate is stable across validation sample counts once the validation window
   is not tiny; very small val windows (n=100) can mis-decide near the threshold.

## Limitations (controlled simulation)

1. **Synthetic only.** No Space-Track TLE, no measured Doppler, no hardware/RF/UART/replay. `reference_is_measured_truth = false`.
2. The systematic function and regime amplitudes are chosen to bracket gate behaviour; they are NOT claimed to match any real satellite's residual structure.
3. This file does NOT demonstrate that learned residual correction improves real BLACK KITE Doppler — the real-data experiments show the opposite (gate closes).
4. Guard-band, outage, and energy/overhead are documented PROXIES, not link-budget or PER/BER/CRC measurements.
5. No live-satellite, gateway-ACK, or net-improvement guarantee is claimed.
