# Evidence-Gate Stress — Compact Summary

> **SOFTWARE-ONLY / CONTROLLED SYNTHETIC SIMULATION DERIVED SUMMARY.**
> Re-summarised from `docs/review/evidence_gate_stress_experiment.md`. No new run,
> no hardware. **NOT real BLACK KITE evidence.** `reference_is_measured_truth=false`.
> Data: `gate_stress_compact.csv`.

*Generated: 2026-06-14 UTC. Source:
`tools/evidence_gate_stress_experiment.py`. MAE in Hz. guard = 2·p99|err|;
outage = P(|err|>500 Hz); gate `G = 1[MAE_ml(V) < γ·MAE_phys(V)]`. Gated MAE at
γ=0.95, val n = largest.*

| Regime | A [Hz] | σ [Hz] | Baseline MAE | ML MAE | Winner | Gated MAE (γ=0.95) | Gate | Guard base→ML [Hz] | Outage base→ML |
|---|---|---|---|---|---|---|---|---|---|
| fresh_low_residual | 0 | 12 | 9.558 | 9.608 | baseline | 9.558 | closed | 62→61 | 0.000→0.000 |
| moderate_staleness | 22 | 75 | 63.174 | 60.176 | ml | 63.174 | closed | 412→387 | 0.000→0.000 |
| extreme_systematic | 2000 | 90 | 1859.296 | 99.748 | ml | 99.748 | open | 14120→950 | 0.814→0.008 |

## One-line result

In a controlled synthetic simulation, the gate **closes** in the noise-dominated
`fresh_low_residual` regime (which matches the real BLACK KITE TLE residual) and
**opens** only in the `extreme_systematic` regime, where a large learnable drift
exists and ML cuts simulated MAE, guard-band, and outage proxies sharply. The
real BLACK KITE data does **not** exhibit the systematic regime; hence the gate
closes on real data.

## What this does NOT say

- ❌ not real-satellite evidence; the positive ML benefit exists **only** in this
  synthetic simulation
- ❌ not a claim that real BLACK KITE has a systematic regime (it does not)
- ❌ guard / outage are **proxies**, not PER/BER/CRC/link-budget
- ❌ no net-improvement guarantee; the gate gives a validation-window inequality
