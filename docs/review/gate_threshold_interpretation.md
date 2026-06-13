# Gate Threshold (γ) Interpretation

> **SOFTWARE-ONLY / SYNTHETIC interpretation note.** Reads the γ-sweep already in
> `docs/review/evidence_gate_stress_experiment.md`; introduces no new result.
> **NOT real BLACK KITE evidence.** `reference_is_measured_truth=false`.

*Generated: 2026-06-14 UTC.*

## The gate

```
G = 1   if   MAE_ml(V) < γ · MAE_phys(V),   γ ∈ (0, 1]      (enable ML)
G = 0   otherwise                                            (use SGP4 physics)
```

γ is a **safety-vs-opportunism knob**, not a performance guarantee. Smaller γ =
stricter = ML must beat physics by a wider margin to be enabled (conservative).
γ→1 = lenient = ML enabled on any apparent gain, including noise.

## γ-sweep decisions (val n = 3000, from the stress experiment)

| Regime | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
| fresh_low_residual (noise-dominated, **real-like**) | closed | closed | closed | closed |
| moderate_staleness | closed | closed | OPEN | OPEN |
| extreme_systematic (dominant drift) | OPEN | OPEN | OPEN | OPEN |

## Interpretation

1. **Robust extremes.** The noise-dominated regime stays **closed** for all γ, and
   the strongly-systematic regime stays **OPEN** for all γ. The gate's two decisive
   regimes are γ-insensitive — the decision is driven by the data, not the knob.
2. **γ matters only at the boundary.** The `moderate_staleness` regime flips
   closed→OPEN between γ=0.95 and γ=0.99. This is exactly where ML's apparent gain
   (test MAE 60.2 vs baseline 63.2, ≈5%) is small; γ decides whether a ~5% margin
   is "enough" to enable learning.
3. **Recommended operating point: γ = 0.95.** It keeps the baseline in the
   marginal regime (accepts a small missed opportunity rather than enabling a
   learner that has not decisively proven itself) and is far from both decisive
   regimes' flip points. The paper should report γ=0.95 as the default and present
   the full sweep so the trade-off is explicit.
4. **Real-data consequence.** The real BLACK KITE residual sits in the noise-
   dominated regime, so the gate closes for **every** γ in the sweep. The γ choice
   does not rescue learning on real data — there is no learnable structure to
   enable.

## What this does NOT say

- ❌ no γ makes learning beat the baseline on real BLACK KITE data
- ❌ γ is not a net-improvement guarantee; the gate property holds only on the
  validation window, not under deployment distribution shift
- ❌ synthetic regimes are illustrative, not real-satellite measurements
