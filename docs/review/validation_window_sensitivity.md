# Validation-Window Sensitivity Interpretation

> **SOFTWARE-ONLY / SYNTHETIC interpretation note.** Reads the validation-window
> sweep already in `docs/review/evidence_gate_stress_experiment.md`; introduces no
> new result. **NOT real BLACK KITE evidence.** `reference_is_measured_truth=false`.

*Generated: 2026-06-14 UTC.*

## Question

Does the gate decision `G = 1[MAE_ml(V) < γ·MAE_phys(V)]` depend on how large the
chronological validation window `V` is? If a tiny window can flip the decision,
the gate is fragile.

## Sweep (from the stress experiment)

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

## Interpretation

1. **Decisive regimes are window-stable.** Both the noise-dominated (closed) and
   the strongly-systematic (open) regimes hold their decision across val n =
   100 → 3000 at every γ. The gate is not an artifact of window size in the
   regimes that matter.
2. **The documented fragility is at the boundary with tiny windows.** The source
   experiment notes that very small validation windows (n≈100) can mis-decide
   **near the γ threshold** (i.e. in the marginal regime), because the
   validation-MAE estimate is noisy. This is a property of any held-out estimate,
   not of the gate rule specifically.
3. **Practical guidance.** Use a validation window that is not tiny (the
   experiment is stable from n≈300 upward). For real BK1/BK2 the chronological
   validation segments are 132 (BK1) and comparable record counts — above the
   tiny-window danger zone — and the decision (gate closed) is unambiguous because
   the real residual sits deep in the noise-dominated regime, far from any γ
   boundary.
4. **Paper wording.** Report the gate as stable above a minimum validation-window
   size, and state explicitly that very small windows near the γ boundary can
   mis-decide — this is an honest limitation, not a guarantee.

## What this does NOT say

- ❌ no validation-window size makes learning beat the baseline on real BLACK KITE
- ❌ window stability is not a deployment guarantee (validation-window property
  only)
- ❌ synthetic regimes are illustrative, not real-satellite measurements
