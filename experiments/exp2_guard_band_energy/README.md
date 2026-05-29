# Experiment 2 — Guard-Band Energy Tradeoff

**Purpose:** Prove that PGRL uncertainty-driven guard bands reduce idle energy versus conservative fixed guards.

## Baselines
1. Fixed conservative guard (30 ms)
2. SGP4-only mean guard
3. PGRL mean-only guard
4. PGRL uncertainty-aware guard (proposed)

## Metrics
- receiver-on time [s]
- guard overhead fraction [%]
- missed opportunity probability
- energy per successful opportunity [J]