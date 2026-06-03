# exp6_robustness — Robustness Evaluation Skeleton

**Status: PLANNED / NOT YET RUN.** This directory defines the robustness
evaluation protocol for the PGRL predictor. No experiment has been executed
and `results.json` contains no numeric results.

The goal is to stress the PGRL residual corrector under conditions that differ
from the nominal trace-driven evaluation, to characterize where the calibrated
uncertainty and Doppler/timing accuracy degrade. All four cases below are
planned only.

## Robustness cases

1. **TLE age perturbation** (`tle_age_days`)
   Re-evaluate predictions using TLE sets of increasing age relative to the
   prediction epoch. SGP4 error grows with TLE staleness; this case measures
   whether PGRL degrades gracefully and whether its uncertainty remains
   calibrated as TLE age increases.

2. **Beacon loss duration** (`beacon_loss_s`)
   Simulate intervals during which no beacon/ephemeris update is available,
   forcing prediction over progressively longer unobserved horizons. Measures
   accuracy and calibration as a function of the gap length.

3. **Satellite-held-out split** (`held_out_satellites`)
   Train on a subset of satellites and evaluate on satellites never seen during
   training. Measures cross-satellite generalization (distinct from the temporal
   split used in the main results).

4. **Orbital regime shift** (`regime`)
   Evaluate across orbital regimes (e.g., altitude / inclination bands) that
   differ from the training regime. Measures robustness to distribution shift in
   orbital dynamics.

## Reproducibility

- Config: `config.yaml` (the four cases are parametrized there with a fixed seed).
- Results: `results.json` (`validation_type: "planned"`; every case has
  `"status": "planned"` and no numeric results — `null` / `"not evaluated"`).

When these experiments are run, numeric results and the `validation_type` label
must be filled in by the run script; until then nothing here should be cited as
a measured result.
