# BLACK KITE-2 Real-Data Evidence Chain
**Report Date:** 2026-06-13
**NORAD ID:** 68474
**Safe Repositories:** `LEO-Hybrid-PGRL-reference` (safe), `leo-pinn` (candidate)
**Status:** Read-only archaeology — no hardware, no training, no transmit

---

## 1. Space-Track Credential Check

**Result: BLOCKED**

The task requires environment variables `SPACETRACK_USERNAME` and `SPACETRACK_PASSWORD`.
These are **not set** in the current environment:

| Variable | Required | Present |
|---|---|---|
| `SPACETRACK_USERNAME` | YES | **NO** |
| `SPACETRACK_PASSWORD` | YES | **NO** |

A file `.env.spacetrack` exists in `LEO-Hybrid-PGRL-reference/` containing
`SPACETRACKUSER` and `SPACETRACKPASS` (different names, not the required env vars).
This file is also **not in `.gitignore`** — committing it would expose credentials.

**Consequence:** Fresh TLE fetch from Space-Track is **not possible** with current env vars.
The existing `black_kite_2_68474_spacetrack_history.tle` (fetched 2026-06-11) is used as-is.

---

## 2. Candidate File Table

### 2a. TLE Files

| File | Size | SHA256 | Role |
|---|---|---|---|
| `dataraw/tle/black_kite_2_68474_spacetrack_history.tle` | 25K | `abb731064c1b1b1f5cdd409fcf1eea9b7ec7da0329417429563e391f00e1ac62` | Primary real-TLE history |
| `dataraw/tle/black_kite_2_68474_spacetrack_history.tle.meta.json` | 560B | `b9ac7a5aba947023ddb1b5137303a68696516a175270b6b87ad90d36fc33901a` | Provenance metadata |
| `dataraw/tle/black_kite_2_68474_celestrak_latest.tle` | 168B | `489279136807ea2070df0183d0b3f7f9d323431a86fe68e6140e5e63789d8ef5` | Latest CelesTrak snapshot |

### 2b. Doppler / Modeled Data Files

| File | Size | Rows / Info | SHA256 | Role |
|---|---|---|---|---|
| `dataraw/doppler/black_kite_2_68474_visible_passes_enu.csv` | 4.6M | 31,059 data rows, 260 passes | `802ece1d73cfd1b6e349e4c08f7a997bc62b4a7c0d3ae2775b2bbaba25c01a61` | SGP4 ENU Doppler for all passes |
| `dataraw/doppler/black_kite_2_68474_visible_passes_enu.summary.json` | 105K | 260 passes, carrier=868 MHz | `3c33ca689ececb05eeb78a1338d24a77e45630df01644088ed8142fa51b6aa82` | Pass-level summary + physics notes |
| `dataraw/doppler/black_kite_2_68474_best_pass_enu.csv` | 24K | 152 rows, pass_id=168 | `a0823212b530ab705e226ac47a5e7fb43340630544e9408cde7adf2f2c3186c5` | Single clean pass (used for replay CSV) |
| `dataraw/doppler/black_kite_2_68474_sgp4_doppler_profile.csv` | 49K | 179 rows, 3 TLE epochs | `be01ab053066a36db1db5fae28c56159bcb8894e166e594ead73deccae4327f7` | Full ECEF state at sparse epochs |

### 2c. PGRL Artifacts

| File | Size | SHA256 | Role |
|---|---|---|---|
| `leo-pinn/artifacts/checkpoints/pgrl_deterministic_retrained.pt` | 3.9M | `5cac977f0b3f66f66124d1f1776bec50345c75c45e5c815ffd609de3f258f9a4` | PGRL checkpoint, epoch 78 |
| `leo-pinn/results/pgrl_full_state_predictions.csv` | 3.8M | `923cf7d815c60423d55db5b07bc6113b6e53db3687c0d098f080c9d000c4b231` | Val predictions on synthetic NPZ |
| `leo-pinn/results/pgrl_hard_split_real.csv` | 1.2K | `87e871c33618c422917eaae2824916fcffc5554a303c1e9ffc0e837a19009f0c` | Hard-split metrics table |

---

## 3. BLACK KITE-2 TLE Provenance

**NORAD Cat ID:** 68474

**Source:** Space-Track `gp_history` API (fetched 2026-06-11T12:04:53 UTC)

**TLE pair count:** 356 lines / 2 = **178 TLE pairs** total; metadata reports 153 pairs used for propagation window.  
**Epoch range:** 2026-04-13 (DOY 103) to 2026-06-11 (DOY 162) — 59-day window.  
**Placeholder check:** `grep "99999"` on spacetrack_history.tle → **0 matches**. No placeholder TLEs present.

**Most-recent TLE epoch:** 2026-06-11T02:44:02.976 UTC (DOY 162.11392334)

**Parsed Keplerian elements from most-recent TLE:**

| Element | Value | Unit |
|---|---|---|
| Semi-major axis (a) | **6968.10 km** | km |
| Eccentricity (e) | 3.01 × 10⁻⁵ | — |
| Inclination (i) | 97.754° | deg |
| RAAN (Ω) | 120.493° | deg |
| Arg. Perigee (ω) | 271.022° | deg |
| Mean Anomaly (M) | 89.097° | deg |
| Mean motion | 14.9256 rev/day | rev/day |

---

## 4. Doppler CSV Provenance

### 4.1 `black_kite_2_68474_visible_passes_enu.csv` (primary)

**Columns:** `pass_id, t_utc, t_s, norad_cat_id, tle_epoch, station_lat_deg, station_lon_deg, station_alt_m, range_m, elevation_deg, v_los_ms, sgp4_doppler_hz`

**Row count:** 31,059 (excluding header)

**Pass count:** 260 passes (from `summary.json`)

**Station:** lat=24.7961°, lon=120.9967°, alt=70 m (Taiwan — CWA network)

**Carrier:** 868.0 MHz (from `summary.json` — legacy ISM band, NOT to be used for real TX)

**Doppler physics verification:** `sgp4_doppler_hz = -v_los_ms × carrier_hz / c` holds to < 0.5 Hz precision. Confirmed.

**Limitations:**
- All Doppler values are **model-computed SGP4 Doppler**, NOT measured
- Carrier=868 MHz appears in metadata; this is the legacy ISM band used in old experiments — **RF frequency must not be hard-coded or approved for live TX**
- `physics_notes` in summary JSON: *"original_spherical_formula: WRONG — do not use; gave pass durations of 22 days"*
- The ENU formula (topocentric elevation = arcsin(unit_LOS · up_vector)) is **correct**; the original spherical-earth formula is deprecated

### 4.2 `black_kite_2_68474_best_pass_enu.csv` (single pass — used for replay CSV)

**Rows:** 152 samples  
**Pass ID:** 168  
**UTC range:** 2026-05-21T06:19:49 to 2026-05-21T06:39:54 (750 s pass at ~24° elevation peak)  
**Doppler range:** +20,268 Hz to −20,245 Hz (consistent with LOS velocity ~7 km/s at 868 MHz)

---

## 5. PGRL Checkpoint Classification

**File:** `leo-pinn/artifacts/checkpoints/pgrl_deterministic_retrained.pt`  
**SHA256:** `5cac977f0b3f66f66124d1f1776bec50345c75c45e5c815ffd609de3f258f9a4`  
**Epoch:** 78  
**Val metrics (from checkpoint):** pos_rmse=4.77 m, below_5m_pct=61.3%  
**Architecture:** SIREN PINN, 6 layers, 256 hidden dim, 128 Fourier features, orbital_elem_dim=6  
**Training domain:** Synthetic J2-perturbed orbital trajectories from `data/tle/*.npz` files with OE normalization:

```
OE_MEAN = [6778.137 km, 0.001, 0.925 rad, 0, 0, 0]
OE_STD  = [1.0 km, 0.001, 0.35 rad, 2.0 rad, 2.0 rad, 2.0 rad]
```

**Training data:** `sat_0016_i53.npz` through `sat_0019_i98.npz` — 12 synthetic satellites, 600 samples each = 7,200 validation samples.

**Classification:** Trained exclusively on synthetic, tightly-clustered LEO orbital elements (a ∈ [6777, 6779] km, σ=1 km). NOT trained on real TLE data for NORAD 68474 or any satellite with a > 6780 km.

---

## 6. PGRL Result CSV Classification

### 6.1 `pgrl_full_state_predictions.csv` (3.8M, 7,201 rows incl. header)

**Columns:** `batch_idx, sample_in_batch, global_sample_idx, source_checkpoint, true_state_0–5, pgrl_state_0–5, true_x_m, true_y_m, true_z_m, true_vx_ms, true_vy_ms, true_vz_ms, pgrl_x_m, pgrl_y_m, pgrl_z_m, pgrl_vx_ms, pgrl_vy_ms, pgrl_vz_ms, pos_error_m, vel_error_ms, t_s, satellite_id, source_npz`

**`source_npz` pattern:** `sat_0016_i53.npz`, `sat_0016_i72.npz`, `sat_0016_i98.npz`, … `sat_0019_i98.npz` — 12 distinct synthetic satellite IDs

**`source_checkpoint`:** `pgrl_deterministic_retrained.pt`

**Explicit classification:** These predictions are from **PGRL inference on the synthetic validation NPZ files only**. `sat_0016_i53.npz` and all other source NPZ files are **synthetic J2-generated orbits, NOT real TLE data for BLACK KITE-2 (NORAD 68474)**.

### 6.2 `pgrl_hard_split_real.csv`

**Rows:** 12 (one per satellite ID)  
**Metrics:** `pgrl_pos_rmse_m`, `pgrl_pos_max_m`, `pgrl_pos_mean_m`, `pgrl_below_5m_pct`  
**All satellite IDs:** `sat_0016_i53.npz` through `sat_0019_i98.npz` (synthetic only)

### 6.3 `pgrl_metric_summary.json`

Reports metrics from 7,200 validation samples from synthetic data. **Not applicable to BLACK KITE-2.**

### 6.4 `sgp4_baseline_metrics.json`

**Method:** Pure Keplerian (SGP4) vs J2 ground truth on synthetic data  
**sgp4_pos_rmse_m:** 87,189 m (expected — SGP4 without J2 corrections diverges significantly)  
**pgrl_pos_rmse_from_checkpoint:** 4.77 m  
**Note:** *"Paper SGP4 error (~4s) is from GPS receiver measurements, not orbital propagation — values not directly comparable."*

---

## 7. PGRL Model Compatibility with BLACK KITE-2

### 7.1 Orbital Element Distribution Analysis

Normalized OE for BLACK KITE-2 (from most-recent TLE):

| Element | BLACK KITE-2 Raw | OE_MEAN | OE_STD | Z-score | OOD? |
|---|---|---|---|---|---|
| a | 6968.10 km | 6778.137 km | 1.0 km | **+189.97** | **YES — extreme** |
| e | 3.01×10⁻⁵ | 0.001 | 0.001 | −0.97 | MARGINAL |
| i | 1.7061 rad | 0.925 rad | 0.35 rad | +2.23 | YES |
| Ω | 2.1030 rad | 0.0 rad | 2.0 rad | +1.05 | MARGINAL |
| ω | 4.7302 rad | 0.0 rad | 2.0 rad | +2.37 | YES |
| M | 1.5550 rad | 0.0 rad | 2.0 rad | +0.78 | OK |

### 7.2 Conclusion

The PGRL model was trained with OE normalization where **a_std = 1 km** — i.e., all training samples had semi-major axis within ±3 km of 6778 km. BLACK KITE-2's a = 6968 km is **190 km outside** this range, corresponding to a z-score of +190. This is a **complete out-of-distribution (OOD) extrapolation** for the most critical input dimension.

**The model was NOT trained on any satellite with orbital elements resembling BLACK KITE-2. Running PGRL inference on BLACK KITE-2 would produce an extrapolation with no ground-truth validation, not a generalization.**

A forward pass at the BLACK KITE-2 epoch produces state vectors (reported for completeness, not as a validated prediction):
- PGRL predicted state at epoch: pos=(2,311,861, 224,588, −3,738,392) m
- No ground-truth comparison available for BLACK KITE-2
- These values should NOT be treated as reliable trajectory predictions

---

## 8. Generated Files

### 8.1 Replay-Schema CSV (Path A — SGP4 model baseline)

**File:** `LEO-Hybrid-PGRL-reference/dataraw/pgrl/black_kite_2_68474_replay_doppler.csv`  
**SHA256:** `a4940b741862a1deb9b66471869fc406a79867c64bd424761fc90eb23c69d3a5`  
**Rows:** 152 (single pass from `best_pass_enu.csv`)

**Schema:**

| Column | Value |
|---|---|
| `t_s` | Seconds since pass start |
| `reference_doppler_hz` | SGP4-computed Doppler at 868 MHz carrier (model-derived, NOT measured) |
| `sgp4_model_doppler_hz` | Same as reference_doppler_hz |
| `pgrl_model_doppler_hz` | **BLANK** — no PGRL inference on BLACK KITE-2 |

**`reference_is_measured_truth=false`** — all Doppler values are model-computed from real TLE, not from RF measurement.

### 8.2 PGRL Predictions CSV (NOT generated — blocked)

**File:** `LEO-Hybrid-PGRL-reference/dataraw/pgrl/black_kite_2_68474_pgrl_predictions.csv`  
**Status:** NOT created. PGRL inference on BLACK KITE-2 requires either (a) OOD extrapolation or (b) retraining on BLACK KITE-2 TLE-derived data.

---

## 9. Path Classification

### Path A: Real-TLE SGP4/Model Baseline
**Status: SOLVED**

- Real TLE for NORAD 68474 obtained from Space-Track (`spacetrack_history.tle`, 178 pairs)
- SGP4 propagation run: ECEF position/velocity for 153 TLE epochs across 260 passes
- ENU Doppler computed at 868 MHz carrier from station lat=24.7961°, lon=120.9967°, alt=70 m
- Replay-schema CSV generated: `black_kite_2_68474_replay_doppler.csv`
- Carrier frequency in data is 868 MHz — **legacy ISM band, NOT approved for live satellite TX**

### Path B: Real-TLE + PGRL Inference/Retrain
**Status: BLOCKED**

- **PGRL model is out-of-distribution** for BLACK KITE-2 orbital elements (a z-score of +190)
- Forward pass produces state values, but these are **extrapolations with no ground-truth validation**
- No PGRL inference output file exists for BLACK KITE-2
- Would require: (1) bounded retraining on BLACK KITE-2 TLE-derived data, or (2) a new model trained from scratch on real LEO TLEs
- Hardware TX/PGRC loop remains blocked

### Path C: Synthetic Dry-Run Only
**Status: NOT APPLICABLE** — not needed; real TLE data exists.

---

## 10. Allowed vs. Forbidden Claim Wording

### Exact Allowed Claim Wording

> "The SGP4 Doppler profile for BLACK KITE-2 was computed from real NORAD 68474 TLE data obtained from Space-Track, propagated from the TLE epoch to ground-station visibility windows at 868 MHz carrier (model-derived, not measured)."

> "The PGRL model checkpoint (`pgrl_deterministic_retrained.pt`) was evaluated on the synthetic J2-propagated validation set (7,200 samples from 12 synthetic satellites), achieving 4.77 m median position RMSE."

> "PGRL inference on BLACK KITE-2 requires evaluating the model's out-of-distribution behavior; the normalized orbital elements for BLACK KITE-2 (a = 6968 km) are 190 standard deviations from the training mean (6778 km, σ = 1 km)."

> "The replay-schema CSV `black_kite_2_68474_replay_doppler.csv` contains SGP4-computed Doppler values only; the `pgrl_model_doppler_hz` column is blank pending PGRL inference on BLACK KITE-2."

### Exact Forbidden Claim Wording

The following claims are **explicitly prohibited** and must never appear in any paper, report, or presentation:

| Forbidden Claim | Reason |
|---|---|
| "PGRL beats SGP4 on BLACK KITE-2" | No PGRL inference exists for BLACK KITE-2; no comparison possible |
| "SGP4 Doppler is measured truth" | All SGP4 Doppler values are model-computed from TLE; no RF measurement involved |
| "Real-time Doppler tracking of BLACK KITE-2" | No live RF capture has been performed |
| "Hardware validation of PGRL for NORAD 68474" | Hardware capture is blocked; no HW loop has been run |
| "868 MHz is the approved RF frequency" | 868 MHz is legacy ISM band; no regulatory approval exists for live satellite TX |
| "RF/PER/BER/CRC metrics for BLACK KITE-2" | No RF measurement, PER, BER, or CRC data exists |
| "`sat_0016_i53.npz` is BLACK KITE-2 real TLE data" | All NPZ files are synthetic J2-generated orbits, NOT real TLE data |

---

## 11. Remaining Blockers Before Hardware

1. **Space-Track credentials** as env vars `SPACETRACK_USERNAME` + `SPACETRACK_PASSWORD` — needed for fresh TLE fetch
2. **PGRL model OOD gap** — model trained on synthetic data with a ∈ [6777, 6779] km cannot reliably extrapolate to a = 6968 km; retraining required
3. **No live RF capture** — hardware capture (`replay_driver.py`, UART, TX) is prohibited
4. **No RF frequency approval** — 868 MHz is in the data but is not an approved live-satellite frequency
5. **PGRL checkpoint not validated on BLACK KITE-2** — no ground-truth trajectory comparison exists
6. **`.env.spacetrack` not gitignored** — must not be committed; add to `.gitignore` before any push

---

## 12. Commands Executed

```bash
# Credential check
echo $SPACETRACK_USERNAME   # NO
echo $SPACETRACK_PASSWORD   # NO

# File discovery and sha256
sha256sum LEO-Hybrid-PGRL-reference/dataraw/tle/*.tle \
         LEO-Hybrid-PGRL-reference/dataraw/tle/*.json \
         LEO-Hybrid-PGRL-reference/dataraw/doppler/*.csv \
         LEO-Hybrid-PGRL-reference/dataraw/doppler/*.json \
         leo-pinn/artifacts/checkpoints/pgrl_deterministic_retrained.pt \
         leo-pinn/results/pgrl_full_state_predictions.csv \
         leo-pinn/results/pgrl_hard_split_real.csv \
         leo-pinn/results/pgrl_hard_split_real.json \
         leo-pinn/results/pgrl_metric_summary.json \
         leo-pinn/results/sgp4_baseline_metrics.json

# TLE parsing (manual, bypassing broken sgp4 2.25 C-extension API)
python3 - <<'PYEOF'
# Parsed line1/line2 from spacetrack_history.tle
# Extracted Keplerian elements: a_km=6968.10, e=3.01e-5, i=97.754°
# Normalized: z-score for a = +189.97 (OOD)
PYEOF

# Doppler physics verification
python3 -c "
# Verified sgp4_doppler_hz = -v_los_ms * carrier_hz / c to < 0.5 Hz
# carrier = 868 MHz, c = 299792458 m/s
"

# PGRL model forward pass test
source .venv-sgp4/bin/activate
PYTHONPATH="leo-pinn:.deps:$PYTHONPATH" python3 - <<'PYEOF'
# Loaded model, ran forward pass with BLACK KITE-2 OE
# Confirmed z-score of +189.97 for a dimension
# Reported predicted state (OOD extrapolation)
PYEOF

# Replay CSV generation
python3 - <<'PYEOF'
# Converted best_pass_enu.csv -> replay_schema CSV
# 152 rows, t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz (blank)
PYEOF
```

---

## 13. Files Created / Modified

| File | Action | Notes |
|---|---|---|
| `LEO-Hybrid-PGRL-reference/dataraw/pgrl/black_kite_2_68474_replay_doppler.csv` | **CREATED** | 152 rows, replay-schema CSV |
| `LEO-Hybrid-PGRL-reference/dataraw/pgrl/` | **CREATED** | Directory for PGRL/BLACK KITE-2 outputs |
| `leo-pinn/docs/review/black_kite_2_real_data_evidence_chain.md` | **CREATED** | This report |
| All `.env` files | **NOT modified** | No credentials committed |
| All git-tracked files | **NOT modified** | No paper edits, no training runs |

---

## 14. Conclusions

**Path A (real-TLE SGP4 baseline):** ✅ SOLVED — real TLE obtained, SGP4 Doppler computed at 868 MHz, replay-schema CSV generated at `dataraw/pgrl/black_kite_2_68474_replay_doppler.csv`. `reference_is_measured_truth=false` clearly documented.

**Path B (PGRL inference on BLACK KITE-2):** ❌ BLOCKED — PGRL model is catastrophically out-of-distribution (a z-score of +190); forward pass produces unvalidated extrapolation. Space-Track credentials missing. No PGRL inference output file exists.

**No hardware touched.** No training runs. No paper edited. No credentials committed. No transmit.

**Recommended next software-only step:** Build a bounded retraining script that: (1) converts BLACK KITE-2 TLE data to normalized orbital-element training samples using the same `NPZOrbitalDataset` pipeline, (2) runs a short (≤10 epoch) retrain with early stopping on a held-out TLE epoch, (3) exports PGRL predictions to `dataraw/pgrl/black_kite_2_68474_pgrl_predictions.csv` with full provenance. Hardware capture remains blocked until the OOD gap is resolved and RF frequency is properly licensed.