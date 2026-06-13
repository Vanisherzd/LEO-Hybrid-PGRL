# Data / Provenance Inventory for OTA-IQ Replay

Software-only inventory of model, orbital, and Doppler inputs needed to generate
a **real** model-derived replay schedule. **No hardware command, capture, TX, or
UART** was performed.

- **Date:** 2026-06-13
- **Repo:** `/Users/laizhendong/Desktop/LEO-Hybrid-PGRL`
- **Branch:** `paper-crosslayer-rewrite`
- **Commit (HEAD):** `086a7da` — *docs(review): add Mac Phase 2 hardware gate runbook*

> Scope: orbital/model inputs only. No live-satellite validation, no measured
> Doppler truth, no PER/BER/CRC, no gateway ACK, no regulatory-compliance claim.

## 1. Candidate PGRL checkpoint / model-weight files
| Search | Result |
|---|---|
| `.pt .pth .ckpt .pkl .npz .npy .safetensors .h5 .onnx .joblib` (excl. `.venv`, `runs/`) | **NONE found** |

- Model **code** exists (`models/orbital_physics.py`, `physics_ml/orbital_physics.py`,
  `controller/doppler_precomp.py`, `controller/pgrl_output_schema.py`) but **no
  trained weights**. PGRL-corrected Doppler cannot be produced without a checkpoint.
- **Status: MISSING** (trained PGRL checkpoint resides off-repo / training machine).

## 2. Candidate TLE / orbit / pass / Doppler / reference-trace files
| File | Nature | Usable as real input? |
|---|---|---|
| `data/examples/sample_tle.txt` | **EXAMPLE/PLACEHOLDER** TLE (NORAD 99999, "NOT a real satellite") | No |
| `data/examples/sample_sgp4_states.csv` | 5 data rows, NORAD 99999, single `doppler_hz`; no SGP4/PGRL split | No (placeholder, no mode columns) |
| `data/manifests/tle_sources.yaml` | catalog metadata only; `committed: false`; raw TLE local-only under git-ignored `data_raw/tle/` | Points to data not in repo |
| `data/schemas/{tle_record,sgp4_state,pass_window}.schema.json` | schemas only | Schema, not data |
| `data_raw/tle/` | **absent** on this host | No real TLE present |
| `hardware/ota_iq/inputs/` | **absent** | No real schedule generated |

- No file contains the replay columns `reference_doppler_hz`,
  `sgp4_model_doppler_hz`, or `pgrl_model_doppler_hz` (those names appear only in
  the generator/schema + review docs, never in a data file).
- **Status: MISSING** real TLE / pass / Doppler trace.

## 3. Scripts that can generate replay / reference CSVs
| Script | Capability | Limit |
|---|---|---|
| `hardware/ota_iq/generate_real_replay_schedule.py` | **consumes** a real predictions CSV → schedule; refuses synthetic/`true_doppler_hz` | does **not** create Doppler; needs §1–2 inputs |
| `models/orbital_physics.py` | TLE parse + Kepler/J2 analytic propagation → ECI pos/vel (→ radial velocity → Doppler) | needs a **real TLE**; physics/SGP4-reference only, not PGRL |
| `controller/doppler_precomp.py` | `compensated_tx_frequency`, `residual_doppler_after_precomp`, `doppler_rate_from_series` | utility math on **given** Doppler inputs; no checkpoint, generates nothing alone |
| `experiments/exp1_pgrl_prediction/addons/doppler_compensation_ablation.*` | summary σ (SGP4/PGRL) from **synthetic** scenarios | "no external TLE"; not per-sample, not real |

- A **physics/SGP4-reference** Doppler series is *generatable* from
  `orbital_physics.py` **once a real TLE is supplied**. The **PGRL** series still
  needs the trained checkpoint (§1).

## 4. Current OTA-IQ config dependencies
All three `hardware/ota_iq/configs/replay_*.yaml`:
- `doppler_profile_csv: null`, `compensation_csv: null` (inputs unset)
- `nominal_center_freq_hz: 0`, `ota_transmission_allowed: false`,
  `frequency_plan: TAIWAN_AS923_REVIEW_REQUIRED` (RF paths blocked)

## 5. Required-input existence matrix
| Required input | Exists? | Source if generated |
|---|---|---|
| `reference_doppler_hz` (model/SGP4-reference Doppler) | **MISSING** | `orbital_physics.py` + **real TLE** |
| `sgp4_model_doppler_hz` (SGP4 open-loop model Doppler) | **MISSING** | `orbital_physics.py` + **real TLE** (w/ epoch staleness) |
| `pgrl_model_doppler_hz` (PGRL-corrected model Doppler) | **MISSING** | trained **PGRL checkpoint** + real TLE |
| trained PGRL checkpoint | **MISSING** | off-repo / training machine |
| real TLE / pass | **MISSING** | provider via `data_raw/tle/` (git-ignored, absent) |

## 6. Minimal required input schema for replay
The generator (`generate_real_replay_schedule.py`) requires a CSV with:

```
t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz
```

Per-mode commanded offsets derived as:
- `no_compensation_offset_hz = reference_doppler_hz`
- `sgp4_only_offset_hz       = reference_doppler_hz − sgp4_model_doppler_hz`
- `pgrl_corrected_offset_hz  = reference_doppler_hz − pgrl_model_doppler_hz`

Prohibited column names (refused): `true_doppler_hz`, `truth_doppler_hz`,
`measured_doppler_hz`, `measured_doppler_truth_hz`.

## 7. Provenance statement — reference is model-derived, NOT measured truth
`reference_doppler_hz` is a **model/reference-derived** Doppler profile (e.g.
SGP4/physics propagation of a TLE). It is **NOT measured on-air Doppler truth**.
Every generated row carries `reference_is_measured_truth=false`. No measured /
live-pass Doppler is claimed anywhere in this pipeline.

## 8. Inventory conclusion
**MISSING DATA.** No trained PGRL checkpoint, no real TLE/pass, and no real
per-sample SGP4/PGRL Doppler trace exist in the repo. A real model-derived
schedule cannot be generated yet. Only synthetic `/tmp` dry-run fixtures are
available (and they are never written into the repo).

## 9. Recommended next software-only step
1. Obtain/select a **real TLE** for the target pass (place under the git-ignored
   `data_raw/tle/` per `tle_sources.yaml`).
2. Obtain the **trained PGRL checkpoint** (off-repo).
3. Add a small, software-only generator step that runs `orbital_physics.py`
   (SGP4/physics reference + SGP4-model Doppler) and the PGRL checkpoint
   (PGRL-model Doppler) to emit the §6 CSV — then `generate_real_replay_schedule.py
   --predictions-csv <that.csv> --nominal-center-hz <APPROVED_F0> --write` (dry-run first).
   None of this transmits; it produces the schedule only.

## 10. Blockers before any capture / TX
- **Data:** trained PGRL checkpoint + real TLE + real §6 predictions CSV (all MISSING).
- **Capture:** user approval · RX-only confirmed · no replay/TX running · no real
  RF frequency change · output directory planned.
- **LR1121 replay/TX:** confirmed NCC/local/gateway channel plan · explicit
  positive `nominal_center_freq_hz` after approval · conducted/coax + attenuator
  (≥30 dB) or 50-Ω load · no antenna TX · firmware/handshake confirmed · lowest
  practical TX power · user approval before non-dry-run replay.
