# Frequency / Doppler Schema / F0 Guard Verification

## Scope

This verification covers only software-side safety checks for the `hardware/ota_iq` replay/capture pipeline.

No hardware was run.
No USRP/UHD capture was started.
No LR1121 UART command was sent.
No OTA experiment was performed.
`paper/icc_main.tex` was intentionally not modified.

## Verified safety changes

### 1. Carrier frequency default removed

The active replay configs no longer use `868 MHz` or `868000000` as a default carrier.
`nominal_center_freq_hz` is set to `0` as a placeholder and must be explicitly replaced only after NCC/local/gateway frequency-plan confirmation.

Related config keys:

- `frequency_plan: "TAIWAN_AS923_REVIEW_REQUIRED"`
- `ota_transmission_allowed: false`
- `frequency_safety_note: ...`

### 2. Schedule generator requires explicit carrier

`hardware/ota_iq/generate_real_replay_schedule.py` blocks execution when `--nominal-center-hz` is not provided.

Verified behavior:

Carrier frequency must be explicitly provided after confirming local frequency plan.

### 3. CW tone analyzer requires explicit carrier

`hardware/ota_iq/analyze_cw_tones.py` now requires `--nominal-center-hz`.

It has no carrier-frequency default.

### 4. Doppler schema no longer uses truth naming

`hardware/ota_iq/generate_real_replay_schedule.py` now uses:

- `reference_doppler_hz`
- `sgp4_model_doppler_hz`
- `pgrl_model_doppler_hz`
- `reference_is_measured_truth = false`

It refuses misleading input columns such as:

- `true_doppler_hz`
- `truth_doppler_hz`
- `measured_doppler_hz`

This prevents accidental claims of measured Doppler truth.

### 5. F0 guard blocks placeholder frequency

The following paths now reject `nominal_center_freq_hz <= 0` before hardware-relevant actions:

- `replay_driver.py`
- `usrp_capture_ota_iq.py plan`
- `usrp_capture_ota_iq.py capture`

Verified behavior:

nominal_center_freq_hz must be explicitly set to a positive carrier frequency after NCC/local/gateway frequency-plan confirmation.

### 6. Environment dependency

`PyYAML` was installed into the repo virtual environment via:

uv pip install --python ./.venv/bin/python "pyyaml>=6.0"

This does not require a dependency-file change because `pyproject.toml` already declares:

pyyaml>=6.0

## Compile verification

All active `hardware/ota_iq/*.py` scripts compile under `.venv/bin/python`.

## Remaining restrictions

Mac OTA / hardware testing remains paused until all of the following are confirmed:

1. NCC/local frequency rules.
2. Gateway/channel plan.
3. AS923-compatible local configuration.
4. Conducted/coax or controlled lab setup.
5. Explicit positive `nominal_center_freq_hz` in config or CLI.

Do not hard-code `923.x MHz` until the actual permitted channel plan is confirmed.

## Status

Frequency default safety: PASS
Doppler schema safety: PASS
F0 placeholder guard: PASS
Hardware execution: NOT RUN
Paper modification: NOT DONE
