# Legacy OTA-IQ Run Artifacts

This directory contains legacy local diagnostic artifacts from earlier OTA-IQ
debugging sessions.

## Scope

These runs are retained for internal debugging and method archaeology only.
They are **not** paper-grade evidence.

Raw IQ files (`*.fc32`) are intentionally git-ignored. Some curated metadata,
summaries, figures, and short config snapshots may be tracked for auditability.

## Important limitations

- These runs predate the current explicit-F0 and safe-schema hardening.
- Any `868000000` / 868 MHz values in this directory are historical lab/debug
  metadata only.
- They are **not** deployment defaults and are **not** Taiwan frequency guidance.
- Runs labeled `demo_synthetic` or synthetic/demo schedule are not real TLE
  schedules.
- These artifacts are **not** measured Doppler truth.
- These artifacts are **not** live-satellite validation.
- These artifacts are **not** PER / BER / CRC / gateway-ACK evidence.
- These artifacts are **not** regulatory-compliance evidence.
- These artifacts are **not** conducted RF validation unless separately
  revalidated under the current hardware gates.

## Do not replay

Do not run or replay schedules from this directory.

Current valid flow is controlled by:

- `hardware/ota_iq/configs/`
- `hardware/ota_iq/README.md`
- `docs/review/mac_phase2_hardware_gate_runbook.md`
- later approved data/provenance documents

Hardware capture/TX remains blocked until both data gates and RF gates are
cleared with explicit user approval.
