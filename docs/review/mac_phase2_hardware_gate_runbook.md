# Mac Phase 2 — Hardware-Gate Runbook (DRAFT)

Hardware-gated runbook for the OTA-IQ replay pipeline. **Nothing in this document
is auto-executed.** Every capture/TX step is gated on explicit user approval and
the blockers below. All commands are **DRAFT placeholders**; no real RF frequency
appears here.

- **Date:** 2026-06-13
- **Repo:** `/Users/laizhendong/Desktop/LEO-Hybrid-PGRL`
- **Branch:** `paper-crosslayer-rewrite`
- **Phase 1 report:** `docs/review/mac_phase1_software_preflight.md` (committed)

> Scope discipline: short-range OTA / conducted **IQ-level diagnostics only**.
> No live-satellite validation, no measured Doppler truth, no PER/BER/CRC, no
> gateway ACK, no regulatory-compliance claim.

## 1. Readiness status

| Component | Status |
|---|---|
| Mac software (`hardware/ota_iq/*.py` compile, deps) | **READY** |
| USRP B210 discovery (`uhd_find_devices` / `uhd_usrp_probe`) | **READY** |
| STM32 / ST-LINK discovery (`/dev/cu.usbmodem1303` VCP) | **READY** |
| Schedule schema guard (safe accepted / unsafe refused / F0=0 blocks) | **READY** |
| USRP **capture** | **BLOCKED — user approval required** |
| LR1121 **replay / TX** | **BLOCKED — user approval required** |

## 2. Blockers — must ALL hold before any USRP capture
1. **User approval** for this specific capture run.
2. **RX-only confirmed** — USRP B210 RX path only; its TX path stays disabled.
3. **No replay / TX running** — LR1121 idle, no UART command session open.
4. **No real RF frequency change** outside an explicitly approved config value.
5. **Output directory planned** — a named `<OUTPUT_DIR>` under
   `hardware/ota_iq/runs/<TS>/...` agreed in advance.

## 3. Blockers — must ALL hold before any LR1121 replay / TX
1. **Confirmed NCC / local / gateway channel plan** for the target region
   (`frequency_plan` currently `TAIWAN_AS923_REVIEW_REQUIRED`).
2. **Explicit positive `nominal_center_freq_hz`** set in the config — **only
   after** approval. Configs ship `0` (all RF paths blocked) by default.
3. **Conducted / coax path with attenuator (≥30 dB)** or a **50-ohm RF load** —
   preferred over antenna OTA. Room OTA stays an uncalibrated diagnostic.
4. **No antenna TX** until the protected RF path above is in place.
5. **LR1121 firmware / handshake confirmed** — `REPLAY_READY` / `RDY` and
   `CARRIER_DONE` (CW) or `BURST_DONE` (LR-FHSS) observed on the VCP.
6. **Lowest practical TX power**.
7. **User approval before any non-dry-run `replay_driver.py`**.

## 4. DRAFT commands (do not run without the gates above)

### 4a. Safe discovery (read-only, no TX) — DRAFT
```bash
uhd_find_devices                 # enumerate USRP (RX-capable discovery only)
uhd_usrp_probe --args type=b200  # device tree; no streaming
ls /dev/cu.usbmodem*             # confirm ST-LINK VCP candidate
uv run python -c "import numpy,yaml,serial;print('deps ok')"
```

### 4b. Software-only schedule dry-run (no repo write, no RF) — DRAFT
```bash
# safe reference schema -> prints rows + md5, writes nothing:
uv run python hardware/ota_iq/generate_real_replay_schedule.py \
    --predictions-csv /tmp/<reference>.csv --nominal-center-hz 1 \
    --n-bursts <N> --dry-run
# (nominal-center-hz 1 = schema/guard placeholder, NOT an experiment frequency)
```

### 4c. FUTURE USRP capture — PLACEHOLDER ONLY (requires §2 gates + approval)
```bash
# Set an APPROVED positive carrier in the config first (no default, no real
# value committed here). <APPROVED_F0_HZ> and <OUTPUT_DIR> are placeholders.
#   config: nominal_center_freq_hz: <APPROVED_F0_HZ>
#   keep ota_transmission_allowed false for RX-only capture; enable TX only after separate replay approval
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config <APPROVED_CONFIG>.yaml \
    --out-dir <OUTPUT_DIR> \
    --device-args serial=8000304
# RX-only. Do NOT run until every §2 blocker is cleared and user approves.
```

### 4d. FUTURE replay — DRY-RUN PLACEHOLDER ONLY (requires §3 gates + approval)
```bash
# Dry-run validates schedule + UART plumbing WITHOUT transmitting:
uv run python hardware/ota_iq/replay_driver.py \
    --schedule <OUTPUT_DIR>/burst_schedule.csv \
    --uart /dev/cu.usbmodem1303 --tx-power-dbm <LOWEST_PRACTICAL> \
    --out <OUTPUT_DIR>/replay_uart_log.csv \
    --dry-run
# Non-dry-run replay (actual TX) is intentionally NOT scripted here. It requires
# every §3 blocker cleared and explicit user approval at run time.
```

## 5. Standing prohibitions (carry over from Phase 1)
- No `usrp_capture_ota_iq.py capture` without §2 + approval.
- No `replay_driver.py` without `--dry-run` and §3 + approval.
- No opening `/dev/cu.usbmodem1303` for command writes outside an approved run.
- No real RF frequency hard-coded in repo configs/docs.
- No `paper/icc_main.tex` edits; no stash pop; no commits without approval.

## 6. Outstanding data blocker (separate from RF gates)
No trained PGRL checkpoint / real per-sample SGP4+PGRL Doppler trace / real
TLE exists in the repo. A **real** model-derived schedule still requires those
inputs fed to `generate_real_replay_schedule.py --write`. Until then, only
synthetic `/tmp` dry-run fixtures are used, and no real schedule is written.
