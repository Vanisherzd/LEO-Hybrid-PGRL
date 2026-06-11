# OTA IQ proxy experiments (CFO residual + adjacent-bin leakage)

Physical-layer **IQ-level proxy** evidence that PGRL-corrected open-loop
feedforward keeps an LR-FHSS emitter better confined to its frequency grid than
SGP4-only feedforward, measured by a USRP B210 from a **short-range room OTA /
near-field** capture.

> **Read [`docs/ota_iq_validation_scope.md`](../../docs/ota_iq_validation_scope.md)
> before reporting anything.** These are not conducted (coax/attenuator)
> measurements and they do **not** decode packets, measure PER/CRC, or validate
> a receiver. Allowed claims and labels are defined there.

## What is measured

| Experiment | Metric | Output |
|---|---|---|
| 1. CFO residual replay | residual carrier-freq offset from the grid, per burst | `cfo_residual_timeseries.csv`, `cfo_residual_summary.json`, `fig_cfo_residual_vs_time.*`, `fig_cfo_residual_cdf.*` |
| 2. Adjacent-bin leakage | ABLR_dB = 10·log10(P_adjacent / P_target), per burst | `ablr_per_burst.csv`, `ablr_summary.json`, `fig_ablr_histogram.*`, `fig_ablr_cdf.*` |
| 3. Two-node waterfall (optional, qualitative) | spectrogram only, fixed vs risk-aware guard | `fig_two_node_waterfall_fixed.*`, `fig_two_node_waterfall_riskaware.*` |

Three compensation modes are compared under an **identical** Doppler-replay
profile: `no_compensation` (A), `sgp4_only` (B), `pgrl_corrected` (C).

## Hardware setup (current room conditions)

- 2× NUCLEO-L476RG + Semtech LR1121 (SWDM001). **Do not change pin wiring.**
- 1× USRP B210 over USB 3.0, **RX only**.
- No SMA coax / no calibrated attenuator → captures are **room OTA / near-field**.
- EU868 ISM band, lowest practical LR1121 TX power, short controlled bursts.

## Safety (before any TX)

1. Attach an antenna **or** a 50-ohm load to the LR1121 RF output before keying.
2. Use the lowest practical LR1121 TX power.
3. USRP B210 is **RX only** in these experiments — never enable its TX.
4. Use only locally permitted ISM/lab frequencies; keep bursts short.

## How residual CFO is defined (replay logic)

There is no real satellite, so the channel Doppler `d(t)` is *emulated* by the
LR1121 feedforward and the terminal *compensates* with `c(t)`. The net commanded
carrier offset per burst is `d(t) − c(t)`; the USRP measures the achieved offset
from the nominal grid `F0`:

```
residual_cfo = measured_carrier_offset_from_F0  (referenced to nearest grid bin)
```

- Mode A (`none`):  `c = 0`            → residual ≈ full Doppler (off grid).
- Mode B (`sgp4`):  `c = SGP4 pred.`   → residual ≈ SGP4 open-loop error.
- Mode C (`pgrl`):  `c = PGRL pred.`   → residual ≈ PGRL open-loop error (tightest).

`d(t)` and `c(t)` come from **real** prediction CSVs you supply (see configs).
The harness never invents them; the only results come from real USRP IQ.

### Offset tuning (`lo_offset_hz`)

The on-grid burst would otherwise sit on the USRP LO/DC spike at `F0`. The USRP
RX therefore tunes to `F0 + lo_offset_hz` (default 200 kHz) and the analyzers
re-reference all peaks back to `F0`. Keep `lo_offset_hz` well inside the captured
band (`< sample_rate/2`).

### Grid spacing (`grid_spacing_hz`)

Adjacent-bin leakage needs the LR-FHSS grid spacing. Set from the **verified**
LR1121 configuration. The SWDM001 ping demo uses `LR_FHSS_V1_GRID_25391_HZ`, so
the configs ship with `grid_spacing_hz: 25391` (verified from firmware source).
**Do not hard-code 137 Hz.** If you change the firmware grid, update the configs.

## Per-burst replay requires host-replay firmware

The stock SWDM001 firmware free-runs at a fixed `RF_FREQUENCY = 868000000` and
**cannot** distinguish the three modes — every burst would be identical. The
per-burst Doppler/CFO command interface is added by the minimal firmware patch in
[`firmware/`](firmware/) (host sets freq + power before each burst over UART).

Execution (two terminals, after flashing the host-replay firmware):

```bash
# T1 — USRP capture over the full replay window (RX only):
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \
    --out-dir hardware/ota_iq/runs/<TS>/pgrl --device-args serial=8000304

# T2 — drive the bursts on the same schedule (logs commanded freq per burst):
uv run python hardware/ota_iq/replay_driver.py \
    --schedule hardware/ota_iq/runs/<TS>/pgrl/burst_schedule.csv \
    --uart /dev/cu.usbmodem1303 --tx-power-dbm -9 \
    --out hardware/ota_iq/runs/<TS>/pgrl/replay_uart_log.csv
```

If `replay_driver.py` reports **0 acks** the host-replay firmware is not active —
do not claim any replay result. Then analyze each run as below.

## Operator procedure

```bash
# 0) supply real Doppler truth + feedforward CSVs, then point the configs at them:
#    doppler_profile_csv: hardware/ota_iq/inputs/pass_doppler_truth.csv   (t_s,doppler_hz)
#    compensation_csv:    hardware/ota_iq/inputs/feedforward_sgp4.csv      (t_s,comp_hz)
#                         hardware/ota_iq/inputs/feedforward_pgrl.csv      (t_s,comp_hz)
#    grid_spacing_hz:     <verified LR-FHSS grid spacing>

# 1) build the burst schedule the LR1121 firmware emits (per mode):
uv run python hardware/ota_iq/usrp_capture_ota_iq.py plan \
    --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \
    --out-dir hardware/ota_iq/runs/pgrl_001

# 2) noise-floor check (LR1121 OFF) then a 10 s smoke capture (LR1121 ON, min power):
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml \
    --out-dir hardware/ota_iq/runs/pgrl_smoke --duration 10
#    inspect capture_meta.json -> clipping_warning must be false; if true, lower rx_gain_db.

# 3) full replay capture for each of the three modes (120-300 s):
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config hardware/ota_iq/configs/replay_no_compensation.yaml --out-dir hardware/ota_iq/runs/nocomp_001
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config hardware/ota_iq/configs/replay_sgp4_only.yaml       --out-dir hardware/ota_iq/runs/sgp4_001
uv run python hardware/ota_iq/usrp_capture_ota_iq.py capture \
    --config hardware/ota_iq/configs/replay_pgrl_corrected.yaml  --out-dir hardware/ota_iq/runs/pgrl_001

# 4) analyze each run:
for r in nocomp_001:no_compensation sgp4_001:sgp4_only pgrl_001:pgrl_corrected; do
  d=${r%%:*}; c=${r##*:}
  uv run python hardware/ota_iq/analyze_cfo_residual.py \
      --run-dir hardware/ota_iq/runs/$d --config hardware/ota_iq/configs/replay_$c.yaml
  uv run python hardware/ota_iq/analyze_adjacent_bin_leakage.py \
      --run-dir hardware/ota_iq/runs/$d --config hardware/ota_iq/configs/replay_$c.yaml
done

# 5) figures:
uv run python hardware/ota_iq/plot_ota_iq_results.py \
    --run no_comp=hardware/ota_iq/runs/nocomp_001 \
    --run sgp4=hardware/ota_iq/runs/sgp4_001 \
    --run pgrl=hardware/ota_iq/runs/pgrl_001 \
    --out-dir hardware/ota_iq/figures
```

If `python-uhd` is not installed on the SDR host, the `capture` step prints the
exact `uhd_rx_cfile` command to run and exits non-zero — it never fabricates IQ.

## Experiment 3 (optional, qualitative only)

Schedule both NUCLEO+LR1121 nodes (fixed small guard vs risk-aware wider guard),
capture each with the USRP, and render spectrograms:

```bash
uv run python hardware/ota_iq/plot_ota_iq_results.py \
    --waterfall fixed=hardware/ota_iq/runs/twonode_fixed/capture_iq.npy \
    --waterfall riskaware=hardware/ota_iq/runs/twonode_riskaware/capture_iq.npy \
    --sample-rate 1e6 --out-dir hardware/ota_iq/figures
```

There is no shared timing/trigger, so **no measured collision probability** is
claimed — these are **qualitative OTA demonstrations only**.

## Files

| File | Role |
|---|---|
| `usrp_capture_ota_iq.py` | `plan` (schedule manifest) + `capture` (USRP B210 RX only) |
| `analyze_cfo_residual.py` | per-burst residual CFO from a capture |
| `analyze_adjacent_bin_leakage.py` | per-burst ABLR from a capture |
| `plot_ota_iq_results.py` | CFO/ABLR figures + optional qualitative waterfalls |
| `ota_common.py` | IQ IO, config load, burst detection, spectra |
| `configs/replay_*.yaml` | the three compensation modes |
