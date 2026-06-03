# LR1121 SDR Validation Workflow

End-to-end procedure for validating LR-FHSS transmissions from a Semtech LR1121
board using a USRP B210 SDR as an **IQ-level** measurement receiver.

> **Scope discipline (read first).** The SDR path is an IQ-level measurement
> tool only. It is **not** a standard-compliant LR-FHSS gateway. Do **not**
> claim LR-FHSS PER, standard decoding, or "successful RF capture" unless the
> signal detector passes (`validation_status == "signal_detected"`). Until then
> the honest claim is "firmware running" and/or "IQ capture done".

---

## 1. Current verified hardware state (ground truth)

- PGRL trace-driven / simulation pipeline is complete.
- USRP B210 works on **Mac native UHD**. Homebrew UHD lacked
  `uhd_rx_cfile` / `rx_samples_to_file` / the Python binding, so a native C++
  capture tool was added:
  `hardware/usrp_scripts/rx_capture_to_file_cpp.cpp`. It records `.fc32`
  (complex64) IQ files.
- `analyze_capture.py` produces a waterfall plus signal detection, residual
  CFO, a QPSK EVM proxy, and SNR. `quick_maxhold.py` produces a max-hold
  spectrum. `scripts/run_lr1121_sdr_sweep.sh` runs multi-frequency / multi-
  antenna sweeps.
- SWDM001 firmware was built on Windows/Keil for target `lr1121_xtal`; it runs
  on a NUCLEO-L476RG + LR1121 and prints `LR11XX-LR-FHSS Ping Init` and
  `Packet sent!` over UART.
- The SWDM001 LR11xx demo default RF frequency is **868 MHz**, set in
  `src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`:
  `#define RF_FREQUENCY ( 868000000 )`.
- Captures have been taken at 868 MHz on both the RX2 and TX/RX antenna ports,
  but the waterfall still looks like noise / DC-only — **no visible LR-FHSS
  hopping or burst has been observed yet.**

---

## 2. SWDM001 firmware

| Item | Value |
|------|-------|
| Build target | `lr1121_xtal` |
| Keil project | `project/keil_polling_STM32L476/STM32L476.uvprojx` |
| Generated binary | `project/keil_polling_STM32L476/STM32L476_lr11xx.bin` |
| Flash target | `NOD_L476RG` |
| UART baud | 115200 |
| Expected UART output | `LR11XX-LR-FHSS Ping Init` then `Packet sent!` |
| Default RF frequency | `868000000` (868 MHz) |

To transmit at 915 MHz instead, change `RF_FREQUENCY` in
`src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`, rebuild, and reflash.
Keep the SDR capture centre frequency in sync with whatever the firmware
actually transmits.

---

## 3. USRP capture and analysis commands

### 3.1 Capture IQ to file

Use the native C++ capture tool to write a `.fc32` complex64 file. Match the
centre frequency to the firmware's `RF_FREQUENCY` (868 MHz by default).

### 3.2 Analyze a single capture

```bash
uv run python hardware/usrp_scripts/analyze_capture.py INPUT.fc32 \
    --sample-rate 1000000 \
    --output-json OUT.json \
    --plot OUT.png \
    --signal-threshold-db 8 \
    --maxhold-plot OUT_maxhold.png
```

This emits the canonical analyzer JSON (see section 5), a waterfall PNG, and a
max-hold spectrum PNG.

### 3.3 Quick max-hold spectrum

```bash
uv run python hardware/usrp_scripts/quick_maxhold.py INPUT.fc32 \
    --sample-rate 1000000 \
    --output OUT_maxhold.png
```

The max-hold view accumulates the peak power per bin across the whole capture,
which makes intermittent LR-FHSS hops easier to spot than a single FFT frame.

---

## 4. Sweep workflow

When a single capture shows nothing, sweep frequency and antenna port to find
where (if anywhere) RF energy appears:

```bash
bash scripts/run_lr1121_sdr_sweep.sh \
    --serial 8000304 \
    --freqs "868e6,915e6,923e6" \
    --antennas "RX2,TX/RX" \
    --rate 2e6 \
    --gain 45 \
    --duration 30
```

This produces `sweep_summary.json` and `sweep_summary.csv`, one row per
(frequency, antenna) combination, each carrying the canonical analyzer fields so
you can see at a glance which configuration (if any) reaches
`validation_status == "signal_detected"`.

---

## 5. Canonical analyzer JSON schema

The analyzer (and each sweep row) reports:

| Field | Meaning |
|-------|---------|
| `validation_status` | `"signal_detected"` \| `"noise_floor_only"` \| `"weak_signal_candidate"` |
| `signal_detected` | bool |
| `peak_to_median_db` | peak bin power over median bin power (dB) |
| `burst_energy_excess_db` | excess energy of detected burst over noise floor (dB) |
| `peak_frequency_offset_hz` | offset of the strongest in-band bin from centre |
| `rx_cfo_hz` | residual carrier frequency offset |
| `rx_cfo_std_hz` | std-dev of the CFO estimate |
| `rx_evm_percent` | QPSK EVM proxy (lower is better) |
| `rx_snr_db` | estimated SNR |

The analyzer **excludes the DC / LO spike** (±2% of the sample rate) before
scoring, so a DC-only capture is **not** flagged as a signal.

---

## 6. Failure interpretation

- **`Packet sent!` on UART** means the firmware command pipeline completed. It
  does **not** prove the SDR received any RF.
- **A DC line only** in the waterfall means there is likely no visible RF
  signal — only the receiver's own LO/DC leakage is present.
- **Do not make any LR-FHSS waveform claim** until the signal detector passes,
  i.e. `validation_status == "signal_detected"`.

---

## 7. RF troubleshooting checklist

- Verify the **sub-GHz / LF RF port** on the LR1121 board. The LR1121 has
  separate HF and sub-GHz ports — transmitting on the wrong port produces no
  sub-GHz energy.
- Verify the antenna matches the band (868 / 915 MHz).
- Try both the **RX2** and **TX/RX** SDR antenna ports.
- Try several gains: **20 / 35 / 45**.
- Try several frequencies: **868 / 915 / 923 MHz**.
- Use an **attenuator** for any conducted (cabled) measurement.
- **Do NOT** cable a high-power TX directly into the USRP without attenuation
  (**30 dB recommended**). This can damage the SDR front end.

---

## 8. Validation-label ladder

Advance the validation label only when its evidence bar is met. Never skip a
rung.

| Label | Evidence required |
|-------|-------------------|
| `firmware_running` | SWDM001 boots on NUCLEO-L476RG + LR1121 and prints `LR11XX-LR-FHSS Ping Init` / `Packet sent!` over UART. Proves firmware, not RF. |
| `iq_capture_done` | A valid `.fc32` IQ file was recorded by the USRP at the right centre frequency and sample rate. Proves the SDR captured samples, not that they contain a signal. |
| `signal_detected` | The analyzer reports `validation_status == "signal_detected"` (DC spike excluded) **and** a waveform figure (waterfall / max-hold) shows the energy. Proves RF energy was received above noise. |
| `hardware_validated` | `signal_detected` plus the measured RF-quality metrics (CFO, EVM proxy, SNR) are consistent with an LR-FHSS transmission and documented with figures. Highest, most defensible claim. |

For the GLOBECOM workshop paper, keep claims conservative: report only the
highest rung whose evidence is actually in hand.
