# MacBook Air M2 Hardware Setup

> **Recommended mode:** Native macOS + Homebrew UHD + `uv` Python environment.
> Docker is used **only** for simulation / offline analysis — not for live USRP hardware.

## Hardware Requirements

| Item | Notes |
|------|-------|
| MacBook Air M2 (or M3) | Apple Silicon or Intel |
| USRP B210 | USB 3.0 connection required |
| USB-C to USB-A adapter | If Mac has USB-C only |
| Coax cable + 30 dB attenuator | For loopback TX→RX test |
| Semtech LR1121/LR11xx dev board | For LR-FHSS TX generation |

**Important:** Connect the USRP B210 directly to a USB 3.0 port. Avoid unpowered USB hubs.

---

## Quick Start

### 1. One-time environment setup

```bash
git clone https://github.com/Vanisherzd/LEO-Hybrid-PGRL.git
cd LEO-Hybrid-PGRL
bash scripts/setup_macos_m2.sh
```

This installs:
- Homebrew UHD (via `brew install uhd`)
- UHD firmware images
- Python environment via `uv sync`

### 2. Verify hardware connectivity

```bash
bash scripts/check_hardware_macos.sh
```

Expected output:
```
[2/5] Searching for USRP devices...
  Device: B210 ...
```

### 3. Dry-run IQ capture

```bash
uv run python hardware/usrp_scripts/dual_mode_trx.py --sim
```

### 4. Live RX capture (with hardware)

```bash
uv run python hardware/usrp_scripts/dual_mode_trx.py --rx-only \
  --freq 915e6 --rate 1e6 --gain 30 --duration 2 \
  --out hardware/captures/baseline.fc32
```

---

## Hardware Validation Artifact Chain

Every hardware run must produce the following artifacts for reproducibility:

```
hardware/captures/YYYYMMDD/
├── baseline.fc32              # Raw IQ from USRP
├── tx_config.json             # PGRL-generated TX parameters
├── uhd_command.log            # uhd_rx_cfile invocation
└── results.json               # CFO, EVM, SNR summary
```

### TX config JSON format

```json
{
  "center_freq_mhz": 915.0,
  "tx_power_dbm": 14,
  "doppler_precomp_hz": -1234,
  "guard_time_ms": 56,
  "timestamp_utc": "2026-XX-XXTXX:XX:XXZ"
}
```

### results.json format

```json
{
  "rx_cfo_hz": 234.5,
  "rx_evm_percent": 67.2,
  "rx_snr_db": 39.8,
  "capture_sample_count": 2000000,
  "validation_type": "hardware"
}
```

---

## Docker vs Native — What Goes Where

| Task | Docker | Native macOS |
|------|--------|--------------|
| USRP B210 live capture | No | Yes (`uhd_rx_cfile`) |
| UHD Python binding | No | Yes (`import uhd`) |
| PGRL training | Yes | Yes |
| Trace-driven simulation | Yes | Yes |
| Figure generation | Yes | Yes |
| Offline IQ analysis | Yes | Yes |
| Semtech TX control | No | Yes (Serial/COM) |

---

## UHD on Apple Silicon — Known Issues

### `uhd_images_downloader` not found

After `brew install uhd`, you may need:

```bash
brew link uhd
# If images are still missing:
# Download from https://files.ettus.com/binaries/uhd/UHD-MACUNTITLED.pkg
```

### Python `import uhd` fails on Apple Silicon

Use the UHD CLI tools as fallback:

```bash
# Capture via CLI → .fc32 file → Python offline analysis
uhd_rx_cfile --args="type=b200" --freq 915e6 --rate 1e6 \
  --gain 30 --duration 2 --output capture.fc32

uv run python hardware/usrp_scripts/analyze_capture.py capture.fc32
```

### GNU Radio on Apple Silicon

GNU Radio may have compatibility issues on newer macOS. First phase uses only UHD + Python without GNU Radio.

---

## Hardware-Claim Checklist

Before labeling any result as "hardware-validated":

- [ ] Raw IQ file captured (`.fc32`)
- [ ] TX configuration JSON recorded
- [ ] `results.json` generated with CFO/EVM/SNR
- [ ] UHD command invocation logged in `uhd_command.log`
- [ ] Figure generated from captured IQ
- [ ] Figure committed to `paper/figures/`

**Do NOT label as "hardware-validated" until all items are checked.**

---

## Next Steps After Capture

```bash
# Analyze captured IQ
uv run python hardware/usrp_scripts/analyze_capture.py \
  hardware/captures/baseline.fc32 \
  --output hardware/captures/baseline_results.json

# Plot waterfall
uv run python sdr_hwil/plot_waterfall.py \
  hardware/captures/baseline.fc32

# Generate new Fig. 5 for paper
# (replace paper/figures/fig5_sdr_synthetic_pipeline.pdf with hardware output)
```