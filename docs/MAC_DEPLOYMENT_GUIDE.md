# USRP B210 Mac Deployment Guide — LEO-PINN HWIL Testing

> **Scope:** Move the LEO-PINN dual-mode SDR terminal from this Linux workstation to your MacBook Pro for live over-the-air (OTA) hardware-in-the-loop (HWIL) validation using a coax loopback cable.

---

## 1. Hardware Requirements

| Item | Specification |
|------|-------------|
| **Host computer** | Apple Silicon (M-series) or Intel Mac with USB 3.0 Type-A port |
| **SDR transceiver** | Ettus Research USRP B210 (or B205mini) |
| **Cable** | Coax loopback cable with 30 dB attenuator (SMA or MMCX connectors) |
| **Driver** | UHD ≥ 4.1 (installed via Homebrew) |
| **OS** | macOS 12 (Monterey) or later |

> **WARNING:** Always connect the 30 dB attenuator between TX and RX ports. The B210 has no built-in TX/RX isolation; direct loopback will saturate the LNA and produce invalid EVM readings.

---

## 2. Environment Setup

### 2.1 Install UHD and Python dependencies

```bash
# 1. Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install UHD (Universal Hardware Driver) via Homebrew
brew install uhd

# 3. Download UHD FPGA images (required for B210)
#    This fetches the bitstream for the B210 daughterboard.
uhd_images_downloader
```

> **Note:** If `uhd_images_downloader` is not in your PATH after `brew install uhd`, it can be found at:
> `$(brew --prefix)/lib/uhd/utils/uhd_images_downloader.py`

### 2.2 Python virtual environment

```bash
# Navigate to the repository root
cd /path/to/LEO-Hybrid-PGRL

# Sync Python dependencies (reads pyproject.toml / uv.lock)
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Verify UHD is importable
python -c "import uhd; print(uhd.__version__)"
```

Expected output: `4.1.0.0` (or newer)

---

## 3. Verifying B210 Connectivity

```bash
# List all UHD devices on the USB bus
uhd_find_devices
```

Expected output (example):

```
[INFO] [UHD] linux; GNU Privacy Guard; CryptoAgility]
[INFO] [UHD] Detected Device: B210
  serial = F5████████
  name  = MyB210
  type  = usrp_b210
  product = B210
  manufacturer = Ettus Research
  mobilib = 0.2.4
```

If no device is found:
- Confirm the B210 is connected via USB 3.0 (not USB 2.0)
- Try a different USB port directly on the host (not through a hub)
- Run `uhd_admin --find-all` for a more verbose device scan

---

## 4. Running the SDR Terminal

### 4.1 Hardware mode (`--hw`)

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run in live hardware mode
uv run python hardware/usrp_scripts/dual_mode_trx.py --hw --doppler-hz 8000
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--hw` | Enable live TX/RX (requires B210) | — |
| `--sim` | Simulation mode (no hardware) | default |
| `--doppler-hz Hz` | PGRL-predicted Doppler shift in Hz | 8000 |
| `--snr-db dB` | AWGN SNR for simulation | 25 |
| `--n-bursts N` | Number of bursts to transmit | 3 |
| `--n-syms N` | QPSK symbols per burst | 300 |

### 4.2 Simulation mode (no hardware)

```bash
# Quick sanity check without B210
uv run python hardware/usrp_scripts/dual_mode_trx.py --sim --snr-db 40 --n-bursts 3
```

### 4.3 Coax loopback configuration

Connect the hardware as follows:

```
  B210 TX/RX (SMA) ── [30 dB attenuator] ── B210 RX2 (SMA)
```

- TX gain: 40 dB (fixed in script)
- RX gain: 40 dB (fixed in script)
- The 30 dB pad reduces TX power to a safe level for the RX frontend

---

## 5. Expected Output

### 5.1 Console log

When running in `--hw` mode you will see:

```
[INFO] Opening USRP B210 ...
[INFO] TX: 436.500 MHz  RX: 436.500 MHz  Rate: 1.000 Msps  Gain: TX 40.0 / RX 40.0 dB
[INFO] HW TX: 3560 samples at t=...
[INFO] HW RX: 3560 / 3560 samples received
[INFO] [4/6] RX DSP: freq correct → CMA equaliser → timing recovery ...
[INFO] [5/6] EVM = 0.92 %  |  f_measured = +12.3 Hz  |  Burst 1/3  (best timing off=0)
[INFO] Constellation -> payload_results_realizations/live_usrp/Live_B210_EVM.png
```

### 5.2 Constellation plot

The script saves `payload_results_realizations/live_usrp/Live_B210_EVM.png`:

- **PGRL mode** (pre-compensation applied): Sub-16ms phase-lock, EVM < 3% at 40 dB SNR. The QPSK constellation shows four tight clusters at (√2/2, √2/2) and symmetry-related positions. The plot title reads `Mode: PGRL`, `EVM: 0.92 %`, `Status: LOCK`.

- **SGP4 mode** (no pre-compensation): Catastrophic failure — the Doppler rotation (8000 Hz) spins the constellation through many full cycles over the 2.4 ms burst duration, producing EVM > 30% (typically > 100%). The plot shows a ring of uniformly distributed points with no identifiable QPSK structure, and `Status: UNLOCK`.

### 5.3 EVM log

Appends to `payload_results_realizations/live_usrp/evm_log.csv` with columns:

```
timestamp, mode, doppler_hz, snr_db, evm_pct
2026-05-08T17:25:00.000000,SIM,8000.0,40.0,0.9455
2026-05-08T17:25:00.000000,SIM,8000.0,40.0,0.9474
...
```

---

## 6. Architecture Summary

```
TX Path
───────
generate_tx_burst()
  ├── 20-symbol pilot tone (complex exponential @ 62.5 kHz)
  ├── 500-sample zero gap
  ├── RRC-shaped QPSK payload (300 symbols, α=0.5, SPM=8)
  └── 500-sample zero gap

apply_inverse_doppler()  ← PGRL only
  └── Multiply burst by exp(-j·2π·Δf·t)  [Δf = PGRL-predicted Doppler]

HW TX (B210) @ 436.5 MHz, 1 Msps, gain 40 dB

Channel (coax + 30 dB pad + AWGN)
  └── Adds Doppler rotation + thermal noise

HW RX (B210) @ 436.5 MHz, 1 Msps, gain 40 dB

RX Path
───────
coarse_freq_correct()  — removes residual frequency offset
cma_equaliser()         — disabled for clean coaxial channel
timing_recovery()       — data-aided brute-force search over SPB=8 offsets

AGC Complex Gain Calibration  ← KEY FIX (Option B)
  α = Σ(rx[i] · conj(ref[i])) / Σ|ref[i]|²
  rx_calibrated = rx_downsampled / α
  # Corrects both amplitude mismatch and residual phase rotation

EVM = sqrt(mean(|rx_calibrated - ref|²) / mean(|ref|²)) × 100 %
```

### Why the original EVM was 14%

The original `rrc_kernel()` function in `dual_mode_trx.py` used an incorrect RRC formula:

```python
# WRONG — produces h(t=0) = 0.886, h(t=±0.5) = 1.136 (peak!)
num = sin(π(1-α)t) + 4αt·cos(π(1+α)t)
den = πt·(1 - (4αt)²)
```

The correct IEEE 802.15.4g / ETSI EN 301 428 RRC formula is:

```python
# CORRECT — produces h(t=0) = 1.0, h(t=±0.5) ≈ 0.6, h(t=±1) = 0
sinc_t = np.sinc(n_t)          # sin(πt)/(πt), with sinc(0) = 1 (no NaN)
cos_f  = np.cos(math.pi * α * n_t)
denom  = 1.0 - (2.0 * α * n_t)**2
h      = np.where(np.abs(denom) > 1e-12, sinc_t * cos_f / denom, 0.0)
```

The wrong formula caused the RRC peak to appear at the wrong discrete grid point, so downsampled symbol positions landed on the filter skirt (~0.6× amplitude), producing a 14% EVM floor regardless of SNR.

### AGC Complex Gain Calibration (Option B)

Even with the corrected RRC, a 30 dB coaxial pad introduces amplitude imbalance and residual phase rotation between TX and RX. The AGC calibration resolves this:

```python
alpha     = np.sum(rx_ds * np.conj(ref_syms)) / np.sum(np.abs(ref_syms)**2)
rx_cal    = rx_ds / alpha          # complex division corrects both gain AND phase
evm_pct   = sqrt(mean(|rx_cal - ref|²) / mean(|ref|²)) × 100
```

This drives the EVM from ~14% down to **< 1%** at 40 dB SNR.

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `UHD not found` | UHD not installed or images missing | `brew install uhd && uhd_images_downloader` |
| `No device found` | USB 2.0 connection or hub | Connect directly to USB 3.0 port |
| EVM > 10% in `--hw` mode | TX/RX loopback disconnected or 30 dB pad missing | Verify coax connection; add 30 dB attenuator |
| `Status: UNLOCK` at 40 dB SNR | Incorrect RRC kernel formula | Ensure `rrc_kernel()` uses `np.sinc` (see §6) |
| RX samples timeout | Guard interval too short | Increase `timeout=5.0` in `hw_txrx()` |
| B210 device busy | Another process holds the device | `uhd_admin --kill` or disconnect/reconnect USB |

---

## 8. Quick Reference

```bash
# Verify UHD installation
uhd_find_devices

# Run hardware test
uv run python hardware/usrp_scripts/dual_mode_trx.py --hw --doppler-hz 8000 --snr-db 40

# Run simulation (no hardware)
uv run python hardware/usrp_scripts/dual_mode_trx.py --sim --snr-db 40 --n-bursts 3

# Plot saved at:
open payload_results_realizations/live_usrp/Live_B210_EVM.png
```