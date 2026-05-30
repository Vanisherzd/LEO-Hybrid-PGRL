#!/usr/bin/env bash
# setup_macos_m2.sh
# Sets up native macOS (Apple Silicon / Intel) environment for USRP B210 hardware testing.
# Run this ONCE on a fresh Mac before connecting the USRP.
# Docker is NOT used for hardware — only for simulation / offline analysis.

set -euo pipefail

echo "=============================================="
echo " Mac M2/M3 Hardware Environment Setup"
echo "=============================================="

# ── Step 1: Homebrew ──────────────────────────────────────────────────────
echo "[1/7] Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew not found. Install from https://brew.sh first."
  exit 1
fi
brew update

# ── Step 2: UHD and dependencies ─────────────────────────────────────────
echo "[2/7] Installing UHD, Python, git, build tools..."
brew install uhd python git pkg-config cmake

# ── Step 3: UHD firmware images ───────────────────────────────────────────
echo "[3/7] Downloading UHD images (needed for USRP B210)..."
if command -v uhd_images_downloader >/dev/null 2>&1; then
  uhd_images_downloader
else
  echo "WARNING: uhd_images_downloader not found."
  echo "  Try: brew link uhd"
  echo "  Or manually: https://files.ettus.com/binaries/uhd/UHD-MACUNTITLED.pkg"
fi

# ── Step 4: Python environment ───────────────────────────────────────────
echo "[4/7] Setting up Python with uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Verify uv works
uv --version

# Sync Python dependencies
cd "$(dirname "$0")/.."  # repo root
uv sync

# ── Step 5: Verify UHD installation ───────────────────────────────────────
echo "[5/7] Verifying UHD..."
uhd_config_info --version || echo "WARNING: uhd_config_info not in PATH"

# ── Step 6: Verify USRP B210 ──────────────────────────────────────────────
echo "[6/7] Looking for USRP B210..."
if uhd_find_devices 2>/dev/null | grep -q "B210"; then
  echo "SUCCESS: USRP B210 found!"
else
  echo "NOTE: No USRP found yet. Connect the B210 via USB 3.0 and retry:"
  echo "  uhd_find_devices"
fi

# ── Step 7: Done ─────────────────────────────────────────────────────────
echo "[7/7] Done."
echo ""
echo "Next steps:"
echo "  1. Connect USRP B210 directly to a USB 3.0 port (avoid unpowered hubs)"
echo "  2. Run: bash scripts/check_hardware_macos.sh"
echo "  3. Run: uv run python hardware/usrp_scripts/dual_mode_trx.py --hw"
echo ""
echo "Docker is used ONLY for:"
echo "  - PGRL training"
echo "  - trace-driven simulation"
echo "  - figure generation"
echo "  - offline IQ analysis (NOT live hardware)"