#!/usr/bin/env bash
# check_hardware_macos.sh
# Smoke-test script to verify USRP B210 connectivity and Python environment.
# Run after setup_macos_m2.sh and after connecting the USRP B210.

set -euo pipefail

echo "=============================================="
echo " USRP B210 Hardware Smoke Test"
echo "=============================================="

# ── UHD version ─────────────────────────────────────────────────────────
echo ""
echo "[1/5] UHD version..."
if command -v uhd_config_info >/dev/null 2>&1; then
  uhd_config_info --version || echo "WARNING: uhd_config_info failed"
else
  echo "WARNING: uhd_config_info not found in PATH"
fi

# ── Find devices ─────────────────────────────────────────────────────────
echo ""
echo "[2/5] Searching for USRP devices..."
if command -v uhd_find_devices >/dev/null 2>&1; then
  uhd_find_devices || echo "NOTE: No devices found (connect B210 first)"
else
  echo "WARNING: uhd_find_devices not in PATH"
fi

# ── Probe device ────────────────────────────────────────────────────────
echo ""
echo "[3/5] Probing USRP..."
if command -v uhd_usrp_probe >/dev/null 2>&1; then
  uhd_usrp_probe 2>&1 | head -20 || echo "NOTE: uhd_usrp_probe may need sudo or device not connected"
else
  echo "WARNING: uhd_usrp_probe not in PATH"
fi

# ── Python environment ───────────────────────────────────────────────────
echo ""
echo "[4/5] Checking Python environment..."
cd "$(dirname "$0")/.."  # repo root
uv run python - <<'PYEOF'
import sys
print(f"  Python {sys.version.split()[0]}")

import numpy as np
print(f"  numpy {np.__version__} OK")

try:
    import uhd
    print(f"  python-uhd OK (UHD {uhd.__version__})")
except ImportError as e:
    print(f"  python-uhd not available: {e}")
    print("  -> Using UHD CLI tools (uhd_rx_cfile) as fallback")

try:
    import matplotlib
    print(f"  matplotlib OK")
except ImportError:
    print("  WARNING: matplotlib not available")

print("  All imports OK")
PYEOF

# ── UHD RX smoke test ───────────────────────────────────────────────────
echo ""
echo "[5/5] UHD RX smoke test (1 Msps, 5 MHz center, 60 dB gain, 0.5 s)..."
if command -v uhd_rx_cfile >/dev/null 2>&1; then
  TMPFILE=$(mktemp /tmp/uhd_smoke_XXXXXX.fc32)
  echo "  Capturing to $TMPFILE ..."
  # Use short duration to avoid blocking; timeout via background
  timeout 10 uhd_rx_cfile \
    --args="type=b200" \
    --freq 5e6 \
    --rate 1e6 \
    --gain 60 \
    --duration 0.5 \
    --output "$TMPFILE" 2>&1 | tail -5 || true
  if [ -f "$TMPFILE" ] && [ -s "$TMPFILE" ]; then
    SIZE=$(du -h "$TMPFILE" | cut -f1)
    echo "  Captured $SIZE ($TMPFILE)"
    echo "  Next: uv run python hardware/usrp_scripts/analyze_capture.py $TMPFILE"
  else
    echo "  No capture file generated (device may not be connected)"
  fi
  rm -f "$TMPFILE"
else
  echo "  uhd_rx_cfile not in PATH — skip RX smoke test"
fi

echo ""
echo "=============================================="
echo " Smoke test complete."
echo " If USRP was found and RX succeeded, hardware is ready."
echo "=============================================="