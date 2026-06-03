#!/usr/bin/env bash
# run_lrfhss_tx.sh — LR-FHSS TX CONFIG / DRY-RUN WRAPPER (does NOT transmit)
#
# ============================================================================
# HONESTY NOTICE
# ============================================================================
# This script is a CONFIG / DRY-RUN WRAPPER ONLY. It does NOT transmit anything
# from this Mac. The Mac CANNOT transmit LR-FHSS. The actual LR-FHSS RF
# transmission is performed by the Semtech LR1121 radio on the SWDM001 board,
# driven by SWDM001 firmware that you build in Keil and flash onto the
# NUCLEO-L476RG (NOD_L476RG) over ST-LINK.
#
# This wrapper only:
#   - prints the SWDM001 build / flash instructions, and
#   - generates the PGRL-derived TX config JSON (advisory; not auto-applied), and
#   - prints the matching USRP capture / analyze commands for cross-checking.
#
# WARNING: "Packet sent!" on the board UART only means the firmware command
# completed. It does NOT prove the SDR received any RF, and it does NOT prove a
# valid LR-FHSS waveform was captured. Confirm reception independently with the
# USRP capture + analyzer (IQ-level validation only).
# ============================================================================
#
# Usage: ./run_lrfhss_tx.sh [baseline|compensated|compare|help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default LR11xx LR-FHSS demo RF frequency (from SWDM001 demo source).
DEFAULT_RF_FREQUENCY_HZ=868000000   # 868 MHz

# SWDM001 build / flash coordinates.
KEIL_BUILD_TARGET="lr1121_xtal"
KEIL_PROJECT="project/keil_polling_STM32L476/STM32L476.uvprojx"
KEIL_BINARY="project/keil_polling_STM32L476/STM32L476_lr11xx.bin"
FLASH_TARGET="NOD_L476RG"
UART_BAUD="115200"
DEMO_SRC="src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c"

mode="${1:-baseline}"

print_warning() {
  echo ""
  echo "WARNING: \"Packet sent!\" on the UART means the firmware command"
  echo "         completed. It does NOT prove the SDR received RF, nor that a"
  echo "         valid LR-FHSS waveform was captured. Verify with USRP capture."
}

do_baseline() {
  echo "[TX] BASELINE — SWDM001 LR-FHSS ping (no PGRL compensation)"
  echo ""
  echo "This Mac does NOT transmit. Build + flash SWDM001 firmware to the board:"
  echo ""
  echo "  1. Open Keil project:   $KEIL_PROJECT"
  echo "  2. Select build target: $KEIL_BUILD_TARGET"
  echo "  3. Build -> produces:   $KEIL_BINARY"
  echo "  4. Flash target:        $FLASH_TARGET (NUCLEO-L476RG via ST-LINK)"
  echo "  5. Open UART @ ${UART_BAUD} 8N1 to observe demo output."
  echo ""
  echo "  Default RF frequency:   ${DEFAULT_RF_FREQUENCY_HZ} Hz (868 MHz)"
  echo "  Defined in:             $DEMO_SRC (RF_FREQUENCY)"
  echo ""
  echo "  Expected UART output:"
  echo "    \"LR11XX-LR-FHSS Ping Init\""
  echo "    \"Packet sent!\""
  print_warning
}

do_compensated() {
  echo "[TX] COMPENSATED — PGRL-derived LR-FHSS TX config"
  echo ""
  echo "Generating PGRL config (advisory only; NOT auto-applied to firmware):"
  ( cd "$PROJECT_ROOT" && uv run python semtech_validation/tx_config_from_pgrl.py )
  echo ""
  echo "To apply the PGRL-compensated frequency you MUST edit firmware by hand:"
  echo ""
  echo "  1. Open:   $DEMO_SRC"
  echo "  2. Change RF_FREQUENCY from ${DEFAULT_RF_FREQUENCY_HZ} (868 MHz)"
  echo "     to 915000000 (915 MHz) or the PGRL-compensated frequency from the"
  echo "     generated config JSON."
  echo "  3. Rebuild in Keil (target $KEIL_BUILD_TARGET) -> $KEIL_BINARY"
  echo "  4. Reflash to $FLASH_TARGET."
  echo ""
  echo "  NOTE: The generated config JSON does NOT auto-apply to the firmware."
  echo "        RF_FREQUENCY is a compile-time constant; it requires a manual"
  echo "        edit + rebuild + reflash to take effect."
  print_warning
}

do_compare() {
  echo "[TX] COMPARE — baseline vs PGRL-compensated (dry-run only)"
  echo ""
  do_baseline
  echo ""
  echo "------------------------------------------------------------------"
  echo ""
  do_compensated
  echo ""
  echo "=================================================================="
  echo "After flashing each firmware variant, capture + analyze with USRP:"
  echo ""
  echo "  # Single capture + analysis:"
  echo "  uv run python hardware/usrp_scripts/analyze_capture.py \\"
  echo "    <capture.fc32> --sample-rate 2000000 \\"
  echo "    --output-json <out.json> --plot <out.png> \\"
  echo "    --maxhold-plot <out_maxhold.png> --signal-threshold-db 8"
  echo ""
  echo "  # Full repeatable sweep (freqs x antennas):"
  echo "  bash scripts/run_lr1121_sdr_sweep.sh \\"
  echo "    --serial 8000304 --freqs \"868e6,915e6,923e6\" \\"
  echo "    --antennas \"RX2,TX/RX\" --rate 2e6 --gain 45 --duration 30"
  echo "=================================================================="
}

case "$mode" in
  -h|--help|help)
    cat <<EOF
Usage: $0 [baseline|compensated|compare|help]

CONFIG / DRY-RUN WRAPPER ONLY. This Mac does NOT transmit LR-FHSS.
Transmission happens on the Semtech LR1121 board via SWDM001 firmware
flashed from Keil onto the NUCLEO-L476RG, NOT from this Mac.

  baseline      Print SWDM001 build/flash instructions + default RF freq (868 MHz).
  compensated   Generate PGRL config, then print manual firmware-edit instructions
                to change RF_FREQUENCY (e.g. to 915 MHz) and reflash.
  compare       Run baseline + compensated, then print USRP capture/analyze cmds.
  help          Show this help.
EOF
    exit 0
    ;;
  baseline)    do_baseline ;;
  compensated) do_compensated ;;
  compare)     do_compare ;;
  *)
    echo "ERROR: unknown mode: $mode" >&2
    echo "Usage: $0 [baseline|compensated|compare|help]" >&2
    exit 1
    ;;
esac
