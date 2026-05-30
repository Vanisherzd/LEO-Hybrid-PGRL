#!/usr/bin/env bash
# run_lrfhss_tx.sh — Baseline SWDM001 LR-FHSS TX + PGRL-compensated runner
# Usage: ./run_lrfhss_tx.sh [baseline|compensated|compare]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mode="${1:-baseline}"

case "$mode" in
  -h|--help|help)
    echo "Usage: $0 [baseline|compensated|compare]"
    echo ""
    echo "baseline      Print baseline SWDM001 TX placeholder"
    echo "compensated   Generate PGRL-compensated LR1121/LR11xx TX config"
    echo "compare       Run baseline then compensated dry-run"
    exit 0
    ;;

  baseline)
    echo "[TX] Baseline LR-FHSS TX (no PGRL compensation)"
    echo "  → TODO: Replace with actual SWDM001 binary + board-specific invocation"
    echo "  → Current script is a dry-run wrapper only; it does NOT transmit yet."
    ;;

  compensated)
    echo "[TX] PGRL-compensated LR-FHSS TX config"
    cd "$PROJECT_ROOT"
    uv run python semtech_validation/tx_config_from_pgrl.py
    echo "  → Apply semtech_validation/lr1121_tx_config_example.json to SWDM001 / LR1121 firmware."
    echo "  → Current script generates config only; it does NOT transmit yet."
    ;;

  compare)
    echo "[TX] Compare baseline vs PGRL-compensated TX dry-run"
    "$0" baseline
    "$0" compensated
    echo "Compare later with hardware captures:"
    echo "  uv run python hardware/usrp_scripts/analyze_capture.py <capture.fc32> --sample-rate 1000000 --output-json <out.json> --plot <out.png>"
    ;;

  *)
    echo "Usage: $0 [baseline|compensated|compare]"
    exit 1
    ;;
esac
