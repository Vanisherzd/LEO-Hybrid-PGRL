#!/usr/bin/env bash
# run_lrfhss_tx.sh — Baseline SWDM001 LR-FHSS TX + PGRL-compensated runner
# Usage: ./run_lrfhss_tx.sh [baseline|compensated|compare]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mode="${1:-baseline}"

case "$mode" in
  baseline)
    echo "[TX] Baseline LR-FHSS TX (no PGRL compensation)"
    echo "  → Replace with actual SWDM001 binary + invocation"
    ;;

  compensated)
    echo "[TX] PGRL-compensated LR-FHSS TX"
    python -c "
import sys, json
sys.path.insert(0, '${PROJECT_ROOT}'
from semtech_validation.tx_config_from_pgrl import load_example, make_semtech_tx_config
cfg = make_semtech_tx_config(load_example(), base_freq_hz=915_000_000)
print(json.dumps(cfg, indent=2))
"
    echo "  → Apply config to SWDM001, then capture with sdr_hwil/capture_iq.py"
    ;;

  compare)
    echo "[TX] Compare baseline vs PGRL-compensated TX"
    "$0" baseline
    "$0" compensated
    echo "Compare with: python sdr_hwil/estimate_cfo.py <baseline.cfile> <compensated.cfile>"
    ;;

  *)
    echo "Usage: $0 [baseline|compensated|compare]"
    exit 1
    ;;
esac