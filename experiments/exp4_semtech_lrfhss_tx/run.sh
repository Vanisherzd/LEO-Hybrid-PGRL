#!/usr/bin/env bash
# run.sh — exp4: Semtech LR-FHSS TX Bring-Up
# Hardware status: Planned. Set config.yaml hardware.status='acquired' when LR1121 is in hand.
# Dry-run mode: generates TX config JSON without hardware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "Exp4: Semtech LR-FHSS TX Bring-Up"
echo "============================================"

MODE="${1:-dry-run}"

case "$MODE" in
  dry-run)
    echo "[dry-run] Generating PGRL TX config JSON (no hardware required)"
    uv run python "$PROJECT_ROOT/semtech_validation/tx_config_from_pgrl.py" \
        --output "$SCRIPT_DIR/logs/dry_run_config.json" 2>&1 || \
    python -c "
import sys, json
sys.path.insert(0, '$PROJECT_ROOT')
from semtech_validation.tx_config_from_pgrl import load_example, make_semtech_tx_config, save_config
cfg = make_semtech_tx_config(load_example(), base_freq_hz=915_000_000)
save_config(cfg, '$SCRIPT_DIR/logs/dry_run_config.json')
print('[dry-run] Config:', json.dumps(cfg, indent=2))
"
    echo "[dry-run] Config written to $SCRIPT_DIR/logs/dry_run_config.json"
    ;;

  baseline|compensated|compare)
    echo "[TX] Use semtech_validation/run_lrfhss_tx.sh $MODE"
    bash "$PROJECT_ROOT/semtech_validation/run_lrfhss_tx.sh" "$MODE"
    ;;

  *)
    echo "Usage: $0 [dry-run|baseline|compensated|compare]"
    exit 1
    ;;
esac

echo "[run] Done."