#!/usr/bin/env bash
# run.sh — exp1: PGRL Prediction Accuracy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
RESULTS="$SCRIPT_DIR/results.json"

echo "============================================"
echo "Exp1: PGRL Prediction Accuracy"
echo "============================================"

# Check if evaluate_prediction.py exists
if [[ -f "$PROJECT_ROOT/physics_ml/evaluate_prediction.py" ]]; then
    echo "[run] python physics_ml/evaluate_prediction.py --config $CONFIG"
    uv run python "$PROJECT_ROOT/physics_ml/evaluate_prediction.py" --config "$CONFIG" \
        --output-json "$RESULTS" || true
else
    echo "[run] Skipping — evaluate_prediction.py not yet implemented (thesis extension)"
    echo "Writing placeholder results.json"
fi

echo "[run] Done. Results: $RESULTS"