#!/usr/bin/env bash
# run.sh — exp2: Guard-Band Energy Tradeoff
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
RESULTS="$SCRIPT_DIR/results.json"

echo "============================================"
echo "Exp2: Guard-Band Energy Tradeoff"
echo "============================================"

python -c "
import sys, json, math
sys.path.insert(0, '$PROJECT_ROOT')
from controller.pgrl_output_schema import PGRLOutput
from controller.guard_band_policy import adaptive_guard_time, guard_overhead_fraction, missed_opportunity_probability
from controller.energy_model import total_opportunity_energy

# Simulate 4 baselines across 100 passes (Monte Carlo)
import numpy as np
np.random.seed(42)

baselines = [
    {'name': 'fixed_guard_30ms',  'base_guard': 0.030,  'use_sigma': False, 'sigma': 0.0},
    {'name': 'sgp4_only',          'base_guard': 0.500,  'use_sigma': False, 'sigma': 1.500},
    {'name': 'pgrl_mean_only',    'base_guard': 0.500,  'use_sigma': False, 'sigma': 0.200},
    {'name': 'pgrl_uncertainty',  'base_guard': 0.500,  'use_sigma': True,  'sigma': 0.016},
]

rows = []
for bl in baselines:
    guards, misses, energies = [], [], []
    for _ in range(200):
        sigma = bl['sigma'] * (1 + 0.2*np.random.randn())
        g = adaptive_guard_time(bl['base_guard'], sigma if bl['use_sigma'] else 0.0, k=3.0)
        guards.append(g)
        misses.append(missed_opportunity_probability(max(sigma, 1e-6), g))
        e = total_opportunity_energy(g, rx_on_s=2.0, tx_s=0.5)
        energies.append(e['total_j'])

    rows.append({
        'name': bl['name'],
        'guard_mean_s': round(float(np.mean(guards)), 4),
        'guard_overhead_pct': round(float(np.mean([guard_overhead_fraction(g, 240.0) for g in guards])) * 100, 2),
        'missed_opp_rate': round(float(np.mean(misses)), 4),
        'energy_per_opportunity_j': round(float(np.mean(energies)), 5),
    })

result = {
    'experiment': 'exp2_guard_band_energy',
    'validation_type': 'simulation',
    'baselines': [r['name'] for r in rows],
    'metrics': {
        'guard_mean_s': {r['name']: r['guard_mean_s'] for r in rows},
        'guard_overhead_percent': {r['name']: r['guard_overhead_pct'] for r in rows},
        'missed_opportunity_rate': {r['name']: r['missed_opp_rate'] for r in rows},
        'energy_per_opportunity_j': {r['name']: r['energy_per_opportunity_j'] for r in rows},
    }
}
with open('$RESULTS', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
"
echo "[run] Results → $RESULTS"