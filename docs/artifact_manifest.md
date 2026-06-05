# Artifact Manifest

Current branch head at manifest creation: `4b5a8850`

This manifest documents the repo artifacts used by the paper
`PGRL-Assisted Uncertainty-Aware LR-FHSS Uplink Control for Direct-to-Satellite IoT`.

## Validation labels

- `trace-driven`: predictor/control evaluation against recorded orbital references or shared baseline residual labels.
- `simulation`: analytical or Monte Carlo controller evaluation with no RF capture.
- `proxy-simulation`: synthetic RF-quality or grid-alignment proxy, not LR-FHSS decoding.
- `hardware-signal-detected`: IQ-level TX ON/OFF RF evidence only.

## Main paper source

- Paper: `paper/globecom_main.tex`
- Bibliography: `paper/refs.bib`
- Main summary table source: `paper/tables/main_results.json`

## Dataset / config anchors

- Predictor config: `experiments/exp1_pgrl_prediction/config.yaml`
- Guard policy config: `experiments/exp2_guard_band_energy/config.yaml`
- LR-FHSS grid proxy config: `experiments/exp3_lrfhss_grid_proxy/config.yaml`
- SDR/QPSK proxy config: `experiments/exp5_sdr_doppler_precomp/config.yaml`
- Robustness scaffold: `experiments/exp6_robustness/config.yaml`

## Figure reproduction

### Fig. 2

- Asset: `paper/figures/fig2_pgrl_uncertainty.pdf`
- Upstream source: `paper/tables/main_results.json` and `experiments/exp1_pgrl_prediction/addons/uncertainty_calibration_results.json`

### Fig. 3

- Asset: `paper/figures/fig3_guard_energy.pdf`
- Upstream source: `experiments/exp2_guard_band_energy/results.json`

### Fig. 4

- Asset: `paper/figures/fig4_lrfhss_grid_proxy.pdf`
- Upstream source: `experiments/exp3_lrfhss_grid_proxy/results.json`

### Fig. 5

- Asset: `paper/figures/hardware/fig_hw_lrfhss_detection_evidence.pdf`
- Generator: `paper/figures/hardware/generate_hw_detection_evidence.py`
- Hardware logs:
  - `hardware/artifacts/lr1121_signal_detected_repeatability_20260604/repeatability_summary.json`
  - `hardware/artifacts/lr1121_signal_detected_repeatability_20260604/run1_000358/run1_comparison.json`
  - `hardware/artifacts/lr1121_signal_detected_repeatability_20260604/run1_000358/run1_on_analysis.json`
  - `hardware/artifacts/lr1121_signal_detected_repeatability_20260604/run1_000358/run1_off_analysis.json`

## New reviewer-defense outputs

- Ablation summary:
  - `results/ablation_summary.json`
  - `results/ablation_summary.csv`
  - `paper/tables/ablation_table.tex`
- Hard split / leakage-control scaffold:
  - `results/hard_split_summary.json`
  - `results/hard_split_summary.csv`
  - `paper/tables/hard_split_table.tex`
- Guard-k sweep:
  - `results/guard_k_sweep.json`
  - `results/guard_k_sweep.csv`
  - `paper/figures/guard_k_sweep.pdf`
  - `paper/tables/guard_k_sweep_table.tex`
- CFO stress proxy:
  - `results/cfo_stress_summary.json`
  - `results/cfo_stress_summary.csv`
  - `paper/figures/cfo_stress_proxy.pdf`
  - `paper/tables/cfo_stress_table.tex`
- Hardware repeatability / negative controls:
  - `results/hardware_repeatability_summary.json`
  - `results/hardware_repeatability_summary.csv`
  - `paper/tables/hardware_repeatability_table.tex`

## Exact commands

Run from repo root.

```bash
python3 experiments/summary_table.py
python3 scripts/generate_ablation_summary.py
python3 scripts/generate_hard_split_summary.py
env MPLCONFIGDIR=/tmp/mpl python3 scripts/generate_guard_k_sweep.py
env MPLCONFIGDIR=/tmp/mpl python3 scripts/generate_cfo_stress_summary.py
python3 scripts/analyze_hardware_repeatability.py
env MPLCONFIGDIR=/tmp/mpl python3 paper/figures/hardware/generate_hw_detection_evidence.py
tectonic -X compile paper/globecom_main.tex --keep-logs
```

## Scope boundaries

- None of the proxy outputs claim LR-FHSS decoding or PER.
- None of the hardware outputs claim gateway interoperability.
- The maximum hardware claim is IQ-level RF signal detection / preliminary RF-path evidence.

## Known TODOs

- `results/hard_split_summary.*` records the leakage-control scaffold, not measured hard-split metrics.
- `results/ablation_summary.*` preserves `TODO` cells where the repo lacks per-variant controller measurements.
- The manifest should be regenerated after future local commits if you need the exact post-edit commit hash embedded here.
