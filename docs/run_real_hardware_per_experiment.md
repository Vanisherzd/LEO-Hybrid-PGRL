# Runbook: Real Hardware PER Experiment (Conducted/Shielded)

Prereq: complete `docs/real_hardware_connection_checklist.md`. Conducted/shielded
only; no OTA; TX power not raised by tooling. Hardware modes require `--armed`.

## A. Dry-run sanity (no hardware)
```
python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/templates/dryrun_experiment.yaml \
  --out validation_runs/pre_hw_dryrun
```
Expected: software PER computed (self-test only; not a hardware claim).

## B. Stage-5 IQ-only replay (requires --armed)
```
python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/real_configs/real_iq_only_stage5_replay.yaml \
  --out validation_runs/real_iq_only_stage5_replay \
  --armed
```
Expected: **PER unavailable** (IQ-level signal detection only).

## C. Real decoded-RX run A (alpha=0.00)
Edit `real_configs/real_runA_alpha0_decoded_rx.yaml`: set `rx.rx_log_path` to the
actual decoded RX log; set `operator` / `safety.acknowledged_by`.
```
python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/real_configs/real_runA_alpha0_decoded_rx.yaml \
  --out validation_runs/real_runA_alpha0 \
  --armed
```

## D. Real decoded-RX run B (alpha=0.25)
Edit `rx.rx_log_path` in `real_runB_alpha025_decoded_rx.yaml`.
```
python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/real_configs/real_runB_alpha025_decoded_rx.yaml \
  --out validation_runs/real_runB_alpha025 \
  --armed
```

## E. Real decoded-RX run C (stress)
Edit `rx.rx_log_path` and `attenuator_db` in `real_runC_stress_decoded_rx.yaml`.
```
python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/real_configs/real_runC_stress_decoded_rx.yaml \
  --out validation_runs/real_runC_stress \
  --armed
```

## F. Finalize (classify commit-safe vs local-only artifacts)
```
python hardware/packet_validation/finalize_hardware_manifest.py \
  --run-dir validation_runs/real_runA_alpha0
# repeat for real_runB_alpha025, real_runC_stress
```

## G. Plot (only after real decoded RX logs exist)
```
python scripts/plot_packet_delivery_metrics.py \
  --summaries validation_runs/real_runA_alpha0/packet_validation_summary.json \
              validation_runs/real_runB_alpha025/packet_validation_summary.json \
              validation_runs/real_runC_stress/packet_validation_summary.json \
  --out paper/figures/fig6_packet_delivery_metrics.pdf
```
**Do NOT add Fig. 6 to the paper** until real decoded RX logs exist and PER is
legitimately measured. The plotter skips any run whose PER is unavailable.

## Notes
- `validation_runs/` and raw IQ (`*.fc32`) are gitignored — never committed.
- Commit only small summaries/manifests (see `recommended_commit_files.txt` from
  finalize); copy them out of `validation_runs/` to a committed location.
