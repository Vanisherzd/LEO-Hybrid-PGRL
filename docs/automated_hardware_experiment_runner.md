# Automated Conducted-Hardware Experiment Runner

## Safety model
- **Conducted or shielded only**: coax + attenuator, dummy load, or shielded
  enclosure. **No over-the-air (OTA) testing.**
- TX power is **never raised by the tooling**; it uses the configured value.
- **Hardware modes (`iq_only`, `decoded_rx`) refuse to run without `--armed`**,
  and the config must set `safety.conducted_or_shielded: true`.
- Real TX (`tx.backend: lr1121_uart`) additionally requires `--i-have-hardware`
  (enforced by the TX runner).
- Preflight runs first and aborts on any fatal check (missing gitignore rules,
  missing scripts, missing input paths, unarmed hardware mode).

## Modes and claims

| Mode | TX / RX | Hardware? | PER | Claim |
|---|---|---|---|---|
| `dryrun` | mock / mock | no | computed (self-test) | software pipeline only; no hardware claim |
| `iq_only` | file_replay / iq_only | yes (`--armed`) | **unavailable** | IQ-level RF signal detection only |
| `decoded_rx` | lr1121_uart or mock / log_parser | yes (`--armed`) | computed from decoded payloads | packet delivery / PER |

PER is reported **only** when decoded RX payloads exist. `iq_only` always
reports "PER unavailable: no receiver-side packet decoding."

## Commands

### Dry-run (no hardware)
```
uv run python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/templates/dryrun_experiment.yaml \
  --out validation_runs/hw_dryrun_001
```

### IQ-only conducted (requires --armed)
```
uv run python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/templates/iq_only_conducted_experiment.yaml \
  --out validation_runs/hw_iq_only_001 --armed
```

### Decoded-RX conducted (requires --armed + real decoded RX log)
```
# edit templates/decoded_rx_conducted_experiment.yaml: set rx.rx_log_path
uv run python hardware/packet_validation/run_hardware_experiment.py \
  --config hardware/packet_validation/templates/decoded_rx_conducted_experiment.yaml \
  --out validation_runs/hw_decoded_rx_001 --armed
```

### Finalize (classify artifacts)
```
uv run python hardware/packet_validation/finalize_hardware_manifest.py \
  --run-dir validation_runs/hw_decoded_rx_001
```

## Commit vs do-not-commit
- **Commit** (after copying out of gitignored `validation_runs/`): small
  summaries, `manifest*.yaml`, `artifact_index.csv`, `sha256sums.txt`,
  `packet_validation_summary.*`, decoded RX log (text), small IQ metadata JSON.
- **Never commit**: raw IQ (`*.fc32`/`*.cfile`/`*.sigmf-data`/`*.dat`), oversized
  files, the whole `validation_runs/` tree, build artifacts.
- `validation_runs/` and raw IQ extensions are gitignored; `finalize` emits
  `recommended_commit_files.txt` and `do_not_commit_files.txt`.

## Status
No measured hardware PER is claimed. The paper is unchanged; decoded-RX PER
requires a real conducted decoded RX log.
