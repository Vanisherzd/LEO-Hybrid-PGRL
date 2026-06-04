# Hardware figures — LR1121 LR-FHSS SDR signal detection

Submission-ready hardware figures use curated successful artifacts from:

`hardware/artifacts/lr1121_signal_detected_repeatability_20260604/`

Preferred paper figures:

| File | Content | Curated source |
|------|---------|----------------|
| `fig_hw_lrfhss_detection_evidence.pdf` | Preferred paper figure: detector-focused ON/OFF spectral evidence, occupancy metrics, repeatability, and score/UART summary | `run1_000358/run1_comparison.png`, `run1_000358/run1_comparison.json`, `run1_000358/run1_on_analysis.json`, `run1_000358/run1_off_analysis.json`, `repeatability_summary.json` |

Legacy figure retained for comparison/debug only:

| File | Content |
|------|---------|
| `fig_hw_lrfhss_waterfall_onoff.pdf` | Waterfall-only diagnostic figure; kept out of the main paper because TX-ON/TX-OFF raw waterfalls are visually too similar |
| `fig_hw_lrfhss_detection_summary.pdf` | Split summary figure from the previous paper pass |
| `fig_hw_lrfhss_evidence_full.pdf` | Older 2x2 composite figure; not preferred for the paper because the panels are too small at final PDF scale |

Generator scripts:

- `generate_hw_detection_evidence.py`
- `generate_hw_detection_summary.py`

Claim boundary for all paper figures:

- IQ-level RF signal detection only
- not LR-FHSS decoding
- not PER
- not a full gateway receiver
