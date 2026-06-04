# Hardware figures — LR1121 LR-FHSS SDR signal detection

Submission-ready hardware figures use curated successful artifacts from:

`hardware/artifacts/lr1121_signal_detected_repeatability_20260604/`

Preferred paper figures:

| File | Content | Curated source |
|------|---------|----------------|
| `fig_hw_lrfhss_waterfall_onoff.pdf` | Fig. 5: TX-ON/TX-OFF waterfall evidence from the same successful trial | `run1_000358/run1_on_waterfall.png`, `run1_000358/run1_off_waterfall.png` |
| `fig_hw_lrfhss_detection_summary.pdf` | Fig. 6: run-1 TX-ON/TX-OFF max-hold comparison plus three-trial repeatability | `run1_000358/run1_comparison.png`, `repeatability_summary.json` |

Legacy figure retained for comparison/debug only:

| File | Content |
|------|---------|
| `fig_hw_lrfhss_evidence_full.pdf` | Older 2x2 composite figure; not preferred for the paper because the panels are too small at final PDF scale |

Generator scripts:

- `generate_hw_waterfall_onoff.py`
- `generate_hw_detection_summary.py`

Claim boundary for all paper figures:

- IQ-level RF signal detection only
- not LR-FHSS decoding
- not PER
- not a full gateway receiver
