# Hardware figures — LR1121 LR-FHSS SDR signal detection

Preliminary IQ-level hardware evidence (USRP B210 @ 868 MHz, TX/RX antenna,
1 Msps, gain 45, 10 s). Source: `hardware/artifacts/lr1121_signal_detected_20260604_000358/`.

| File | Content |
|------|---------|
| `fig_hw_lrfhss_evidence_full.pdf` | Preferred paper figure: TX-ON waterfall, TX-OFF reference, max-hold ON/OFF, and three-trial repeatability |
| `fig_hw_lrfhss_onoff_comparison.png` | TX-ON vs TX-OFF max-hold / occupancy comparison (8.88 dB delta) |
| `fig_hw_lrfhss_on_waterfall.png` | TX-ON waterfall — sparse hop-like time-frequency dashes |
| `fig_hw_lrfhss_on_maxhold.png` | TX-ON max-hold spectrum — multiple narrow peaks across band |
| `fig_hw_lrfhss_repeatability.png` | ON/OFF delta bar chart across three trials (8.88, 11.87, 9.82 dB) |

The individual waterfall/max-hold assets remain in this directory for artifact inspection.

## Suggested caption

> Preliminary IQ-level LR1121 LR-FHSS signal detection using USRP B210 at
> 868 MHz. TX-ON shows sparse hop-like time-frequency occupancy and an 8.88 dB
> ON/OFF max-hold occupancy delta relative to TX-OFF. This figure demonstrates
> RF signal presence only; it is not a standard-compliant LR-FHSS decoding or
> PER result.

## LaTeX snippet (single-column)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/hardware/fig_hw_lrfhss_onoff_comparison.png}
    \caption{Preliminary IQ-level LR1121 LR-FHSS signal detection using USRP
    B210 at 868~MHz. TX-ON shows sparse hop-like time-frequency occupancy and an
    8.88~dB ON/OFF max-hold occupancy delta relative to TX-OFF. This figure
    demonstrates RF signal presence only; it is not a standard-compliant
    LR-FHSS decoding or PER result.}
    \label{fig:hw_lrfhss}
\end{figure}
```

## Repeatability caption (three trials)

> Preliminary IQ-level LR1121 LR-FHSS signal detection using a USRP B210 at
> 868~MHz. TX-ON/OFF controls show consistent sparse-hop RF evidence across
> three trials, with ON/OFF deltas of 8.88, 11.87, and 9.82~dB. This validates
> RF signal presence only and does not constitute standard-compliant LR-FHSS
> decoding or PER measurement.

### Option A — comparison plot only (single column)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/hardware/fig_hw_lrfhss_onoff_comparison.png}
    \caption{Preliminary IQ-level LR1121 LR-FHSS signal detection using a USRP
    B210 at 868~MHz. TX-ON shows sparse hop-like time-frequency occupancy and an
    8.88~dB ON/OFF max-hold occupancy delta relative to TX-OFF. RF signal
    presence only; not standard-compliant LR-FHSS decoding or PER.}
    \label{fig:hw_lrfhss}
\end{figure}
```

### Option B — comparison + repeatability bar chart (two panels)

```latex
\begin{figure}[t]
    \centering
    \begin{subfigure}{0.49\linewidth}
        \includegraphics[width=\linewidth]{figures/hardware/fig_hw_lrfhss_onoff_comparison.png}
        \caption{TX-ON vs TX-OFF (run 1).}
        \label{fig:hw_lrfhss_a}
    \end{subfigure}\hfill
    \begin{subfigure}{0.49\linewidth}
        \includegraphics[width=\linewidth]{figures/hardware/fig_hw_lrfhss_repeatability.png}
        \caption{ON/OFF delta across three trials.}
        \label{fig:hw_lrfhss_b}
    \end{subfigure}
    \caption{Preliminary IQ-level LR1121 LR-FHSS signal detection using a USRP
    B210 at 868~MHz. TX-ON/OFF controls show consistent sparse-hop RF evidence
    across three trials, with ON/OFF deltas of 8.88, 11.87, and 9.82~dB. This
    validates RF signal presence only and does not constitute standard-compliant
    LR-FHSS decoding or PER measurement.}
    \label{fig:hw_lrfhss}
\end{figure}
```

(Option B needs `\usepackage{subcaption}` in the preamble.)

**Claim boundary:** IQ-level signal detection only — NOT LR-FHSS decoding, NOT
PER, NOT a full gateway receiver. Status: **repeated** hardware-signal-detected
(bring-up), ON/OFF corroborated across three trials. Paper claims must not
exceed this.
