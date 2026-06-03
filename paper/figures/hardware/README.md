# Hardware figures — LR1121 LR-FHSS SDR signal detection

Preliminary IQ-level hardware evidence (USRP B210 @ 868 MHz, TX/RX antenna,
1 Msps, gain 45, 10 s). Source: `hardware/artifacts/lr1121_signal_detected_20260604_000358/`.

| File | Content |
|------|---------|
| `fig_hw_lrfhss_onoff_comparison.png` | TX-ON vs TX-OFF max-hold / occupancy comparison (8.88 dB delta) |
| `fig_hw_lrfhss_on_waterfall.png` | TX-ON waterfall — sparse hop-like time-frequency dashes |
| `fig_hw_lrfhss_on_maxhold.png` | TX-ON max-hold spectrum — multiple narrow peaks across band |

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

**Claim boundary:** IQ-level signal detection only — NOT LR-FHSS decoding, NOT
PER, NOT a full gateway receiver. Status: hardware signal-detected (bring-up),
ON/OFF corroborated. Paper claims must not exceed this.
