# SDR / HWIL Validation Module

The SDR is used as an **IQ-level measurement platform** for RF-quality assessment under LEO-like Doppler, CFO, noise, and interference impairments. It does **not** implement a full standard-compliant LR-FHSS gateway receiver.

## Provided Capabilities
- `capture_iq.py`         — IQ sample capture from USRP B210
- `estimate_cfo.py`        — CFO estimation from captured IQ
- `evm_proxy.py`           — EVM as RF-quality proxy (QPSK reference)
- `packet_detection_proxy.py` — Energy-based packet detection
- `plot_waterfall.py`     — Waterfall / spectrogram visualization
- `inject_doppler.py`     — Programmable Doppler ramp injection

## Naming Convention
All SDR metrics are reported as **RF-quality proxies**, not as full-system PER/LLR metrics. EVM is the QPSK EVM proxy; packet detection is energy-based; orthogonalities are grid proxies.

## EVM Proxy vs LR-FHSS PER
> `evm_proxy.py` uses a QPSK constellation as an RF chain quality indicator. It is **NOT** a replacement for LR-FHSS packet-error-rate (PER) measured on a standards-compliant LR-FHSS decoder. The EVM proxy demonstrates that Doppler/CFO pre-compensation improves RF signal quality at the physical layer, which translates to better LR-FHSS demodulation robustness in practice.