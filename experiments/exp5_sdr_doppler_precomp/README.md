# Experiment 5 — SDR HWIL Doppler Pre-Compensation

**Purpose:** SDR-based hardware-in-the-loop test under controllable LEO-like Doppler, CFO, AWGN, and interference impairment.

## Impairments
- Doppler ramp (simulated LEO pass)
- Residual CFO after pre-compensation
- AWGN (configurable SNR)
- Optional co-channel interference

## Baselines
1. No compensation
2. SGP4 compensation
3. PGRL compensation
4. Oracle compensation

## Metrics
- Residual CFO [Hz]
- EVM proxy [%]
- Spectral alignment error
- Packet detection probability

## Framework
Uses SDR (USRP B210) as the RX front-end — captures IQ samples, applies impairment models in software, evaluates RF-quality proxies. This is a D2S-like validation platform, NOT a full standard-compliant gateway.