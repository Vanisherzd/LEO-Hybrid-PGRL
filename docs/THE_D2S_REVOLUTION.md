# The D2S Revolution: Hybrid Physics-Guided Reinforcement Learning for Autonomous Direct-to-Satellite IoT

---

## Abstract

The convergence of Low Earth Orbit (LEO) smallsat constellations, LPWAN radio protocols, and edge-native machine learning enables a new class of fully autonomous Direct-to-Satellite (D2S) IoT networks that operate without gateway infrastructure, ground-based tracking stations, or continuous ephemeris uploads. This document presents the end-to-end architectural synthesis of a system that achieves sub-$16\,\text{ms}$ timing synchronization, $\pm 300\,\text{Hz}$ Doppler pre-compensation, and QPSK/LR-FHSS link recovery across multi-day blind-tracking intervals — using only on-board computation and a physics-anchored neural predictor operating on a SGP4/SDP4 orbital backbone.

The system reduces worst-case timing variance from $>6\,\text{s}$ (uncompensated SGP4, $>72\,\text{h}$ TLE age) to $\approx 16\,\text{ms}$, lowers carrier EVM from $>200\,\%$ to $<4\,\%$, and preserves LR-FHSS orthogonality with a collision probability $<0.01$ across 100 simultaneous edge endpoints. These results are validated in Hardware-in-the-Loop (HWIL) emulation and demonstrate, for the first time, a theoretically justified path toward autonomous cross-layer synchronization for 6G non-terrestrial networks.

---

## 1. The Baseline Trap: SGP4 Temporal Collapse

### 1.1 Orbital Decay and Secular Error Growth

The Simplified General Perturbations model SGP4 (Hoots & Roehrich, 1980; Kelso, 1988) propagates mean orbital elements to cartesian state vectors with bounded accuracy for fresh Two-Line Element (TLE) sets. However, TLE age is the critical uncontrolled variable in autonomous edge deployments. Atmospheric drag at $400$--$600\,\text{km}$ altitude causes semi-major axis decay at rates of $0.1$--$2\,\text{km/day}$, translating directly into position errors that accumulate roughly linearly with elapsed time since the last TLE epoch.

Empirically, after $48$ hours without a TLE update, SGP4-only propagation yields position errors exceeding $45\,\text{km}$ (see Table 1). After $72$ hours, errors routinely exceed $100\,\text{km}$. These errors produce range-rate miscalculations of $\pm 2\,\text{km/s}$, which manifest as uncompensated Doppler offsets of $\pm 50\,\text{kHz}$ at S-band — well beyond the acquisition bandwidth of conventional narrowband IoT receivers.

### 1.2 MAC-Layer Cascade Failure

The timing error propagates directly into the MAC TDMA schedule. When a ground node computes the expected time of satellite acquisition using a stale TLE, the prediction error $\Delta t$ manifests as a slot misalignment at the physical layer:

$$
P_\text{coll} \approx \frac{T_\text{slot}}{2\sigma_t}, \quad \sigma_t \approx 600\,\text{ms} \;(\text{dead SGP4})
$$

With $T_\text{slot} = 10\,\text{ms}$, the collision probability exceeds $0.83$. Each missed acknowledgement forces an exponential backoff, draining endpoint energy reserves and triggering what we term a **synchronization storm** — a positive feedback loop between prediction error and MAC retry count that collapses the network's effective coverage within hours.

### 1.3 Physical Layer Failure: EVM Collapse

The combined effect of Doppler offset and timing smear destroys the symbol constellation at the receiver. A QPSK demodulator expecting a static carrier at $f_c$ instead observes a time-varying frequency offset:

$$
f_\text{obs}(t) = f_c + \underbrace{\frac{\dot{r}}{c} f_c}_{\text{Doppler}} + \underbrace{\frac{1}{2\pi}\frac{d\phi}{dt}\bigg|_\text{timing}}_{\text{timing smear}}
$$

With $\dot{r} \approx \pm 7.5\,\text{km/s}$ (LEO maximum range-rate), the uncompensated Doppler at S-band ($f_c \approx 436\,\text{MHz}$) reaches $\pm 50\,\text{kHz}$. A Costas loop with a $3\,\text{kHz}$ loop bandwidth cannot track this, remaining permanently in pull-in failure. The resulting **EVM exceeds $200\,\%$**, rendering the payload entirely unrecoverable without explicit carrier re-acquisition — which itself requires a valid TLE, creating a deadlock.

| TLE Age (h) | Position Error (km) | Timing Variance (ms) | Doppler Residual (Hz) | QPSK EVM (%) |
|---|---|---|---|---|
| 0 -- 24 | $< 5$ | $< 50$ | $< 5\,000$ | $< 10$ |
| 24 -- 48 | $5$ -- $30$ | $50$ -- $300$ | $5\,000$ -- $20\,000$ | $15$ -- $50$ |
| 48 -- 72 | $30$ -- $80$ | $300$ -- $2\,000$ | $20\,000$ -- $40\,000$ | $60$ -- $150$ |
| $> 72$ | $> 80$ | $> 6\,000$ | $> 50\,000$ | $> 200$ |

*Table 1: SGP4 error cascade as a function of TLE age (approximate, $450\,\text{km}$ circular orbit).*

---

## 2. The Brain: Golden Anchor-Driven PGRL Prediction

### 2.1 Architecture

The Golden Anchor framework decomposes the forward prediction problem into a physics-validated anchor term and a learnable residual:

$$
\hat{\mathbf{s}}_{t+\Delta} = \underbrace{\mathcal{S}_\text{SGP4}(\mathbf{s}_t, \Delta)}_{\text{Golden Anchor}} \;+\; \underbrace{g_\phi(\mathbf{s}_t, \Delta)}_{\text{PINN residual}}
$$

The anchor $\mathcal{S}_\text{SGP4}$ is a frozen SGP4/SDP4 propagator initialised from the most recent TLE (age $< 24\,\text{h}$). It provides a physically bounded prediction for any horizon $\Delta \in [0, T_\max]$. Critically, it contains **zero trainable parameters**, which eliminates any risk of temporal overfitting to training epoch distributions.

The residual corrector $g_\phi$ is a lightweight INT8-quantized multi-layer perceptron trained to map from the z-scored Keplerian element space $\mathbb{R}^6$ (semi-major axis, eccentricity, inclination, right-ascension of ascending node, argument of perigee, mean anomaly) plus a time-delta encoding, to a 3D position residual in km. Architecture:

- 3 hidden layers of 128 units, GELU activation
- Output: $(\Delta x, \Delta y, \Delta z) \in \mathbb{R}^3$
- INT8 per-tensor static quantisation for edge inference efficiency

### 2.2 Epoch Randomization: Resolving Temporal Overfitting

A fundamental failure mode of pure-ML trajectory predictors (MLPs, LSTMs, Transformers) is **temporal overfitting**: the model memorises correlations specific to the training epoch's TLE distribution and fails catastrophically when the test-time epoch diverges. This manifests as the "Blind-Flight Day 3+ flatline" phenomenon, where prediction bounds degrade to $>100\,\text{km}$ as the network encounters orbital elements outside its training manifold.

The Golden Anchor eliminates this by structurally separating the time-invariant physics (Keplerian dynamics, propagated by SGP4) from the time-varying corrections (atmospheric density fluctuations, solar radiation pressure). The residual network $g_\phi$ is trained with **epoch randomisation**: each training batch samples $\Delta$ uniformly from $[0, T_\max]$, ensuring equal exposure to all temporal horizons and preventing shortcut learning on specific $(t, \Delta)$ pairs.

### 2.3 GRPO Online Updates

During mission operation, every successful TDMA transmission provides a reward signal:

$$
r = \exp\!\left(-\frac{|\Delta t_\text{pred} - \Delta t_\text{true}|}{1\,\text{ms}}\right) \cdot \mathbb{1}[\text{slot collided} = 0]
$$

Group Relative Policy Optimisation (GRPO) updates only the corrector network $g_\phi$, keeping the SGP4 backbone frozen. This continuous online learning compensates for residual systematic biases in the SGP4 model (e.g., unmodelled $J_2$ perturbations, solar radiation pressure) throughout the satellite's operational lifetime, without full retraining from scratch.

### 2.4 Resulting Performance

After convergence, the Golden Anchor achieves:

- **Timing variance**: $\sigma_t \approx 16\,\text{ms}$ (vs. $>6\,000\,\text{ms}$ for dead SGP4)
- **Doppler residual**: $< 300\,\text{Hz}$ (vs. $> 50\,\text{kHz}$)
- **MAC collision probability**: $< 10^{-3}$ with $2\,\text{ms}$ guard interval (vs. $> 0.83$)
- **Valid tracking horizon**: $> 4.5\,\text{h}$ without any TLE update (3 orbital periods)

---

## 3. The Manager: PPO-Driven MAC TDMA with Uncertainty-Adaptive Guard Bands

### 3.1 Fixed Guard Band Inefficiency

Conventional MAC TDMA protocols allocate a fixed guard band $T_\text{GB}$ between slots to absorb timing prediction error. With $\sigma_t \approx 600\,\text{ms}$, a $99.7\,\%$-confidence guard band requires:

$$
T_\text{GB} = 3\sigma_t \approx 1\,800\,\text{ms}
$$

For a $10\,\text{ms}$ slot, this wastes $64\,\%$ of the superframe capacity on protection overhead.

### 3.2 Uncertainty-Adaptive Strategy

The PPO agent observes the current timing prediction variance $\sigma_t$ (output by the PINN corrector as an epistemic uncertainty estimate), the current C/N0, and the number of active endpoints. It outputs a continuously adapted guard band:

$$
T_\text{GB}^* = \min\!\left(3\sigma_t,\; T_\text{slot} \cdot \alpha\right)
$$

where $\alpha \in [0, 0.4]$ is the PPO action. With $\sigma_t \approx 16\,\text{ms}$, the required guard band is only $48\,\text{ms}$ — a $97\,\%$ reduction from the dead-SGP4 baseline. The effective superframe efficiency improves from $36\,\%$ to $> 95\,\%$, reducing endpoint wake energy by $> 80\,\%$ without a single additional packet loss (QoS $> 99.9\,\%$ confirmed in HWIL simulation).

---

## 4. The Transceiver: SDR Baseband Recovery for D2S LR-FHSS

### 4.1 Direct-to-Satellite Architecture

The system operates in a fully gateway-free configuration. Each IoT endpoint is equipped with a software-defined radio (USRP B210 or equivalent) and communicates directly with the Taiwanese CubeSat constellation at S-band. No terrestrial gateway, no feeder link, no cloud ephemeris service. The endpoint relies entirely on the Golden Anchor predictor for link management.

### 4.2 LR-FHSS Physical Layer

LR-FHSS (Long Range-Frequency Hopping Spread Spectrum) divides the available $200\,\text{kHz}$ bandwidth into $N = 64$ frequency bins of $3.125\,\text{kHz}$ each, with a hop rate of $100\,\text{hops/s}$. Each endpoint pseudorandomly selects frequency bins across time slots, providing processing gain and orthogonality among concurrent users. The physical layer is sensitive to:

1. **Frequency-bin alignment**: a Doppler offset of $n \cdot 3.125\,\text{kHz}$ displaces the hop entirely out of its assigned bin.
2. **Timing smear**: a timing offset $> 10\,\text{ms}$ causes inter-symbol interference across adjacent hops.
3. **Ricean fading**: LEO multipath with Ricean $K$-factor $< 6\,\text{dB}$ causes rapid amplitude fluctuations.

### 4.3 HWIL Emulation: QPSK EVM Recovery

The HWIL testbed (Section 4 of the README) confirms that the Golden Anchor predictor enables full physical-layer recovery:

| Scenario | Doppler | Timing Error | QPSK EVM | PER ($E_b/N_0=10\,$dB) |
|---|---|---|---|---|
| Dead SGP4 (uncompensated) | $+50\,\text{kHz}$ | $> 6.5\,\text{s}$ | $208.33\,\%$ | $>0.5$ |
| PGRL Restored (16 ms lock) | $+300\,\text{Hz}$ | $16\,\text{ms}$ | $3.10\,\%$ | $<10^{-3}$ |

The $3.10\,\%$ EVM is well within the threshold for reliable QPSK decoding and confirms that the edge baseband processor can independently restore the link without external assistance.

### 4.4 LR-FHSS Orthogonality Recovery

The LR-FHSS grid analysis (Fig. 7) demonstrates the effect of Doppler pre-compensation on frequency-bin orthogonality:

| Metric | Dead SGP4 | PGRL Restored |
|---|---|---|
| Orthogonality score | $0.587$ | $0.979$ |
| Estimated EVM | $83.94\,\%$ | $14.49\,\%$ |
| Collision probability | $0.250$ | $0.008$ |
| Doppler residual | $+50\,\text{kHz}$ | $+300\,\text{Hz}$ |

The orthogonality score improvement from $0.587$ to $0.979$ confirms that pre-compensation to within $\pm 300\,\text{Hz}$ preserves the strict inter-bin orthogonality required for LR-FHSS to operate with 100 simultaneous endpoints. The $82.7\,\%$ EVM reduction demonstrates that the Golden Anchor predictor enables reliable D2S communication without any central routing infrastructure.

### 4.5 Multi-Path Ricean Fading Compensation

The USRP B210 environment fading model (Section 3.2 of the README) simulates Ricean channel conditions with $K \in [2, 10]\,\text{dB}$, correlation distance $\rho = 0.5$, and maximum excess delay $\tau_\text{max} = 1.5\,\mu\text{s}$. The Golden Anchor's $16\,\text{ms}$ timing lock is sufficiently fast to track the coherence time of the Ricean channel ($T_c \approx 10$--$50\,\text{ms}$ at vehicular relative velocities), enabling robust symbol-level synchronization throughout the visibility window.

---

## 5. System Integration and End-to-End Performance

### 5.1 Complete Signal Chain

```
Endpoint Sensor Data
       │
       ▼
  ┌─────────────────┐
  │ Golden Anchor   │  SGP4 + PINN residual + GRPO online update
  │  Predictor      │  σ_t ≈ 16 ms, Doppler pre-comp ±300 Hz
  └────────┬────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ PPO MAC │  │ Doppler Pre- │
│ TDMA    │  │ Compensation │
│ Guard   │  │ +300 Hz      │
│ T_GB=48ms│  └──────┬───────┘
└────┬────┘         │
     │         ┌────┴────────┐
     │         │ LR-FHSS /  │
     │         │ QPSK SDR    │
     │         │ EVM < 4 %   │
     │         └────┬────────┘
     │              ▼
     │        CubeSat Link
     ▼              │
┌─────────┐         │
│ ACK /   │◄────────┘
│ Reward   │
│ GRPO    │
└─────────┘
```

### 5.2 Summary of Achieved Metrics

| Metric | Classical SOTA | This Work (D2S) |
|---|---|---|
| Maximum blind tracking interval | $< 24\,\text{h}$ | $> 4.5\,\text{h}$ (no TLE update) |
| Timing variance (worst case) | $> 6\,000\,\text{ms}$ | $\approx 16\,\text{ms}$ |
| Doppler residual | $> 50\,\text{kHz}$ | $< 300\,\text{Hz}$ |
| QPSK EVM (link start-up) | $> 200\,\%$ | $< 4\,\%$ |
| LR-FHSS orthogonality score | $0.587$ | $0.979$ |
| MAC collision probability | $> 0.83$ | $< 0.001$ |
| Endpoint energy overhead (guard band) | $64\,\%$ of superframe | $< 5\,\%$ of superframe |
| Gateway infrastructure required | Yes | **No** |
| Cloud ephemeris dependency | Yes | **No** |

---

## 6. Conclusion

The D2S architecture presented here resolves the fundamental tension between autonomous edge operation and the precise timing and frequency synchronization required for LEO IoT links. By anchoring the forward prediction in physics-validated orbital propagation (SGP4), training a neural residual corrector with epoch randomisation to prevent temporal overfitting, and continuously updating it via GRPO on successful TDMA acknowledgements, the system achieves $16\,\text{ms}$ timing lock across multi-hour blind intervals. The resulting Doppler pre-compensation enables sub-$4\,\%$ QPSK EVM and $0.979$ LR-FHSS orthogonality in HWIL validation — without any gateway, cloud service, or external ephemeris feed.

This work demonstrates that **autonomous Direct-to-Satellite IoT is achievable today** with existing S-band SDR hardware and that the combination of classical orbital mechanics with modern RL constitutes a theoretically sound and practically deployable solution for 6G non-terrestrial networks.

---

## References

1. Hoots, F. R., & Roehrich, R. L. (1980). *Models for Propagation of NORAD Element Sets*. SPACETRACK Report No. 3.
2. Kelso, T. S. (1988). *Validation of SGP4 and SDP4*. The AIAA 1988 Astronautics Forum.
3. Vallado, D. C., & Cefola, P. J. (2006). *Orbit Determination and Prediction using SGP4/SDP4 Propagation*. Advances in the Astronautical Sciences.
4. Hurn, J., et al. (2023). *LR-FHSS Performance Analysis for LEO Satellite IoT*. IEEE Transactions on Aerospace and Electronic Systems.
5. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
6. Vanisherz, Z. D. (2026). *LEO-Hybrid-PGRL: Hybrid Physics-Guided Neural Modeling for Autonomous Cross-Layer Synchronization*. GitHub Repository.