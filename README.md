# 🛰️ PINN Protocol: Hybrid PGRL for Autonomous LEO Operations

[![Project Status: Research Grade](https://img.shields.io/badge/Status-Research_Grade-blue.svg)](https://github.com/Vanisherzd/LEO-Hybrid-PGRL)
[![Stability: State-of-the-Art](https://img.shields.io/badge/Stability-35m_RMSE-green.svg)](docs/FINAL_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌌 Overview

Autonomous satellite operations in Low Earth Orbit (LEO) demand trajectory prediction with sub-100m precision. While conventional analytical models (SGP4) are globally stable, they suffer from unmodeled residuals (e.g., atmospheric drag). Conversely, pure data-driven approaches (Neural ODEs) exhibit **Lyapunov Instability**, leading to exponential divergence.

The **PINN Protocol** implements a **Hybrid Physics-Guided Residual Learning (PGRL)** framework. By anchoring a high-capacity neural corrector to a stable SGP4 baseline, we achieve **meter-level precision** over multi-orbital integration windows.

---

## 🔬 Core Methodology: Hybrid PGRL

The framework follows a **Residual Correction** strategy instead of direct state prediction:

1.  **Global Anchor**: SGP4 provides a gravitationally consistent, bounded baseline $\mathbf{s}_{sgp4}(t)$.
2.  **Neural Residual Learner**: A 3-layer MLP $\mathcal{N}_{\theta}$ (GeLU, [256, 256, 256]) is trained to predict the instantaneous position error $\Delta \mathbf{r}$ induced by non-conservative forces (Atmospheric Drag, SRP).
3.  **Synthesis**: The final estimated state is:
    $$\mathbf{s}_{hybrid}(t) = \mathbf{s}_{sgp4}(t) + \mathbf{s}_{corrector}(t)$$

This approach "domesticates" the neural network, preventing the quadratic error accumulation typical of unconstrained integrators.

---

## 📊 Scientific Benchmarks

Integrated performance over a continuous **120-minute window** (Formosat-5 & Formosat-7 baseline):

| Model Architecture        | F5 RMSE (120m)  | F7 RMSE (100m) | Stability Profile    |
| :------------------------ | :-------------- | :------------- | :------------------- |
| **SGP4 (Baseline)**       | 1.14 km         | 73.0 m         | Globally Stable      |
| **Pure Neural ODE (MLP)** | **> 100 km**    | **Divergent**  | Catastrophic         |
| **Hybrid PGRL (Ours)**    | **35.4 meters** | **3.8 meters** | **State-of-the-Art** |

### The 95% Performance Leap

On the drag-intensive Formosat-7 (F7) scenario, the Hybrid model achieves a **95% reduction in error** compared to SGP4, proving that AI can actively refine classical orbitography rather than simply replacing it.

---

## 📁 Repository Structure

- **`src/pinn/`**: Core PGRL implementation, including high-precision L-BFGS training loops.
- **`src/physics/`**: High-fidelity Golden Truth solvers (J2-J4, Atmospheric Drag, SRP).
- **`src/rl/`**: Autonomous MAC protocols optimized for the stable predictor.
- **`plots/`**: Definitive scientific figures (RIC Breakdown, Global Head-to-Head).
- **`weights/`**: Pre-trained high-precision correctors.

---

## 🛠️ Usage

### ⚙️ Reproduction

Execute the full academic suite (Data Synthesis -> Benchmark -> Refinement -> RL):

```bash
uv run python src/reproduce_study.py --mode paper
```

### 📈 Scientific Validation

Generate definitive Figs A-C using the unified validation pipeline:

```bash
uv run python src/final_validation.py
```

---

**Detailed Technical Report**: [Analysis of Lyapunov Instability and RIC Convergence](docs/FINAL_REPORT.md)
