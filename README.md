# Hybrid Physics-Guided Residual Learning (PGRL) for Autonomous LEO Operations

[![Project Status: Research Grade](https://img.shields.io/badge/Status-Research_Grade-blue.svg)](https://github.com/Vanisherzd/LEO-Hybrid-PGRL)
[![Reproducibility: Verified](https://img.shields.io/badge/Reproducibility-Verified-green.svg)](docs/CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌌 Overview

Autonomous satellite operations in Low Earth Orbit (LEO) demand trajectory prediction with unprecedented precision. Conventional analytical models (SGP4) suffer from unmodeled residuals (e.g., atmospheric drag), while pure data-driven approaches (Neural ODEs) exhibit **Lyapunov Instability**, leading to exponential divergence.

This repository implements a **Hybrid PGRL** framework that anchors high-fidelity neural correctors to stable analytical physics. By modeling residuals ($\Delta \mathbf{s}$) instead of absolute states ($\mathbf{s}$), we achieve meter-level precision over multi-orbital integration windows.

---

## 🔬 Core Methodology

The framework follows a dual-stage integration strategy:

1. **Global Anchor**: SGP4 provides a stable, gravitationally consistent baseline $\mathbf{s}_{sgp4}(t)$.
2. **Neural Corrector**: A deep MLP $\mathcal{N}_{\theta}$ predicts the instantaneous residual discrepancy induced by non-conservative forces (SRP, Drag).
3. **Synthesis**: The true state is estimated as:
   $$\mathbf{s}_{true}(t) = \mathbf{s}_{sgp4}(t) + \mathcal{N}_{\theta}(\mathbf{s}_{sgp4}(t), \mu, \dots)$$

---

## 📊 Scientific Benchmarks

Integrated performance over a continuous **100-minute window** (Formosat-5 Characteristics):

| Model Architecture        | Local Force MSE          | Global Trajectory RMSE | Stability Profile        |
| :------------------------ | :----------------------- | :--------------------- | :----------------------- |
| **SGP4 (Baseline)**       | N/A                      | 1.52 km                | Globally Stable          |
| **Pure Neural ODE (MLP)** | **$1.2 \times 10^{-6}$** | **$> 10,000$ km**      | **Divergent** (Lyapunov) |
| **Hybrid PGRL (Ours)**    | $4.5 \times 10^{-5}$     | **157 meters**         | **Stable & Accurate**    |

> [!IMPORTANT]
> **The MLP Paradox**: While pure AI models show superior _local_ force learning (lowest MSE), they lack the Hamiltonian structure required for integration stability. Our Hybrid approach "domesticates" this neural flexibility via physics anchoring.

---

## 🛠️ Operational Pipeline

### 1. Reproduce Full Study

Executes the E2E suite: Data Ingestion -> Benchmark Training -> Hybrid Fine-tuning -> RL Policy Optimization.

```bash
uv run python src/reproduce_study.py --mode paper
```

### 2. Scientific Visualization

Generates definitive Figs 1-5 using empirical integration and high-fidelity styling.

```bash
uv run python src/generate_figures.py
```

## 📁 Project Anatomy

- **`src/pinn/`**: PGRL implementation and residual training loops.
- **`src/rl/`**: Autonomous MAC protocols using the stable predictor.
- **`src/physics/`**: High-fidelity Golden Truth solvers (J2-J4, SRP).
- **`plots/`**: Standardized Fig1-Fig5 outputs.

---

**Technical Report**: [Empirical Analysis of Lyapunov Instability](docs/FINAL_REPORT.md)
