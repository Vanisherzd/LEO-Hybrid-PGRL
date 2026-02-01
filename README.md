# Physics-Guided Residual Learning (PGRL) for Autonomous LEO Operations

[![Project Status: Verified](https://img.shields.io/badge/Status-Reproducibility_Verified-green.svg)](https://github.com/Vanisherzd/LEO-Hybrid-PGRL)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Abstract

Autonomous satellite operations in Low Earth Orbit (LEO) require trajectory prediction with sub-meter precision. Traditional analytical models (e.g., SGP4) suffer from atmospheric drag unmodeled residuals, while pure neural-integration strategies (Neural ODEs) exhibit Lyapunov instability. This research presents a **Physics-Guided Residual Learning (PGRL)** framework:

1.  **Classical Physics**: Utilization of SGP4 (analytical propagation), which is constrained by simplified atmospheric drag and J2-J4 geopotential models.
2.  **Standard Neural ODEs**: Direct state-space integration using Differentiable RK4 Solvers. Empirical tests revealed **Lyapunov Instability**, wherein pure neural dynamics diverge significantly over long prediction horizons.
3.  **Physics-Guided Residual Learning (PGRL)**: Optimization of the modeling objective to target $\Delta \mathbf{s}$ (residuals) rather than the absolute state $\mathbf{s}$. By anchoring the model with SGP4, we achieve stable, sub-200m RMSE.

## Mathematical Formulation

The satellite state $\mathbf{s} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$ is modeled as:
$$\mathbf{s}_{true}(t) = \mathbf{s}_{sgp4}(t) + \mathcal{N}_{\theta}(\mathbf{s}_{sgp4}(t))$$
where $\mathcal{N}_{\theta}$ is the neural corrector trained to minimize the residual $\Delta \mathbf{s}$ against high-fidelity SP3 ephemerides.

## Experiment Setup

- **Hardware**: NVIDIA RTX 5080 GPU (Ampere/Blackwell optimized).
- **Optimizer**: Two-stage optimization (AdamW followed by L-BFGS).
- **Baselines**: MLP-Neural-ODE, LSTM, GRU, and Attention-based sequence models.

## Results

Detailed benchmarks over a 100-minute integration window:

| Model Architecture | RMSE (km) | Stability      |
| :----------------- | :-------- | :------------- |
| SGP4 (WGS72)       | 1.52      | Stable         |
| Pure Neural ODE    | 0.09      | Divergent      |
| **Hybrid PGRL**    | **0.12**  | **Convergent** |
| LSTM (Recurrent)   | 120.4     | Divergent      |

_Note: Hybrid PGRL maintains stability across multiple orbital periods, whereas pure neural models diverge exponentially._

## Repository Structure

- `src/pinn/`: Core PGRL and Physics modules.
- `src/rl/`: Autonomous MAC protocols and Gymnasium environments.
- `src/models/`: Neural network architecture variants.
- `logs/`: Deterministic training metrics (CSV).
- `plots/`: Standardized scientific figures (Fig1-Fig5).
- `weights/`: Verified model checkpoints (.pth).

## Usage

### 1. Reproduce Full Study

Executes training for all benchmarks, hybrid models, and RL agents.

```bash
uv run python src/reproduce_study.py --mode paper
```

### 2. Generate Figures

Generates the 5 definitive figures from verified logs and weights.

```bash
uv run python src/generate_figures.py
```
