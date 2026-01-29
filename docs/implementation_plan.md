# Formosat-5 PINN Protocol Implementation Plan

## Goal Description

Implement a Real-Time Data Pipeline and Physics-Informed Neural Network (PINN) for Formosat-5 orbit prediction. The system must run on NVIDIA RTX 5080 using PyTorch Nightly (CUDA 12.x), leveraging TensorFloat-32 and Mixed Precision. The model will combine LSTM (temporal) and ResNet (spatial) architectures, trained with a loss function incorporating J2 perturbation physics.

## Proposed Changes

### Environment & Dependencies

- **Project Structure**: `pinn_protocol/` initialized via `uv`.
- **Dependencies**:
  - `requests` (Data fetching)
  - `sgp4` (Orbit propagation/Ground Truth)
  - `numpy`, `pandas`, `scipy`, `tqdm`, `matplotlib` (Scientific stack)
  - **PyTorch Nightly**: `torch>=2.6.0` with CUDA 12.6 support for RTX 5080.

### Data Engineering (`src/data_fetcher.py`)

- Fetch TLE from CelesTrak for NORAD ID 42920 (Formosat-5).
- Propagate orbit using SGP4 for 300 minutes at 1.0s intervals.
- Save as `real_training_data.npz`.

### Model Architecture (`src/model.py`)

- **HybridPINN Class**:
  - **LSTM Branch**: Captures temporal state dynamics.
  - **ResNet Branch**: Refines features/residual correction.
  - **Fusion**: Combine outputs for final state prediction.
- Support for Transfer Learning (Backbone freezing).

### Training Pipeline (`src/train.py`)

- **Optimization**:
  - Enable `torch.set_float32_matmul_precision('high')`.
  - Use `torch.amp.autocast('cuda')`.
- **Physics Loss**:
  - Compute $\ddot{r}$ using Autograd.
  - specialized loss function: $L_{total} = L_{data} + \lambda L_{J2}$.
  - $L_{J2}$ compares network acceleration with analytical J2 perturbed acceleration.

## Verification Plan

### Automated Tests

- `validate_env.py`: Assert `torch.cuda.is_available()` and check version.
- `data_fetcher.py`: Run and verify `real_training_data.npz` is created and contains valid data shapes.
- `train.py`: Run a short training loop to verify loss convergence and hardware utilization (no errors with AMP/TF32).

## Phase B: High-Precision SP3 Data

### Goal

Upgrade from SGP4 (km-level) to Precise Ephemeris (SP3, cm-level).

### Status

**IMPLEMENTED & VERIFIED**

- Infrastructure ready.
- Awaiting user upload of `.sp3` files to `raw_data/sp3/`.

### Changes

1.  **Dependencies**: `astropy` (Coordinates, Time), `h5py` (Performance).
2.  **Loader**: `src/data/sp3_loader.py`.
    - Support SP3-c/d format.
    - Implement differentiation for Velocity ($\mathbf{v} = d\mathbf{r}/dt$).
    - Convert ITRF (Earth Fixed) $\to$ GCRS (Inertial).
3.  **Ingestion**: `src/ingest_sp3.py`.
    - Batch convert `raw_data/sp3/*.sp3` to `data/precise_training_data.npz`.

## Phase C: Golden Physics Engine

### Goal

Achieve < 50m accuracy by modeling environmental forces explicitly, leaving only residuals for the Neural Network.

### Changes

1.  **Advanced Forces** (`src/physics/advanced_forces.py`):
    - **Lunisolar**: Third-body perturbations (Sun/Moon) using `astropy`.
    - **SRP**: Cannonball model with Cylindrical Earth Shadow.
    - **Gravity**: J2, J3, J4 terms.
2.  **Solver** (`src/physics/golden_solver.py`):
    - Integrate $\mathbf{a}_{total} = \mathbf{a}_{geo} + \mathbf{a}_{3rd} + \mathbf{a}_{SRP} + \mathbf{a}_{drag} + \mathbf{a}_{NN}$.
3.  **Training** (`src/train_golden.py`):
    - Optimize on SP3 data using the Golden Solver.
