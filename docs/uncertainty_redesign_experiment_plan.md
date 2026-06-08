# Uncertainty Redesign Experiment Plan — Option 2

**Date:** 2026-06-07
**Branch:** `uncertainty-redesign-proposal` (from `evidence-retrain-pgrl`)
**Status:** DESIGN ONLY — do not implement until explicitly approved
**SHA256 of deterministic checkpoint:** `5cac977f0b3f66f66124d1f1776bec50345c75c45e5c815ffd609de3f258f9a4`

---

## 1. Current Deterministic Baseline

### Checkpoint
| Property | Value |
|----------|-------|
| File | `artifacts/checkpoints/pgrl_deterministic_retrained.pt` |
| Size | 3.9 MB |
| SHA256 | `5cac977f0b3f66f66124d1f1776bec50345c75c45e5c815ffd609de3f258f9a4` |
| Training epochs | 78 (early stop triggered) |
| Architecture | TrajectoryPINN, orbital_elem_dim=6, hidden=256, layers=6, fourier=128 |
| Parameter count | 332,050 |

### Validated Metrics (7,200-sample validation set)
| Metric | Value |
|--------|-------|
| Position RMSE | **5.35 m** (full validation set) |
| Position RMSE (training best) | **4.77 m** (epoch 78) |
| Below-5m rate | **61.3 %** |
| Below-3m rate | **24.7 %** |
| Position median | 4.29 m |
| Position max | 16.44 m |
| Position p90 | 8.10 m |
| Velocity RMSE | 5.50 m/s |
| Per-satellite RMSE range | 5.27–5.47 m (consistent across 12 satellites) |

### Known Limitations of Deterministic Model
1. No uncertainty quantification — cannot claim calibrated σ bounds
2. No NLL / ECE / coverage metrics possible
3. Paper must remove all Bayesian / Gaussian NLL / calibrated-σ claims
4. Cannot regenerate paper's reported timing 16.4 ms, Doppler 234 Hz, NLL 0.17, ECE 0.0028/0.0012

---

## 2. Proposed Uncertainty Model

### Architecture Change: Dual-Head TrajectoryPINN

**Current (deterministic):**
```
forward(t, oe) → (batch, 6)   # 6 position/velocity values
final_layer: Linear(hidden_dim → 6)
```

**Proposed (uncertainty-aware):**
```
forward(t, oe) → (batch, 12)  # 6 mean + 6 logvar
final_layer: Linear(hidden_dim → 12)
```

**New output layout (12 values):**
| Index | Meaning |
|-------|---------|
| 0 | μ_x (km) |
| 1 | μ_y (km) |
| 2 | μ_z (km) |
| 3 | μ_vx (km/s) |
| 4 | μ_vy (km/s) |
| 5 | μ_vz (km/s) |
| 6 | logvar_x |
| 7 | logvar_y |
| 8 | logvar_z |
| 9 | logvar_vx |
| 10 | logvar_vy |
| 11 | logvar_vz |

**Output mode controlled by** `output_uncertainty: bool = False` constructor flag.
When `False`: Behaves identically to deterministic model (for inference/compatibility).
When `True`: Outputs 12 values; calling code must split mean/logvar.

### LogVar Clipping
```
logvar_i = clamp(logvar_i, min=-10.0, max=5.0)   # σ ∈ [0.00045, 12.2]
```
This prevents numerical instability while allowing σ up to ~12 km for high-uncertainty predictions.

### Loss Function: GaussianNLL + Physics Residual

**Combined loss:**
```
L_total = λ_NLL * L_NLL + λ_physics * L_physics + λ_energy * L_energy + λ_angmom * L_angmom
```

**Gaussian NLL per sample:**
```
L_NLL = (1/6) * Σ_i [logvar_i + (y_i - μ_i)^2 / exp(logvar_i)]
```

**λ schedule (warmup):**
| Epochs | λ_NLL | λ_physics |
|--------|-------|-----------|
| 1–10 | 0.01 | 0.0 |
| 11–30 | 0.1 | 0.01 |
| 31–end | 1.0 | 0.1 |

Rationale: Physics loss must not fight NLL during early uncertainty learning. Warmup lets the network learn mean structure before uncertainty.

### Calibration Metrics (computed on validation set)

After each epoch, compute on held-out validation set:

| Metric | Formula | Target |
|--------|---------|--------|
| NLL | mean of Gaussian NLL over all samples | < 0.5 |
| ECE (timing) | \|empirical_coverage(68%) − 0.68\| + \|empirical_coverage(95%) − 0.95\| averaged | < 0.02 (2%) |
| ECE (Doppler) | same for velocity components | < 0.02 (2%) |
| 68% coverage | fraction of samples where |y_i − μ_i| < σ_i | 60–75% |
| 95% coverage | fraction of samples where |y_i − μ_i| < 2σ_i | 90–98% |
| Sharpness | mean(σ_i) over position components | report only |

**Note on units:** NLL and ECE are computed on normalized units (pos/1e4, vel/10), not meters. Raw NLL will differ from paper-reported values which were on different target units.

---

## 3. Exact Files That Would Change

### 3.1 `physics_ml/pinn_core.py` — TrajectoryPINN class

**Changes:**

```python
# Line 78-86: __init__ signature
def __init__(
    self,
    orbital_elem_dim: int = 6,
    time_dim: int = 1,
    hidden_dim: int = 256,
    num_layers: int = 6,
    fourier_features: int = 128,
    omega0: float = 30.0,
    use_fourier: bool = True,
    output_uncertainty: bool = False,   # NEW
):
    # ...
    self.output_uncertainty = output_uncertainty
    if output_uncertainty:
        self.final_layer = nn.Linear(hidden_dim, 12)  # 6 mean + 6 logvar
    else:
        self.final_layer = nn.Linear(hidden_dim, 6)   # deterministic

# Line 120-147: forward method
def forward(self, t, orbital_elems) -> torch.Tensor:
    # ... existing logic ...
    state = self.final_layer(x)
    
    if self.output_uncertainty:
        mean = state[:, :6]
        logvar = torch.clamp(state[:, 6:], min=-10.0, max=5.0)
        # Return concatenated mean+logvar for loss computation
        # Shape: (batch, 12)
        return torch.cat([mean, logvar], dim=-1)
    else:
        return state  # (batch, 6) — unchanged
```

**What stays the same:** SirenLayer, FourierFeatureEmbedding, MultiScaleTrajectoryPINN, elem_scale/elem_bias, all SIREN layers.

### 3.2 `physics_ml/losses.py` — PINNTotalLoss class

**Changes:**

Add new class `GaussianNLLWithPhysicsLoss`:

```python
class GaussianNLLWithPhysicsLoss(nn.Module):
    """
    Combined loss for uncertainty-aware TrajectoryPINN.
    
    L = λ_NLL * L_NLL + λ_physics * L_physics + λ_energy * L_energy + λ_angmom * L_angmom
    
    L_NLL = (1/6) * Σ_i [logvar_i + (y_i - μ_i)^2 / exp(logvar_i)]
    """
    
    def __init__(
        self,
        w_data: float = 1.0,
        w_physics: float = 0.1,
        w_energy: float = 0.01,
        w_angmom: float = 0.01,
        lambda_nll_schedule: list = None,   # NEW: [(epoch, lambda), ...]
    ):
        super().__init__()
        # ... existing init ...
        self.lambda_nll_schedule = lambda_nll_schedule or [(1, 0.01), (11, 0.1), (31, 1.0)]
        self.current_nll_lambda = 0.01
    
    def gaussian_nll(self, pred_12: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian NLL for 12-output uncertainty model."""
        mean = pred_12[:, :6]
        logvar = pred_12[:, 6:]
        var = torch.exp(logvar)  # stable: logvar already clamped in forward
        
        # (y - μ)^T Σ^{-1} (y - μ) for diagonal Σ = Σ_i (y_i - μ_i)^2 / var_i + logvar_i
        diff = target - mean
        nll_per_dim = logvar + diff ** 2 / var
        return 0.5 * nll_per_dim.mean()
    
    def update_lambda(self, epoch: int):
        """Update λ_NLL based on schedule."""
        for max_epoch, lam in reversed(self.lambda_nll_schedule):
            if epoch >= max_epoch:
                self.current_nll_lambda = lam
                return
    
    def forward(self, pred_12: torch.Tensor, target: torch.Tensor, t: torch.Tensor, oe: torch.Tensor) -> dict:
        """Forward for 12-output uncertainty model."""
        mean = pred_12[:, :6]
        
        # Data loss (MSE on mean, as in deterministic case)
        l_data = F.mse_loss(mean, target)
        
        # Physics residual (on mean prediction only)
        l_physics = self.physics_residual(mean, t, oe)
        
        # Energy conservation (on mean prediction)
        l_energy = self._energy_loss(mean, oe)
        
        # Angular momentum conservation (on mean prediction)
        l_angmom = self._angmom_loss(mean, oe)
        
        # Gaussian NLL
        l_nll = self.gaussian_nll(pred_12, target)
        
        total_loss = (
            self.w_data * l_data
            + self.current_nll_lambda * l_nll
            + self.w_physics * l_physics
            + self.w_energy * l_energy
            + self.w_angmom * l_angmom
        )
        
        return {
            "total_loss": total_loss,
            "data_loss": l_data.item(),
            "nll_loss": l_nll.item(),
            "physics_loss": l_physics.item(),
            "energy_loss": l_energy.item(),
            "angmom_loss": l_angmom.item(),
        }
```

**Also add calibration metric computation to `compute_calibration_metrics(pred_12, target)`:**

```python
def compute_calibration_metrics(pred_12, target, sigma_scales=[1.0, 2.0]):
    """
    Compute NLL, ECE, and coverage metrics from uncertainty model outputs.
    
    Args:
        pred_12: (N, 12) — 6 mean + 6 logvar
        target: (N, 6) — ground truth
        sigma_scales: list of k values for k-sigma intervals
    
    Returns:
        dict with nll, ece_timing, ece_doppler, coverage_{68,95}_timing/doppler, sharpness
    """
    mean = pred_12[:, :6].detach().cpu().numpy()
    logvar = pred_12[:, 6:].detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    sigma = np.exp(0.5 * logvar)  # std dev
    errors = np.abs(target_np - mean)
    
    nll = np.mean(logvar + errors**2 / np.exp(logvar))
    
    # Coverage by dimension group (position: 0-2, velocity: 3-5)
    results = {"nll": nll}
    for group_name, idx in [("timing", [0,1,2]), ("doppler", [3,4,5])]:
        for k in sigma_scales:
            covered = np.mean(np.all(errors[:, idx] < k * sigma[:, idx], axis=1))
            results[f"coverage_{int(k*100)}_{group_name}"] = covered
        
        # ECE: |empirical - nominal| averaged across confidence levels
        ece = (abs(results.get(f"coverage_68_{group_name}", 0) - 0.68) +
               abs(results.get(f"coverage_95_{group_name}", 0) - 0.95)) / 2
        results[f"ece_{group_name}"] = ece
        
        # Sharpness: mean σ for position components
        results[f"sharpness_{group_name}"] = np.mean(sigma[:, idx])
    
    return results
```

### 3.3 `scripts/pretrain_fixed.py`

**Changes:**

1. Add `--uncertainty` CLI flag:

```python
parser.add_argument("--uncertainty", action="store_true",
    help="Train with uncertainty output head (12 outputs: 6 mean + 6 logvar)")
parser.add_argument("--lambda_nll", type=float, default=1.0,
    help="Weight for Gaussian NLL loss term")
```

2. Change loss initialization:

```python
if args.uncertainty:
    loss_fn = GaussianNLLWithPhysicsLoss(
        w_data=args.w_data,
        w_physics=args.w_physics,
        w_energy=args.w_energy,
        w_angmom=args.w_angmom,
        lambda_nll_schedule=[(1, 0.01), (11, 0.1), (31, 1.0)],
    )
else:
    loss_fn = PINNTotalLoss(...)
```

3. Change model initialization:

```python
model = TrajectoryPINN(
    orbital_elem_dim=6,
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    fourier_features=args.fourier_features,
    output_uncertainty=args.uncertainty,   # NEW
).to(device)
```

4. Update train_epoch to call `loss_fn.update_lambda(epoch)` before each epoch.

5. Store uncertainty metrics in checkpoint:

```python
ckpt = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_metrics": val_m,
    "train_metrics": train_m,
    "output_uncertainty": args.uncertainty,   # NEW
}
```

### 3.4 `scripts/validate_fixed.py`

**Changes:**

1. Add `--uncertainty` flag; pass to `TrajectoryPINN(output_uncertainty=args.uncertainty)`.

2. If `--uncertainty` and checkpoint has uncertainty outputs, call `compute_calibration_metrics` and print:
   - NLL
   - ECE timing / Doppler
   - 68% / 95% coverage timing / Doppler
   - Sharpness timing / Doppler

3. Store calibration results in output JSON.

### 3.5 `scripts/export_pgrl_predictions.py`

**Changes:**

If checkpoint has `output_uncertainty=True`:
- Export additional columns: `pred_sigma_pos_m`, `pred_sigma_vel_ms` per sample
- Export NLL per sample: `nll_per_sample`
- Add `compute_calibration_metrics` output to summary JSON

---

## 4. Acceptance Criteria

All must be satisfied simultaneously for Option 2 to be accepted as a replacement for the deterministic model.

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Position RMSE | ≤ 6.0 m | 7,200-sample validation set |
| Degradation vs deterministic | ≤ 10% | 5.35 m × 1.10 = 5.885 m max |
| Gaussian NLL | < 0.5 | Mean NLL on validation set |
| ECE (timing) | < 0.02 (2%) | 0.5 × (\|cov68 − 0.68\| + \|cov95 − 0.95\|) |
| ECE (Doppler) | < 0.02 (2%) | Same formula for velocity dimensions |
| 68% interval coverage | 60–75% | Empirical fraction within 1σ |
| 95% interval coverage | 90–98% | Empirical fraction within 2σ |
| Below-5m rate | ≥ 55% | ≥ 90% of deterministic 61.3% |

**Critical:** If position RMSE > 6.0 m OR degradation > 10% vs deterministic baseline → **abort**, keep deterministic.

**NLL note:** NLL is computed on normalized units (÷ 1e4 for position, ÷ 10 for velocity). The paper's reported NLL of 0.17 was computed on different target units (likely timing/doppler residuals, not ECI state). This experiment validates NLL as a calibration metric, not as a reproduction of the paper's synthetic-model NLL.

---

## 5. Training Command (RTX 5080 / PC GPU)

### Single-line command
```bash
cd /opt/data/workspace/leo-pinn

PYTHONPATH="$(pwd)/.deps_fresh:$(pwd)/.deps:$(pwd)/.venv/lib/python3.11/site-packages" \
~/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3 scripts/pretrain_fixed.py \
  --data_dir data/tle/ \
  --epochs 100 \
  --batch_size 512 \
  --hidden_dim 256 \
  --num_layers 6 \
  --fourier_features 128 \
  --lr 5e-4 \
  --wd 1e-5 \
  --w_data 1.0 \
  --w_physics 0.1 \
  --w_energy 0.01 \
  --w_angmom 0.01 \
  --output_dir outputs/pretrain_uncertainty \
  --uncertainty \
  2>&1 | tee outputs/pretrain_uncertainty/train.log
```

### Expected GPU time
| Hardware | Est. time for 100 epochs |
|----------|--------------------------|
| RTX 5080 (CUDA 12, ~16k CUDA cores) | ~2–4 min |
| RTX 3090 / 4090 | ~3–5 min |
| CPU (3.11.15) | ~15–25 min |

### Key differences from deterministic run
- Batch size increased from 256 → 512 (GPU memory ~4GB for 12-output model)
- LR increased from 1e-4 → 5e-4 (NLL needs stronger gradients for logvar learning)
- `output_uncertainty=True` → final layer outputs 12 instead of 6
- `GaussianNLLWithPhysicsLoss` instead of `PINNTotalLoss`

---

## 6. Fallback Plan (If Uncertainty Fails)

### If any acceptance criterion is NOT met:

1. **Discard** `outputs/pretrain_uncertainty/` — do not overwrite deterministic checkpoint
2. **Revert** all 5 modified files to their deterministic versions
3. **Do NOT** commit uncertainty checkpoint
4. **Proceed with Option 1** (deterministic paper cleanup) using the already-validated:
   - `artifacts/checkpoints/pgrl_deterministic_retrained.pt`
   - 5.35 m position RMSE
   - 61.3% below-5m rate
   - Remove all Bayesian/Gaussian NLL claims per patch plan items 1–25

### When to abort during training:
| Observation | Action |
|-----------|--------|
| Val RMSE > 8 m at epoch 20 | Abort — model is diverging |
| NLL > 5.0 at epoch 50 (still rising) | Abort — uncertainty not converging |
| Loss is NaN | Abort — logvar numerical explosion; try lower LR or stricter clipping |
| ECE > 0.1 (10%) at epoch 80 | Abort — intervals severely miscalibrated |

---

## 7. File Change Summary

| File | Lines Changed | Type |
|------|--------------|------|
| `physics_ml/pinn_core.py` | ~15 | Modified: `__init__`, `forward` |
| `physics_ml/losses.py` | ~120 | Added: `GaussianNLLWithPhysicsLoss`, `compute_calibration_metrics` |
| `scripts/pretrain_fixed.py` | ~25 | Modified: argparser, model init, loss init, train loop |
| `scripts/validate_fixed.py` | ~30 | Modified: `--uncertainty` flag, calibration output |
| `scripts/export_pgrl_predictions.py` | ~20 | Modified: sigma/NLL export columns |

**Total new code:** ~190–220 lines
**No existing logic deleted** — only wrapped in `if output_uncertainty` conditionals.

---

## 8. What This Does NOT Enable

- **Does NOT** claim the paper's original NLL=0.17 / ECE=0.0028/0.0012 (those were from synthetic model, computed on different targets)
- **Does NOT** require changing the paper's timing (16 ms) or Doppler (234 Hz) values — those require GPS pseudorange computation
- **Does NOT** enable LR-FHSS decoding, PER measurement, or gateway interoperability
- **Does NOT** change hardware validation pipeline

**What it DOES enable:**
- Real calibrated uncertainty intervals from the actual trained model
- NLL and ECE computed on ECI position/velocity targets
- Legitimate claim of "uncertainty-aware residual learning" with validated bounds
- Restoration of Bayesian/calibrated-σ paper claims only if acceptance criteria are met

---

## 9. Decision Point

**Do NOT implement until user approves this plan explicitly.**

To approve: say "APPROVE OPTION 2 — RUN EXPERIMENT"
To reject: say "REJECT OPTION 2 — PROCEED WITH OPTION 1 PAPER CLEANUP ONLY"