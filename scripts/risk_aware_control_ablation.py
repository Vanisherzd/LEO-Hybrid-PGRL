#!/usr/bin/env python3
"""
Stage 4 — Risk-Aware Control Ablation
=======================================
Compare deterministic (fixed margin) vs uncertainty-aware (adaptive margin)
policies for LR-FHSS/PGRL control robustness.

Policy:  guard_m = guard_base + alpha * radial_sigma_m

Metrics per alpha:
  - Val RMSE (identical across alpha — mean is unchanged)
  - mean_guard_overhead: mean guard margin in metres
  - outage_proxy: fraction of samples where residual_error_m > guard_m
  - collision_proxy: fraction where |Doppler_error_hz| > k_doppler * bin_spacing_hz
  - success_proxy: 1 - outage_proxy
  - risk_adjusted_reward: success_proxy - lambda_energy * normalized_guard_overhead

  where normalized_guard_overhead = mean(guard_m) / mean(guard_base)

Ablation alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
  alpha=0.0 → deterministic baseline (fixed guard = guard_base)
  alpha>0   → uncertainty-aware (guard grows with radial sigma)

Usage:
    python scripts/risk_aware_control_ablation.py \
        --checkpoint artifacts/checkpoints/pgrl_uncertainty_stage3e_20ep.pt \
        --output results/risk_aware_control_ablation_stage4.txt
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from data.dataset import create_dataloader
from physics_ml.pinn_core import TrajectoryPINN, count_parameters
from physics_ml.losses import DU, VU
from controller.guard_band_policy import adaptive_guard_time, guard_overhead_fraction
from controller.energy_model import total_opportunity_energy


# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT    = 299792458.0       # m/s
F_CARRIER  = 436.5e6           # Hz  (S-band)
# LR-FHSS: 242 Hz BW, chip_rate 8393 cps → bin spacing ≈ 242 Hz
LRFHSS_BIN_HZ = 242.0          # Hz per frequency bin
# Timing-to-position: v_orbital ≈ 7.67 km/s at 400 km altitude
V_ORBITAL_KMS = 7.67           # km/s (approximate ISS orbital speed)
V_ORBITAL_MS  = V_ORBITAL_KMS * 1000.0  # m/s

ALPHAS            = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_ENERGY     = 0.1        # energy penalty weight in reward
DOPPLER_K_FACTOR  = 2.0        # collision threshold: k * bin_spacing_hz


def radial_sigma_from_logvar(logvar: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    Compute radial (3-D) sigma in metres from (B,6) logvar tensor.

    Only uses xyz components (dim 0..2).
    Returns (B,) tensor of radial sigma in metres.
    """
    sig_xyz = torch.sqrt(torch.exp(logvar[:, :3]))            # (B, 3) normalized
    radial   = torch.norm(sig_xyz, dim=1)                     # (B,) normalized
    return radial * DU                                        # (B,) metres


def doppler_error_from_velocity_err(
    vel_err_norm: torch.Tensor,
    f_carrier: float = F_CARRIER,
    v_orbital_ms: float = V_ORBITAL_MS,
) -> torch.Tensor:
    """
    Approximate Doppler error in Hz from velocity prediction error.

    Doppler_shift = (v_r / c) * f_carrier
    Residual Doppler error ≈ (vel_error / v_orbital) * doppler_budget
    where vel_error_norm is in normalized VU units.
    Returns (B,) tensor of Doppler error in Hz.
    """
    # vel_err_norm is in normalized VU (1 VU = 10 km/s)
    # Convert to m/s: vel_err_ms = vel_err_norm * VU * 1000
    vel_err_ms = vel_err_norm * VU * 1000.0          # (B,) m/s
    # Fraction of orbital velocity
    frac = vel_err_ms / v_orbital_ms                 # (B,) dimensionless
    # Doppler error = frac * (v_orb/c) * f_carrier
    doppler_hz = frac * (v_orbital_ms / C_LIGHT) * f_carrier  # (B,) Hz
    return torch.abs(doppler_hz)


@torch.no_grad()
def collect_predictions(model, val_loader, device):
    """
    Run model on all validation samples and collect per-sample metrics.
    Returns dict of (N,) tensors on CPU.
    """
    model.eval()
    residual_m_list   = []
    radial_sigma_list = []
    vel_err_norm_list = []    # velocity error magnitude in normalized VU
    guard_base_list   = []    # fixed guard = p95 of residual across all samples

    for batch_t, batch_oe, batch_state in val_loader:
        batch_t     = batch_t.squeeze(1).to(device)
        batch_oe    = batch_oe.squeeze(1).to(device)
        batch_state = batch_state.squeeze(1).to(device)

        out_12 = model(batch_t, batch_oe)
        if out_12.ndim == 3:
            out_12 = out_12.squeeze(1)

        mean   = out_12[:, :6]     # (B, 6)
        logvar = out_12[:, 6:]     # (B, 6)

        # Position error (residual) in metres
        residual = torch.norm(mean[:, :3] - batch_state[:, :3], dim=1) * DU  # (B,)

        # Radial sigma in metres
        rad_sig = radial_sigma_from_logvar(logvar, mean)                    # (B,)

        # Velocity error magnitude in normalized VU
        vel_err_norm = torch.norm(mean[:, 3:] - batch_state[:, 3:], dim=1)   # (B,)

        residual_m_list.append(residual.cpu())
        radial_sigma_list.append(rad_sig.cpu())
        vel_err_norm_list.append(vel_err_norm.cpu())

    residual_m   = torch.cat(residual_m_list)    # (N,)
    radial_sigma = torch.cat(radial_sigma_list)  # (N,)
    vel_err_norm = torch.cat(vel_err_norm_list)  # (N,)

    # Fixed base guard = p95 of residuals (the deterministic worst-case margin)
    guard_base = torch.quantile(residual_m, 0.95).item()  # metres

    return {
        "residual_m":    residual_m,
        "radial_sigma":  radial_sigma,
        "vel_err_norm":  vel_err_norm,
        "guard_base":    guard_base,
        "N":             residual_m.shape[0],
    }


def evaluate_alpha(alpha, data, guard_base):
    """
    Evaluate metrics for a given alpha.

    guard_m = guard_base + alpha * radial_sigma_m
    """
    residual_m    = data["residual_m"]
    radial_sigma  = data["radial_sigma"]
    vel_err_norm = data["vel_err_norm"]
    N            = data["N"]

    # Guard margin for this alpha
    guard_m = guard_base + alpha * radial_sigma           # (N,) metres

    # Outage proxy: residual > guard_m
    outage_mask  = residual_m > guard_m                   # (N,) bool
    outage_proxy = outage_mask.float().mean().item()      # scalar
    success_proxy = 1.0 - outage_proxy

    # Collision proxy: |Doppler error| > k * bin_spacing
    doppler_err_hz = doppler_error_from_velocity_err(vel_err_norm)  # (N,) Hz
    collision_mask  = doppler_err_hz > DOPPLER_K_FACTOR * LRFHSS_BIN_HZ  # (N,)
    collision_proxy  = collision_mask.float().mean().item()

    # Guard overhead (normalised)
    mean_guard_m     = guard_m.mean().item()
    norm_guard       = mean_guard_m / guard_base           # dimensionless

    # Risk-adjusted reward
    # reward = success_proxy - lambda_energy * (norm_guard - 1)
    # Using (norm_guard - 1) so that alpha=0 baseline has 0 energy penalty
    reward = success_proxy - LAMBDA_ENERGY * (norm_guard - 1.0)

    # RMSE (same for all alpha — computed from raw residual)
    rmse = torch.sqrt((residual_m ** 2).mean()).item()

    return {
        "alpha":          alpha,
        "rmse_m":         rmse,
        "mean_guard_m":   mean_guard_m,
        "norm_guard":     norm_guard,
        "outage_proxy":   outage_proxy,
        "collision_proxy": collision_proxy,
        "success_proxy":  success_proxy,
        "risk_reward":    reward,
        "cov68_proxy":    0.0,   # placeholder for chi2 coverage (informational)
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 70}")
    print(f"Stage 4 — Risk-Aware Control Ablation  |  device={device}")
    print(f"{'=' * 70}")

    # ── Load model & checkpoint ────────────────────────────────────────────────
    model = TrajectoryPINN(
        orbital_elem_dim=6,
        hidden_dim=256,
        num_layers=6,
        fourier_features=128,
        output_uncertainty=True,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model._load_compatible(ckpt["model_state"])
        print(f"  Loaded: {args.checkpoint}")
        print(f"  Epoch:  {ckpt.get('epoch', '?')}")
    else:
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"  Model params: {count_parameters(model):,}")

    # ── Validation data ────────────────────────────────────────────────────────
    val_loader = create_dataloader(
        args.data_dir, split="val",
        batch_size=args.batch_size, num_workers=0
    )
    print(f"  Val: {len(val_loader.dataset):,} samples, {len(val_loader)} batches")

    # ── Collect per-sample predictions once ────────────────────────────────────
    print(f"\n  Collecting predictions (one pass) ...")
    data = collect_predictions(model, val_loader, device)
    guard_base = data["guard_base"]
    N          = data["N"]
    print(f"  N = {N:,}  |  guard_base (p95 residual) = {guard_base:.2f} m")

    # ── Sweep alpha ────────────────────────────────────────────────────────────
    print(f"\n{'α':>5} | {'RMSE':>7} | {'guard':>6} | {'norm':>5} | "
          f"{'outage':>7} | {'coll':>6} | {'success':>8} | "
          f"{'reward':>7}")
    print("-" * 75)

    results = []
    for alpha in ALPHAS:
        m = evaluate_alpha(alpha, data, guard_base)
        results.append(m)
        print(
            f"{m['alpha']:5.2f} | "
            f"{m['rmse_m']:6.2f}m | "
            f"{m['mean_guard_m']:5.2f}m | "
            f"{m['norm_guard']:4.3f} | "
            f"{m['outage_proxy']:6.3f} | "
            f"{m['collision_proxy']:5.3f} | "
            f"{m['success_proxy']:7.4f} | "
            f"{m['risk_reward']:6.4f}"
        )

    # ── Identify Pareto frontier (minimise outage, minimise norm_guard) ────────
    # Pareto: no other point has both lower outage AND lower norm_guard
    pareto_mask = []
    for i, r in enumerate(results):
        dominated = any(
            j != i and
            results[j]["outage_proxy"] <= r["outage_proxy"] and
            results[j]["norm_guard"]   <= r["norm_guard"] and
            (results[j]["outage_proxy"] < r["outage_proxy"] or
             results[j]["norm_guard"]   < r["norm_guard"])
            for j in range(len(results))
        )
        pareto_mask.append(not dominated)

    pareto_alphas = [results[i]["alpha"] for i in range(len(results)) if pareto_mask[i]]

    # ── Select best by risk-adjusted reward ───────────────────────────────────
    best_reward = max(results, key=lambda r: r["risk_reward"])
    best_outage = min(results, key=lambda r: r["outage_proxy"])

    print(f"\n  Pareto alphas (outage vs overhead): {pareto_alphas}")
    print(f"  Best reward α={best_reward['alpha']:.2f}  reward={best_reward['risk_reward']:.4f}  "
          f"outage={best_reward['outage_proxy']:.3f}")
    print(f"  Lowest outage α={best_outage['alpha']:.2f}  outage={best_outage['outage_proxy']:.3f}  "
          f"guard={best_outage['mean_guard_m']:.2f}m")

    # ── Save report ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("Stage 4 — Risk-Aware Control Ablation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Data split : val (same as Stage 3E/3F)\n")
        f.write(f"Batch size : {args.batch_size}\n")
        f.write(f"Samples N  : {N:,}\n")
        f.write(f"Doppler k× : {DOPPLER_K_FACTOR}  (collision if |Δf| > {DOPPLER_K_FACTOR}×{LRFHSS_BIN_HZ:.0f} Hz)\n")
        f.write(f"λ_energy   : {LAMBDA_ENERGY}\n")
        f.write(f"guard_base : {guard_base:.2f} m  (p95 of position residual)\n")
        f.write(f"F_CARRIER  : {F_CARRIER/1e6:.1f} MHz  |  V_orbital : {V_ORBITAL_KMS:.2f} km/s\n\n")

        f.write(
            f"{'α':>5} | {'RMSE':>7} | {'guard':>6} | {'norm':>5} | "
            f"{'outage':>7} | {'coll':>6} | {'success':>8} | {'reward':>7}\n"
        )
        f.write("-" * 75 + "\n")
        for i, r in enumerate(results):
            marker = " ◀" if pareto_mask[i] else ""
            f.write(
                f"{r['alpha']:5.2f} | "
                f"{r['rmse_m']:6.2f}m | "
                f"{r['mean_guard_m']:5.2f}m | "
                f"{r['norm_guard']:4.3f} | "
                f"{r['outage_proxy']:6.3f} | "
                f"{r['collision_proxy']:5.3f} | "
                f"{r['success_proxy']:7.4f} | "
                f"{r['risk_reward']:6.4f}{marker}\n"
            )

        f.write("\n")
        f.write(f"Pareto frontier α: {pareto_alphas}\n")
        f.write(f"Best risk-adjusted reward: α={best_reward['alpha']:.2f}\n")
        f.write(f"  reward={best_reward['risk_reward']:.4f}  outage={best_reward['outage_proxy']:.4f}\n")
        f.write(f"  guard={best_reward['mean_guard_m']:.2f}m  norm_guard={best_reward['norm_guard']:.4f}\n")
        f.write(f"\nLowest outage policy: α={best_outage['alpha']:.2f}\n")
        f.write(f"  outage={best_outage['outage_proxy']:.4f}  guard={best_outage['mean_guard_m']:.2f}m\n")
        f.write(f"\nPolicy: guard_m = guard_base + α × radial_sigma_m\n")
        f.write(f"Collision threshold: |Doppler_err| > {DOPPLER_K_FACTOR}×{LRFHSS_BIN_HZ:.0f} Hz\n")
        f.write(f"Reward = success_proxy − λ_energy × (norm_guard − 1)\n")
        f.write(f"NOTE: RMSE is identical across all α (mean prediction unchanged)\n")

    print(f"\n  Report saved: {args.output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4 — Risk-Aware Control Ablation"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/checkpoints/pgrl_uncertainty_stage3e_20ep.pt")
    parser.add_argument("--data_dir",   type=str, default="data/tle/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output",    type=str,
                        default="results/risk_aware_control_ablation_stage4.txt")
    args = parser.parse_args()
    main(args)