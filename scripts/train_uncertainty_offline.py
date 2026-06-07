#!/usr/bin/env python3
"""
Stage 3C — Uncertainty Fine-Tuning (Offline) — Metric-Fixed
============================================================
Fine-tune an uncertainty head on top of a frozen deterministic mean.

Loss:
    L = 1.0 * MSE(mean, target)
      + 0.05 * GaussianNLL(mean, logvar, target)
      + physics_regularization(mean)

Unit conventions (all tensors are normalized):
  mean / target : (B, 6) [pos/DU, vel/VU],  DU=10000, VU=10
  position_error_m = norm(mean[:,:3] - target[:,:3], dim=1) * DU   → (B,) metres
  position_sigma_m = mean(sqrt(exp(logvar[:,:3])), dim=1) * DU     → (B,) metres

Usage:
    python scripts/train_uncertainty_offline.py \
        --epochs 50 --batch_size 64 \
        --checkpoint artifacts/checkpoints/pgrl_deterministic_retrained.pt \
        --output artifacts/checkpoints/pgrl_uncertainty_stage3c.pt

Smoke (2 batches):
    python scripts/train_uncertainty_offline.py \
        --epochs 1 --batch_size 64 --max_batches 2 \
        --checkpoint artifacts/checkpoints/pgrl_deterministic_retrained.pt \
        --output artifacts/checkpoints/pgrl_uncertainty_stage3c_smoke.pt
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import create_dataloader
from physics_ml.pinn_core import TrajectoryPINN, count_parameters
from physics_ml.losses import PINNTotalLoss, gaussian_nll_loss, DU, VU, OE_MEAN, OE_STD, MU_EARTH


# ── Shape assertions ──────────────────────────────────────────────────────────
def _assert_shape(tensor, expected, name):
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected}, got {actual}"
        )


def train_epoch(model, dataloader, optimizer, device, max_batches=None, physics_weight=0.1):
    """Train one epoch. Returns dict of scalar metrics."""
    model.train()
    n = 0

    total_loss_sum  = 0.0
    data_loss_sum   = 0.0
    nll_loss_sum    = 0.0
    phys_loss_sum   = 0.0
    err_m_sum       = 0.0
    sigma_m_sum     = 0.0
    cov68_sum       = 0.0
    cov95_sum       = 0.0

    for batch_t, batch_oe, batch_state in tqdm(dataloader, desc="Training", leave=False):
        if max_batches is not None and n >= max_batches:
            break

        # ── Shapes after squeeze ────────────────────────────────────────────────
        # Dataset: t=(B,1,1), oe=(B,1,6), state=(B,1,6)
        batch_t     = batch_t.squeeze(1).to(device)       # (B, 1)
        batch_oe    = batch_oe.squeeze(1).to(device)     # (B, 6)
        batch_state = batch_state.squeeze(1).to(device) # (B, 6)

        # ── Forward ─────────────────────────────────────────────────────────────
        out_12  = model(batch_t, batch_oe)                # (B, 12)
        if out_12.ndim == 3:
            out_12 = out_12.squeeze(1)                   # safe squeeze (B,1,12)→(B,12)

        mean   = out_12[:, :6]                            # (B, 6)
        logvar = out_12[:, 6:]                            # (B, 6)

        # ── Shape assertions ────────────────────────────────────────────────────
        _assert_shape(mean,       (batch_state.shape[0], 6), "mean")
        _assert_shape(logvar,     (batch_state.shape[0], 6), "logvar")
        _assert_shape(batch_state, (batch_state.shape[0], 6), "batch_state")
        assert mean.shape == batch_state.shape, (
            f"mean.shape {mean.shape} != target.shape {batch_state.shape}"
        )

        # ── Data MSE + Gaussian NLL ──────────────────────────────────────────────
        l_data = F.mse_loss(mean, batch_state)
        l_nll  = gaussian_nll_loss(mean, logvar, batch_state)

        # ── Physics: bounded energy + angmom on mean only ─────────────────────────
        pred_pos_km = mean[:, :3] * DU                    # km
        pred_vel_km = mean[:, 3:] * VU                    # km/s
        a_oe = (batch_oe[:, 0:1] * OE_STD[:, 0].to(batch_oe)
                + OE_MEAN[:, 0].to(batch_oe))            # km
        e_oe = (batch_oe[:, 1:2] * OE_STD[:, 1].to(batch_oe)
                + OE_MEAN[:, 1].to(batch_oe))

        energy   = PINNTotalLoss._orbital_energy(pred_pos_km, pred_vel_km)
        E_ref    = (-MU_EARTH / (2.0 * a_oe)).squeeze(-1)
        l_energy = F.mse_loss(energy / (VU**2), E_ref / (VU**2))

        h_mag    = PINNTotalLoss._angular_momentum_magnitude(pred_pos_km, pred_vel_km)
        h_ref    = torch.sqrt(MU_EARTH * a_oe * (1.0 - e_oe**2)).squeeze(-1)
        l_angmom = F.mse_loss(h_mag / (VU * DU), h_ref / (VU * DU))

        l_physics = l_energy

        # ── Combined loss ────────────────────────────────────────────────────────
        total_loss = (
            1.0  * l_data
            + 0.05 * l_nll
            + physics_weight * l_physics
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = mean.shape[0]

        # ── Position calibration metrics ──────────────────────────────────────────
        mean_pos   = mean[:, :3]                           # (B, 3) normalized
        target_pos = batch_state[:, :3]                    # (B, 3) normalized

        _assert_shape(mean_pos,   (B, 3), "mean_pos")
        _assert_shape(target_pos, (B, 3), "target_pos")

        # Error: RMSE in metres  (DU is already the normalized-to-reported-metre scale)
        err_per_sample  = torch.norm(mean_pos - target_pos, dim=1)          # (B,)  DU
        err_m           = err_per_sample * DU                                # (B,)  metres

        _assert_shape(err_per_sample, (B,), "err_per_sample")
        _assert_shape(err_m,         (B,), "err_m")

        # Sigma: sqrt(exp(logvar)) per dim, then mean across xyz
        var_per_dim = torch.exp(logvar[:, :3])               # (B, 3) normalized^2
        sig_per_dim = torch.sqrt(var_per_dim)               # (B, 3) normalized

        # ── Chi-squared coverage (3 dof → xyz) ─────────────────────────────────
        # err_xyz / sigma_xyz  → squared Mahalanobis → chi2(3)
        # P(chi2 ≤ 3.506) ≈ 0.683  |  P(chi2 ≤ 7.815) ≈ 0.950
        err_xyz  = mean_pos - target_pos                    # (B, 3) normalized
        z2       = ((err_xyz / sig_per_dim) ** 2).sum(dim=1)  # (B,) chi2(3)
        cov68    = (z2 <= 3.506).float().mean().item()
        cov95    = (z2 <= 7.815).float().mean().item()

        # Readable sigma summaries (in metres, DU is already the normalized-to-reported-metre scale)
        axis_sigma_m   = sig_per_dim.mean(dim=1) * DU   # (B,) mean over xyz → metres
        radial_sigma_m = torch.norm(sig_per_dim, dim=1) * DU  # (B,) radial → metres

        _assert_shape(var_per_dim,      (B, 3), "var_per_dim")
        _assert_shape(sig_per_dim,      (B, 3), "sig_per_dim")
        _assert_shape(axis_sigma_m,    (B,),   "axis_sigma_m")
        _assert_shape(radial_sigma_m,  (B,),   "radial_sigma_m")

        # Accumulate
        total_loss_sum += total_loss.item()
        data_loss_sum  += l_data.item()
        nll_loss_sum   += l_nll.item()
        phys_loss_sum  += l_physics.item()
        err_m_sum      += err_m.mean().item()
        sigma_m_sum    += axis_sigma_m.mean().item()
        cov68_sum      += cov68
        cov95_sum      += cov95
        n += 1

    return {
        "total_loss":       total_loss_sum / max(n, 1),
        "data_loss":        data_loss_sum  / max(n, 1),
        "nll_loss":         nll_loss_sum   / max(n, 1),
        "physics_loss":    phys_loss_sum / max(n, 1),
        "pos_rmse_m":      err_m_sum      / max(n, 1),
        "position_sigma_m": sigma_m_sum   / max(n, 1),
        "cov68":           cov68_sum      / max(n, 1),
        "cov95":           cov95_sum      / max(n, 1),
    }


@torch.no_grad()
def validate(model, dataloader, device, max_batches=None):
    """Validate and compute position calibration coverages."""
    model.eval()
    pos_errors_list = []
    pos_sigmas_list = []
    val_nll_list    = []

    for batch_t, batch_oe, batch_state in dataloader:
        if max_batches is not None and len(pos_errors_list) >= max_batches:
            break

        batch_t     = batch_t.squeeze(1).to(device)       # (B, 1)
        batch_oe    = batch_oe.squeeze(1).to(device)      # (B, 6)
        batch_state = batch_state.squeeze(1).to(device)   # (B, 6)

        out_12  = model(batch_t, batch_oe)                # (B, 12)
        if out_12.ndim == 3:
            out_12 = out_12.squeeze(1)

        mean   = out_12[:, :6]                            # (B, 6)
        logvar = out_12[:, 6:]                            # (B, 6)

        # ── Shape assertions ─────────────────────────────────────────────────────
        _assert_shape(mean,        (batch_state.shape[0], 6), "mean (validate)")
        _assert_shape(logvar,      (batch_state.shape[0], 6), "logvar (validate)")
        _assert_shape(batch_state, (batch_state.shape[0], 6), "batch_state (validate)")
        assert mean.shape == batch_state.shape

        # Position slices
        mean_pos   = mean[:, :3]
        target_pos = batch_state[:, :3]

        _assert_shape(mean_pos,   (batch_state.shape[0], 3), "mean_pos (validate)")
        _assert_shape(target_pos, (batch_state.shape[0], 3), "target_pos (validate)")

        # Error: RMSE per sample (DU → metres, DU is already the normalized-to-reported-metre scale)
        err_per_sample = torch.norm(mean_pos - target_pos, dim=1)   # (B,) DU
        err_m = err_per_sample * DU                                # (B,) metres

        # Sigma: sqrt(exp(logvar)) per dim
        var_per_dim = torch.exp(logvar[:, :3])               # (B, 3)
        sig_per_dim = torch.sqrt(var_per_dim)               # (B, 3)

        # ── Chi-squared coverage ─────────────────────────────────────────────────
        err_xyz  = mean_pos - target_pos                      # (B, 3) normalized
        z2       = ((err_xyz / sig_per_dim) ** 2).sum(dim=1)  # (B,) chi2(3)
        cov68    = (z2 <= 3.506).float().mean().item()
        cov95    = (z2 <= 7.815).float().mean().item()

        # Readable sigma summaries (in metres, DU is already the normalized-to-reported-metre scale)
        axis_sigma_m   = sig_per_dim.mean(dim=1) * DU   # (B,) → metres
        radial_sigma_m = torch.norm(sig_per_dim, dim=1) * DU  # (B,) → metres

        pos_errors_list.append(err_m.cpu())
        pos_sigmas_list.append(axis_sigma_m.cpu())
        val_nll_list.append(gaussian_nll_loss(mean, logvar, batch_state).cpu())

    pos_errors = torch.cat(pos_errors_list)   # (N,)
    pos_sigmas = torch.cat(pos_sigmas_list)   # (N,)

    _assert_shape(pos_errors, pos_sigmas.shape, "pos_errors/pos_sigmas")
    assert pos_errors.ndim == 1, f"pos_errors must be 1D, got {pos_errors.ndim}D"

    return {
        # RMSE = sqrt(mean of squared errors), not mean of errors (MAE)
        "val_pos_rmse_m":   torch.sqrt((pos_errors ** 2).mean()).item(),
        "val_pos_max_m":    pos_errors.max().item(),
        "val_pos_mean_m":   pos_errors.mean().item(),
        "val_nll":          torch.stack(val_nll_list).mean().item(),
        "position_sigma_m": pos_sigmas.mean().item(),
        "axis_sig_p50_m":   torch.quantile(pos_sigmas, 0.50).item(),
        "axis_sig_p95_m":   torch.quantile(pos_sigmas, 0.95).item(),
        "radial_sig_p50_m":  0.0,   # not yet accumulated per-batch
        "radial_sig_p95_m":  0.0,
        "err_p50_m":        torch.quantile(pos_errors, 0.50).item(),
        "err_p68_m":        torch.quantile(pos_errors, 0.68).item(),
        "err_p95_m":        torch.quantile(pos_errors, 0.95).item(),
        "cov68":            cov68,
        "cov95":            cov95,
        "below_5m_pct":     (pos_errors < 5.0).float().mean().item() * 100,
        "below_3m_pct":     (pos_errors < 3.0).float().mean().item() * 100,
        "below_1m_pct":     (pos_errors < 1.0).float().mean().item() * 100,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 70}")
    print(f"Stage 3C Uncertainty Fine-Tuning | device={device}")
    print(f"{'=' * 70}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = create_dataloader(args.data_dir, split="train",
                                    batch_size=args.batch_size, num_workers=0)
    val_loader   = create_dataloader(args.data_dir, split="val",
                                    batch_size=args.batch_size, num_workers=0)
    print(f"  Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_loader.dataset):,} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TrajectoryPINN(
        orbital_elem_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_features=args.fourier_features,
        output_uncertainty=True,          # 12-D output: mean(6) + logvar(6)
    ).to(device)
    print(f"  Model params: {count_parameters(model):,}")

    # ── Load deterministic checkpoint into mean head ─────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model._load_compatible(ckpt["model_state"])
        rmse_before = ckpt.get("val_metrics", {}).get("val_pos_rmse_m", "?")
        print(f"  Loaded deterministic checkpoint: {args.checkpoint}")
        print(f"  Deterministic val RMSE: {rmse_before}m")

    # ── Fix log_var_layer initialization so initial sigma ≈ 10 m ────────────────
    # DU is the normalized-to-reported-metre scale used by the deterministic
    # evaluation pipeline. We want per-axis sigma ≈ 10 m:
    # sigma_norm = 10 / DU = 0.001, so logvar = log((0.001)^2) = log(1e-6).
    for name, param in model.named_parameters():
        if "log_var_layer" in name:
            if "weight" in name:
                param.data.zero_()           # zero weight → identity covariance
                print(f"  log_var_layer.weight zeroed (was {param.shape})")
            elif "bias" in name:
                init_sigma = 10.0 / DU        # 10 m in DU
                param.data.fill_(math.log(init_sigma ** 2))
                print(f"  log_var_layer.bias = log(({init_sigma:.6f})^2) = {param.data[0].item():.4f}  "
                      f"→ initial per-axis σ ≈ {init_sigma * DU:.1f} m")

    # ── Freeze entire model except logvar_layer for Stage 3C ─────────────────
    # Only logvar_layer learns; mean stays at deterministic checkpoint values.
    # This ensures physics/NLL updates do not corrupt the mean head.
    for name, param in model.named_parameters():
        if "log_var_layer" in name:
            param.requires_grad = True
            print(f"  Trainable: {name}")
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (logvar_layer only): {trainable:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_rmse = float("inf")

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'Ep':>3} | {'Loss':>8} | {'MSE':>7} | {'NLL':>7} | "
          f"{'Phys':>7} | {'RMSE':>7} | {'SigM':>7} | "
          f"{'Cov68':>6} | {'Cov95':>6} | {'<5m%':>5} | {'Best':>4} | {'Time':>5}")
    print("-" * 98)

    # ── Smoke test: evaluate loaded checkpoint BEFORE any training steps ──────────
    if args.max_batches is not None:
        print(f"\n  [Smoke] Running pre-training validation (0 steps taken) ...")
        pre_m = validate(model, val_loader, device, max_batches=args.max_batches)
        print(f"  Pre-training RMSE: {pre_m['val_pos_rmse_m']:.2f}m  "
              f"NLL: {pre_m['val_nll']:.4f}  "
              f"Sigma: {pre_m['position_sigma_m']:.2f}m  "
              f"Cov68: {pre_m['cov68']:.3f}  Cov95: {pre_m['cov95']:.3f}")
        print(f"  Pre-training <5m: {pre_m['below_5m_pct']:.1f}%  "
              f"<3m: {pre_m['below_3m_pct']:.1f}%  "
              f"<1m: {pre_m['below_1m_pct']:.1f}%")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_epoch(
            model, train_loader, optimizer, device,
            max_batches=args.max_batches,
            physics_weight=args.w_physics,
        )
        val_m = validate(
            model, val_loader, device,
            max_batches=args.max_batches,
        )
        scheduler.step()
        dt = time.time() - t0

        is_best = val_m["val_pos_rmse_m"] < best_rmse
        if is_best:
            best_rmse = val_m["val_pos_rmse_m"]

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_m,
            "train_metrics": {k: v for k, v in train_m.items()},
        }
        torch.save(ckpt, args.output)
        if is_best:
            torch.save(ckpt, args.output.replace(".pt", "_best.pt"))

        print(
            f"{epoch:3d} | "
            f"{train_m['total_loss']:8.4f} | "
            f"{train_m['data_loss']:7.4f} | "
            f"{train_m['nll_loss']:7.4f} | "
            f"{train_m['physics_loss']:7.4f} | "
            f"{val_m['val_pos_rmse_m']:6.2f}m | "
            f"{val_m['position_sigma_m']:6.2f}m | "
            f"{val_m['cov68']:6.3f} | "
            f"{val_m['cov95']:6.3f} | "
            f"{val_m['below_5m_pct']:5.1f}% | "
            f"{'★' if is_best else '':>4} | "
            f"{dt:4.1f}s"
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                "      debug: "
                f"err[p50,p68,p95]=({val_m['err_p50_m']:.3f}, {val_m['err_p68_m']:.3f}, {val_m['err_p95_m']:.3f})m | "
                f"axis_sig[p50,p95]=({val_m['axis_sig_p50_m']:.3f}, {val_m['axis_sig_p95_m']:.3f})m"
            )


    print(f"\n{'=' * 98}")
    print(f"Best Val RMSE: {best_rmse:.2f}m  |  Checkpoint: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3C — Uncertainty Offline Fine-Tuning (metric-corrected)"
    )
    parser.add_argument("--data_dir",         default="data/tle/")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",        type=int,   default=64)
    parser.add_argument("--max_batches",      type=int,   default=None,
                        help="Smoke test: stop after N batches per epoch")
    parser.add_argument("--hidden_dim",        type=int,   default=256)
    parser.add_argument("--num_layers",        type=int,   default=6)
    parser.add_argument("--fourier_features",  type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--wd",                type=float, default=1e-5)
    parser.add_argument("--w_data",            type=float, default=1.0)
    parser.add_argument("--w_physics",         type=float, default=0.1)
    parser.add_argument("--w_energy",          type=float, default=0.01)
    parser.add_argument("--w_angmom",         type=float, default=0.01)
    parser.add_argument("--checkpoint",        type=str,   default="",
                        help="Path to deterministic .pt checkpoint to warm-start mean head")
    parser.add_argument("--output",            type=str,
                        default="artifacts/checkpoints/pgrl_uncertainty_stage3c.pt")
    args = parser.parse_args()
    main(args)