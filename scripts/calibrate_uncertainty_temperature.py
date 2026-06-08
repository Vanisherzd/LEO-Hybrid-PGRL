#!/usr/bin/env python3
"""
Stage 3F — Post-hoc Uncertainty Temperature Calibration
=======================================================
Post-hoc calibrate the uncertainty head by applying a scalar temperature T
to log-variance only. The deterministic mean prediction stays unchanged.

Temperature law:
    logvar_T = logvar + 2 * log(T)

For Gaussian with variance σ², scaling by T² gives:
    NLL_T = NLL_original + log(T)    (extra constant per sample)
    Cov68/95 use chi2(3) thresholds 3.506 / 7.815 on Mahalanobis distance.

Usage:
    python scripts/calibrate_uncertainty_temperature.py \
        --checkpoint artifacts/checkpoints/pgrl_uncertainty_stage3e_20ep.pt \
        --output results/uncertainty_temperature_calibration_stage3f.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from data.dataset import create_dataloader
from physics_ml.pinn_core import TrajectoryPINN, count_parameters
from physics_ml.losses import DU, gaussian_nll_loss


TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
CHI2_68 = 3.506   # P(chi2(3) <= 3.506) ≈ 0.683
CHI2_95 = 7.815   # P(chi2(3) <= 7.815) ≈ 0.950


@torch.no_grad()
def evaluate_temperature(model, val_loader, device, T):
    """
    Evaluate model at temperature T.
    - mean is NOT modified
    - logvar_T = logvar + 2*log(T)
    Returns dict of per-sample metrics (all on CPU).
    """
    model.eval()
    pos_errors_list = []
    pos_sigmas_list = []   # axis sigma (mean over xyz) in metres
    z2_list = []           # squared Mahalanobis per sample
    nll_T_list = []        # NLL with temperature-adjusted logvar

    for batch_t, batch_oe, batch_state in val_loader:
        batch_t     = batch_t.squeeze(1).to(device)
        batch_oe    = batch_oe.squeeze(1).to(device)
        batch_state = batch_state.squeeze(1).to(device)

        out_12 = model(batch_t, batch_oe)
        if out_12.ndim == 3:
            out_12 = out_12.squeeze(1)

        mean   = out_12[:, :6]
        logvar = out_12[:, 6:]

        # ── Temperature calibration (logvar only) ─────────────────────────────
        logvar_T = logvar + 2.0 * torch.log(torch.tensor(T, device=device))

        # ── Position error (mean stays same regardless of T) ───────────────────
        mean_pos   = mean[:, :3]                       # (B, 3) normalized
        target_pos = batch_state[:, :3]                # (B, 3) normalized

        err_per_sample = torch.norm(mean_pos - target_pos, dim=1)   # (B,) DU
        err_m          = err_per_sample * DU                         # (B,) metres

        # ── Sigma from temperature-adjusted logvar ─────────────────────────────
        var_per_dim    = torch.exp(logvar_T[:, :3])      # (B, 3) normalized²
        sig_per_dim    = torch.sqrt(var_per_dim)          # (B, 3) normalized
        axis_sigma_m   = sig_per_dim.mean(dim=1) * DU    # (B,) metres

        # ── Squared Mahalanobis (chi2(3)) ─────────────────────────────────────
        err_xyz = mean_pos - target_pos                 # (B, 3) normalized
        z2      = ((err_xyz / sig_per_dim) ** 2).sum(dim=1)  # (B,)

        # ── NLL with temperature-adjusted variance ─────────────────────────────
        # Temperature law: logvar_T = logvar + 2*log(T)
        # NLL per sample = mean over 6 dims of 0.5*(log(2π) + logvar_T + (err/σ_T)²)
        # With reduction='none': nll_per_dim = 0.5*(log(2π) + logvar_T + (err/σ_T)²) → (B,6)
        nll_per_dim_T = gaussian_nll_loss(mean, logvar_T, batch_state, reduction="none")  # (B, 6)
        nll_T = nll_per_dim_T.mean(dim=1)   # (B,) mean NLL per sample

        pos_errors_list.append(err_m.cpu())
        pos_sigmas_list.append(axis_sigma_m.cpu())
        z2_list.append(z2.cpu())
        nll_T_list.append(nll_T.cpu())

    pos_errors = torch.cat(pos_errors_list)   # (N,)
    pos_sigmas = torch.cat(pos_sigmas_list)   # (N,)
    z2_all     = torch.cat(z2_list)           # (N,)
    nll_all    = torch.cat(nll_T_list)         # (N,)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    rmse        = torch.sqrt((pos_errors ** 2).mean()).item()
    nll_mean    = nll_all.mean().item()
    cov68       = (z2_all <= CHI2_68).float().mean().item()
    cov95       = (z2_all <= CHI2_95).float().mean().item()
    axis_p50    = torch.quantile(pos_sigmas, 0.50).item()
    axis_p95    = torch.quantile(pos_sigmas, 0.95).item()
    axis_mean   = pos_sigmas.mean().item()

    cal_error = abs(cov68 - 0.683) + abs(cov95 - 0.950)

    return {
        "T":          T,
        "rmse_m":     rmse,
        "nll":        nll_mean,
        "cov68":      cov68,
        "cov95":      cov95,
        "axis_mean":  axis_mean,
        "axis_p50":   axis_p50,
        "axis_p95":   axis_p95,
        "cal_error":  cal_error,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 70}")
    print(f"Stage 3F — Uncertainty Temperature Calibration  |  device={device}")
    print(f"{'=' * 70}")

    # ── Load model & checkpoint ────────────────────────────────────────────────
    model = TrajectoryPINN(
        orbital_elem_dim=6,
        hidden_dim=256,
        num_layers=6,
        fourier_features=128,
        output_uncertainty=True,   # 12-D output: mean(6) + logvar(6)
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model._load_compatible(ckpt["model_state"])
        print(f"  Loaded: {args.checkpoint}")
        print(f"  Epoch:  {ckpt.get('epoch', '?')}")
        prior = ckpt.get("val_metrics", {})
        if prior:
            print(f"  Prior RMSE: {prior.get('val_pos_rmse_m', '?')} m")
    else:
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"  Model params: {count_parameters(model):,}")

    # ── Validation data (same split as train_uncertainty_offline.py) ──────────
    val_loader = create_dataloader(
        args.data_dir, split="val",
        batch_size=args.batch_size, num_workers=0
    )
    print(f"  Val: {len(val_loader.dataset):,} samples, {len(val_loader)} batches")

    # ── Print header ───────────────────────────────────────────────────────────
    print(f"\n"
          f"{'T':>5} | {'RMSE':>7} | {'NLL':>7} | "
          f"{'Cov68':>6} | {'Cov95':>6} | "
          f"{'axMean':>7} | {'axP50':>6} | {'axP95':>6} | "
          f"{'CalErr':>7}")
    print("-" * 75)

    results = []
    for T in TEMPS:
        m = evaluate_temperature(model, val_loader, device, T)
        results.append(m)
        print(
            f"{m['T']:5.2f} | "
            f"{m['rmse_m']:6.2f}m | "
            f"{m['nll']:7.4f} | "
            f"{m['cov68']:6.3f} | "
            f"{m['cov95']:6.3f} | "
            f"{m['axis_mean']:6.2f}m | "
            f"{m['axis_p50']:5.2f}m | "
            f"{m['axis_p95']:5.2f}m | "
            f"{m['cal_error']:7.4f}"
        )

    # ── Select best T ─────────────────────────────────────────────────────────
    best = min(results, key=lambda r: r["cal_error"])
    print(f"\n  Best T = {best['T']:.2f}  (cal_error = {best['cal_error']:.4f})")
    print(f"  Cov68 = {best['cov68']:.3f}  (target 0.683)")
    print(f"  Cov95 = {best['cov95']:.3f}  (target 0.950)")

    # ── Save report ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("Stage 3F — Uncertainty Temperature Calibration Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Data split : val (same as Stage 3E)\n")
        f.write(f"Batch size : {args.batch_size}\n")
        f.write(f"Temperatures tested: {TEMPS}\n\n")
        f.write(
            f"{'T':>5} | {'RMSE':>7} | {'NLL':>7} | "
            f"{'Cov68':>6} | {'Cov95':>6} | "
            f"{'axMean':>7} | {'axP50':>6} | {'axP95':>6} | "
            f"{'CalErr':>7}\n"
        )
        f.write("-" * 75 + "\n")
        for r in results:
            f.write(
                f"{r['T']:5.2f} | "
                f"{r['rmse_m']:6.2f}m | "
                f"{r['nll']:7.4f} | "
                f"{r['cov68']:6.3f} | "
                f"{r['cov95']:6.3f} | "
                f"{r['axis_mean']:6.2f}m | "
                f"{r['axis_p50']:5.2f}m | "
                f"{r['axis_p95']:5.2f}m | "
                f"{r['cal_error']:7.4f}\n"
            )
        f.write("\n")
        f.write(f"Selected best T = {best['T']:.2f}\n")
        f.write(f"  Calibration error : {best['cal_error']:.4f}\n")
        f.write(f"  Cov68             : {best['cov68']:.4f}  (target 0.683)\n")
        f.write(f"  Cov95             : {best['cov95']:.4f}  (target 0.950)\n")
        f.write(f"  Axis sigma mean   : {best['axis_mean']:.2f} m\n")
        f.write(f"  Axis sigma p50    : {best['axis_p50']:.2f} m\n")
        f.write(f"  Axis sigma p95    : {best['axis_p95']:.2f} m\n")
        f.write(f"\nNote: logvar_T = logvar + 2*log(T). Mean is unchanged.\n")
        f.write(f"Chi2 thresholds: 68%->3.506, 95%->7.815 (3 dof)\n")

    print(f"\n  Report saved: {args.output}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3F — Uncertainty Temperature Calibration"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/checkpoints/pgrl_uncertainty_stage3e_20ep.pt")
    parser.add_argument("--data_dir",   type=str, default="data/tle/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output",     type=str,
                        default="results/uncertainty_temperature_calibration_stage3f.txt")
    args = parser.parse_args()
    main(args)