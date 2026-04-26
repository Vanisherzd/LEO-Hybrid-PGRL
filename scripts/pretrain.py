"""
Offline Pretraining Script — Train PINN on J2 ground-truth data
================================================================
Uses J2-perturbed propagation as ground truth with physics-informed losses.

Usage:
 python scripts/pretrain.py --data_dir data/tle/ --epochs 100
"""

import argparse, os, sys, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import create_dataloader
from models.pinn_core import TrajectoryPINN, count_parameters
from training.losses import PINNTotalLoss


def train_epoch(model, loss_fn, dataloader, optimizer, device, use_autograd=False):
    model.train()
    keys = ["total_loss", "data_loss", "physics_loss", "energy_loss",
            "angmom_loss", "pos_rmse_m", "below_5m_pct"]
    epoch_losses = {k: 0.0 for k in keys}
    n = 0

    for batch_t, batch_oe, batch_state in tqdm(dataloader, desc="Training", leave=False):
        # Dataset: t=(batch,1,1), oe=(batch,1,6) -> squeeze to (batch,1), (batch,6)
        batch_t = batch_t.squeeze(1).to(device)
        batch_oe = batch_oe.squeeze(1).to(device)
        batch_state = batch_state.squeeze(1).to(device)  # (batch, 1, 6) -> (batch, 6)

        optimizer.zero_grad()
        pred_state = model(batch_t, batch_oe)

        if use_autograd and hasattr(loss_fn, 'forward_with_autograd_physics'):
            total_loss, result = loss_fn.forward_with_autograd_physics(
                model, batch_t, batch_oe, batch_state)
        else:
            result = loss_fn(pred_state, batch_state, batch_t, batch_oe)
            # total_loss is a Python float (.item() was already called in forward).
            # Reconstruct the tensor version for .backward().
            l_data = torch.nn.functional.mse_loss(pred_state, batch_state)
            l_physics = result["physics_loss"]
            l_energy = result["energy_loss"]
            l_angmom = result["angmom_loss"]
            total_loss = (
                loss_fn.w_data    * l_data
                + loss_fn.w_physics * l_physics
                + loss_fn.w_energy  * l_energy
                + loss_fn.w_angmom  * l_angmom
            )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in keys:
            if k in result:
                epoch_losses[k] += result[k]
        n += 1

    for k in keys:
        epoch_losses[k] /= max(n, 1)
    return epoch_losses


@torch.no_grad()
def validate(model, loss_fn, dataloader, device):
    model.eval()
    pos_errors, vel_errors = [], []

    for batch_t, batch_oe, batch_state in dataloader:
        batch_t = batch_t.squeeze(1).to(device)
        batch_oe = batch_oe.squeeze(1).to(device)
        batch_state = batch_state.squeeze(1).to(device)  # (batch, 1, 6) -> (batch, 6)

        pred_state = model(batch_t, batch_oe)

        # Both pred and GT are normalized: pos/1e4, vel/10
        # Error in normalized units: multiply by scale to get real units
        pos_err = torch.norm(pred_state[:, :3] - batch_state[:, :3], dim=-1)   # normalized-km
        vel_err = torch.norm(pred_state[:, 3:] - batch_state[:, 3:], dim=-1)   # normalized-km/s

        pos_errors.append(pos_err.cpu())
        vel_errors.append(vel_err.cpu())

    pos_errors = torch.cat(pos_errors)
    vel_errors = torch.cat(vel_errors)

    # Denormalize: pos_err_norm_km * 10000 = m; vel_err_norm_km/s * 10 * 1000 = m/s
    return {
        "val_pos_rmse_m": (pos_errors * 10000).mean().item(),
        "val_pos_max_m": (pos_errors * 10000).max().item(),
        "below_5m_pct": (pos_errors * 10000 < 5.0).float().mean().item() * 100,
        "below_3m_pct": (pos_errors * 10000 < 3.0).float().mean().item() * 100,
        "below_1m_pct": (pos_errors * 10000 < 1.0).float().mean().item() * 100,
        "val_vel_rmse_ms": (vel_errors * 10 * 1000).mean().item(),
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 70}")
    print(f"LEO-PINN Pretraining | device={device}")
    print(f"{'=' * 70}")

    # Data
    print(f"\nLoading data from: {args.data_dir}")
    train_loader = create_dataloader(args.data_dir, split="train",
                                     batch_size=args.batch_size, num_workers=0)
    val_loader = create_dataloader(args.data_dir, split="val",
                                   batch_size=args.batch_size, num_workers=0)
    print(f" Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # Model — 6 orbital elements: a, e, i, Omega, omega, M0
    model = TrajectoryPINN(
        orbital_elem_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_features=args.fourier_features,
    ).to(device)
    print(f" Model: {count_parameters(model):,} parameters")

    # Loss
    loss_fn = PINNTotalLoss(
        w_data=args.w_data,
        w_physics=args.w_physics,
        w_energy=args.w_energy,
        w_angmom=args.w_angmom,
        use_adaptive=True,
        include_j2=True,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_rmse = float("inf")

    print(f"\n{'Epoch':>5} | {'Total':>12} | {'Data':>10} | {'Phys':>10} | "
          f"{'Ener':>10} | {'Pos RMSE':>10} | {'<5m%':>6} | {'Val RMSE':>10} | {'Best':>4} | {'Time':>6}")
    print("-" * 105)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, loss_fn, train_loader, optimizer, device)
        val_m = validate(model, loss_fn, val_loader, device)
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
            "train_metrics": train_m,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))

        print(
            f"{epoch:>5} | {train_m['total_loss']:>12.4f} | "
            f"{train_m.get('data_loss', 0):>10.4f} | {train_m.get('physics_loss', 0):>10.4f} | "
            f"{train_m.get('energy_loss', 0):>10.4f} | {val_m['val_pos_rmse_m']:>9.2f}m | "
            f"{val_m['below_5m_pct']:>5.1f}% | {val_m['val_pos_rmse_m']:>9.2f}m | "
            f"{'★' if is_best else '':>4} | {dt:>5.1f}s"
        )

        if val_m["val_pos_rmse_m"] < 5.0:
            print(f"\n✓ TARGET ACHIEVED! Val RMSE = {val_m['val_pos_rmse_m']:.2f}m < 5.0m")
            torch.save(ckpt, os.path.join(args.output_dir, "final_model.pt"))
            break

    print(f"\n{'=' * 105}")
    print(f"Training complete. Best Val RMSE: {best_rmse:.2f}m")
    print(f"Checkpoints: {args.output_dir}/")
    print(f"Best model: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tle/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--fourier_features", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--w_data", type=float, default=1.0)
    parser.add_argument("--w_physics", type=float, default=0.1)
    parser.add_argument("--w_energy", type=float, default=0.01)
    parser.add_argument("--w_angmom", type=float, default=0.01)
    parser.add_argument("--output_dir", default="outputs/pretrain")
    args = parser.parse_args()
    main(args)