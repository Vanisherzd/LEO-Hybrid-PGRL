"""
Validation Entry Point — GPS benchmark and PINN accuracy verification
========================================================================
Usage:
  python scripts/validate.py --checkpoint checkpoints/pinn_pretrained.pt \
      --num_samples 100 --output_json validation/results.json
"""

import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from data.dataset import parse_tle
from models.pinn_core import TrajectoryPINN, count_parameters
from models.orbital_physics import generate_synthetic_tle
from validation.gps_benchmark import run_validation


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate or use TLE
    line1, line2 = generate_synthetic_tle(
        altitude_km=args.altitude_km,
        inclination_deg=args.inclination_deg,
    )
    tle = parse_tle(line1, line2)

    # Load model
    model = TrajectoryPINN(
        orbital_elem_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_features=args.fourier_features,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint.get("epoch", "?")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"WARNING: No checkpoint at {args.checkpoint} — using random weights")

    print(f"\nRunning validation with {args.num_samples} samples...")

    metrics = run_validation(model, tle, num_samples=args.num_samples, device=device)

    print("\n" + "=" * 60)
    print("LEO PINN — VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Position RMSE:    {metrics['pos_rmse_m']:.3f}m  "
          f"{'✓ PASS' if metrics['target_met'] else '✗ FAIL'} (target: <5m)")
    print(f"  Position Mean:    {metrics['pos_mean_m']:.3f}m")
    print(f"  Position Max:     {metrics['pos_max_m']:.3f}m")
    print(f"  P50 (median):     {metrics['pos_p50_m']:.3f}m")
    print(f"  P90:              {metrics['pos_p90_m']:.3f}m")
    print(f"  P95:              {metrics['pos_p95_m']:.3f}m")
    print(f"  P99:              {metrics['pos_p99_m']:.3f}m")
    print(f"  Threshold compliance:")
    print(f"    < 1m: {metrics['below_1m_pct']:.1f}%")
    print(f"    < 3m: {metrics['below_3m_pct']:.1f}%")
    print(f"    < 5m: {metrics['below_5m_pct']:.1f}%  {'✓' if metrics['below_5m_pct'] > 95 else '✗'}")
    print(f"    < 10m: {metrics['below_10m_pct']:.1f}%")
    print(f"  GPS comparison:")
    print(f"    GPS mean error:  {metrics['gps_mean_error_m']:.3f}m")
    print(f"    GPS RMSE:        {metrics['gps_rmse_m']:.3f}m")
    print(f"    PINN vs GPS:     {metrics['pinn_vs_gps_improvement']:.1f}% improvement")
    vel_rmse = metrics.get('vel_rmse_ms')
    if isinstance(vel_rmse, float):
        print(f"  Velocity RMSE:    {vel_rmse:.4f} m/s")
    print("=" * 60)

    if metrics["target_met"]:
        print("\n✓ PINN TRAJECTORY PREDICTION TARGET MET: RMSE < 5m")
    else:
        print(f"\n✗ PINN RMSE {metrics['pos_rmse_m']:.2f}m exceeds 5m target")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate LEO PINN model")
    parser.add_argument("--checkpoint", type=str, default="outputs/pretrain/best_model.pt")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--fourier_features", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--altitude_km", type=float, default=400.0)
    parser.add_argument("--inclination_deg", type=float, default=53.0)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    main(args)