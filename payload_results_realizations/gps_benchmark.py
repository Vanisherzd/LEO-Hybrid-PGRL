"""
GPS Benchmark & Validation — Compare PINN vs GPS ephemeris error
==============================================================
Validates that PINN prediction error is below 5m target.

Key metrics:
- RMSE (Root Mean Square Error) in meters
- MAXE (Maximum Absolute Error) in meters
- Availability (% of predictions within 5m)
- Comparison with GPS standalone error budget
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class GPSErrorModel:
    """
    GPS error budget (per ICD-GPS-200).
    Total User Range Error (URE) ~ 3.6m (95th percentile).
   """
    satellite_clock: float = 1.5    # m (1-sigma)
    ephemeris: float = 2.1          # m (1-sigma)
    ionospheric: float = 2.0        # m (1-sigma, L1)
    tropospheric: float = 0.7       # m (1-sigma)
    multipath: float = 1.2          # m (1-sigma, ground receiver)
    receiver_noise: float = 0.5     # m (1-sigma)

    @property
    def total_ure_1sigma(self) -> float:
        return math.sqrt(
            self.satellite_clock**2 + self.ephemeris**2 +
            self.ionospheric**2 + self.tropospheric**2
        )

    @property
    def total_ure_95(self) -> float:
        return 2.0 * self.total_ure_1sigma  # ~95th percentile

    def range_error_m(self, seed: int = None) -> float:
        """Sample a realistic GPS range error in meters."""
        rng = np.random.default_rng(seed)
        components = [
            rng.normal(0, self.satellite_clock),
            rng.normal(0, self.ephemeris),
            rng.normal(0, self.ionospheric),
            rng.normal(0, self.tropospheric),
            rng.normal(0, self.multipath),
            rng.normal(0, self.receiver_noise),
        ]
        return float(np.sqrt(sum(c**2 for c in components)))

    def position_error_m(self, seed: int = None) -> float:
        """Sample GPS position error from DOP + range error."""
        # Simplified: position error ≈ URE * HDOP (assume HDOP ~ 1.5)
        range_err = self.range_error_m(seed)
        return range_err * 1.5


class ValidationMetrics:
    """Compute comprehensive trajectory prediction metrics."""

    @staticmethod
    def compute_metrics(
        pred_positions: np.ndarray,   # (N, 3) km
        gt_positions: np.ndarray,     # (N, 3) km
        pred_velocities: np.ndarray = None,  # (N, 3) km/s
        gt_velocities: np.ndarray = None,    # (N, 3) km/s
    ) -> dict:
        """Compute all trajectory error metrics."""
        pos_errors = np.linalg.norm(pred_positions - gt_positions, axis=1) * 1000  # m
        pos_rmse = float(np.sqrt(np.mean(pos_errors ** 2)))
        pos_max = float(np.max(pos_errors))
        pos_mean = float(np.mean(pos_errors))
        pos_median = float(np.median(pos_errors))
        pos_std = float(np.std(pos_errors))

        # Percentiles
        p50 = float(np.percentile(pos_errors, 50))
        p90 = float(np.percentile(pos_errors, 90))
        p95 = float(np.percentile(pos_errors, 95))
        p99 = float(np.percentile(pos_errors, 99))

        # Threshold compliance
        below_1m = float(np.mean(pos_errors < 1.0)) * 100
        below_3m = float(np.mean(pos_errors < 3.0)) * 100
        below_5m = float(np.mean(pos_errors < 5.0)) * 100
        below_10m = float(np.mean(pos_errors < 10.0)) * 100

        metrics = {
            "pos_rmse_m": pos_rmse,
            "pos_max_m": pos_max,
            "pos_mean_m": pos_mean,
            "pos_median_m": pos_median,
            "pos_std_m": pos_std,
            "pos_p50_m": p50,
            "pos_p90_m": p90,
            "pos_p95_m": p95,
            "pos_p99_m": p99,
            "below_1m_pct": below_1m,
            "below_3m_pct": below_3m,
            "below_5m_pct": below_5m,
            "below_10m_pct": below_10m,
            "target_met": pos_rmse < 5.0,
        }

        # Velocity errors
        if pred_velocities is not None and gt_velocities is not None:
            vel_errors = np.linalg.norm(pred_velocities - gt_velocities, axis=1) * 1000  # m/s
            metrics["vel_rmse_ms"] = float(np.sqrt(np.mean(vel_errors ** 2)))
            metrics["vel_max_ms"] = float(np.max(vel_errors))
            metrics["vel_mean_ms"] = float(np.mean(vel_errors))

        return metrics

    @staticmethod
    def compare_with_gps(
        pinn_metrics: dict,
        gps_error_model: GPSErrorModel,
        num_samples: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Compare PINN predictions vs GPS standalone error."""
        rng = np.random.default_rng(seed)
        gps_errors = [gps_error_model.position_error_m(seed=rng.integers(1e9)) for _ in range(num_samples)]
        gps_errors = np.array(gps_errors)

        return {
            "gps_mean_error_m": float(np.mean(gps_errors)),
            "gps_max_error_m": float(np.max(gps_errors)),
            "gps_rmse_m": float(np.sqrt(np.mean(gps_errors ** 2))),
            "pinn_vs_gps_improvement": (
                (float(np.mean(gps_errors)) - pinn_metrics["pos_rmse_m"])
                / float(np.mean(gps_errors)) * 100
            ),
            "pinn_rmse_m": pinn_metrics["pos_rmse_m"],
            "pinn_target_met": pinn_metrics["target_met"],
        }


def run_validation(
    model: torch.nn.Module,
    tle_data,
    num_samples: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Run full validation against ground truth.

    Args:
        model: TrajectoryPINN
        tle_data: TLEData object
        num_samples: number of validation samples
        device: compute device

    Returns:
        full metrics dict
    """
    from models.orbital_physics import OrbitalElements, propagate_j2

    model.eval()

    # Generate validation times (cover 1.5 orbits)
    orbit_period = 2 * math.pi / tle_data.mean_motion_rad_s
    times = np.linspace(0, orbit_period * 1.5, num_samples)

    orbital_elements = np.array([
        tle_data.semi_major_axis,
        tle_data.eccentricity,
        tle_data.inclination,
        tle_data.raan,
        tle_data.omega,
        tle_data.mean_anomaly,
        tle_data.mean_motion_rad_s,
    ], dtype=np.float32)

    kep = OrbitalElements(*orbital_elements)

    pred_positions, gt_positions = [], []
    pred_velocities, gt_velocities = [], []

    for t in times:
        # Ground truth
        pos, vel = propagate_j2(kep, t)

        # PINN prediction
        with torch.no_grad():
            t_t = torch.tensor([[t]], dtype=torch.float32, device=device)
            oe_t = torch.tensor([orbital_elements], dtype=torch.float32, device=device)
            pred_state = model(t_t, oe_t).cpu().numpy()[0]

        pred_positions.append(pred_state[:3])
        pred_velocities.append(pred_state[3:])
        gt_positions.append(pos)
        gt_velocities.append(vel)

    pred_positions = np.array(pred_positions)
    gt_positions = np.array(gt_positions)
    pred_velocities = np.array(pred_velocities)
    gt_velocities = np.array(gt_velocities)

    # Compute metrics
    metrics = ValidationMetrics.compute_metrics(
        pred_positions, gt_positions,
        pred_velocities, gt_velocities,
    )

    # GPS comparison
    gps_model = GPSErrorModel()
    gps_comparison = ValidationMetrics.compare_with_gps(metrics, gps_model)
    metrics.update(gps_comparison)

    return metrics


if __name__ == "__main__":
    from data.dataset import parse_tle
    from models.orbital_physics import generate_synthetic_tle
    from models.pinn_core import TrajectoryPINN

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate TLE
    line1, line2 = generate_synthetic_tle(400, inclination_deg=53.0)
    tle = parse_tle(line1, line2)

    # Load pretrained model
    model = TrajectoryPINN().to(device)
    ckpt_path = "outputs/pretrain/best_model.pt"

    if __import__("os").path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print("No checkpoint found — using random model (expect high error)")

    metrics = run_validation(model, tle, num_samples=500, device=device)

    print("\n" + "=" * 60)
    print("PINN VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Position RMSE:          {metrics['pos_rmse_m']:.3f}m  {'✓' if metrics['target_met'] else '✗'} (target: <5m)")
    print(f"  Position Mean Error:    {metrics['pos_mean_m']:.3f}m")
    print(f"  Position Max Error:     {metrics['pos_max_m']:.3f}m")
    print(f"  Position Std Dev:       {metrics['pos_std_m']:.3f}m")
    print(f"  Position Median:        {metrics['pos_median_m']:.3f}m")
    print(f"  ─── Percentiles ───")
    print(f"    P50 (median):         {metrics['pos_p50_m']:.3f}m")
    print(f"    P90:                  {metrics['pos_p90_m']:.3f}m")
    print(f"    P95:                  {metrics['pos_p95_m']:.3f}m")
    print(f"    P99:                  {metrics['pos_p99_m']:.3f}m")
    print(f"  ─── Threshold Compliance ───")
    print(f"    < 1m:                 {metrics['below_1m_pct']:.1f}%")
    print(f"    < 3m:                 {metrics['below_3m_pct']:.1f}%")
    print(f"    < 5m:                 {metrics['below_5m_pct']:.1f}%  {'✓' if metrics['below_5m_pct'] > 95 else '✗'}")
    print(f"    < 10m:                {metrics['below_10m_pct']:.1f}%")
    print(f"  ─── GPS Comparison ───")
    print(f"  GPS mean error:         {metrics['gps_mean_error_m']:.3f}m")
    print(f"  GPS RMSE:              {metrics['gps_rmse_m']:.3f}m")
    print(f"  PINN vs GPS improvement: {metrics['pinn_vs_gps_improvement']:.1f}%")
    print(f"  ─── Velocity ───")
    print(f"  Velocity RMSE:          {metrics.get('vel_rmse_ms', 'N/A'):.4f} m/s" if isinstance(metrics.get('vel_rmse_ms'), float) else "  Velocity RMSE: N/A")
    print("=" * 60)