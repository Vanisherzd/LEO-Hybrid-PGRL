"""
Online GRPO Training Loop — Real-time weight updates on successful TDMA events
================================================================================
Simulates satellite pass with TDMA communication events triggering GRPO updates.

Usage:
 python scripts/train_online.py --checkpoint outputs/pretrain_golden/best_model.pt
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics_ml.pinn_core import TrajectoryPINN
from models.grpo_agent import create_grpo_agent, GRPOConfig
from models.orbital_physics import generate_synthetic_tle, SGP4Propagator
from protocols.mac_tdma import MACTDMAProtocol, TDMAConfig, CommEvent


def run_online_grpo_loop(
    model,
    grpo_agent,
    mac_protocol,
    sgp4,
    orbital_elements,
    ground_stations,
    num_frames: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run the online GRPO training loop over a satellite pass.
    Each successful TDMA communication:
    1. Observe (t, orbital_elements, GPS position, GPS velocity)
    2. Compute reward (negative GPS error)
    3. If error > threshold → GRPO policy gradient update
    4. Log metrics
    """
    all_errors = []
    update_count = 0
    comm_event_count = 0
    epoch_start = time.time()

    start_time = datetime.utcnow()

    for frame_idx in range(num_frames):
        t_s = frame_idx * mac_protocol.config.frame_duration_s

        # Ground truth from SGP4 propagator
        gt_pos, gt_vel = sgp4.propagate(t_s)

        # Build and simulate TDMA frame
        frame = mac_protocol.build_frame(
            frame_idx,
            start_time + timedelta(seconds=frame_idx * mac_protocol.config.frame_duration_s),
        )

        # Simplified elevation (use first station)
        ref_station = ground_stations.get(100, np.array([6378.137, 0, 0]))
        elevation = mac_protocol._compute_elevation(gt_pos, ref_station)

        events = mac_protocol.simulate_frame(
            frame, gt_pos, ground_stations, elevation,
            orbital_elements, t_s,
        )

        comm_event_count += len(events)

        # Process each successful communication event
        for event in events:
            reward, position_error_m, should_update = grpo_agent.observe_and_reward(
                t=event.t_since_epoch_s,
                orbital_elems=event.orbital_elements,
                gps_pos=gt_pos,
                gps_vel=gt_vel,
            )
            all_errors.append(position_error_m)

            if should_update:
                update_metrics = grpo_agent.online_update()
                update_count += 1

        if verbose and update_count % 10 == 0:
            print(f" Frame {frame_idx:3d} | Events:{len(events):2d} | "
                  f"Error:{position_error_m:.2f}m | Updates:{update_count}")

    elapsed = time.time() - epoch_start
    errors = np.array(all_errors) if all_errors else np.array([0.0])

    return {
        "total_frames": num_frames,
        "total_comm_events": comm_event_count,
        "total_grpo_updates": update_count,
        "mean_error_m": float(np.mean(errors)),
        "max_error_m": float(np.max(errors)),
        "min_error_m": float(np.min(errors)),
        "std_error_m": float(np.std(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "below_5m_pct": float(np.mean(errors < 5.0) * 100),
        "below_3m_pct": float(np.mean(errors < 3.0) * 100),
        "elapsed_s": elapsed,
        "updates_per_second": update_count / elapsed if elapsed > 0 else 0,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading pretrained model from: {args.checkpoint}")

    # Load pretrained model — use orbital_elem_dim=6 (matches training)
    model = TrajectoryPINN(
        orbital_elem_dim=6,  # Fixed: was 7
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_features=args.fourier_features,
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded pretrained model from epoch {checkpoint.get('epoch', '?')} "
              f"(Val RMSE: {checkpoint.get('val_metrics', {}).get('val_pos_rmse_m', '?')}m)")
    else:
        print("WARNING: No pretrained model found, starting from scratch")

    # GRPO Agent
    grpo_config = GRPOConfig(
        learning_rate=args.grpo_lr,
        error_threshold_m=5.0,
        kl_target=0.01,
        gradient_bound=0.1,
        device=device,
    )
    grpo_agent = create_grpo_agent(model, **vars(grpo_config))

    # MAC TDMA Protocol
    tdma_config = TDMAConfig(
        frame_duration_s=args.frame_duration_s,
        num_iot_slots=args.num_iot_slots,
    )
    mac_protocol = MACTDMAProtocol(tdma_config)

    # Satellite propagator — generate synthetic TLE at ~400 km
    line1, line2 = generate_synthetic_tle(altitude_km=400, inclination_deg=53.0)
    sgp4 = SGP4Propagator(line1, line2)

    orbital_elements = np.array([
        sgp4.tle["a"],
        sgp4.tle["eccentricity"],
        sgp4.tle["inclination"],
        sgp4.tle["raan"],
        sgp4.tle["omega"],
        sgp4.tle["mean_anomaly"],
        sgp4.tle["n"],
    ], dtype=np.float32)

    # Ground stations
    ground_stations = mac_protocol.generate_ground_stations(args.num_stations)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"Online GRPO Training — LEO Satellite IoT PINN")
    print(f"{'─'*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frames: {args.num_frames}, IoT slots: {args.num_iot_slots}, "
          f"Stations: {args.num_stations}")
    print(f"GRPO LR: {args.grpo_lr}, Frame duration: {args.frame_duration_s}s")
    print(f"Output: {args.output_dir}")
    print(f"{'─'*60}\n")

    # Track best
    best_rmse = float("inf")
    total_passes = args.num_passes

    for pass_idx in range(1, total_passes + 1):
        print(f"\n{'='*60}")
        print(f"Satellite Pass {pass_idx}/{total_passes}")
        print(f"{'='*60}")

        metrics = run_online_grpo_loop(
            model=model,
            grpo_agent=grpo_agent,
            mac_protocol=mac_protocol,
            sgp4=sgp4,
            orbital_elements=orbital_elements,
            ground_stations=ground_stations,
            num_frames=args.num_frames,
            verbose=True,
        )

        print(f"\n Pass {pass_idx} Results:")
        print(f" ├─ Comm events: {metrics['total_comm_events']}")
        print(f" ├─ GRPO updates: {metrics['total_grpo_updates']}")
        print(f" ├─ RMSE: {metrics['rmse_m']:.2f}m")
        print(f" ├─ Mean error: {metrics['mean_error_m']:.2f}m")
        print(f" ├─ Max error: {metrics['max_error_m']:.2f}m")
        print(f" ├─ <5m threshold: {metrics['below_5m_pct']:.1f}%")
        print(f" └─ Updates/sec: {metrics['updates_per_second']:.1f}")

        # Save checkpoint
        is_best = metrics["rmse_m"] < best_rmse
        if is_best:
            best_rmse = metrics["rmse_m"]
            ckpt_path = os.path.join(args.output_dir, "best_online_model.pt")
            grpo_agent.save(ckpt_path)
            print(f" ★ New best! RMSE = {best_rmse:.2f}m")

        if metrics["rmse_m"] < 5.0:
            print(f"\n✓ Target achieved! RMSE = {metrics['rmse_m']:.2f}m < 5.0m")
            break

    print(f"\n{'='*60}")
    print(f"Training complete. Best RMSE: {best_rmse:.2f}m")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online GRPO training for LEO PINN")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/pretrain_golden/best_model.pt")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--fourier_features", type=int, default=128)
    parser.add_argument("--grpo_lr", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--num_passes", type=int, default=10)
    parser.add_argument("--num_iot_slots", type=int, default=32)
    parser.add_argument("--num_stations", type=int, default=32)
    parser.add_argument("--frame_duration_s", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs/online_grpo")
    args = parser.parse_args()

    main(args)