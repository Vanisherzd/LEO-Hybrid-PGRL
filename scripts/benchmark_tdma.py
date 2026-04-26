"""
TDMA MAC Benchmark — Smart RL-TDMA vs Standard SGP4 Baseline
============================================================
Models the FULL TDMA frame time budget to compute true payload efficiency.

Key insight: the guard band competes DIRECTLY with payload time within each
1-second frame. With 500ms guard band, only ~0.5 slots fit per frame vs 32 with
2.14ms guard. This is why traditional SGP4 MAC is so inefficient.

Collision model (Gaussian):
  P_collision = 2 * (1 - Φ(coverage_m / sigma_error_m))
where coverage = guard_ms * V_LEO_KM_S / 1000 (in km).
"""

import argparse
import math
import os
import sys
from scipy.stats import norm

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pinn_core import TrajectoryPINN

# ─── Normalization constants ──────────────────────────────────────────────────
DU = 10000.0
VU = 10.0
T_SCALE = 16661.0
OE_MEAN = np.array([6778.137, 0.001, 0.925, 0.0, 0.0, 0.0], dtype=np.float32)
OE_STD = np.array([1.0, 0.001, 0.35, 2.0, 2.0, 2.0], dtype=np.float32)

# ─── Physical / Protocol constants ────────────────────────────────────────────
V_LEO_KM_S = 7.67        # Orbital speed at ~400 km LEO (km/s)
TX_POWER_W = 1.0         # Satellite TX (W)
IDLE_POWER_W = 0.05      # Idle/guard band (W)
SLOT_DATA_BYTES = 16     # Payload per IoT slot (bytes)
FRAME_MS = 1000.0        # Frame duration (ms)
BEACON_MS = 10.0         # Beacon slot (ms)
PRIORITY_MS = 20.0       # Priority slot (ms) × 2
N_PRIORITY = 2
GAP_MS = 0.1             # TX turnaround gap (ms)
NUM_IOT = 32             # IoT slots per frame (design target)
TX_MS = 5.0              # Nominal TX duration per IoT slot (ms)


def compute_guard_band_ms(rmse_m: float) -> float:
    """3-sigma guard band with 20% margin. PINN: 2.14ms, SGP4: 782ms."""
    return 1.2 * 3.0 * rmse_m / V_LEO_KM_S


def p_collision(guard_ms: float, sigma_m: float) -> float:
    """P(collision) = 2*(1-Φ(coverage_km/sigma_km)) — two-sided."""
    coverage_km = guard_ms * V_LEO_KM_S / 1000.0
    sigma_km = sigma_m / 1000.0
    if sigma_km < 1e-12:
        return 0.0
    return 2.0 * (1.0 - norm.cdf(coverage_km / sigma_km))


def slot_time_budget(guard_ms: float) -> dict:
    """
    How many IoT slots actually fit in the usable frame time, and what
    fraction of each slot is payload vs guard overhead?

    Frame budget:
      beacon(10ms) + 2×priority(20ms+gap) + N×iot_slot(budget) ≤ 1000ms

    Each IoT slot budget: guard(guard_ms) + tx(remaining)
    We WANT 32 IoT slots ideally, but the guard may force fewer.

    Returns: {n_slots, effective_tx_ms, payload_efficiency, guard_overhead_pct}
    """
    overhead_ms = BEACON_MS + N_PRIORITY * (PRIORITY_MS + GAP_MS)
    usable_ms = FRAME_MS - overhead_ms  # 950ms

    # Budget per slot = guard + nominal_tx
    budget_per_slot = guard_ms + TX_MS

    # Maximum slots that fit
    n_fit = int(usable_ms / budget_per_slot)
    n_fit = max(n_fit, 1)  # at least 1

    # What fraction of the frame is payload vs overhead?
    # Time actually spent transmitting data
    tx_time_ms = n_fit * TX_MS
    # Time spent in guard bands (wasted)
    guard_time_ms = n_fit * guard_ms
    # Turnaround gaps
    gap_time_ms = n_fit * GAP_MS

    # Payload efficiency = TX / total_frame
    payload_eff = tx_time_ms / FRAME_MS * 100.0

    # Guard overhead = guard_time / total_frame
    guard_oh_pct = guard_time_ms / FRAME_MS * 100.0

    return {
        "n_slots": n_fit,
        "tx_ms": tx_time_ms,
        "guard_ms": guard_time_ms,
        "gap_ms": gap_time_ms,
        "payload_eff": payload_eff,
        "guard_oh_pct": guard_oh_pct,
        "budget_per_slot": budget_per_slot,
        "usable_ms": usable_ms,
    }


def simulate_pass(
    guard_ms: float,
    sigma_m: float,
    num_frames: int,
    num_slots_design: int,  # what we WANT to schedule
) -> dict:
    """
    Full analytical TDMA pass simulation.
    Computes collision probability, throughput, energy, efficiency.
    """
    p_col = p_collision(guard_ms, sigma_m)

    # Time budget analysis
    tb = slot_time_budget(guard_ms)
    n_slots = tb["n_slots"]

    # Successful transmissions per slot (geometric retry model)
    # Expected # of TX attempts for 1 success: 1/(1-p_col)
    # Time per attempt = guard_ms + TX_MS + GAP_MS
    slot_duration_ms = guard_ms + TX_MS + GAP_MS
    expected_attempts = 1.0 / max(1.0 - p_col, 1e-9)
    expected_slot_ms = slot_duration_ms * expected_attempts

    # Expected bytes delivered per slot
    expected_bytes_per_slot = SLOT_DATA_BYTES * (1.0 - p_col)

    total_slots = num_frames * n_slots
    expected_throughput_bytes = total_slots * expected_bytes_per_slot

    # Collision count
    expected_collisions = total_slots * p_col

    # Energy per frame
    # Guard/idle energy: always on during guard
    guard_j_per_slot = (guard_ms / 1000.0) * IDLE_POWER_W
    # TX energy: only on successful TX
    tx_j_per_slot = (TX_MS / 1000.0) * TX_POWER_W

    # Energy = guard (always) + TX (on success)
    j_per_slot = guard_j_per_slot + tx_j_per_slot * (1.0 - p_col)
    total_energy_j = total_slots * j_per_slot

    # Guard waste
    guard_j_total = guard_j_per_slot * total_slots
    # TX energy
    tx_j_total = tx_j_per_slot * total_slots * (1.0 - p_col)

    return {
        "guard_ms": guard_ms,
        "p_col": p_col,
        "pdr": (1.0 - p_col) * 100.0,
        "n_slots": n_slots,
        "tx_ms": tb["tx_ms"],
        "guard_ms_total": tb["guard_ms"],
        "payload_eff": tb["payload_eff"],
        "guard_oh_pct": tb["guard_oh_pct"],
        "throughput_bytes": expected_throughput_bytes,
        "throughput_mb": expected_throughput_bytes / 1e6,
        "total_energy_j": total_energy_j,
        "guard_energy_j": guard_j_total,
        "tx_energy_j": tx_j_total,
        "collisions": expected_collisions,
        "successful": total_slots * (1.0 - p_col),
        "total_slots": total_slots,
        "budget_per_slot": tb["budget_per_slot"],
        "usable_ms": tb["usable_ms"],
        "pdr_per_slot": (1.0 - p_col),
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*70}")
    print(f"   TDMA MAC Benchmark  —  Smart RL-TDMA vs Standard SGP4 Baseline")
    print(f"{'═'*70}")
    print(f"   Device: {device}")
    print(f"{'═'*70}\n")

    # ── Load PINN ───────────────────────────────────────────────────────────
    model = TrajectoryPINN(
        orbital_elem_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_features=args.fourier_features,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        val_rmse = ckpt.get("val_metrics", {}).get("val_pos_rmse_m", None)
        epoch = ckpt.get("epoch", "?")
        print(f"[✓] Loaded: {args.checkpoint}")
        print(f"     Epoch {epoch} | Val RMSE {val_rmse:.2f}m" if val_rmse else "     (no val metrics)")
    else:
        print(f"[!] Not found: {args.checkpoint}")
        val_rmse = 1000.0

    model.eval()

    # ── Guard band parameters ────────────────────────────────────────────────
    pinm_guard = compute_guard_band_ms(val_rmse or 4.55)
    sgp4_guard = 500.0  # SGP4: 500ms mandatory guard

    pinm_sigma = val_rmse or 4.55
    sgp4_sigma = 2000.0  # Typical SGP4 standalone error

    pinm_pcol = p_collision(pinm_guard, pinm_sigma)
    sgp4_pcol = p_collision(sgp4_guard, sgp4_sigma)

    # Time budgets
    tb_pinm = slot_time_budget(pinm_guard)
    tb_sgp4 = slot_time_budget(sgp4_guard)

    NUM_FRAMES = args.num_frames
    NUM_SLOTS = args.num_slots

    print(f"[i] Pass: {NUM_FRAMES}s | Target: {NUM_SLOTS} IoT slots/frame")
    print(f"     Frame budget: {FRAME_MS:.0f}ms = beacon({BEACON_MS}ms) + "
          f"{N_PRIORITY}×priority({PRIORITY_MS}ms) + IoT × (guard+{TX_MS}ms)")
    print(f"     Usable for IoT: {tb_pinm['usable_ms']:.0f}ms/frame")

    print(f"\n{'─'*70}")
    print(f"   Guard Band & Frame Time Budget")
    print(f"{'─'*70}")
    print(f"   {'Approach':<18} {'Guard/slot':>10} {'σ_pos':>8} {'Slots fit':>10} "
          f"{'TX/slot':>8} {'Guard waste':>12}")
    print(f"   {'─'*18} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*12}")

    print(f"   {'Smart RL-TDMA':<18} {pinm_guard:>9.2f}ms "
          f"{pinm_sigma:>7.1f}m "
          f"{tb_pinm['n_slots']:>10d} "
          f"{TX_MS:>7.1f}ms "
          f"{tb_pinm['guard_oh_pct']:>10.1f} %")
    print(f"   {'Standard SGP4-MAC':<18} {sgp4_guard:>9.1f}ms "
          f"{sgp4_sigma:>7.0f}m "
          f"{tb_sgp4['n_slots']:>10d} "
          f"{TX_MS:>7.1f}ms "
          f"{tb_sgp4['guard_oh_pct']:>10.1f} %")
    print(f"   {'─'*18} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*12}")
    print(f"   PINN: {tb_pinm['n_slots']}/{NUM_SLOTS} slots fit "
          f"({tb_pinm['payload_eff']:.1f}% payload efficiency)")
    print(f"   SGP4: {tb_sgp4['n_slots']}/{NUM_SLOTS} slots fit "
          f"({tb_sgp4['payload_eff']:.1f}% payload efficiency)")
    print(f"{'─'*70}\n")

    # ── Run simulations ──────────────────────────────────────────────────────
    smart = simulate_pass(pinm_guard, pinm_sigma, NUM_FRAMES, NUM_SLOTS)
    standard = simulate_pass(sgp4_guard, sgp4_sigma, NUM_FRAMES, NUM_SLOTS)

    # ── Comparison table ─────────────────────────────────────────────────────
    tp_gain = smart["throughput_mb"] / max(standard["throughput_mb"], 1e-9)
    pdr_gain = smart["pdr"] / max(standard["pdr"], 0.01)
    nrg_save = standard["total_energy_j"] - smart["total_energy_j"]
    col_save = standard["collisions"] - smart["collisions"]

    # Physical maximum if NO guard bands (theoretical): 950ms / 5ms = 190 slots/frame
    ideal_slots_frame = tb_pinm["usable_ms"] / TX_MS
    ideal_tp_mb = (NUM_FRAMES * ideal_slots_frame * SLOT_DATA_BYTES) / 1e6

    print(f"{'═'*70}")
    print(f"   RESULTS  —  {NUM_FRAMES}s LEO Satellite Pass")
    print(f"{'═'*70}\n")
    print(f"   {'Metric':<38} {'Smart RL-TDMA':>13} {'Standard SGP4':>13} "
          f"{'Improvement':>12}")
    print(f"   {'─'*38} {'─'*13} {'─'*13} {'─'*12}")
    print(f"   {'Guard Band / Slot':<38} {pinm_guard:>10.2f} ms "
          f"{sgp4_guard:>10.1f} ms "
          f"{(sgp4_guard-pinm_guard):>+9.1f} ms")
    print(f"   {'Guard Band Overhead':<38} {smart['guard_oh_pct']:>10.2f} % "
          f"{standard['guard_oh_pct']:>10.1f} % "
          f"{(smart['guard_oh_pct']-standard['guard_oh_pct']):>+9.1f} %")
    print(f"   {'Slots that Fit per Frame':<38} {smart['n_slots']:>13d} "
          f"{standard['n_slots']:>13d} {smart['n_slots']-standard['n_slots']:>+12d}")
    print(f"   {'Payload Efficiency':<38} {smart['payload_eff']:>10.1f} % "
          f"{standard['payload_eff']:>10.1f} % "
          f"{(smart['payload_eff']-standard['payload_eff']):>+9.1f} %")
    print(f"   {'─'*38} {'─'*13} {'─'*13} {'─'*12}")
    print(f"   {'Collision Probability':<38} {pinm_pcol*100:>10.2f} % "
          f"{sgp4_pcol*100:>10.2f} % "
          f"{(sgp4_pcol-pinm_pcol)*100:>+9.2f} %")
    print(f"   {'Packet Delivery Rate (PDR)':<38} {smart['pdr']:>10.1f} % "
          f"{standard['pdr']:>10.1f} % "
          f"{(smart['pdr']-standard['pdr']):>+9.1f} %")
    print(f"   {'Data Throughput (Mbytes/pass)':<38} {smart['throughput_mb']:>10.3f} MB "
          f"{standard['throughput_mb']:>10.3f} MB "
          f"{(smart['throughput_mb']-standard['throughput_mb'])*1000:>+9.2f} KB")
    print(f"   {'vs Physical Ideal (no guard)':<38} {smart['throughput_mb']/ideal_tp_mb*100:>10.1f} % "
          f"{standard['throughput_mb']/ideal_tp_mb*100:>10.1f} % "
          f"{(smart['throughput_mb']-standard['throughput_mb'])/ideal_tp_mb*100:>+9.1f} %")
    print(f"   {'─'*38} {'─'*13} {'─'*13} {'─'*12}")
    print(f"   {'TX Energy (J/pass)':<38} {smart['tx_energy_j']:>10.1f} J "
          f"{standard['tx_energy_j']:>10.1f} J "
          f"{(standard['tx_energy_j']-smart['tx_energy_j']):>+9.1f} J")
    print(f"   {'Guard-Band Idle Energy (J/pass)':<38} {smart['guard_energy_j']:>10.1f} J "
          f"{standard['guard_energy_j']:>10.1f} J "
          f"{(standard['guard_energy_j']-smart['guard_energy_j']):>+9.1f} J")
    print(f"   {'Total Energy (J/pass)':<38} {smart['total_energy_j']:>10.1f} J "
          f"{standard['total_energy_j']:>10.1f} J "
          f"{nrg_save:>+9.1f} J")
    print(f"   {'Energy per Mbyte':<38} "
          f"{smart['total_energy_j']/max(smart['throughput_mb'],1e-9):>10.1f} J/MB "
          f"{standard['total_energy_j']/max(standard['throughput_mb'],1e-9):>10.1f} J/MB "
          f"{'—':>12}")
    print(f"   {'─'*38} {'─'*13} {'─'*13} {'─'*12}")
    print(f"   {'Expected Collisions':<38} {smart['collisions']:>13.0f} "
          f"{standard['collisions']:>13.0f} {'—':>12}")
    print(f"   {'Successful Transmissions':<38} {smart['successful']:>13.0f} "
          f"{standard['successful']:>13.0f} {'—':>12}")
    print(f"   {'─'*38} {'─'*13} {'─'*13} {'─'*12}")
    print(f"   {'Physical Ideal Throughput (no guard)':<38} {ideal_tp_mb:>10.3f} MB "
          f"{'—':>13} {'—':>12}")

    print(f"\n{'═'*70}")
    print(f"   ★ Guard band: {sgp4_guard:.0f}ms → {pinm_guard:.2f}ms  "
          f"({(1-pinm_guard/sgp4_guard)*100:.1f}% tighter)")
    print(f"   ★ Slots per frame: {standard['n_slots']} → {smart['n_slots']}  "
          f"({smart['n_slots']/max(standard['n_slots'],1):.0f}× more)")
    print(f"   ★ PDR: {standard['pdr']:.1f}% → {smart['pdr']:.1f}%  "
          f"({pdr_gain:.2f}× relative)")
    print(f"   ★ Throughput: {standard['throughput_mb']:.3f} → {smart['throughput_mb']:.3f} MB/pass  "
          f"({tp_gain:.1f}× more data)")
    print(f"   ★ Energy saved: {nrg_save:.1f} J/pass  "
          f"({(nrg_save/max(standard['total_energy_j'],0.01))*100:.1f}% reduction)")
    print(f"   ★ Collision reduction: {standard['collisions']:.0f} → {smart['collisions']:.0f}  "
          f"({col_save:.0f} fewer)")
    print(f"{'═'*70}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "benchmark_results.txt")
    with open(out, "w") as f:
        f.write("Smart RL-TDMA vs Standard SGP4 MAC Benchmark\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"PINN Val RMSE: {val_rmse:.2f}m | SGP4 baseline: 2000m\n")
        f.write(f"Pass: {NUM_FRAMES}s | Frames: {NUM_FRAMES} | Target slots: {NUM_SLOTS}/frame\n\n")
        f.write(f"Smart guard: {pinm_guard:.2f}ms | Standard: {sgp4_guard:.1f}ms\n")
        f.write(f"PINN collision: {pinm_pcol*100:.2f}% | SGP4: {sgp4_pcol*100:.2f}%\n\n")
        f.write(f"{'Metric':<32} {'Smart':>10} {'Standard':>10}\n")
        f.write("-" * 54 + "\n")
        f.write(f"{'PDR':<32} {smart['pdr']:>9.1f}% {standard['pdr']:>9.1f}%\n")
        f.write(f"{'Throughput (MB)':<32} {smart['throughput_mb']:>10.3f} {standard['throughput_mb']:>10.3f}\n")
        f.write(f"{'Slot Efficiency':<32} {smart['payload_eff']:>9.1f}% {standard['payload_eff']:>9.1f}%\n")
        f.write(f"{'Total Energy (J)':<32} {smart['total_energy_j']:>10.1f} {standard['total_energy_j']:>10.1f}\n")
        f.write(f"{'Collisions':<32} {smart['collisions']:>10.0f} {standard['collisions']:>10.0f}\n")
    print(f"[i] Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDMA MAC Benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/pretrain_golden/best_model.pt")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--fourier_features", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    args = parser.parse_args()
    main(args)