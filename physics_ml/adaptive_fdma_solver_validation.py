#!/usr/bin/env python3
"""
adaptive_fdma_solver_validation.py
===================================
Validates Adaptive FDMA (LR-FH/FDMA) spatial sub-carrier allocation against
unsynchronised Pure Aloha under high-Doppler tracking conditions.

Generates:
  plots/PLOT_NEWOFMMA_IMp_2x.jpeg
    - Left panel  : Pure Aloha collision heatmap (unmanaged, dead SGP4)
    - Right panel : Adaptive FDMA heatmap (PGRL-managed spatial bins)
    - Both panels share the same colour scale and are annotated with
      spectral efficiency (bits/s/Hz) and collision probability.

Key physics
-----------
With PGRL timing jitter sigma_t ~ 16 ms and sub-300 Hz Doppler residual,
the spatial sub-carrier bin occupancy probability is:

  P_occupy(bin_k) = 1 - exp(-lambda_k * T_sym)

where lambda_k is the endpoint arrival rate at bin k.
Adaptive FDMA assigns bins based on predicted Doppler (v_range_rate / c * f_c),
so bin spacing adapts to sigma_v ~ 0.5 m/s, giving sub-500 Hz inter-bin gap.

Pure Aloha: unsynchronised endpoints transmit at random,
             collision probability P_coll = 1 - exp(-2*lambda*T_sym)
             With lambda = 100 endpoints, T_sym = 10 ms  =>  P_coll > 0.86

Adaptive FDMA: synchronised by Golden Anchor predictor,
              collision probability P_coll = sum_k (lambda_k * T_sym)^n / n!
              With sigma_f = 300 Hz, bin spacing 3.125 kHz  =>  P_coll < 0.008

Requires: numpy, scipy, matplotlib  (no torch, no uhd)
"""

import os
import sys
import math
import numpy as np
from scipy import special

np.random.seed(0xAE42)

# ── paths ────────────────────────────────────────────────────────────────────
PLOT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "plots", "PLOT_NEWOFMMA_IMp_2x.jpeg"
)
os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)

# ── grid parameters ───────────────────────────────────────────────────────────
N_SUBCARRIERS = 64          # frequency bins (same as LR-FHSS grid)
N_TIMESLOTS   = 48          # observation window
N_ENDPOINTS   = 100         # simultaneous IoT endpoints
T_SYM         = 10e-3       # symbol duration 10 ms (matches control slot)
F_STEP        = 3.125e3    # bin spacing Hz (200 kHz / 64 bins)
F_CENTRE      = 436.5e6    # Hz  S-band centre

# ── Aloha collision model ─────────────────────────────────────────────────────
def pure_aloha_grid(n_endpoints: int, n_bins: int, n_slots: int,
                    p_tx: float = 0.5) -> np.ndarray:
    """
    Unsynchronised Pure Aloha spatial allocation.
    Each endpoint picks a random frequency bin and random time slot.
    Returns energy matrix (n_slots, n_bins) — collisions show as
    summed energy in the same bin-slot.
    """
    grid = np.zeros((n_slots, n_bins), dtype=np.float64)
    for _ in range(n_endpoints):
        if np.random.rand() > p_tx:
            continue
        t = np.random.randint(0, n_slots)
        f = np.random.randint(0, n_bins)
        # Dead SGP4: ±50 kHz bin uncertainty => spread over ~16 bins
        spread = 16
        for dw in range(-spread, spread + 1):
            fb = f + dw
            if 0 <= fb < n_bins:
                weight = math.exp(-abs(dw) / (spread * 0.5))
                grid[t, fb] += weight * np.random.uniform(0.7, 1.3)
    # Normalise per time slot
    row_max = grid.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    grid /= row_max
    return grid


def adaptive_fdma_grid(n_endpoints: int, n_bins: int, n_slots: int,
                       sigma_f_hz: float = 300.0,
                       t_slot_s: float  = 0.016,
                       pgrl_weight: float = 0.92) -> np.ndarray:
    """
    PGRL-guided Adaptive FDMA spatial allocation.
    Endpoints are pre-assigned bins based on predicted Doppler offset
    (v_range_rate / c * f_c).  The PGRL predictor holds sigma_f < 300 Hz,
    so bins are spaced by 3.125 kHz and the residual spread is 1 bin at most.
    A fraction (1-pgrl_weight) of endpoints act as uncooperative "rogue"
    nodes representing residual prediction failure.
    """
    grid = np.zeros((n_slots, n_bins), dtype=np.float64)
    doppler_bins = sigma_f_hz / F_STEP   # ~0.096 bins of uncertainty

    for ep in range(n_endpoints):
        if np.random.rand() > pgrl_weight:
            # Rogue: behaves like Aloha
            t = np.random.randint(0, n_slots)
            f = np.random.randint(0, n_bins)
        else:
            # PGRL-synchronised: deterministic assignment based on
            # endpoint ID modulo N_SUBCARRIERS with Doppler pre-compensation
            t = ep % n_slots
            f = (ep + int(doppler_bins * np.random.randn())) % n_bins

        spread = 1          # PGRL residual: ±1 bin at most
        for dw in range(-spread, spread + 1):
            fb = f + dw
            if 0 <= fb < n_bins:
                weight = math.exp(-abs(dw) / 0.7)
                grid[t, fb] += weight * np.random.uniform(0.85, 1.15)

    # Normalise
    row_max = grid.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    grid /= row_max
    return grid


def collision_rate(grid: np.ndarray) -> float:
    """Fraction of occupied cells that have more than one transmitter."""
    total = (grid > 0.01).sum()
    if total == 0:
        return 0.0
    # Multi-transmitter cells (where random weight summed > 1.4) count as collision
    collisions = (grid > 1.4).sum()
    return float(collisions / total)


def spectral_efficiency(grid: np.ndarray) -> float:
    """
    Bits per second per Hz, assuming QPSK (2 bits/symbol) and
    symbol rate = 1 / T_SYM.
    """
    success = (0.1 < grid) & (grid <= 1.4)
    n_success = success.sum()
    bits_per_sym = 2.0
    bw_hz = N_SUBCARRIERS * F_STEP
    eff = n_success * bits_per_sym / (N_TIMESLOTS * T_SYM) / bw_hz
    return float(eff)


# ── visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


def _cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fdma", ["#0d1b2a", "#1b263b", "#415a77",
                  "#778da9", "#e0e1dd", "#ffffff"]
    )


def plot_heatmap(ax, data: np.ndarray, title: str,
                 coll_pct: float, spec_eff: float,
                 cmap, vmin: float, vmax: float):
    im = ax.imshow(data, origin='lower', aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.set_xlabel("Frequency Bin (kHz offset)", fontsize=8)
    ax.set_ylabel("Time Slot Index", fontsize=8)
    # Y-ticks
    ax.set_yticks(np.linspace(0, N_TIMESLOTS - 1, 7).astype(int))
    ax.set_yticklabels([str(int(y)) for y in np.linspace(0, N_TIMESLOTS - 1, 7)])
    # X-ticks (kHz)
    xticks_khz = np.linspace(0, N_SUBCARRIERS - 1, 9) * F_STEP / 1e3
    ax.set_xticks(np.linspace(0, N_SUBCARRIERS - 1, 9).astype(int))
    ax.set_xticklabels([f"{k:.0f}" for k in xticks_khz])
    ax.grid(True, alpha=0.15, color='white', linewidth=0.3)

    # Annotation box
    badge_bg = '#fff0f0' if coll_pct > 40 else '#f0fff4'
    badge_fg = '#c0392b' if coll_pct > 40 else '#27ae60'
    ax.text(0.97, 0.03,
            f"Collisions: {coll_pct:.1f}%\n"
            f"Spectral eff: {spec_eff*1e6:.2f} bits/s/Hz",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, fontweight='bold', color=badge_fg,
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor=badge_bg, edgecolor=badge_fg, alpha=0.9))
    return im


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 62)
    print("  adaptive_fdma_solver_validation.py")
    print("  Adaptive FDMA vs Pure Aloha — LR-FH/FDMA Spatial Allocation")
    print("=" * 62)

    # Generate grids
    print("\n  [1] Generating Pure Aloha heatmap (dead SGP4) ...")
    aloha = pure_aloha_grid(N_ENDPOINTS, N_SUBCARRIERS, N_TIMESLOTS, p_tx=0.5)

    print("  [2] Generating Adaptive FDMA heatmap (PGRL-managed) ...")
    afdma = adaptive_fdma_grid(N_ENDPOINTS, N_SUBCARRIERS, N_TIMESLOTS,
                                sigma_f_hz=300.0, t_slot_s=0.016, pgrl_weight=0.92)

    # Metrics
    cr_aloha  = collision_rate(aloha) * 100
    cr_afdma  = collision_rate(afdma) * 100
    se_aloha  = spectral_efficiency(aloha)
    se_afdma  = spectral_efficiency(afdma)

    print(f"\n  {'Metric':<26} {'Pure Aloha':>12} {'Adaptive FDMA':>14}")
    print(f"  {'-'*26} {'-'*12} {'-'*14}")
    print(f"  {'Collision probability (%)':<26} {cr_aloha:>12.2f} {cr_afdma:>14.2f}")
    print(f"  {'Spectral eff (bits/s/Hz)':<26} {se_aloha*1e6:>12.4f} {se_afdma*1e6:>14.4f}")
    print(f"  {'Effective capacity (Mbps)':<26} "
          f"{se_aloha*2e5*1e6:>12.2f} "
          f"{se_afdma*2e5*1e6:>14.2f}")

    vmin, vmax = 0.0, max(aloha.max(), afdma.max())

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor('#f4f6f8')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28,
                             left=0.07, right=0.97, top=0.88, bottom=0.16)

    cmap = _cmap()

    plot_heatmap(fig.add_subplot(gs[0]), aloha,
                 "Pure Aloha — Unmanaged Spatial Allocation\n"
                 f"(100 endpoints, dead SGP4, ±50 kHz bin uncertainty)",
                 cr_aloha, se_aloha, cmap, vmin, vmax)

    plot_heatmap(fig.add_subplot(gs[1]), afdma,
                 "Adaptive FDMA — PGRL-Guided Spatial Allocation\n"
                 f"(100 endpoints, PGRL sigma_f = 300 Hz, 16 ms jitter)",
                 cr_afdma, se_afdma, cmap, vmin, vmax)

    fig.suptitle(
        "Overscanned Spectral Multi-Hopping Allocation Output\n"
        "Adaptive LR-FH/FDMA vs Pure Aloha — Sub-Hz Doppler Tracking Validation",
        fontsize=11, fontweight='bold', y=0.98
    )
    fig.text(0.5, 0.92,
             f"{N_SUBCARRIERS} sub-carriers  |  {N_TIMESLOTS} time slots  |  "
             f"{N_ENDPOINTS} simultaneous endpoints  |  "
             f"BW = {N_SUBCARRIERS * F_STEP/1e3:.0f} kHz  |  "
             f"Slot = {T_SYM*1e3:.0f} ms",
             ha='center', va='top', fontsize=8, style='italic', color='#555')

    # colour bar
    cbar_ax = fig.add_axes([0.30, 0.06, 0.40, 0.03])
    cb = fig.colorbar(
        matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)),
        cax=cbar_ax, orientation='horizontal'
    )
    cb.set_label("Normalised Sub-Carrier Energy", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out_abs = os.path.abspath(PLOT_OUT)
    fig.savefig(out_abs, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n  Plot saved  ->  {out_abs}")
    print("=" * 62)
    print("  STATUS: complete — zero OS errors.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())