import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacfwd

# Earth constants
MU_EARTH = 3.986004418e5
R_EARTH  = 6378.137

# Neural-net normalisation scales (must match dataset.py)
DU = 10000.0   # 1 unit = 10 000 km
VU = 10.0      # 1 unit = 10 km/s
TU = 16661.0   # 1 unit = 16 661 s

OE_MEAN = torch.tensor([[6778.137, 0.001, 0.925, 0.0, 0.0, 0.0]])
OE_STD  = torch.tensor([[1.0, 0.001, 0.35, 2.0, 2.0, 2.0]])


class PINNTotalLoss(nn.Module):
    """
    Combined loss for satellite trajectory PINN.

    Loss = w_data * L_data
         + w_physics * L_physics
         + w_energy * L_energy
         + w_angmom * L_angmom

    L_data    : MSE between network output and ground-truth state (normalised units)
    L_physics : MSE between autograd-computed acceleration and Newtonian target
                (non-dimensionalised to NN output scale, no 9e20 explosion)
    L_energy  : orbital-energy conservation (bounded, O(1))
    L_angmom  : angular-momentum magnitude conservation (bounded, O(1))
    """

    def __init__(
        self,
        w_data: float = 1.0,
        w_physics: float = 0.01,
        w_energy: float = 0.0,
        w_angmom: float = 0.0,
        position_weight: float = 1.0,
        velocity_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.w_data     = w_data
        self.w_physics  = w_physics
        self.w_energy   = w_energy
        self.w_angmom   = w_angmom
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

    # ------------------------------------------------------------------ #
    #  Autograd physics helpers                                          #
    # ------------------------------------------------------------------ #
    def _physics_gradients(
        self,
        model: nn.Module,
        t_norm: torch.Tensor,
        oe_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predicted velocity and acceleration via double automatic
        differentiation through the network forward pass.

        All outputs are in NON-DIMENSIONAL units (DU/TU for velocity,
        DU/TU^2 for acceleration) so they sit in the same O(1) range as
        the NN outputs [-0.7, 0.7].

        Args:
            model   : TrajectoryPINN
            t_norm  : (batch, 1) time — already normalised by TU
            oe_norm : (batch, 6) orbital elements — z-score normalised

        Returns:
            v_autograd : (batch, 3) velocity in DU/TU
            a_autograd : (batch, 3) acceleration in DU/TU^2
        """
        B = t_norm.shape[0]
        b_idx = torch.arange(B, device=t_norm.device)

        # d(pos) / d(t)  →  velocity in DU/TU
        def _model_pos(t_):
            return model(t_, oe_norm)[..., :3]          # (B, 3)

        full_jac = jacfwd(_model_pos)(t_norm)          # (B, 3, B, 1)
        v_autograd = full_jac[b_idx, :, b_idx, 0]       # (B, 3)

        # d2(pos) / d(t)2  →  acceleration in DU/TU^2
        def _model_pos_jac(t_):
            return jacfwd(_model_pos)(t_)               # (B, 3, B, 1)

        full_hess = jacfwd(_model_pos_jac)(t_norm)      # (B, 3, B, 1, B, 1)
        a_autograd = full_hess[b_idx, :, b_idx, 0, b_idx, 0]  # (B, 3)

        return v_autograd, a_autograd

    # ------------------------------------------------------------------ #
    #  Orbital mechanics helpers (physical units → normalised units)    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _orbital_energy(
        pos_km: torch.Tensor, vel_kms: torch.Tensor
    ) -> torch.Tensor:
        """Specific orbital energy in km^2/s^2."""
        r = torch.norm(pos_km, dim=-1).clamp(min=1e-8)
        v2 = torch.sum(vel_kms ** 2, dim=-1)
        return v2 / 2.0 - MU_EARTH / r

    @staticmethod
    def _angular_momentum_magnitude(
        pos_km: torch.Tensor, vel_kms: torch.Tensor
    ) -> torch.Tensor:
        """Specific angular momentum magnitude in km^2/s."""
        hx = pos_km[:, 1] * vel_kms[:, 2] - pos_km[:, 2] * vel_kms[:, 1]
        hy = pos_km[:, 2] * vel_kms[:, 0] - pos_km[:, 0] * vel_kms[:, 2]
        hz = pos_km[:, 0] * vel_kms[:, 1] - pos_km[:, 1] * vel_kms[:, 0]
        return torch.sqrt(hx**2 + hy**2 + hz**2)

    # ------------------------------------------------------------------ #
    #  Bounded autograd-physics entry point                              #
    # ------------------------------------------------------------------ #
    def forward_with_autograd_physics(
        self,
        model: nn.Module,
        t_norm: torch.Tensor,
        oe_norm: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss using bounded non-dimensional physics.

        The autograd path differentiates through the network to get
        a_ag in DU/TU^2.  The target acceleration is converted to the
        SAME units so the MSE is O(1) instead of 9e20.

        Args:
            model        : TrajectoryPINN
            t_norm       : (batch, 1) normalised time (real_s / TU)
            oe_norm      : (batch, 6) normalised orbital elements
            target_state : (batch, 6) normalised ground-truth [pos, vel]

        Returns:
            (total_loss, metrics_dict)
        """
        # ---- 1. Data loss (normalised MSE) ----
        pred_state_full = model(t_norm, oe_norm)            # (batch, 6) or (batch, 12)
        pred_state = pred_state_full[..., :6]                # always take first 6 (mean)
        l_data = F.mse_loss(pred_state, target_state)

        # ---- 2. Physics loss (autograd acceleration, bounded) ----
        v_ag, a_ag = self._physics_gradients(model, t_norm, oe_norm)
        #   v_ag is in DU/TU;  a_ag is in DU/TU^2

        # Target acceleration in the SAME units (DU/TU^2):
        #   a_physical = -MU * pos_km / |pos_km|^3   [km/s^2]
        #   a_nominal  = a_physical / (DU/TU^2)      [DU/TU^2]
        pos_km  = pred_state[..., :3] * DU            # DU → km
        r_mag   = torch.norm(pos_km, dim=-1, keepdim=True).clamp(min=1e-8)
        a_phys  = -MU_EARTH * pos_km / (r_mag ** 3)   # km/s^2
        AU      = DU / (TU ** 2)                      # ~0.0006 km/s^2 per DU/TU^2
        a_target = a_phys / AU                        # → DU/TU^2

        # Newton error in NN's own derivative space — bounded O(1–100)
        l_physics = F.mse_loss(a_ag, a_target)

        # ---- 3. Energy conservation loss (bounded, non-dim) ----
        vel_km  = pred_state[..., 3:] * VU            # km/s
        energy  = self._orbital_energy(pos_km, vel_km)            # (batch,) km^2/s^2
        a_oe    = oe_norm[:, 0:1] * OE_STD[:, 0] + OE_MEAN[:, 0]  # (batch, 1) km
        e_oe    = oe_norm[:, 1:2] * OE_STD[:, 1] + OE_MEAN[:, 1]  # (batch, 1)
        E_ref   = (-MU_EARTH / (2.0 * a_oe)).squeeze(-1)          # (batch,)
        l_energy = F.mse_loss(energy / (VU**2), E_ref / (VU**2))

        # ---- 4. Angular momentum loss (bounded, non-dim) ----
        h_mag   = self._angular_momentum_magnitude(pos_km, vel_km)  # (batch,) km^2/s
        h_ref   = torch.sqrt(MU_EARTH * a_oe * (1.0 - e_oe**2)).squeeze(-1)  # (batch,)
        l_angmom = F.mse_loss(h_mag / (VU * DU), h_ref / (VU * DU))

        # ---- 5. Combine ----
        total_loss = (
            self.w_data    * l_data
            + self.w_physics * l_physics
            + self.w_energy  * l_energy
            + self.w_angmom  * l_angmom
        )

        # Metrics
        pos_rmse_m = (
            torch.norm(pred_state[:, :3] - target_state[:, :3], dim=-1).mean()
            * DU * 1000
        ).item()
        below_5m = (
            (torch.norm(pred_state[:, :3] - target_state[:, :3], dim=-1) * DU * 1000 < 5.0)
            .float()
            .mean()
            .item()
            * 100
        )

        metrics = {
            "total_loss":      total_loss.item(),
            "data_loss":       l_data.item(),
            "physics_loss":    l_physics.item(),
            "energy_loss":     l_energy.item(),
            "angmom_loss":     l_angmom.item(),
            "pos_rmse_m":      pos_rmse_m,
            "below_5m_pct":    below_5m,
            "v_autograd_norm": v_ag.norm().item(),
            "a_autograd_norm": a_ag.norm().item(),
            "a_target_norm":   a_target.norm().item(),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------ #
    #  Legacy forward (backward-compatible with existing train_epoch)    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred_state: torch.Tensor,
        gt_state: torch.Tensor,
        t_norm: torch.Tensor,
        oe_norm: torch.Tensor,
    ) -> dict:
        """
        Compatible with existing callers that pass
        (pred_state, gt_state, t_norm, oe_norm).

        Physics terms computed WITHOUT double-autograd overhead
        (energy + angmom only; no Jacobian needed here).
        All bounded to O(1) ranges.
        """
        # ---- 1. Data loss ----
        l_data = F.mse_loss(pred_state, gt_state)

        # ---- 2. Bounded physics: energy + angmom ----
        pos_km = pred_state[..., :3] * DU            # km
        vel_km = pred_state[..., 3:] * VU            # km/s

        energy   = self._orbital_energy(pos_km, vel_km)              # (batch,)
        a_oe     = oe_norm[:, 0:1] * OE_STD[:, 0] + OE_MEAN[:, 0]    # (batch, 1) km
        e_oe     = oe_norm[:, 1:2] * OE_STD[:, 1] + OE_MEAN[:, 1]    # (batch, 1)
        E_ref    = (-MU_EARTH / (2.0 * a_oe)).squeeze(-1)            # (batch,)
        l_energy = F.mse_loss(energy / (VU**2), E_ref / (VU**2))

        h_mag    = self._angular_momentum_magnitude(pos_km, vel_km)  # (batch,)
        h_ref    = torch.sqrt(MU_EARTH * a_oe * (1.0 - e_oe**2)).squeeze(-1)  # (batch,)
        l_angmom = F.mse_loss(h_mag / (VU * DU), h_ref / (VU * DU))

        # Physics residual = energy conservation (bounded proxy)
        l_physics = l_energy

        total_loss = (
            self.w_data    * l_data
            + self.w_physics * l_physics
            + self.w_energy  * l_energy
            + self.w_angmom  * l_angmom
        )

        pos_rmse_m = (
            torch.norm(pred_state[:, :3] - gt_state[:, :3], dim=-1).mean()
            * DU * 1000
        ).item()
        below_5m = (
            (torch.norm(pred_state[:, :3] - gt_state[:, :3], dim=-1) * DU * 1000 < 5.0)
            .float()
            .mean()
            .item()
            * 100
        )

        return {
            "total_loss":   total_loss.item(),
            "data_loss":    l_data.item(),
            "physics_loss": l_physics.item(),
            "energy_loss":  l_energy.item(),
            "angmom_loss":  l_angmom.item(),
            "pos_rmse_m":   pos_rmse_m,
            "below_5m_pct": below_5m,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian NLL Loss for uncertainty head
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for 6D state prediction.

    Args:
        pred_mean:    (batch, 6) — predicted mean
        pred_log_var: (batch, 6) — predicted log-variance (clamped by caller)
        target:       (batch, 6) — ground-truth state
        reduction:    'mean' | 'sum' | 'none'

    Returns:
        NLL scalar (or per-sample if reduction='none')
    """
    # var = exp(log_var) — already clamped outside if needed
    var = torch.exp(pred_log_var)
    # NLL = 0.5 * [log(2π * var) + (y - μ)² / var]
    nll = 0.5 * (torch.log(2 * math.pi * var) + (pred_mean - target) ** 2 / var)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll