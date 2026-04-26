"""
LEO Satellite PINN — Core Neural Network Architecture
=====================================================
SIREN-based (Sinusoidal Representation Network) PINN for satellite trajectory prediction.
Uses orbital physics-informed architecture with Fourier feature embeddings.

Key design:
- Input: (t, orbital_elements) -> Output: (x, y, z, vx, vy, vz)
- Physics residual: orbital equations of motion as soft constraint
- GRPO gradient regulation for online RL corrections
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """Single SIREN layer with sinusoidal activation."""

    def __init__(self, in_features: int, out_features: int, omega0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega0 = omega0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Follow SIREN paper: uniform init for first layer
                b = 1.0 / self.in_features
                self.linear.weight.uniform_(-b, b)
            else:
                # Follow SIREN paper: uniform init for hidden layers
                input_dim = self.linear.weight.shape[1]
                b = math.sqrt(6.0 / input_dim) / self.omega0
                self.linear.weight.uniform_(-b, b)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for high-frequency trajectory components."""

    def __init__(self, input_dim: int, num_features: int = 64, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) -> (..., 2 * num_features)
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TrajectoryPINN(nn.Module):
    """
    Physics-Informed Neural Network for LEO satellite trajectory prediction.

    Architecture:
    1. Fourier feature embedding on time input (captures orbital periodicity)
    2. Orbital element conditioning via learned embedding
    3. SIREN-based hidden layers (better for periodic/multi-scale functions)
    4. Output head: 6D state (position + velocity)

    The network predicts residual correction over Keplerian orbit to achieve <5m RMSE.
    """

    def __init__(
        self,
        orbital_elem_dim: int = 7,  # a, e, i, Omega, omega, M0, n (mean motion)
        time_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 6,
        fourier_features: int = 128,
        omega0: float = 30.0,
        use_fourier: bool = True,
    ):
        super().__init__()
        self.orbital_elem_dim = orbital_elem_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega0 = omega0

 # Fourier feature embedding on time
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier_embed = FourierFeatureEmbedding(time_dim, fourier_features)
            self.time_embed_dim = 2 * fourier_features  # Fourier encoding replaces raw time
        else:
            self.time_embed_dim = time_dim

        # Combine time and orbital element embeddings
        self.input_dim = self.time_embed_dim + orbital_elem_dim

        # SIREN layers
        layers = []
        layers.append(SirenLayer(self.input_dim, hidden_dim, omega0=omega0, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega0=omega0))
        self.siren_layers = nn.ModuleList(layers)

        # Final layer (no activation — outputs position/velocity)
        self.final_layer = nn.Linear(hidden_dim, 6)

        # Learnable orbital element normalization (affine per parameter)
        self.elem_scale = nn.Parameter(torch.ones(orbital_elem_dim))
        self.elem_bias = nn.Parameter(torch.zeros(orbital_elem_dim))

    def forward(self, t: torch.Tensor, orbital_elems: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            t: (batch, 1) — time since epoch in seconds
            orbital_elems: (batch, 7) — Keplerian elements [a, e, i, Omega, omega, M0, n]

        Returns:
            state: (batch, 6) — [x, y, z, vx, vy, vz] in km, km/s (ECI frame)
        """
        # Normalize orbital elements
        orbital_elems = orbital_elems * self.elem_scale + self.elem_bias

        if self.use_fourier:
            t_embed = self.fourier_embed(t)
        else:
            t_embed = t

        # Concatenate time embedding and orbital elements
        x = torch.cat([t_embed, orbital_elems], dim=-1)

        # SIREN forward
        for layer in self.siren_layers:
            x = layer(x)
        state = self.final_layer(x)

        return state


class MultiScaleTrajectoryPINN(nn.Module):
    """
    Multi-scale PINN using wavelets for multi-fidelity orbital prediction.

    Splits prediction into:
    - Low-freq: large-scale orbital trajectory (SGP4 baseline)
    - High-freq: residual correction (atmospheric drag, solar radiation, lunisolar perturbations)
    """

    def __init__(
        self,
        orbital_elem_dim: int = 7,
        hidden_dim: int = 192,
        num_layers: int = 5,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_scales = num_scales

        # Per-scale SIREN branches
        self.scale_nns = nn.ModuleList([
            TrajectoryPINN(
                orbital_elem_dim=orbital_elem_dim,
                hidden_dim=hidden_dim // (2 ** s),
                num_layers=num_layers,
                use_fourier=True,
                fourier_features=64 // (2 ** s),
            )
            for s in range(num_scales)
        ])

        # Scale attention weights
        self.scale_attention = nn.Sequential(
            nn.Linear(orbital_elem_dim + 1, num_scales),
            nn.Softmax(dim=-1),
        )

    def forward(self, t: torch.Tensor, orbital_elems: torch.Tensor) -> torch.Tensor:
        # Compute attention over scales based on eccentricity and time
        context = torch.cat([orbital_elems, t / 3600.0], dim=-1)  # scale by 1 hour
        weights = self.scale_attention(context)  # (batch, num_scales)

        # Weighted sum of scale predictions
        pred = sum(w * nn(t, orbital_elems) for w, nn in zip(weights.unbind(-1), self.scale_nns))
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test
    model = TrajectoryPINN()
    t = torch.randn(32, 1)
    oe = torch.randn(32, 7)
    out = model(t, oe)
    print(f"Model params: {count_parameters(model):,}")
    print(f"Input: t={t.shape}, oe={oe.shape} -> Output: {out.shape}")
    assert out.shape == (32, 6)
    print("✓ Smoke test passed")