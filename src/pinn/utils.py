import numpy as np
import torch

# --- Astrodynamics Constants ---
# Earth Radius (Distance Unit)
R_EARTH = 6378.137  # km
DU = R_EARTH

# Earth Gravitational Parameter
GM_EARTH = 3.986004418e5  # km^3/s^2

# Time Unit
# TU = sqrt(DU^3 / GM)
TU = np.sqrt(DU**3 / GM_EARTH)  # seconds ~806.8s

# J2 Perturbation Constant (Dimensionless)
J2 = 1.08262668e-3

class Normalizer:
    def __init__(self):
        self.DU = DU
        self.TU = TU
        # Scale factors
        self.scale_r = self.DU
        self.scale_v = self.DU / self.TU
        self.scale_a = self.DU / (self.TU ** 2)
        self.scale_t = self.TU

    def normalize_state(self, state_vector):
        """
        Args:
            state_vector: (N, 6) [x, y, z, vx, vy, vz] in km, km/s
        Returns:
            normalized_state: (N, 6) dimensionless
        """
        # Ensure input is numpy or torch
        if isinstance(state_vector, torch.Tensor):
            r = state_vector[:, 0:3] / self.scale_r
            v = state_vector[:, 3:6] / self.scale_v
            return torch.cat([r, v], dim=1)
        else:
            r = state_vector[:, 0:3] / self.scale_r
            v = state_vector[:, 3:6] / self.scale_v
            return np.concatenate([r, v], axis=1)
    
    def normalize_time(self, t):
        """
        Args:
            t: (N,) or (N, 1) in seconds
        Returns:
            t_norm: dimensionless time
        """
        return t / self.scale_t

    def denormalize_state(self, state_vector_norm):
        """
        Args:
            state_vector_norm: (N, 6) dimensionless
        Returns:
            state_vector: (N, 6) in km, km/s
        """
        if isinstance(state_vector_norm, torch.Tensor):
            r = state_vector_norm[:, 0:3] * self.scale_r
            v = state_vector_norm[:, 3:6] * self.scale_v
            return torch.cat([r, v], dim=1)
        else:
            r = state_vector_norm[:, 0:3] * self.scale_r
            v = state_vector_norm[:, 3:6] * self.scale_v
            return np.concatenate([r, v], axis=1)

    def denormalize_time(self, t_norm):
        return t_norm * self.scale_t
