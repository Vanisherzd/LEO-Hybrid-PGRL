import torch
import torch.nn as nn
from src.physics.advanced_forces import (
    compute_j2_j4_gravity, 
    compute_third_body, 
    compute_srp, 
    CelestialEphemeris,
    MU_SUN, MU_MOON
)

class GoldenDynamics(nn.Module):
    def __init__(self, model_nn, ephemeris, drag_model=None, Cr=1.2, Am=0.01):
        super().__init__()
        self.model_nn = model_nn # OrbitalForceNet (Residuals)
        self.ephemeris = ephemeris # CelestialEphemeris object
        self.drag_model = drag_model # Optional
        
        # Learnable Physics Params (Inverse Problem)
        # Using Log space to ensure positivity if trained
        self.log_Cr = nn.Parameter(torch.tensor(np.log(Cr), dtype=torch.float32))
        self.log_Am = nn.Parameter(torch.tensor(np.log(Am), dtype=torch.float32))
        
    def forward(self, t, state):
        """
        compute d(state)/dt
        t: scalar or tensor relative time (s)
        state: (Batch, 6) [rx, ry, rz, vx, vy, vz]
        """
        r = state[:, 0:3]
        v = state[:, 3:6]
        
        # 1. Geopotential (J2-J4)
        a_geo = compute_j2_j4_gravity(r)
        
        # 2. Celestial Bodies (Interpolated)
        # Assuming t is relative to ephemeris start
        r_sun, r_moon = self.ephemeris.get_bodies(t)
        
        a_sun = compute_third_body(r, r_sun, MU_SUN)
        a_moon = compute_third_body(r, r_moon, MU_MOON)
        
        # 3. SRP
        Cr = torch.exp(self.log_Cr)
        Am = torch.exp(self.log_Am)
        a_srp = compute_srp(r, r_sun, Cr, Am)
        
        # 4. Neural Residuals
        # Input state to NN (normalized? usually yes. But here we assume NN handles logic or we norm outside)
        # Wait, OrbitalForceNet usually expects normalized inputs. 
        # But this solver works in PHYSICAL units (km, s).
        # We must manage normalization if NN is trained on normalized data.
        # DESIGN CHOICE: The Golden Solver runs in PHYSICAL space. The NN must adapt or be wrapped.
        # Option A: Normalize inputs for NN, Denormalize outputs.
        
        # Let's assume we pass a wrapper or handle it here?
        # Ideally, we pass simple (r,v) to NN. The NN inside handles scaling.
        # But our current OrbitalForceNet is raw MLP.
        # Let's assume the NN outputs acceleration in km/s^2 DIRECTLY or via scaling.
        # For Phase C, let's keep it simple: NN takes (r,v), outputs a_res.
        
        a_nn = self.model_nn(state) 
        
        # Total Acceleration
        a_total = a_geo + a_sun + a_moon + a_srp + a_nn
        
        return torch.cat([v, a_total], dim=1)

def rk4_step_golden(dynamics, t, state, dt):
    """
    RK4 integrator step using GoldenDynamics module.
    """
    k1 = dynamics(t, state)
    k2 = dynamics(t + 0.5*dt, state + 0.5*dt*k1)
    k3 = dynamics(t + 0.5*dt, state + 0.5*dt*k2)
    k4 = dynamics(t + dt, state + dt*k3)
    
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

import numpy as np # For log init
