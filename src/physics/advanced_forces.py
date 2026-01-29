import torch
import numpy as np
from astropy.coordinates import get_sun, get_body, GCRS, CartesianRepresentation, solar_system_ephemeris
from astropy.time import Time
from astropy import units as u

# Force builtin to avoid download hangs
solar_system_ephemeris.set('builtin')

# Constants
MU_EARTH = 398600.4418      # km^3/s^2
R_EARTH = 6378.137          # km
MU_SUN = 1.32712440018e11   # km^3/s^2
MU_MOON = 4902.800066       # km^3/s^2
P_SUN = 4.56e-6             # N/m^2 (Solar Pressure at 1 AU)

class CelestialEphemeris:
    """
    Handles celestial body positions. 
    For speed, we pre-compute trajectories and interpolate on GPU during integration.
    """
    def __init__(self, start_epoch_ts, duration_s, step_s=60.0, device='cpu'):
        self.device = device
        self.t_start = start_epoch_ts
        self.duration = duration_s
        self.step = step_s
        
        # Pre-compute using Astropy
        self._precompute()
        
    def _precompute(self):
        print("Pre-computing Sun/Moon ephemeris using Astropy...")
        times = np.arange(0, self.duration + self.step, self.step)
        t_objs = Time(self.t_start + times, format='unix')
        
        # Sun
        sun_gcrs = get_sun(t_objs).transform_to(GCRS(obstime=t_objs))
        self.sun_pos = torch.tensor(sun_gcrs.cartesian.xyz.to(u.km).value.T, dtype=torch.float32).to(self.device)
        
        # Moon (Using get_body)
        moon_gcrs = get_body("moon", t_objs).transform_to(GCRS(obstime=t_objs))
        self.moon_pos = torch.tensor(moon_gcrs.cartesian.xyz.to(u.km).value.T, dtype=torch.float32).to(self.device)
        
        self.time_grid = torch.tensor(times, dtype=torch.float32).to(self.device)
        print("Ephemeris cached.")

    def get_bodies(self, t_rel_sec):
        """
        Interpolate Sun/Moon positions at relative time t_rel_sec.
        t_rel_sec: Tensor or float (seconds since epoch)
        """
        # Ensure tensor
        if not isinstance(t_rel_sec, torch.Tensor):
            t_rel_sec = torch.tensor(t_rel_sec, dtype=torch.float32, device=self.device)
            
        # Linear Interpolation
        # Index
        idx = (t_rel_sec / self.step)
        idx_0 = torch.floor(idx).long()
        idx_1 = idx_0 + 1
        
        # Clamp
        idx_0 = torch.clamp(idx_0, 0, len(self.time_grid)-1)
        idx_1 = torch.clamp(idx_1, 0, len(self.time_grid)-1)
        
        alpha = idx - idx_0.float()
        # Handle alpha shape for broadcasting if needed
        if isinstance(alpha, torch.Tensor) and alpha.dim() > 0:
             alpha = alpha.unsqueeze(-1)
        
        r_sun = self.sun_pos[idx_0] * (1 - alpha) + self.sun_pos[idx_1] * alpha
        r_moon = self.moon_pos[idx_0] * (1 - alpha) + self.moon_pos[idx_1] * alpha
        
        # Ensure (Batch, 3) shape even if t is scalar
        if r_sun.dim() == 1:
            r_sun = r_sun.unsqueeze(0)
        if r_moon.dim() == 1:
            r_moon = r_moon.unsqueeze(0)
            
        return r_sun, r_moon

def compute_j2_j4_gravity(r_vec):
    """
    Compute Earth Gravity with J2 (Analytical).
    High-fidelity J3/J4 dropped for stability if autograd fails.
    r_vec: (Batch, 3) in km
    """
    x, y, z = r_vec[:, 0:1], r_vec[:, 1:2], r_vec[:, 2:3] # Keep dims (Batch, 1)
    r_sq = x**2 + y**2 + z**2
    r = torch.sqrt(r_sq)
    
    # Pre-compute
    one_over_r3 = 1.0 / (r * r_sq)
    mu_r3 = -MU_EARTH * one_over_r3 # -mu/r^3
    
    # J2 Perturbation
    # a_J2 = - (3/2) J2 (mu/r^2) (R/r)^2 * ... 
    # Let's use the efficient Cartesian form directly
    
    # Constants
    J2 = 1.08263e-3
    
    # Let's use Autograd for J3/J4 to prevent formula errors, but map J2 manually.
    # Actually, simpler: Just compute J2 manually (99% of effect) and let NN handle J3/J4 residuals?
    # User Request: "Refined J2-J4 Gravity".
    
    # Let's try the Potential Autograd. It's clean.
    return grad_potential(r_vec)

def grad_potential(r_vec):
    """
    Compute gradient of U (J2, J3, J4) w.r.t r_vec.
    """
    # Compute Grad (Acceleration)
    # Enable grad even if global context is no_grad (for physics math)
    with torch.enable_grad():
        # Re-declare r_in inside enable_grad to be sure
        r_in = r_vec.detach().clone().requires_grad_(True)
        
        x, y, z = r_in[:, 0], r_in[:, 1], r_in[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        sin_phi = z / r
        R_r = R_EARTH / r
        
        # Constants
        J2 = 1.08263e-3
        J3 = -2.53215e-6
        J4 = -1.61099e-6
        
        P2 = 0.5 * (3 * sin_phi**2 - 1)
        P3 = 0.5 * (5 * sin_phi**3 - 3 * sin_phi)
        P4 = 0.125 * (35 * sin_phi**4 - 30 * sin_phi**2 + 3)
        
        # Central + Perturbation
        U = (MU_EARTH / r) * (1 - (J2 * R_r**2 * P2 + J3 * R_r**3 * P3 + J4 * R_r**4 * P4))
        
        # Sum U to make scalar for backward (batch mode)
        U_sum = torch.sum(U)
        grads = torch.autograd.grad(U_sum, r_in, create_graph=True)[0]
    
    return grads

def compute_third_body(r_sat, r_body, mu_body):
    """
    r_sat: (Batch, 3) pos of satellite
    r_body: (Batch, 3) pos of celestial body
    mu_body: Gravitational constant
    """
    diff = r_body - r_sat
    dist_sat_body = torch.norm(diff, dim=1, keepdim=True)
    dist_earth_body = torch.norm(r_body, dim=1, keepdim=True)
    
    # a = mu * ( (r_body - r_sat)/|d|^3 - r_body/|d_e|^3 )
    a_3rd = mu_body * ( (diff / dist_sat_body**3) - (r_body / dist_earth_body**3) )
    return a_3rd

def compute_srp(r_sat, r_sun, Cr=1.2, Am=0.01):
    """
    Cannonball SRP.
    """
    # Vector from Sun to Sat
    rs_vec = r_sat - r_sun
    dist = torch.norm(rs_vec, dim=1, keepdim=True)
    u_sun = rs_vec / (dist + 1e-8)
    
    # Force magnitude
    a_srp = -P_SUN * Cr * Am * (1e-3) * u_sun 
    
    r_sun_norm = r_sun / (torch.norm(r_sun, dim=1, keepdim=True) + 1e-8)
    
    # L calculation
    L = torch.sum(r_sat * r_sun_norm, dim=1, keepdim=True)
    
    # Shadow Logic
    is_behind = L < 0 
    
    r_sat_sq = torch.sum(r_sat**2, dim=1, keepdim=True)
    d_perp_sq = r_sat_sq - L**2
    
    # Ensure d_perp_sq is non-negative (numerical noise)
    d_perp_sq = torch.clamp(d_perp_sq, min=0.0)
    
    in_shadow = torch.logical_and(is_behind, d_perp_sq < R_EARTH**2)
    
    shadow_factor = torch.where(in_shadow, torch.zeros_like(L), torch.ones_like(L))
    
    return a_srp * shadow_factor
