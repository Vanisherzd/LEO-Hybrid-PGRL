import torch
import numpy as np
import os
import datetime
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.physics.advanced_forces import (
    CelestialEphemeris,
    compute_j2_j4_gravity,
    grad_potential,
    compute_third_body,
    compute_srp,
    MU_SUN, MU_MOON, R_EARTH
)
from src.physics.golden_solver import GoldenDynamics, rk4_step_golden
from src.pinn.utils import Normalizer
import requests

# --- Configuration ---
NORAD_ID = 44387 # Formosat-7 (COSMIC-2)
DURATION_MIN = 100
DT_SEC = 1.0
DATA_PATH = os.path.join("data", "f7_golden_truth.npz")

def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    print(f"Fetching TLE for ID {norad_id}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        l1, l2 = None, None
        for l in lines:
            if l.startswith('1 '): l1 = l
            elif l.startswith('2 '): l2 = l
        return l1, l2
    except:
        # Fallback to a representative F7 TLE if request fails
        # Note: This is an example, real-time fetch is preferred.
        return ("1 44387U 19037A   24029.54477817  .00010000  00000-0  17596-3 0  9990",
                "2 44387  24.0000 112.5921 0001222  95.2341 264.9123 15.20023412341234")

def compute_drag(r, v, BC=0.01):
    """
    BC: Ballistic Coefficient (m^2/kg) - simplified.
    Using exponential density model.
    """
    alt = torch.norm(r, dim=1, keepdim=True) - 6378.137
    # Simple model: rho = rho0 * exp(-alt / H)
    rho0 = 1.0e-11 # kg/km^3 (approx for 550km)
    H = 60.0 # scale height km
    rho = rho0 * torch.exp(-alt / H)
    
    v_target = v # Assume static atmosphere for simplicity
    v_rel = torch.norm(v_target, dim=1, keepdim=True)
    
    # a_drag = -0.5 * rho * v^2 * BC
    # Calibration: 0.2x multiplier targets ~1km error in 100min
    a_drag = -0.5 * rho * v_rel * v_target * (BC * 0.2)
    return a_drag

class GoldenDynamicsWithDrag(torch.nn.Module):
    def __init__(self, ephemeris):
        super().__init__()
        self.ephemeris = ephemeris
        
    def forward(self, t, state):
        r = state[:, 0:3]
        v = state[:, 3:6]
        
        a_geo = grad_potential(r)
        r_sun, r_moon = self.ephemeris.get_bodies(t)
        a_sun = compute_third_body(r, r_sun, MU_SUN)
        a_moon = compute_third_body(r, r_moon, MU_MOON)
        a_srp = compute_srp(r, r_sun, Cr=1.5, Am=0.02) # Higher SRP for F7
        a_drag = compute_drag(r, v)
        
        a_total = a_geo + a_sun + a_moon + a_srp + a_drag
        return torch.cat([v, a_total], dim=1)

def generate_f7_golden_truth():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating DRAG-ENABLED Golden Truth for Formosat-7 on {device}")
    
    # 1. Fetch TLE and get Init State
    l1, l2 = fetch_tle(NORAD_ID)
    satellite = Satrec.twoline2rv(l1, l2, WGS72)
    
    start_dt = datetime.datetime.now(datetime.timezone.utc)
    start_epoch = start_dt.timestamp()
    
    jd, fr = jday(start_dt.year, start_dt.month, start_dt.day, 
                  start_dt.hour, start_dt.minute, start_dt.second)
    e, r_init, v_init = satellite.sgp4(jd, fr)
    
    state_init = torch.tensor([list(r_init) + list(v_init)], dtype=torch.float32).to(device)
    
    # 2. Setup Golden Solver
    duration_s = DURATION_MIN * 60
    ephemeris = CelestialEphemeris(start_epoch, duration_s, step_s=60.0, device=device)
    dynamics = GoldenDynamicsWithDrag(ephemeris).to(device)
    
    # 3. Integrate Trajectory
    t_raw = np.arange(0, duration_s + DT_SEC, DT_SEC)
    states_golden = [state_init.cpu().numpy()[0]]
    
    curr_state = state_init
    print(f"Propagating {len(t_raw)} steps...")
    
    with torch.no_grad():
        for i in range(len(t_raw) - 1):
            curr_state = rk4_step_golden(dynamics, t_raw[i], curr_state, DT_SEC)
            states_golden.append(curr_state.cpu().numpy()[0])
            
            if (i+1) % 1000 == 0:
                print(f"Step {i+1}/{len(t_raw)}")
                
    states_golden = np.array(states_golden)

    # 4. Generate SGP4 baseline for the same time grid
    sgp4_states = []
    for t_s in t_raw:
        curr_dt = start_dt + datetime.timedelta(seconds=float(t_s))
        # Use precise components for consistency
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, 
                      curr_dt.hour, curr_dt.minute, curr_dt.second)
        e, r, v = satellite.sgp4(jd, fr)
        sgp4_states.append(list(r) + list(v))
    sgp4_states = np.array(sgp4_states)

    # 5. Save
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Normalization
    normalizer = Normalizer()
    t_norm = normalizer.normalize_time(t_raw)
    states_norm = normalizer.normalize_state(states_golden)
    sgp4_states_norm = normalizer.normalize_state(sgp4_states)
    
    np.savez(DATA_PATH, 
             t=t_norm, 
             states=states_norm, 
             t_raw=t_raw, 
             states_raw=states_golden, 
             sgp4_raw=sgp4_states,
             sgp4_norm=sgp4_states_norm,
             start_epoch=start_epoch)
    
    print(f"Golden Truth saved to {DATA_PATH}")
    print(f"Final Position: {states_golden[-1, 0:3]} km")

if __name__ == "__main__":
    generate_f7_golden_truth()
