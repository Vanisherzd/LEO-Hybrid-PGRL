import numpy as np
import os
import datetime
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.pinn.utils import Normalizer
import requests

DATA_PATH = os.path.join("data", "precise_training_data.npz")
NORAD_ID = 42920

def fetch_tle(norad_id):
    try:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
        response = requests.get(url, timeout=10)
        lines = response.text.strip().splitlines()
        l1, l2 = None, None
        for l in lines:
            if l.startswith('1 '): l1 = l
            elif l.startswith('2 '): l2 = l
        return l1, l2
    except:
        return ("1 42920U 17049A   24029.54477817  .00004500  00000-0  17596-3 0  9990",
                "2 42920  98.2435 112.5921 0001222  95.2341 264.9123 14.50023412341234")

def synthesize_high_fidelity_data():
    print("--- Synthesizing High-Fidelity Hybrid Benchmark Data ---")
    l1, l2 = fetch_tle(NORAD_ID)
    satellite = Satrec.twoline2rv(l1, l2, WGS72)
    
    # Use modern epoch (Now)
    start_dt = datetime.datetime.now(datetime.timezone.utc)
    start_epoch = start_dt.timestamp()
    
    t_raw = np.arange(0, 6000, 10) # 100 minutes, 10s steps (600 points)
    
    sgp4_r = []
    sgp4_v = []
    
    print(f"Generating SGP4 baseline for {len(t_raw)} points...")
    for t_sec in t_raw:
        curr_dt = start_dt + datetime.timedelta(seconds=float(t_sec))
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, 
                      curr_dt.hour, curr_dt.minute, curr_dt.second)
        e, r, v = satellite.sgp4(jd, fr)
        sgp4_r.append(r)
        sgp4_v.append(v)
        
    sgp4_r = np.array(sgp4_r)
    sgp4_v = np.array(sgp4_v)
    
    # Create "Truth" (SP3) = SGP4 + Residuals (Unmodeled Physics)
    # residual = bias + 1km oscillations
    t_norm = t_raw / 3600.0 # hours
    res_x = 0.5 * np.sin(2 * np.pi * t_norm * 2) + 0.3 * np.cos(2 * np.pi * t_norm * 5)
    res_y = 0.4 * np.cos(2 * np.pi * t_norm * 1.5) + 0.1 * np.sin(2 * np.pi * t_norm * 10)
    res_z = 0.6 * np.sin(2 * np.pi * t_norm * 3) + 1.2 # Bias
    
    residuals = np.stack([res_x, res_y, res_z], axis=1) # km
    
    states_sp3_r = sgp4_r + residuals
    states_sp3_v = sgp4_v # Assume velocity residuals are negligible for this demo
    states_sp3 = np.hstack([states_sp3_r, states_sp3_v])
    
    # Normalize
    normalizer = Normalizer()
    t_norm_final = normalizer.normalize_time(t_raw)
    states_norm = normalizer.normalize_state(states_sp3)
    
    np.savez(DATA_PATH, t=t_norm_final, states=states_norm, t_raw=t_raw, states_raw=states_sp3, start_epoch=start_epoch)
    print(f"Synthesized data saved to {DATA_PATH}")
    print(f"Synthetic Residual Mean: {np.mean(np.linalg.norm(residuals, axis=1)):.4f} km")

if __name__ == "__main__":
    synthesize_high_fidelity_data()
